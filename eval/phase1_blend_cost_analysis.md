# 为什么短 prompt 下 CacheBlend 更差 —— 完整分析

> 第一阶段效率对比(10 题 × 2 轮)的核心发现:CacheBlend 在复用段较短时
> **慢于**原生 vLLM prefix cache,复用段足够长时反超。本文给出机制分解、
> 全部证据链与改进建议。
>
> 实验环境:Qwen3-30B-A3B-Instruct-2507,TP=4 @ GPU 1,3,4,5,vLLM 0.18.0,
> LMCache(分支版,blend + blend_store_generated),CUDA graphs 两模式均开启。
> 复现步骤见 [AB_EXPERIMENT.md](AB_EXPERIMENT.md)。

---

## 一、两轮实验结果回顾

### 轮 1:小上下文(复用段 ~1k token,demo 默认压缩式摘要)

| | A(CacheBlend) | B(原生 prefix cache) |
|---|---|---|
| 端到端均值/题 | 91.3s | **64.2s**(B 快) |
| 服务端 prefill 均值/请求 | 0.310s | **0.171s** |
| 服务端 decode 均值/请求 | 0.645s | 0.892s(输出更长所致,按 token 相当) |
| blend(KV加载+融合) | **501ms/次**(50 次,均值命中 919 tok) | 无此阶段 |
| GPU 干扰(邻卡均值 util) | 38.2% | 52.6%(B 干扰更大仍更快) |

### 轮 2:大上下文(详细摘要模式,复用段 8-18k token)

| | A(CacheBlend) | B(原生 prefix cache) |
|---|---|---|
| 端到端均值/题 | **93.6s**(A 快 12%) | 106.3s |
| prefill >1s 的请求数 | **3** | **10**(含 1-1.5s 7 个、2-5s 2 个) |
| blend | 542ms/次(57 次,均值命中 7918 tok) | 无此阶段 |
| GPU 干扰 | 0.0% | 0.4%(两窗口均干净) |
| decode 占推理总时间 | 93% | 95% |

---

## 二、机制:blend 成本模型(107 个真实事件回归)

把两轮全部 blend 计时日志(`Blend (KV load + fuse) took X ms for N hit tokens`,
只取 Worker_TP0)做线性回归:

```
blend_ms = 455 + 0.0146 × hit_tokens        (n=107, R² = 0.52)
```

| 复用规模 | 实测均值 | 原生 prefill 同规模(估) | 净效果 |
|---|---|---|---|
| ~550 tok(n=63) | 481ms | ~100-200ms | **亏 ~300ms/请求** |
| ~2.9k(n=14) | 525ms | ~300-500ms | 接近平衡 |
| ~8.9k(n=16) | 509ms | ~1-2s | 赚 0.5-1.5s |
| ~20k(n=14) | 728ms | ~2-5s | 赚 1.5-4s |

**核心事实:blend 成本几乎是一条水平线 —— 固定开销 ~455ms,边际成本仅
14.6μs/token。** 短 prompt 时省下的 prefill 计算(几十 ms)远小于固定开销;
**盈亏交叉点约 4-6k token**。

### ⚠️ 2026-06-12 修正:分项计时器实测推翻了下面的归因猜测

给 blender 加了分项计时(`Blend timing: load=…, compute=…`,
`LMCACHE_BLEND_TIMER_SYNC=1` 做 CUDA 同步精确归因)。
完整 10 题归因跑(`ab_blend_10_split_timing`,n=57 个 blend,零失败,
已剔除首个编译预热点)的回归:

```
load_ms    = 347 + 2.4μs/token    (R: 搬运侧,固定开销几乎全在这)
compute_ms = 180 + 12.1μs/token   (重算侧:固定 ≈ 48层 eager 启动,边际 = GPU 计算)
```

| 命中规模 | n | 搬运 load | 重算 compute | 搬运占比 |
|---|---|---|---|---|
| <1k | 18 | 318ms | 216ms | 60% |
| 1-4k | 8 | 389ms | 206ms | 65% |
| 4-10k | 17 | 393ms | 233ms | 63% |
| >10k | 14 | 380ms | 429ms | 47% |

(sync 模式自身的测量开销约 +70ms/次,分摊在两相;async 边界模式总固定
~455ms 与 sync 总固定 527ms 的差即此。)

**固定开销的大头在"搬运"侧(~350ms,与 token 数无关),不是 MoE 重算编排。**
搬运慢不是带宽问题(2.4μs/token 边际 ≈ ~10GB/s 有效带宽,尚可),而是
retrieve_layer 逐层逐段的小拷贝/分配/同步的机械成本——60 token 和 20k token
的 load 时间几乎一样。compute 侧固定 ~180ms(≈3.8ms/层 的 host 启动链,
MoE kernel 多放大了它),边际 12.1μs/token 是真实 GPU 重算。
**优化第一优先级:合并/预取 KV 装载路径**(每层一次大拷贝或全 blend 预取
流水线,目标把 347ms 压到 <50ms);第二:编译/图化 compute 循环(省 ~180ms)。
下节的逐层编排分析仍正确描述 compute 侧构成,但其总占比此前被高估。

### 固定开销 ~455ms 的构成(原分析,见上方修正)

依据代码结构(`lmcache/v1/compute/blend/blender.py`、
`lmcache/v1/compute/models/base.py`、`cache_engine.retrieve_layer`):

1. **48 层逐层 Python 生成器流水线**:`blend()` 同步驱动 50 次 `next()`,
   每层 = 取数步(该层各 segment 的 CPU→GPU 小拷贝,延迟受限而非带宽受限)
   + 计算步(qkv 投影、q/k norm、rope、topk 选点、contiguous attention、
   o_proj、**MoE MLP**)。约 9.5ms/层 × 48 ≈ 455ms,与回归截距吻合。
2. **MoE 小批量低效**:recompute 子集只有命中数的 15%(短 prompt 时
   ~100 token),MoE 路由 + 分组 GEMM 在小 M 下固定开销占比极高。
3. **不在编译/图覆盖内**:主 forward 有 CUDA graphs;blender 路径在
   forward 之外手工驱动,逐层 kernel launch + 协程切换 + 动态 shape 的
   compile guard。
4. **对照**:原生 prefill 是单次融合批量 pass(flash attention + 编译),
   2k token 在 4 卡上约 100-200ms,GPU 吃满。

### 为什么大上下文端到端只快 12%(Amdahl 上限)

本工作负载 decode 占推理总时间 ~93%(详细摘要模式生成量大),
prefill 全部免费也只能省约 7% 推理时间。CacheBlend 的收益本质上在
**TTFT/prefill 侧**:轮 2 中复用请求(13-18k prompt)的 prefill 从原生
1-5s 降到 0.5-1s,**快 2-4 倍**;但被 decode 稀释后端到端差距收窄。

---

## 三、结论依赖的数据与文件(可复核清单)

| 证据 | 文件 | 支撑的结论 |
|---|---|---|
| **blend 计时日志**(107 事件) | `/tmp/vllm_p1A_serve.log`(50 次)、`/tmp/vllm_p1bigA_serve.log`(57 次),`grep "Blend (KV load + fuse)"` | 回归成本模型(主证据) |
| **vLLM /metrics 差分** | `results/ab_meta/p1{A,B,bigA,bigB}_metrics_{before,after}.txt` | prefill/decode/TTFT/queue 均值与桶分布(>1s prefill:A 3 vs B 10);**交叉核验**:轮 1 A−B prefill 均值差 0.14s ≈ blend 总耗时摊每请求 25.1s/201=0.125s,两个独立测量互相印证 |
| **客户端逐调用日志** | `results/ab_meta/p1{A,B,bigA,bigB}_calls.jsonl` | 角色级延迟与 prompt/completion token 数;量化轨迹分叉(轮 1:A 89 次摘要 vs B 120 次) |
| 端到端延迟 | `results/ab_{blend,native}_10{,_big}/reports.jsonl` | 91.3 vs 64.2(轮1)、93.6 vs 106.3(轮2) |
| GPU 占用 | `results/ab_meta/gpu_monitor.csv`(含 PHASE_MARKER;启停脚本 [gpu_monitor.sh](gpu_monitor.sh)) | 排除干扰解释(轮 2 两窗口邻卡 ~0%;轮 1 B 窗口干扰反而更大) |
| 分析脚本 | [analyze_phase1.py](analyze_phase1.py)、[compare_ab.py](compare_ab.py) | 上述全部统计的产出工具 |

### 诚实声明(结论边界)

- 轮 1 的端到端差(27s/题)**不全是** blend 的开销——blend 摊到每题只有
  ~2.8s,其余是**轨迹分叉噪声**(A/B prompt 不同 → 模型走不同搜索/迭代
  路径、生成量不同)。"短 prompt 更差"的结论锚定在**服务端逐请求信号**
  (blend 计时回归 + prefill 直方图)上,这两个不受轨迹噪声影响。
- A 的逐角色客户端统计中,DECIDE/FINAL 各混入每题 1 次 warmup 微调用
  (均值被稀释);服务端直方图无此问题。
- 原生 prefill 的"同规模耗时"列来自轮 2 桶分布的区间推断,非逐请求配对。

---

## 四、改进建议(按性价比排序)

1. **零代码:调大 blend 启用阈值**。LMCache 已有 `blend_min_tokens`
   (默认 256)→ 调到 ~4000,低于阈值放弃复用走原生 prefill,自动避开
   亏损区。⚠️ 需先验证该配置在 lookup/blend 决策路径的**实际生效点**
   (尚未实测它是否真正门控 blend)。
2. **小改动:成本模型自适应开关**。lookup 后用实测模型
   (455ms + 0.015ms/tok vs 原生 ~0.06-0.1ms/tok)动态决策 blend or 重算。
3. **工程优化(中):消除逐层编排开销**(455ms 的大头):
   - KV 预取流水线:独立 CUDA stream 提前 2-4 层异步预取,计算与搬运重叠
     (现状是"拷第 i 层→算第 i 层"串行);
   - 48 层循环纳入单个 `torch.compile` 区域 + shape 桶化(padding),
     或对 blend 前向做 CUDA graph 捕获,消除逐层 launch/guard。
4. **工程优化(小):合并小拷贝**。段级 MemoryObj 在 CPU 侧拼成单块再一次
   H2D;staging buffer 跨请求复用。
5. **MoE 特定实验**:recompute 子集小时 GPU 远未吃满,
   `blend_recompute_ratios` 0.15→0.3 可能**不增时延却提升质量**,
   值得跑质量/时延曲线。
6. **场景层面**:decode 占 93% 封顶了端到端收益;要放大收益,选 prefill
   密集场景(高并发、长文档、短输出),或限制 final report 长度。
7. **实验方法**:固定语料回放(缓存搜索结果,两模式喂完全相同 prompt)
   消除轨迹分叉,让端到端数字也成为干净信号——需小改 demo。
