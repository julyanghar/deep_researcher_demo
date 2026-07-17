# 混合模式税(大白话):为什么交、源码在哪、破案怎么破的、我们治了什么

> **缘起**:三个追问——①graph+eager 混合(piecewise)为什么会交税,vLLM 代码里能不能找到对应位置?②破案过程和优化,用大白话讲一遍;③五常数表里"草稿费 激活 5.8ms"到底定位了没有?外加把 Ceager 的调参结果补上。
> **读法**:§一〜§三 讲机制+源码钉子;§四 破案故事;§五 优化与调参结果;§六 诚实交代没收口的 4.5ms。细节全文在 [how-we-found-the-fixed-tax.md](how-we-found-the-fixed-tax.md),只要结论看 [fixed-tax-conclusions.md](fixed-tax-conclusions.md)。

---

## 一、一句话 + 一个比喻

**开 suffix 投机后,每个 decode 步凭空多花 ~10ms;其中 8.1ms 是"graph+eager 混合执行"的逐层换班费,`--enforce-eager` 一个启动参数把它一刀消掉,端到端 +28%。**

比喻:GPU 干活有两种方式——**放录像带**(CUDA graph:把一步里几百个 kernel 的发射预先录好,一次回放,不用 CPU 逐个喊)和**现场喊号子**(eager:CPU 逐个发 kernel)。纯放带快,纯喊号子其实也不慢(GPU 反正在等显存,喊的声音被掩盖)。**最糟的是每层换一次班**:放一段带 → 停下来喊几嗓子 → 再放一段带……64 层换 128 次班,换班本身把驱动搞得团团转——这就是混合模式税。

## 二、为什么开投机就被踢进"混合模式"(源码链,一环扣一环)

vLLM 有两种 graph 形态:**FULL**(整步一盘带,最快)和 **PIECEWISE**(切片带:注意力被挖出来现场喊,其余 FFN/norm 还是带)。谁用哪种,是每步动态分派的:

1. **门槛**:FULL 带要求"这一步每个请求处理的 token 数完全一样"(形状固定才能录带)。开了投机,这个门槛变成 `1+k`:[gpu_model_runner.py:751](../../../../../anaconda3/envs/lmcache/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py#L751) `self.uniform_decode_query_len = 1 + self.num_spec_tokens`。
2. **检查**:[gpu_model_runner.py:3338-3356](../../../../../anaconda3/envs/lmcache/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py#L3338-L3356) `_is_uniform_decode()`——要求 `max_num_scheduled_tokens == uniform_decode_query_len` 且总 token 数=请求数×它。**suffix 的草稿长度每步都变**(0〜k 个,树上匹配到多少发多少)→ 这个检查几乎永远不过 → `uniform_decode=False`。
3. **判决**:[cudagraph_dispatcher.py:139-144](../../../../../anaconda3/envs/lmcache/lib/python3.12/site-packages/vllm/v1/cudagraph_dispatcher.py#L139-L144)——非 uniform 直接走 else 分支,只能匹配 **PIECEWISE** 的 key → 这一步进混合模式。
4. **切片是在编译期准备好的**:[compilation.py:681-682](../../../../../anaconda3/envs/lmcache/lib/python3.12/site-packages/vllm/config/compilation.py#L681-L682) 默认 `splitting_ops = ["vllm::unified_attention", ...]`(把注意力算子标为切割点);[backends.py:405-471](../../../../../anaconda3/envs/lmcache/lib/python3.12/site-packages/vllm/compilation/backends.py#L405-L471) 按切割点把整图切成子图;每个非注意力子图包上 [CUDAGraphWrapper](../../../../../anaconda3/envs/lmcache/lib/python3.12/site-packages/vllm/compilation/cuda_graph.py#L145)。
5. **运行时的换班动作**:每层执行 = `entry.cudagraph.replay()` 放一段带([cuda_graph.py:352](../../../../../anaconda3/envs/lmcache/lib/python3.12/site-packages/vllm/compilation/cuda_graph.py#L352))→ eager 发注意力 kernel → 下一层再放带……**64 层 × 每层两次域切换**。

> 对照:**不开投机**时每请求每步恒 1 token,uniform 检查通过 → FULL 整图,没有换班。**全 eager**(`--enforce-eager`)则从头到尾喊号子,也没有换班。só混合模式独有这笔税。

## 三、换班为什么这么贵(4.5ms 驱动锁)

换班费的主体**不在 vLLM 代码里,在 NVIDIA 驱动里**:`cuGraphLaunch`(放带)和 `cuLaunchKernel`(喊号子)交错时,驱动内部一把**私有条件变量锁**被反复争抢,nsys 栈采样实锤:**锁等待 4.5ms/步**(全 eager 对照 = 0.00s);再加图段启动 1.4ms + 杂项 2.2ms ≈ **8.1ms/步**。vLLM 侧能定位的是"交错结构"(§二的 5 处源码),锁本身是驱动闭源行为——所以治法不是改 vLLM,是**别让它交错**(全 eager 或全 graph)。

> 📖 **零基础版**:cuLaunchKernel / cuGraphLaunch 是什么、"桶形状"是什么、锁怎么产生——从零讲解见 [cuda-graph-lock-from-zero.md](cuda-graph-lock-from-zero.md)(本文只放账本与物证)。

### 三b、"锁等待 4.5ms"怎么实锤的 + 物证怎么看(2026-07-09 追问回填,已现场复现)

**原理**:`nsys profile` 开 **OSRT 追踪**会逐事件记录每次系统级等待(如 `pthread_cond_timedwait`)的起止,**并附调用栈**。实锤三段论:①C0(混合)里发射线程的大量 cond 等待,栈穿过 `cuLaunchKernel`(`cond_timedwait ← libcuda 内部帧 ← cuLaunchKernel` = 在"发射函数内部"被驱动挂起等锁,不是闲等活);②**全 eager 对照同一查询 = 0 次**(这个桶不存在);③结论=交错提交触发驱动串行化。

**数字(2026-07-09 复现)**:C0 栈内含 cuLaunchKernel 的 cond 等待 = **50,184 次 / 13.41s**(26s 窗口、TP4 全进程树);Ceager 同查询 **0 次**。换算:13.41 ÷ 4 worker ≈ 3.35s/worker;26s ÷ 34ms/步 ≈ 765 步 → **≈4.4ms/步**。⚠️ 必须 ÷4——四个 TP worker 并行等,墙钟只算一份(与"traj 按 worker ×N"同款口径坑,见 [decode-step-compute-anatomy.md 测量陷阱](decode-step-compute-anatomy.md))。

**物证与看法**(文件在 `~/modify-code-runs/suffix-spec-decode/tax-split/`):
- `cpusample_C0`(混合模式,**无扩展名但就是 sqlite**,375MB)、`nsys/cpusample_Ceager.sqlite`(eager 对照)、`cpusample_C0eager.{nsys-rep,sqlite}`(终审);
- **命令行复现**(30 秒):三张表 `OSRT_API`(每次等待起止+函数)、`OSRT_CALLCHAINS`(调用栈,id↔callchainId)、`StringIds`(符号字典)——

```sql
sqlite3 cpusample_C0 "
WITH cond AS (SELECT o.callchainId cc,(o.end-o.start) dur FROM OSRT_API o
  JOIN StringIds s ON o.nameId=s.id WHERE s.value='pthread_cond_timedwait'),
launch_cc AS (SELECT DISTINCT c.id FROM OSRT_CALLCHAINS c
  JOIN StringIds s ON c.symbol=s.id WHERE s.value LIKE '%cuLaunchKernel%')
SELECT CASE WHEN cc IN (SELECT id FROM launch_cc)
  THEN '栈内有cuLaunchKernel' ELSE '普通等待' END, COUNT(*), SUM(dur)/1e9
FROM cond GROUP BY 1;"
-- C0: 栈内有cuLaunchKernel | 50184 | 13.41s   /  Ceager: 该桶为 0
```

- **GUI**:`~/modify-code-runs/tools/nsight-systems-2025.5.1/host-linux-x64/nsys-ui` 打开 `.nsys-rep` → Worker 进程发射线程 → OS runtime libraries 泳道,`pthread_cond_timedwait` 的块直接嵌在 `cuLaunchKernel` 区间内;右键 Show in Events View 看逐条栈。
- **Windows 上看**:装 Nsight Systems Host **≥2025.5.1**,拷走 `tax-split/nsys/cpusample_C0.nsys-rep`(88MB,混合=有锁,已验证与 13.41s 证据同一录像)+ `nsys/cpusample_Ceager.nsys-rep`(279MB,eager 对照=无锁);找 CUDA API 行密集的发射线程 → 放大稳态 → 宽 `cuLaunchKernel` 块(~200-400μs)正下方嵌 `cond_timedwait` 块 → Show in Events View 看栈;拖选时段 Filter and Zoom 后用 Stats System View 的 `osrt_sum`/`cuda_api_sum` 看聚合。逐事件预期:宽块中位 ~272μs、每步每 worker ~16 个;eager 侧 0。**GUI 两个坑**:①Events View 的搜索框是"已加载行里查找/跳转"不是全表过滤(表是懒加载,"2 matches"≠只有 2 条——看行号已到百万级);要统计用 **Stats System View → cuda_api_sum**(给出每 API 的 Num Calls/均值/重尾)。②发射在 worker **主线程**上(Ceager 四 worker 各 ~50.6 万次 launch,TID=1025035-38 那批);空闲期主线程挂大 cond_timedwait 属"等活"无害形态,要把时间线移到 CUDA HW 实心的负载区看。

## 四、破案过程(大白话浓缩版,8 轮修正)

1. **立靶**:造了个 C0(投机开着但草稿永远为空)——前向和原生 A 一模一样大,却每步多 10ms。→ 税和"猜没猜"无关,是机制费。
2. **先查 CPU**:把 python 侧能插桩的全插了(propose/采样/metadata/RPC),合计才 1.6ms。→ 大头不在 CPU 明面上。
3. **上显微镜踩坑**:nsys 时间线上每层注意力前有 455μs"空隙",一度定罪"GPU 空转 29ms/步"。**错了**——nsys 默认不记录 graph 内部的 kernel,那段"空隙"里 FFN 正在跑(开 `--cuda-graph-trace=node` 重录,空隙里冒出 34k 个 gemm,GPU 活跃率 91.5%)。真税只是每层 ~100μs 的边界开销。
4. **2×2 定罪**:{suffix, ngram} × {混合, 全 eager} 四格对照——suffix 混合 0.83x → suffix 全 eager 1.00x;ngram 混合却不慢(靠接受多摊薄)。→ **慢的是"混合",不是"eager"、不是"有 gap"**。
5. **终审**:C0eager(空草稿+全 eager)实测 26.4ms/步 → 固定税拆成 **混合边界 8.1 + spec 管线 1.9**;驱动锁栈实锤 4.5ms 是边界税主体。

## 五、我们优化了什么(含 Ceager 调参结果)

**主刀:`--enforce-eager`**(一个启动参数):replay summary 0.83x→1.00x、report 2.29x;**真实 e2e(40 题、并发 3)墙钟中位 +28%,decode 口径零回归;唯一代价 TTFT +14%**(prefill 失去 graph)。

**Ceager 的参数调节(k × pad 扫点,补记)**:

| 调法 | 混合模式下 | 全 eager(Ceager)下 |
|---|---|---|
| 草稿上限 k:24→8 | ✅ 赚(接受在 k=8 就饱和,k24 多出的全是白验证;k8+pad8 0.97x > k24 0.85x) | ➖ 无所谓(CeagerK8 151 ≈ Ceager 152 char/s) |
| pad(草稿补齐定长) | ✅ 赚 +2ms(稳住 graph 桶形状) | ❌ 倒亏 3.5ms(纯按量成本,无桶可稳) |
| 两刀合一 CeagerK8pad | — | ❌ 被否(pad 洪水把接受率打到 13%) |

**三旋钮扫参——两轮,结论不同(按时间线,别混)**:

*第一轮(早,混合模式下,replay 冷树)*:`min_token_prob ∈ {0.05,0.1,0.3,0.5}`、`max_spec_factor=0.5`、关全局树,6 配置——**全部不能让 summary 翻正**(prob 调低=冷门也猜、命中掉到 28%、0.889x;调高=下注太少 0.917x;factor 0.5=胆小一半、最慢 0.838x)。→ 当时结论"参数不是病根,病根是混合边界税"。详见 [questions-log.md Q15/Q16](../questions-log.md)。

*第二轮(7/6-7/7 夜,**深注扫描**,`--enforce-eager` 之后,Ceager replay40)*:边界税消掉后重新下注,`factor 4/8/16 × prob 0.1/0.05/0.02 × depth 24/48` 8 配置,预注册"双涨判据"(tokens/step↑ 且 char/s↑ 才算胜):

| 配置 | summary char/s | report char/s | tok/step | 接受率 |
|---|--:|--:|--:|--:|
| g0base(默认 1/0.1/24) | 150 | 291 | 2.33 | 49.6% |
| **gf4(factor=4)** | **156** | **345(+18.6%)** | 2.62 | 24.3% |
| gf8(factor=8) | 110 ❌ | 350 | 2.87 | 17.2% |
| gd48(depth=48) | 150 | 284 | 2.27 | 45.8% |
| gmax(16/0.02/48) | 140 | 311 | 2.77 | 7.9% |

**→ 修正后的结论:`factor` 是真杠杆——gf4 双涨甜点(summary +4%、report +18.6%),接受率虽从 50% 掉到 24% 但净赚(位置便宜 0.62ms、盈亏线 2.4%);factor=8 只赢 report、summary 塌;depth/prob 是哑弹。** 物证:`~/modify-code-runs/suffix-spec-decode/tax-split/deepscan_status.txt`;step-time 侧算的坑与修正见 [decode-step-compute-anatomy.md 测量陷阱](decode-step-compute-anatomy.md)。

**→ 推荐配置 = `suffix + --enforce-eager` + `max_spec_factor=4` + 其余默认、不 pad。**(⚠️ 注意:gf4 是 replay40 口径,预注册的 census/e2e 复核未跑;且**当前生产 config_suffix.yaml 还是默认 factor=1**——40 题 e2e 和加速对比都是 factor=1 跑的,gf4 的 report +18.6% 是**还没吃进生产的已知余粮**。)"k=8 甜点/pad 补形状"是混合模式的伤药,全 eager 下不需要。

## 六、诚实交代:草稿费"激活 5.8ms"定位了吗?——**没完全定位**

五常数表第⑤行:草稿费 = **激活 5.8 + 0.62×草稿位置数**(这一步只要有草稿就交 5.8,再按位置数计件)。现状:

- **已归因 ~1.3ms**:sampler 0.82 + metadata 0.44(CPU 插桩实测);
- **~4.5ms 未归因**——嫌疑人:logits 多行处理(有草稿时 logits 从 1 行变 k+1 行)、spec 采样(rejection sampling)kernel、输出拷贝。**这是税表最后一块没砌的砖。**
- **值多少**:消掉它,冷树 summary 可从 1.00x 抬到 ~1.15x。
- **下一步怎么测**(未做):Ceager 带真草稿 vs C0eager 空草稿,nsys + 分段埋点差分,把 4.5ms 摊到嫌疑人头上——方法和 eagle3 成本解剖(进行中)同款。

## 七、五常数速查表(TP4,summary 段,预测=实测残差 0)

| 常数 | ms/步 | 什么时候交 |
|---|--:|---|
| ① graph 基线 | 24.0 | 每步(读一遍 32B 权重) |
| ② eager 附加 | +0.5 | 用 enforce-eager 时 |
| ③ spec 管线 | +1.9 | 投机开着就交 |
| ④ **混合边界税** | **+8.1** | **仅混合模式交(主体=驱动锁 4.5);全 eager 一刀消掉** |
| ⑤ 草稿费 | 激活 5.8(**其中 4.5 未归因**)+ 0.62×位置 | 有草稿就交 |

## 相关

- 结论精简版:[fixed-tax-conclusions.md](fixed-tax-conclusions.md) · 破案全文:[how-we-found-the-fixed-tax.md](how-we-found-the-fixed-tax.md)
- 每步解剖表:[suffix-vs-native-cost-anatomy.md](suffix-vs-native-cost-anatomy.md) · e2e 判决:[overnight-verdicts-2026-07-06.md](overnight-verdicts-2026-07-06.md)
- eagle3 版成本解剖(进行中,同款方法):[../eagle-idea.md](../eagle-idea.md)
- suffix 机制源码讲解:[18-vllm-suffix-decoding.md](../../../18-vllm-suffix-decoding.md)
