# Qwen3-32B A/B 效率实验总结(CacheBlend vs 原生,research 走缓存)

> 2026-06-14。Qwen3-32B dense,deep_researcher generate,web 搜索全部走 replay 缓存
> (20 题、零网络、每题固定 3 文档),把端到端计时从搜索噪声里隔离出来。
> 两个复用规模档(大/小)各跑 A(CacheBlend)/B(原生 prefix cache)。
> 原始数据:`results/RUN_32b_v1/ab32b_{big,small}_{blend,native}_20/`、计量
> `results/ab_meta/RUN_32b_v1/{big,small}_{A,B}32_*`、服务日志 `/home/yilin/tmp/RUN_32b_v1/ab32b_*_serve.log`。
> (注:这是**修复前**的首轮 32B-only 实验,big 档有 27% 摘要截断污染;干净结论见 ab_32b_vs_30b_summary.md。)

## 1. 跑了什么

| 轮 | 档 | 模式 | 关键参数 | 服务 |
|---|---|---|---|---|
| 1 | 大 | A blend | `SUMMARY_DETAILED=1 RESEARCH_SUMMARY_MAX_TOKENS=2000 MAX_ITERATIONS=2`,sep=`<\|fim_pad\|>` | LMCache blend |
| 2 | 大 | B native | 同上,无 sep | 原生 vLLM |
| 3 | 小 | A blend | `RESEARCH_SUMMARY_MAX_TOKENS=800 MAX_ITERATIONS=1`(压缩摘要),sep | LMCache blend |
| 4 | 小 | B native | 同上,无 sep | 原生 vLLM |

公共:`LIMIT=20 SEARCH_CACHE=replay SEARCH_CACHE_FIX_N=3 MAX_FOLLOWUPS=2
MAX_QUERIES_PER_RESEARCHER=2 SAMPLE_CONCURRENCY=1`,每轮前重启服务(缓存从空)。
**80 个样本全部成功,0 失败**(32B 上下文 40960 无溢出)。

## 2. 端到端结果(每题平均延迟)

| 档 | A blend | B native | 表面差异 |
|---|---|---|---|
| 大档 | **102.1s** | 115.3s | blend 快 ~11% |
| 小档 | **31.6s** | 35.1s | blend 快 ~10% |

⚠️ **端到端差异不可直接归因于 blend**:`KV_REUSE_SEPARATOR` 在 A 模式改了 prompt 布局
(摘要前置),会改变模型生成的内容与**输出长度**,两模式轨迹分叉(大档 A 写报告
平均输出 442 token vs B 770;小档反过来)。decode 占总时间 ~93%,而 decode 量由输出长度
决定 → 端到端那 ~10% 主要是**轨迹/输出长度漂移**的噪声,不是干净的 blend 收益。

## 3. 干净信号:逐请求 prefill(同角色、同 prompt 规模对比)

用服务端逐请求计时(`VLLM_RESP_TIMING`,落在 calls.jsonl 的 `prefill_s`),按角色+prompt
规模对齐,隔离掉 decode/输出长度的混淆:

| 档 | 角色 | prompt | A blend prefill | B native prefill | blend 效果 |
|---|---|---|---|---|---|
| **大档** | FINAL_REPORT(写报告,复用全部摘要) | ~8.5k | **1.506s** | 2.220s | **−32%** ✓ |
| 大档 | SUPERVISOR_DECISION | ~7.3k | 1.229s | 1.314s | −6% |
| **小档** | SUPERVISOR_DECISION | ~2k | 0.501s | 0.410s | **+22%**(blend 反而慢) |
| 小档 | FINAL_REPORT | ~2k | 0.264s | 0.413s | (n=3/4,样本太小,噪声大) |

**核心结论**:
- **大复用(写报告 ~8.5k token):blend 把该请求的 prefill 砍掉 ~32%**(1.5s vs 2.2s)——
  这是干净、可复现的 blend 收益,落在写报告这个复用最重的请求上。
- **小复用(~2k):blend 的固定开销(load+fuse ~200ms)盖过 prefill 省的那点**,
  supervisor prefill 反而 +22%。与盈亏交叉点 ~4-6k token 的成本模型一致。

## 4. blend 成本与正确性

- **blend(load+fuse)单次**:大档 mean 247ms(hit ~2036 token),小档 mean 199ms(hit ~426)。
  **load(CPU→GPU 搬运)占 62-69%**,compute(重算+编排)占其余——transport 主导,与之前结论一致。
- **请求级 opt-in 存储验证通过**:blend 日志里 `Queued final save` 大档 162 次/20 题 ≈ 8/题
  (= researcher 摘要 + 2 warmup),**最终报告 / decide / JSON 一律没存**——证明改成请求级
  opt-in 后,只存"会被复用"的段,砍掉了之前"全存"的浪费。
- **blend 命中规模**:大档 blend 事件 max 命中 9184 token、21 个事件 >4k(交叉点以上),
  确认大档确实在大规模复用。

## 5. GPU 占用(顺带)

| 档 | A blend 本卡均利用率 | B native |
|---|---|---|
| 大档 | **52.2%** | 74.7% |
| 小档 | 73.3% | 74.8% |

大档 blend 利用率明显更低(52% vs 75%)——blend 跳过重算 prefill → 少算,GPU 更闲;
原生每次都重算整段前缀 → 更热。小档因复用小,两者接近。
(同期 GPU 0 有别的用户 nicholas 在跑,~30-49% 干扰负载,已记录在 gpu_monitor.csv。)

## 6. 总结一句话

在 **Qwen3-32B dense + 固定缓存文档**下:CacheBlend 在**大复用请求(写报告 ~8k)上把
prefill 砍 ~32%**、GPU 更省;在**小复用(~2k)上因固定开销净亏**。但因 decode 占 93% 且
受 prompt 布局改动导致的输出长度漂移影响,**端到端那 ~10% 的差异不能干净归因于 blend**——
要看 blend 真实价值,应看大复用请求的 prefill,而非端到端均值。

## 7. 已知局限 / 下一步

- A/B prompt 布局不同 → 轨迹分叉,端到端被输出长度混淆。要更干净,可固定两模式生成相同
  输出(如对齐 max_tokens 或用相同 decode 强制),或只比 prefill/TTFT 直方图。
- 小档"真大请求"样本少(n=3-4),小档 prefill 结论噪声大;如需结实结论可加大复用或题量。
- 本轮只 generate 未判分(qwen-flash);质量对比(report 分数)是另一阶段。
