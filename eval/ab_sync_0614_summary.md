# CacheBlend A/B(`LMCACHE_BLEND_TIMER_SYNC=1` 重测,干净 load/compute 拆分)

> 2026-06-15。{Qwen3-32B dense, Qwen3-30B-A3B MoE} × {A=CacheBlend, B=原生 prefix cache}
> × {大档/小档},八轮,每格 20 题,research 全走 replay 缓存(零网络、固定 3 文档)。
> **关键改动:所有 blend(A)轮加 `LMCACHE_BLEND_TIMER_SYNC=1`**,在 blend 每个 phase 边界插
> `torch.cuda.synchronize()`,拿到**真实的 load vs compute 拆分**(非 sync 下 GPU 异步会把
> compute 的时间错算进 load)。native(B)轮无 blend、不受此 env 影响。
> 原始:`results/RUN_sync_0614/{32b,30b}_{big,small}_{blend,native}/`、calls/metrics
> `results/ab_meta/RUN_sync_0614/*_calls.jsonl`、server 日志 `/home/yilin/tmp/RUN_sync_0614/*serve.log`;
> 聚合脚本 `/home/yilin/tmp/RUN_sync_0614/analyze.py`。**80 个样本 79 成功(32B 大档 1 题超时)。**

## 0. 一句话

开 GPU 同步看清后,**blend 的成本结构是模型相关的**:**dense(32B)是 compute 主导(71%)、
MoE(30B-A3B)是 load/transport 主导(73-74%)**——推翻了之前(非 sync)"transport 一律主导
75-82%"的结论。收益侧也更干净:**blend 只在 32B dense 大复用上真赢(prefill −51~67%),
32B 小档、30B 全档都净亏**。

## 1. ★ blend 固定开销分项(sync=True,本轮重点)

server 日志 `Blend timing: load=… compute=… (sync=True)` 聚合(load=KV 搬运+逐层编排,
compute=选择性重算 ~15% 重要 token + 融合):

| 轮 | events | load(ms) | compute(ms) | 合计(ms) | load% | compute% | 均复用 token |
|---|---|---|---|---|---|---|---|
| **32B dense 大** | 368 | 138 | **331** | 470 | 29% | **71%** | 5817 |
| 32B dense 小 | 312 | 150 | 140 | 291 | 52% | 48% | 559 |
| **30B MoE 大** | 388 | **358** | 133 | 491 | **73%** | 27% | 3430 |
| **30B MoE 小** | 312 | **354** | 122 | 476 | **74%** | 26% | 488 |

**两条干净结论:**
1. **dense(32B)是 compute 主导**:compute 随"复用 token 数"涨(331ms@5817tok vs 140ms@559tok)
   ——选择性重算的 ~15% 重要 token 要过**完整稠密 FFN**,token 越多越贵。
2. **MoE(30B)是 load 主导**:compute **几乎不动**(~125ms,稀疏 FFN 重算极便宜),
   固定的逐层 load 编排(~355ms)成了大头。注意 30B 的 KV 更小(96KB/tok < 32B 256KB)、层更少
   (48 < 64),**load 反而更高**(355 vs 138ms,7.4 vs 2.2 ms/层)→ 坐实 load 是**逐层 orchestration
   开销(含 MoE workspace lock/unlock 等),不是带宽/字节瓶颈**。

> 对照非 sync 旧数(32B load=322/comp=106、30B load=450/comp=96):旧测把 32B 的 compute(真值 331ms)
> 大部分错算进了 load(322ms),于是误得"transport 主导"。MoE 因 compute 本就小,旧测偏差不大。

## 2. 逐请求 prefill:A blend(仅 hit>0)vs B native

口径:A 列**只统计真实请求**(`max_tokens!=8` 排掉同 tag 的 warmup_kv_prefix 预热小请求)
**且 server 日志 `LMCache hit tokens>0`**(⋈ calls.jsonl `req_id`,排 0 命中如首样本第一次 decide);
B 列 native 同样排 warmup(native 本不发 warmup)。`A/B<1` = blend 更快。

| 模型/档/角色 | A blend(真实,hit>0) | B native | **A/B** |
|---|---|---|---|
| **32B 大 / 写报告** | 0.998s (n19) | 3.044s (n20) | **0.33**(−67%)✓ |
| **32B 大 / 决策** | 0.903s (n35) | 1.859s (n33) | **0.49**(−51%)✓ |
| 32B 小 / 写报告 | 0.401s (n20) | 0.229s (n20) | **1.75**(亏) |
| 32B 小 / 决策 | 0.433s (n20) | 0.231s (n20) | **1.88**(亏) |
| 30B 大 / 写报告 | 0.597s (n20) | 0.385s (n20) | **1.55**(亏) |
| 30B 大 / 决策 | 0.596s (n39) | 0.217s (n37) | **2.75**(亏) |
| 30B 小 / 写报告 | 0.534s (n20) | 0.064s (n20) | **8.3**(亏) |
| 30B 小 / 决策 | 0.558s (n20) | 0.065s (n20) | **8.6**(亏) |

> **口径订正(重要)**:warmup_kv_prefix 的预热请求与真实请求同 tag(`FINAL_REPORT_MARKDOWN`/
> `SUPERVISOR_DECISION_JSON`)、`max_tokens=8`、prefill≈0.74s;native 模式不发 warmup。早先未排
> warmup → A 列被污染(32B 大档曾误为 0.26/0.40、小档 2.36/1.95)。**排 warmup 后方向不变、数值订正如上。**

**blend 只在 32B dense 大档真赢(prefill −51~67%);32B 小档、30B 全档都净亏。**
32B 小档:真实写报告 A blend 0.401s > native 0.229s —— prompt 才 ~1.1k token,**原生 prefill 已比
blend 固定开销(~0.3-0.4s)便宜**,故净亏(本轮每角色 n=20,结论稳)。

**为什么**(成本模型):blend 赢 ⟺ `B native prefill > A blend 成本`。
- 32B 大:native 1.9~3.0s **远超** blend ~0.47s → 赢。
- 32B 小:native 0.23s **< blend 0.29~0.54s** → 亏(小 prompt 原生 prefill 已很便宜)。
- 30B 全:native 0.06~0.39s **全 < blend ~0.48s** → 亏(MoE 稀疏激活让原生 prefill 太便宜,
  连 blend 固定开销都够不到;小档差到 ~8.5×)。

## 3. 端到端(每题平均延迟,秒)——被 decode/输出漂移混淆,只作趋势

| 模型 | 大档 A | 大档 B | 小档 A | 小档 B |
|---|---|---|---|---|
| 32B dense | **138.1**(19) | 154.5 | **30.5** | 43.5 |
| 30B MoE | 31.8 | **27.5** | 9.0 | **6.7** |

⚠️ 注意 **32B 小档端到端 A 更快(30.5<43.5)、但其 prefill 反而更慢(§2 A/B=2.0~2.4)**——
这正说明端到端被 **decode(占 ~90%)+ `KV_REUSE_SEPARATOR` 改 prompt 布局导致的输出长度漂移**
主导,**不能干净归因于 blend**。干净信号看 §2 的 prefill。

## 4. 机制根因(与 §1 一致)

prefill compute-bound。算力差在 FFN:dense 32B 每 token 每层过完整 FFN(全 32B 参数);
MoE 30B 每 token 只过 top-8/128 小专家(~3B 激活)→ FFN FLOPs/token 约 **3.6 vs 50 GFLOP(~14×)**。
- **dense**:native 全量重算贵(3.0s)、blend 只重算 ~15% 也得过稠密 FFN(compute 331ms,但远小于全量)
  → blend 用"便宜搬运 + 少量重算"换掉"昂贵全量重算",大复用上大赚。
- **MoE**:native 本就便宜(稀疏),blend 省不下多少;反被固定 load 开销(355ms,MoE 逐层编排更重)拖累
  → 双重吃亏(成本更高、收益更低),全面净亏。

## 5. 对 07 文档的修正

- §3.3 旧表"transport 主导 75-82%"基于**非 sync** 测量,对 dense **错误**。已用本轮 sync=True 数据更正为
  **"dense compute 主导(71%)/ MoE load 主导(73-74%)"**,并保留旧数差异的解释。
- §3.1 prefill A/B:32B 小档由"赢"修正为"亏"(本轮 n≥38 比旧 n=3 可靠);结论"dense 大复用赢、
  其余亏"更聚焦。

## 6. 局限

- **sync=True 给 A 真实加了 barrier 串行开销**(blend phase 合计比非 sync 高 ~40~90ms),
  → A 的**绝对** prefill/端到端比生产态(sync=0)略高;但 **load/compute 拆分比例**与各档**方向结论**不受影响。
- A/B prompt 布局不同 → 输出长度漂移,端到端被混淆(§3);prefill 口径(§2)不受影响,为准。
- 32B 大档 19/20(1 题超时);其余 20/20。仅 generate 未判分。
