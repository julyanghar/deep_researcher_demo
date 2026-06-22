# CacheBlend A/B:32B dense vs 30B MoE(截断修复后重跑 + 逐请求复用核实)

> 2026-06-14。{Qwen3-32B dense, Qwen3-30B-A3B MoE} × {A=CacheBlend, B=原生 prefix cache}
> × {大档/小档},每格 20 题,research 全走 replay 缓存(零网络、固定 3 文档)。
> 改进:① 截断修复(`KV_REUSE_TOKENIZER`,截断摘要也可复用);② 自然 EOS(大档 cap=4000);
> ③ `LLM_TIMEOUT=300`。big-A 两模型用独立 server log 重跑,逐请求复用已核实。
> 原始:`results/RUN_32b_vs_30b/ab2_*`、calls/metrics `results/ab_meta/RUN_32b_vs_30b/ab2_*`、server 日志 `/home/yilin/tmp/RUN_32b_vs_30b/ab2_*serve.log`。

## 1. 端到端(每题平均延迟,秒)

| 模型 | 大档 A blend | 大档 B native | 小档 A blend | 小档 B native |
|---|---|---|---|---|
| **32B dense** | **134.4**(18/20) | 149.7 | 33.3 | 34.6 |
| **30B MoE** | 31.0 | **27.8** | 9.6 | **6.8** |

32B 大档 blend 快、30B 全面 native 快。⚠️ 端到端受输出长度漂移混淆(decode 占 ~90%),
只作趋势;干净归因看 §3 的 prefill。

## 2. 逐请求复用已核实(不是推断)

方法:server 日志每请求一行 `Reqid: <id> … LMCache hit tokens: H`
([adapter:1493](../../LMCache/lmcache/integration/vllm/vllm_v1_adapter.py#L1493))⋈ calls.jsonl 的
`req_id`(=响应 `chatcmpl-<id>`,服务端多一段实例后缀,去掉即相等)→ 每请求拿到"角色 + 复用 token 数"。

| 角色 | 32B 大档 | 30B 大档 | 32B 小档 | 30B 小档 |
|---|---|---|---|---|
| FINAL_REPORT | 36/37 命中,均 5211,复用 **94%** | 39/40,均 3249,**95%** | 39/40,均 508,**89%** | 39/40,均 500,89% |
| SUPERVISOR | 50/51 命中,均 5415,**93%** | 57/58,均 3332,**96%** | 39/40,均 545,89% | 39/40,均 538,89% |

→ **SUPERVISOR/FINAL_REPORT 每个请求都真复用了**(大档复用其 prompt 的 ~94%、max 单请求 ~18-19k token);
唯一 0 命中的是首样本第一次 decide(那时还没东西可复用)。摘要本身命中 0(它是 producer 不是 consumer),正确。

## 3. ★ prefill 速度详细分析(本节是重点)

### 3.1 逐请求 prefill:A blend vs B native(`A/B` < 1 = blend 更快)

**口径(重要)**:A blend 一列**只统计真正命中 blend 的请求**(按 server 日志 `LMCache hit
tokens>0` ⋈ calls.jsonl `req_id` 筛出),排除 0 命中的(如首样本第一次 decide,走原生)。
另:`LLM_CALL_LOG` 是追加写、big-A 重跑复用了同一 calls.jsonl 路径→里面混了多轮记录,
**必须靠 server 日志 join 才能只取当前轮命中的请求**(否则会被陈旧记录污染均值)。

| 模型/档/角色 | A blend(仅 hit>0) | B native | **A/B** |
|---|---|---|---|
| 32B big / 写报告 | 0.880s (n18) | 3.077s | **0.29**(−71%) |
| 32B big / 决策 | 0.839s (n32) | 1.861s | **0.45**(−55%) |
| 32B small / 写报告 | 0.375s (n3) | 0.585s | **0.64**(−36%) |
| 32B small / 决策 | 0.381s (n3) | 0.588s | **0.65** |
| 30B big / 写报告 | 0.620s (n19) | 0.411s | **1.51**(+51%) |
| 30B big / 决策 | 0.655s (n31) | 0.269s | **2.44**(+144%) |
| 30B small / 写报告 | 0.600s (n4) | 0.132s | **4.53** |
| 30B small / 决策 | 0.658s (n4) | 0.133s | **4.94** |

**铁律:dense 4/4 全部 blend 更快(−36~71%);MoE 4/4 全部 blend 更慢(+51~394%)。** 没有例外。
(小档 n=3-4 较小,因 prompt≥1500 + hit>0 只剩复用摘要的写报告/决策,噪声偏大。)

### 3.2 为什么:blend prefill ≈ 固定,native prefill 随"prompt×密度"线性涨

- **A blend 的 prefill ≈ 一个近似固定值**(只搬运 KV + 选择性重算 ~15% 重要 token,不随 prompt 线性涨):
  32B 0.38~0.88s、**30B 0.60~0.66s 几乎不动**。它由 blend 固定开销主导(见 §3.3)。
- **B native 的 prefill = `prompt_tokens × 每 token 算力`**,随 prompt **线性**且**与模型密度强相关**:
  - 32B dense ≈ **0.25 ms/token**(全部 32B 参数都算)→ 12k token = 3.08s。
  - 30B MoE ≈ **0.06 ms/token**(只激活 ~3B)→ 6k token = 0.41s。

→ **blend 赚不赚 = `native prefill > blend 固定开销` 是否成立**:
- **32B**:native 0.59~3.08s **全都 >** blend 0.38~0.79s → 4/4 赚。
- **30B**:native 0.13~0.41s **全都 <** blend 0.60~0.66s → 4/4 亏(MoE 的 native prefill 连 blend 固定开销都够不到)。

### 3.3 blend 固定开销分项(big-A 日志,load=搬运 / compute=重算编排)

| 模型 | load(搬运) | compute(重算编排) | 合计 | load 占比 |
|---|---|---|---|---|
| 32B dense | 322ms | 106ms | **429ms** | 75% |
| 30B MoE | 450ms | 96ms | **546ms** | 82% |

**transport(搬运)主导(75-82%)**,且和算力/字节都不强相关(30B 的 KV 只有 96KB/token < 32B 256KB,
但 load 反而更高)——说明它是**逐层 orchestration 固定开销**(eager 逐层 kernel launch+同步),不是带宽瓶颈。
**MoE 的 blend 固定开销还更高(546 vs 429ms),所以 MoE 双重吃亏:成本更高、收益更低。**

### 3.4 盈亏交叉点(以"原生需 prefill 的 token 数"为横轴)

`blend 赚 ⟺ prompt_tokens × 每token算力 > blend固定开销(~0.5s)` ⟹ 交叉点 `≈ 0.5s / 每token算力`:
- **32B dense**:0.5s / 0.25ms ≈ **2000 token**。复用型请求 prompt 普遍 2-12k > 2000 → 基本都赚。
- **30B MoE**:0.5s / 0.06ms ≈ **8000-9000 token**。本工作负载 prompt 多在 2-7k < 8000 → 基本都亏;
  只有单请求复用 >~9k 的极少数才可能打平(大档 max ~19k 那种)。

**一句话:dense 的交叉点低(~2k,轻松跨过),MoE 的交叉点高(~9k,基本够不到)。**

## 4. 机制根因(dense vs MoE 的 prefill 算力)

prefill 是 compute-bound。算力差距来自 FFN:稠密 32B 每 token 每层过完整 FFN(inter 25600,全 32B 参数);
MoE 30B 每 token 只过 top-8/128 个小专家(inter 768,仅 ~3B 参数)→ FFN FLOPs/token 约 **3.6 vs 50 GFLOP**(~14×)。
**MoE 用稀疏激活换来廉价 prefill——这正是 CacheBlend 前提("拿便宜搬运换昂贵计算")在 MoE 上塌掉的原因。**

## 5. 改进方向(详见专项调研文档,本处摘要)

1. **自适应门控(do-no-harm)**:按 `预估 native prefill > blend 开销` 才复用;MoE/小复用走原生。立即止损。
2. **砍 blend 固定开销(治本,overhead-bound 非 bytes-bound)**:把 blender 逐层 load+compute
   ([blender.py:167-176](../../LMCache/lmcache/v1/compute/blend/blender.py#L167) 当前**串行**)改 CUDA graph/批量 +
   **层间流水线重叠**(load 第 i+1 层 ‖ compute 第 i 层)→ ~0.5s 砍到 ~0.3s,让 MoE 逼近 break-even。
3. **GPU 常驻热 KV**:跳过 CPU→GPU 搬运。4. **KV 量化**:收益有限(非 bytes-bound)。

## 6. 结论

- **CacheBlend 是 dense + 大复用的优化**:prefill 上 dense −36~74%、MoE +51~394%,**逐请求复用已核实(~94%)**。
- **MoE(30B-A3B)上不建议开 blend**:稀疏激活让 native prefill 太便宜,连 blend 固定开销都够不到,全面净亏。
- 端到端被 decode(~90%)+ 输出长度漂移稀释;blend 真实价值看 prefill。

## 7. 局限

- 32B 大档 18/20(2 题超时/瞬断,cap=4000 个别摘要 >300s);其余 20/20。
- A/B prompt 布局不同 → 轨迹/输出分叉,端到端混淆(prefill 口径不受影响,为准)。
- 仅 generate 未判分。
