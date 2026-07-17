# exp-test-supervisor 结果:supervisor 复用是不是元凶?——不是

**问题**:四臂实验里全复用 C 整体 F1 比真算低 ~8 分。复用发生在 supervisor(改研究轨迹)和 writer(改报告生成)两处。本实验隔离:**SW 臂 = supervisor 复用 + writer 纯 prefill**,看是不是 supervisor 复用拖累了 report。

**实验有效性**(两日志对齐法,见 ../experiment_notes.md §1):SW 跑 40/40,`SUPERVISOR_DECISION_JSON` 94/94 调用窗口内有 blend(100% 复用)、`FINAL_REPORT_MARKDOWN` 28/28 无 blend(writer 纯 prefill)。配置确认无误。

## 打分方法(统一,修掉上一轮三个毛病)
- **评委 kimi-k2.5**(一把尺子),不再用 qwen-flash —— 后者跨会话漂移巨大(同批 C 报告两次打分逐题 |Δ|=0.236、均值差 0.09~0.20,比要测的效应 0.03 还大,导致"C 低 8 分"基本是假象)。
- **每臂逐题打分各一次、落盘 `sw_unified/ratings_<arm>.jsonl`**(不再走 merge 合成目录把 SW 重复打 3 遍;中间结果不丢、可 resume)。
- 照 deepsearchqa 官方链路:`score_answer→build_item_rating_from_report→rate_report→reduce_autorater_response→ItemRating`,`aggregate_ratings` 汇总。脚本 `../../server/vllm/Exp_sw/exp_sw_score.py`。

## ① 整体 F1(aggregate_ratings 官方口径,kimi-k2.5)
| 臂 | 配置 | 整体 F1 | 全对率 |
|---|---|---|---|
| **B** | 不复用·带分隔符(基线)| **24.82%** | 12.5% (5/40) |
| **SW** | 只 supervisor 复用 | **30.28%** | 17.5% (7/40) |
| **C** | 全复用 | 28.18% | 15.0% (6/40) |
| A_orig | prefill·无分隔符 | 31.12% | 15.0% (6/40) |
| A_orig2 | prefill·无分隔符(第二条)| 30.53% | 17.5% (7/40) |

→ **复用臂(SW 30.3% / C 28.2%)不比不复用 B(24.8%)差、甚至略高**;SW 跟两条 prefill 臂(30-31%)挤在一起。

## ② 逐题配对差 + bootstrap 95% CI(Set,主力)
| 对比 | 含义 | 差 | 95% CI |
|---|---|---|---|
| A_orig − A_orig2 | 噪声底 | 0.043 | [−0.061, 0.169] |
| B − C | 纯复用(总)| −0.000 | [−0.098, 0.107] |
| **B − SW** | **supervisor 复用效应** | **−0.007** | [−0.073, 0.055] |
| SW − C | writer 复用效应 | 0.006 | [−0.084, 0.104] |

(Single 题 n=13、方差大:B−SW=−0.154 CI[−0.385,0]、SW−C=0.051,同样不显著且 B−SW 为负=SW 不比 B 差。)

→ **所有效应 ≈0、CI 跨 0、在噪声底(0.043)以内**。supervisor 效应 −0.007(SW 还略高于 B)。

## ③ 崩题归因(B−C ≥ 0.5)
| 题 | B | SW | C | 锅在谁 |
|---|---|---|---|---|
| q27 | 1.00 | 0.80 | 0.00 | **writer**(SW 救回)|

kimi 口径下只剩 q27 一道崩题,且 SW(writer prefill)救回 → 归 writer。

## ④ 评委自一致性(kimi-k2.5 同一批 report 打两遍,验证结论可信)
| 臂 | mean\|Δf1\|(run1 vs run2)| 变了的题 | run1 均值 | run2 均值 |
|---|---|---|---|---|
| B | 0.037 | 6/40 | 0.248 | 0.277 |
| SW | 0.018 | 3/40 | 0.303 | 0.286 |
| C | 0.000 | 1/40 | 0.282 | 0.281 |
| **合计** | **0.018** | | | |

**kimi 自漂移 0.018,vs qwen-flash 的 0.236 —— 稳约 13 倍。** 效应在两次独立打分里都 |·|≤0.024、贴 0 抖(supervisor −0.007/+0.024、writer +0.006/−0.018),无定向。→ kimi 评委可信,本实验结论站得住。脚本 `../../server/vllm/Exp_sw/exp_sw_stability.py`,数据 `sw_unified/ratings_<arm>_run2.jsonl`。

## 结论
**"report 变差是不是 supervisor 复用引起"——否。**
- SW(supervisor 复用)≈ C ≈ A_orig/A2 ≥ B,supervisor 复用**没有**让报告变差(配对效应 −0.007,SW 甚至略高)。
- 唯一崩题 q27 归 writer(且被 SW 救回),不归 supervisor。
- 之前 qwen-flash 的"C 低 8 分"主要是**评委漂移假象**;换稳的 kimi-k2.5 后复用与真算无可辨差异。

**给 KV reuse 项目的启示**:supervisor 的 decide(continue/complete + followup)用复用 KV 是安全的,不会把研究轨迹带偏到伤害最终答案。报告级的整体差距落在 agent 非确定性 + 评委噪声里。

数据:`sw_unified/ratings_{B,SW,C,A_orig,A_orig2}.jsonl`(逐题)、`metrics_<arm>.json`(汇总)。
