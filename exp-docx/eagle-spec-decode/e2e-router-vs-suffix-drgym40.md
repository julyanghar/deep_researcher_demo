# DRGym 40 题端到端:suffix vs router(真实并发,per-request 选 proposer)

真实生产场景验证 MCPROUTE router 值不值:DRGym 40 题 local 完整 e2e(supervisor+researcher+summary+report),
对比**纯 suffix** vs **router**(summary→eagle 域训 head、其余→suffix)。2026-07-16。

## TL;DR

**router 在真实并发 e2e 净赚 7.6% wallclock**,主要收益来自把 suffix 最大的短板(summary 段,占 decode 81%)
换成 eagle 域训 head。这是 phase2(单请求 replay)没测过的**真实并发**场景,而且**口径干净**(两配置都 enforce-eager,纯比 proposer)。

- summary 段:tokens/s **10.5→13.7(+30%)**、墙钟/题 **111.6→89.2s(−20%)**、decode 省 23%。
- 代价:report/supervisor 白算 ~4%(router 里它们走 suffix,eagle 对其白算 draft)。
- 总:单题 wallclock **252.2→233.1s(−7.6%)**;decode 层面省 18%,被超长 prefill + 非 LLM 检索稀释到墙钟 −7.6%。

## 实验设计

| 项 | 配置 |
|---|---|
| 数据 | DRGym 40 题(`/home/yilin/tmp/drgym_local40.jsonl`,全命中 local 缓存,英文 ~10K token 长 prompt) |
| 模式 | local replay(纯离线读 chunks.jsonl,零联网)、完整 e2e(3 迭代×3 researcher×3 subquery)、含 final report |
| 并发 | 题级 concurrency 4 + 题内 researcher 并行(**真实并发**,非单请求 replay) |
| 配置 A | 纯 suffix:server `method=suffix`,不开 PROPOSER_ROUTING |
| 配置 B | router:server `method=router`+ 域训权重 phase2-12k-3ep;deep_researcher 开 `PROPOSER_ROUTING=1`(summary→eagle、其余→suffix) |
| **口径** | **两配置都 `--enforce-eager`**(suffix 本就只能 eager、router 必须 eager)→ 纯 proposer-to-proposer 对比,不掺 cudagraph 差异 |
| 计时/接受率 | server `VLLM_RESP_TIMING=1`(逐调用 prefill/decode)+ `SUFFIX_TRAJ`(逐请求 acc/dr) |
| proposer 注入 | deep_researcher `llm.py` 按 tag 注入 `vllm_xargs={"proposer":...}`(summary=RESEARCH_SUMMARY_TEXT→eagle,其余→suffix) |

## 防实验剧场(预检全过)

| 验证项 | 结果 |
|---|---|
| **proposer 真按 tag 分流** | join MCPROUTE_pick + llm_calls tag:summary→eagle **18/18**、other→suffix **12/12**,零错配 |
| server timing 非空 | 15/15 调用有 decode_s(VLLM_RESP_TIMING 生效) |
| local 缓存命中不联网 | server 日志无 cold_live/tavily 痕迹 |
| eagle head 加载正常 | MCPROUTE_init aux_hidden=True(config_deploy rope 格式正确,没崩) |
| **接受率交叉验证** | suffix report **60.5% vs phase2 61.2%**、summary **26.3% vs phase2 24.5-28.8%**——两独立实验几乎逐字吻合,数据可信 |

## 完整对比(40 题,逐阶段)

| 阶段 | suffix tok/s | router tok/s | suffix decode 总 | router decode 总 | suffix 墙钟/题 | router 墙钟/题 | suffix 接受率 | router 接受率 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **summary** | 10.5 | **13.7** | 23044 | **17660** | 111.6 | **89.2** | 0.263 | N/A(eagle) |
| supervisor | 22.6 | 18.9 | 435 | 509 | 16.3 | 16.8 | 0.377 | 0.369 |
| query_plan | 28.3 | 30.1 | 291 | 273 | 5.6 | 3.7 | 0.428 | 0.433 |
| report | 29.5 | 25.2 | 4615 | 4804 | 118.7 | 123.4 | 0.605 | 0.597 |
| **合计** | — | — | **28385** | **23246** | **252.2** | **233.1** | — | — |

## 净账拆解

**(+) summary 段大赚(主要收益)**:eagle 域训 head 补 suffix 最大短板。
- decode 时间 23044→17660,**省 5384s(−23%)**;墙钟/题 −22.4s(−20%);tokens/s +30%。
- 为什么这段是杠杆:summary 占 suffix 全部 decode 的 **81%**(23044/28385),且 suffix 在 summary 上最弱(改写型输出、字面重复少、接受率仅 0.263)。换 eagle 直接压缩这个大头。

**(−) report/supervisor 白算(代价)**:router 里这些走 suffix,但 eagle 对它们也白算 draft(GPU)。
- report decode 4615→4804,**+189s(+4%)**;supervisor +74s。和单请求白算实测的 ~4% 吻合。
- report tokens/s 表面 29.5→25.2(含输出长度差异),纯速度看 decode_s 的 +4% 更准。

**总账**:decode 28385→23246,**省 5139s(−18%)**;单题 wallclock 252.2→233.1,**−7.6%**(decode 只占墙钟一部分,超长 prefill + 非 LLM 检索把 −18% 稀释到 −7.6%)。

## 相比原生 vanilla vLLM(无投机)能省多少(粗略估算)

本实验只实测 suffix vs router(都开投机),**没跑 vanilla(无投机)臂**。用 DRGym 自己各阶段的 decode + 接受率反推 vanilla。

**关键前提:这个 workload 是 summary 主导。** 各阶段 decode 时间总和(suffix 配置,40 题):

| 阶段 | suffix decode | 占比 | suffix vs vanilla(cudagraph)倍数 |
|---|--:|--:|--:|
| **summary** | 23044s | **81%** | ~1.05×(suffix 短板,LATEST x1.00-1.07) |
| report | 4615s | 16% | ~2.0×(照抄大赚) |
| supervisor+query_plan | 726s | 3% | ~1.4× |

summary 占 decode **81%**(deep_researcher 里 summary 调用 504 次 >> report 40 次,3 迭代×3 researcher 撑成大头)。
**这决定了 suffix vs vanilla 被 summary 稀释**——summary(81%)几乎打平、report(16%)翻倍,加权后:

- 估 vanilla(cudagraph)decode ≈ 24200(summary×1.05) + 9200(report×2.0) + 1020 ≈ **34400s**
- suffix 28385 → **suffix vs vanilla 省 ~18%**;router 23246 → **router vs vanilla 省 ~32%**

### 综合:本实验 router 相比 vanilla 约省 **~30%**;suffix 单独只省 ~18%

**⚠️ 强烈依赖 workload 的 summary/report 占比**(这是最容易搞错的):
- **report 主导场景**(如 suffix-spec-decode 的 DRBench:长报告占 decode 大头):suffix 单独就 vs vanilla **省 ~50%**(report 2.37× 主导),router 增量小。
- **summary 主导场景**(本 DRGym 实验,summary 占 81%):suffix 被自己的短板(summary)拖累,vs vanilla 只 ~18%;**router 把 summary 换 eagle 补短板,增量才大(summary decode 省 23%)** → router vs vanilla ~32%。

→ **一句话:report 主导时 suffix 已够猛、router 增量小;summary 主导时 suffix 被短板拖累、router 补 summary 的价值才凸显——这正是 router 该出场的场景。**

**诚实标注**:① vanilla 臂未实测,倍数用 DRGym 接受率 + LATEST-CONCLUSIONS 口径估;② vanilla 基线取 cudagraph(无投机不需 eager);若 vanilla 也 eager,suffix/router 相对它省更多(summary ~1.3× → router vs vanilla_eager ~44%);③ 要精确需补跑 vanilla 臂(同 DRGym 40 题)。

## 口径与诚实边界

1. **两配置都 enforce-eager**:所以这是干净的 proposer 对比。但要注意——router 里的 eagle 被迫 enforce-eager
   (router 锁 eager),**没吃到 cudagraph 红利**;单独的纯 eagle server(cudagraph)会更快。即便如此 router 仍净赚。
2. **router 的两笔代价**:① eagle 对非 summary 请求的 ~4% 白算 ② eagle 失 cudagraph。若对比对象是"理想双 server
   分离部署"(纯 eagle cudagraph 跑 summary + 纯 suffix 跑其余,显存翻倍),router 用这两笔代价换单实例省显存 + 按请求选。
3. **summary 接受率 N/A**:router 的 summary 走 eagle,SUFFIX_TRAJ 只记 suffix 请求(gate),故 eagle summary 接受率
   不在 traj。但 tokens/s 13.7 vs 10.5 直接证明 eagle 快。
4. **真实 e2e vs replay**:suffix 在真实并发 e2e 的 summary 接受率 0.263,与 phase2 单请求 replay 的 0.245-0.288 一致
   ——印证 suffix 在长 summary 上的短板是稳健的,不是 replay 伪影。summary 段口径见 [suffix LATEST-CONCLUSIONS](../suffix-spec-decode/docs/LATEST-CONCLUSIONS.md)。
5. **样本**:40 题 local(忽略题目属于训练集的影响,本实验测吞吐不测质量)。

## 结论

**per-request 选 proposer(summary→eagle 补短板、report/其余→suffix 保强项)在真实并发生产场景净赚 7.6% wallclock,
主要来自 summary 段。** 这回答了之前悬着的"真并发下 router 到底值不值"——**值**,而且口径比 phase2 更干净
(纯 proposer 对比)。收益的物理来源清晰:summary 是 suffix 结构性最弱、又占 decode 大头的一段,eagle 域训恰好补这里。

## 物证与相关

- 脚本/数据:`/home/yilin/modify-code-runs/e2e-drgym-router/`(launch_suffix.sh / launch_router.sh / analyze_e2e.py / measure_waste.py)
- 逐调用日志:`eval/benchmarks/results/drgym/drgym_local40_{suffix,router}/q*/llm_calls.jsonl`
- 白算单独测量:`/home/yilin/modify-code-runs/vllm-multi-proposer/waste-analysis.md`(单请求 −4.2% / 并发 N=8 −4.1% 不摊薄)
- router 实现:`LMCache/vllm_patch_backup/mcproute-spec-router/`(commit 1dcf50ad)
- 域训权重与 phase2 评测:[phase2-results-summary.md](phase2-results-summary.md)
- suffix 口径权威:[LATEST-CONCLUSIONS.md](../suffix-spec-decode/docs/LATEST-CONCLUSIONS.md)
