# L0/L3 实验报告(V2 多轮画像)· suffix 投机解码上线判决

> 承接 [optimization-proposal.md](optimization-proposal.md) 的 L0/L3 方案与 [summary.md](summary.md) 的 v1 结论。本轮回答:**v2 多轮画像(18.1 summary/题)下,全局开 suffix 到底赚不赚(L0)?"逐字引用"共设计能不能把 summary 翻正(L3)?** 三个实验递进,2026-07-02 完成。

## 最终判决(一句话)
**L0 采用:v2 端到端全局开 suffix,墙钟 −29%、decode +16%,summary 在真实运行里不亏(x1.00)。L3 否决(当前形态):引用模式确实让 suffix 吞吐 +49%,但输出膨胀 +72% 把收益吃光倒贴,质量平手,还伤了 1/10 题。**

---

## E1:v2 replay(冷树 · batch=1 · 单条口径 = 保守下界)
60 summary(R1/R2/R3 各 20,按 harvest 里 SUPERVISOR_DECISION 计数归轮)+ 20 report,baseline/suffix 各一轮全新 server。

| | baseline→suffix | 加速 | v1 对照 |
|---|--:|--:|--:|
| SUMMARY(60)| 142→115 char/s | **x0.81**(R1 0.79/R2 0.83/R3 0.81,无轮次漂移)| v1 x0.955,v2 更伤 |
| REPORT(20)| 134s→69s/条 | **x1.87** | v1 单条省 37s,v2 省 65s |

单条实测 × v2 调用数(18.1+1):**每题 371→358s,仅 +3.5%** ——冷树口径下勉强为正。
(v2 report 更大:prompt 中位 ~85K 字符、输出中位 5818 token,27 份 summary 喂入。)

### E1 vs E2 到底差在哪(同一 suffix 配置,常见疑问)

| 维度 | E1 replay | E2 端到端 | 对 suffix 的影响 |
|---|---|---|---|
| **同题密度(主因)** | 60 条 summary 来自 ~47 个不同题(树里几乎无同题兄弟)| 每题 18 条相继流过同一 server,第 k 条生成时树里已有前 k−1 条(union 覆盖 31.7%)| summary 接受率 26.4%→42.5% |
| 并行度(次要) | 串行 batch=1 | summary 段峰值 9 路(9 路在飞时间占 54.6%) | 税摊薄 ↔ 收益稀释,方向相反 |
| 请求来源 | harvest 录制 prompt 原样重发 | demo 动态生成 | — |

注意:喂树靠的是**同题内容相继(以及同时)过同一台 server**,不是并行本身——E2 串行跑同样有此效应;E1 改并发跑也不会有(树里还是没同题兄弟)。树是逐 token 喂的(每步 `add_active_response`),同时在飞的请求可实时命中彼此已生成的前半段。判别证据:report 不依赖全局树,E1/E2 接受率一致(59.0%/56.1%)。

## E2:L0-V2 端到端(demo 完整跑 10 题 × 2 配置 = 真实口径)
q1-10,v2 同款参数(3 轮迭代;题间串行),RESEARCH_MODE=local 回放搜索缓存(两臂输入一致),逐调用计时(llm_calls.jsonl)。两臂各 10/10 报告零错。
**并发结构(按代码+时间戳实测)**:semaphore 只卡 researcher 层(`MAX_CONCURRENCY=3`,workflow.py:212),每个 researcher 的 3 个 sub-query 用 `asyncio.gather` **并行**(agents.py:467-480)→ summary 理论并发 3×3=9;**实测两臂瞬时最大并发均为 9;在"至少 1 条 summary 在飞"的时长里,同时 9 条在飞的时间占比:baseline 43.6% / suffix 54.6%**(请求级并发≈该时段引擎 decode batch,不是 GPU 算力利用率;decode 是显存带宽瓶颈,SM 利用率本就不高且未记录)。逐臂完整分布见 `~/modify-code-runs/suffix-spec-decode/concurrency_profile.txt`。

| 阶段 | baseline | suffix | 加速 |
|---|--:|--:|--:|
| RESEARCH_SUMMARY(171 次)| 3849s | 3851s | **x1.00** |
| FINAL_REPORT(10 次)| 1367s | 638s | **x2.14**(输出归一 x1.94)|
| QUERY_PLAN(57 次)| 71s | 56s | x1.26 |
| **TOTAL decode** | 5382s | 4639s | **x1.16** |
| **整臂墙钟** | 57.6min | 41.2min | **x1.40(−29%)** |

输出量对等已验:summary token 比 1.00(98110 vs 98254),report 比 0.90(temp0 非确定,故另给归一口径)。

> **⚠️ 订正(2026-07-06,决定性实验推翻本节"兄弟→翻盘"叙事)**:①"接受率 26.4%→42.5%"是 server 手工切窗混进 report 的**测量假象**——干净轨迹口径 e2e summary 接受率仅 28.3% ≈ 冷树 ~29%;②下方"union 覆盖 31.7%"是**字符**口径,换 token 口径兄弟 cov24 ~0%(和输入↔输出一样,见本文档 §3 的 char→token 塌陷规律,本节漏做了);③控制实验(接受率 vs 完整兄弟数)实测**全平**(Δ+0.3pp),兄弟连全预热都抬不了接受率。**真相:summary 翻身主因是并发摊薄+换 eager,全局树/兄弟对接受率贡献 ≈0、只经"更多短草稿"给 +9% tokens/step 且早饱和。** 详见 [examine-spec-tax/replay-vs-e2e-reversal.md](examine-spec-tax/replay-vs-e2e-reversal.md) 与 [questions-log-2.md Q3](questions-log-2.md)。下方原文保留作历史。

**为什么端到端远好于 replay(+3.5% → −29% 墙钟)——证据链(2026-07-02 补,修正初版两处说法)**:

差异集中在 **summary(x0.81→x1.00)**;report 两边一致(E1 x1.87 ≈ E2 归一 x1.94),无需额外解释。~~summary 翻身的主因是同题 summary 的"输出间逐字重合"喂饱了全局树~~(**已订正,见上方横幅**),四件证据:

| # | 证据 | 数值 | 出处 |
|---|---|---|---|
| 1 | 同题 summary **输出**两两逐字重合(≥8字块)| pairwise 22.9%,同 researcher 34.6% | [redundancy_summary.md](../../researcher-redundancy-v2/redundancy_summary.md) 的 out_contain8 |
| 2 | **union 覆盖**:每条 summary 被"同题更早生成的 summary"逐字覆盖 | **mean 31.7%(中位 31.7%,P90 56%)——比被自己 prompt 覆盖(27.8%)还高** | 本轮实算,l0v2_suffix 155 条 |
| 3 | **summary 段实际接受率**(server 周期日志按时间窗切)| E1 冷树(**对兄弟冷≠空树**;负载散在 47 题)26.4%(接受长度 1.51)→ **E2**(每题18条密兄弟)**42.5%(2.59)** | server_v2_suffix.log vs server_l0v2_suffix.log |
| 4 | report 段接受率两边相同(59.0% vs 56.1%)| 变化只发生在 summary、且是**草稿质量**变化(batch 不改变接受率)| 同上 |
| 5 | E1 workload 同题密度≈0(34 题 1 条、13 题 2 条)vs E2 全 18 条陆续入树 | 树内"食物"结构性差异 | workload_v2.jsonl |

**两处修正**(初版说法不准,经 redundancy-v2 数据对质):
- ✗"共享来源":同题 summary 的**输入**其实几乎不共享(url_jac 中位 0)。✓ 真正重合的是**输出**——不同来源讲同一主题,吐出相同的实体/数字/术语/句式,同 researcher 跨 sub-query 重复表述更多(34.6%)。全局树缓存的是**输出**,所以照样吃到。
- ✗"report 双保险(全局树也帮 report)":全局树对 report 基本无增益——v1 消融关全局树 report 仅 1.84→1.76,且 E1(树里无同题 summary)与 E2 的 report 接受率几乎相同。report 的加速来自 **prompt 树**(它 prompt 里本来就含全部 summary)+ 自己已生成前文。
- 次要贡献(未单独测量):summary 段多数时间 8-9 条请求同时在飞(9 路占 54.6%),"关异步调度"的固定税被高并发摊薄。但注意 batch 变大同时**稀释**投机收益(抢算力)——两个方向相互抵消的合力未拆分;接受率近乎翻倍(26.4%→42.5%)是唯一实测的、且只能由树内容解释的信号,故列为主因。

墙钟加速(1.40)大于 decode 总和(1.16)的原因不变:summary 段高并发(实测多数时间 9 路在飞,时间被并行吸收),**report 是单线程串行段(每题只 1 个调用、无并行伙伴),省的 73s/题 1:1 落墙钟**。

> **测量方法学结论:评估投机解码对 agent 工作负载的价值,必须端到端测。** 隔离 replay 的失真不止"冷树",更在**丢失了工作负载的内部结构**(同题输出相继生成、彼此重合)——本实验里低估一个量级(3.5% vs 29%)。

## E3:L3-V2 引用共设计(SUMMARY_QUOTE=1,机制成立、经济账不成立)
代码:[agents.py:351-361](../../../deep_researcher_demo/agents.py#L351-L361) env 门控"逐字引用+URL+限长"(/modify-code 全流程物证:`~/modify-code-runs/l3-quote-prompt/`,含 smoke#1 撑爆上下文→AMEND 限长的过程)。同 10 题、冷 suffix server,对照=E2 的 suffix 臂。

| | suffix(对照)| quote+suffix | 读法 |
|---|--:|--:|---|
| summary decode tok/s | 25.5 | **37.9(x1.49)** | **机制验证成功:引用确实让 suffix 咬上了** |
| **summary 接受率**(窗口口径,两臂对等) | 37.5% | **46.8%(+9.3pp)** | **强制引用是唯一能抬接受率的杠杆**(兄弟/热树抬不动,见 [questions-log-2.md Q3](questions-log-2.md));⚠️窗口法高估绝对值(对照干净轨迹口径仅 28.3%),**+9pp 差值方向可信、绝对值折算约 28%→~37%**;每步 token 2.04→2.44(+20%) |
| summary 输出 token | 98K | **168K(+72%)** | 限 600 词没管住(936 vs 595 tok/条,res 实测);**tok/s 快 49% 但要吐的 token 多 72% → 净时间反倒 +15%,加速与膨胀在同一旋钮上对冲** |
| summary decode 总时间 | 3851s | **4442s(+15%)** | 膨胀吃光提速还倒贴 |
| TOTAL decode / 墙钟 | 4639s / 41.2min | 5150s / 46.9min | 全面变差 |
| 完成率 | 10/10 | **9/10**(q3 report prompt 撑爆 40960→400)| 膨胀的连带伤害 |
| **质量(kimi-k2.5 pairwise,A/B 随机序,9 题)** | — | **ctrl 4 / quote 3 / tie 2 ≈ 平手** | 无质量红利;评审还点名 quote 版"同一来源伪装多源"的引用问题(q10)|

**判决:按当前形态不采用。** 要救活的方向(未做):更硬的长度控制(砍 `RESEARCH_SUMMARY_MAX_TOKENS` 而非靠 prompt 自觉)、只对数字/定义类关键句引用。但既然 E2 证明 summary 端到端已不亏,L3 的必要性从"前提"降回"锦上添花",优先级低。

## 部署结论(v2 多轮画像)
1. **直接全局开 suffix,默认参数**(`{"method":"suffix"}`):墙钟 −29%,report x2、summary 不亏、小调用还白赚(QUERY_PLAN x1.26);
2. 全局树是核心收益来源之一 → **别设 `max_cached_requests=0`**;长跑 server 越跑越赚;
3. 本实验题内并发 3 已含在结果里;更高并发上线前仍建议小流量验证;
4. L3/L2 不再是前提;真要继续优化,先做"更高并发档位"和"LMCache 兼容"两个一次性验证。

## 物证
- E1:`~/modify-code-runs/suffix-spec-decode/`(workload_v2.jsonl、res_v2_*.jsonl、metrics_v2_*.txt、v2_status.txt)
- E2/E3:`eval/results/drbench/{l0v2_baseline,l0v2_suffix,l3v2_quote}/q*/`(report.md/harvest.jsonl/llm_calls.jsonl)+ `~/modify-code-runs/suffix-spec-decode/{l0v2_status.txt,l3v2_status.txt,judge_l3_results.jsonl,judge_l3.py}`
- L3 代码改动审查包:`~/modify-code-runs/l3-quote-prompt/review-report.md`(交接闸待确认)
