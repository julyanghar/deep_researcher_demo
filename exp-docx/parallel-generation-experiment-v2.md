# 实验报告 v2:大纲驱动的并行生成(summary / report)

> **取代 v1** [parallel-generation-experiment.md](parallel-generation-experiment.md)(v1 保留存档)。v2 增补:去冗余 prompt 改动、逐请求 acc/draft/输出(真实 traj)、膨胀数纠正(1.94×→1.59×)、run 间变异的诚实边界。
> **代码**:[../deep_researcher_demo/agents.py](../deep_researcher_demo/agents.py)、[../deep_researcher_demo/schemas.py](../deep_researcher_demo/schemas.py);**物证**:[../../modify-code-runs/parallel-gen/](../../modify-code-runs/parallel-gen/)。

## TL;DR

给 summary/report 生成各加 env 开关,把"一次整块生成"改成"**大纲→每节指定源→并行→拼接**"。核心结论(跨 TP2/TP4/新旧 prompt **一致、稳健**):

1. **report 并行 = 赚**:报告段 TP2 **1.56×** / TP4 **~2.8-3×**,e2e −13~25%。因 report 是单流阶段、GPU 空,并发分节填空闲算力。
2. **summary 并行 = 看长度**(⚠️ 受控实验修正,§五):去冗余 prompt 让它提速(逐条 **−34%** / batch **−12%**);并发下并行比单次快(0.55×),但**长 summary(≥4s)才赚、短的(≤1s)因大纲固定开销倒亏**——早先"净亏"是 e2e interval-merge 的指标伪影。
3. **report 大纲 = 稳赚**(单流空 GPU);**summary 大纲该按长度门控**(长的开、短的别开),非一刀切。
4. **prompt 去冗余**(专注本节)在受控 A/B 下把分节输出砍 **−24%**,但 **e2e 层面被 ~20% run 间变异淹没、测不出**。

## 一、改了什么(代码索引)

**A. 并行特性(双开关)** —— [agents.py](../deep_researcher_demo/agents.py):`outline_then_parallel`([:146](../deep_researcher_demo/agents.py#L146))+ `summarize_results`/`FinalWriter.write` 两分支([:457](../deep_researcher_demo/agents.py#L457)、[:711](../deep_researcher_demo/agents.py#L711))+ `Outline` 模型([schemas.py:28](../deep_researcher_demo/schemas.py#L28))。默认关=字节不变,KV-reuse 下禁用。

**B. 去冗余 prompt(专注本节)** —— 给每分节注入**兄弟节清单 `<outline>` + `_SECTION_FOCUS`**(只写本节/不铺垫/不重复)([agents.py:172](../deep_researcher_demo/agents.py#L172)、[:245](../deep_researcher_demo/agents.py#L245));summary 分节改 terse([:501](../deep_researcher_demo/agents.py#L501)),report 分节 "detailed"→"concise"([:696](../deep_researcher_demo/agents.py#L696))。审查报告 [review-report-section-focus.md](../../modify-code-runs/parallel-gen/review-report-section-focus.md)。

## 二、e2e 结果(local + Ceager,3 配置 × 3 题)

| 平台/prompt | off | rep(只report) | both | report 段(off→rep) | summary 段(off→both) |
|---|---|---|---|---|---|
| **TP2** 旧 | 945 | **823(−13%)** | 1103(+17%) | 202→129 = **1.56×** | 404→516 = +28% |
| **TP4** 旧 | 875 | **656(−25%)** | 769 | 251→90 = **2.79×** | 297→338 = +14% |
| **TP4** 新prompt | 771 | **641** | 847 | 225→72 = **3.1×** | 233→373 |

- **rep 始终最优**;report 段并行加速随 GPU 增多而增大(1.56×→2.8-3×)。
- **⚠️ run 间变异 ~20%**:`off` 配置(prompt 不影响它)在 TP4 新旧 run 间也差 875→771(−12% e2e)、summary token −20%——TP4 浮点归约不确定 + 子查询生成非确定性所致。**故 e2e 层面的"新 prompt vs 旧"对比不可靠,3 题不足以压过噪声**。

## 三、逐请求 acc / draft / 实际输出(真实 e2e,SUFFIX_TRAJ)

server 带 `SUFFIX_TRAJ` 重跑,逐请求逐步记 `{r:req_id, acc, dr}`;按 `r` 聚合(accepted=Σacc、draft=Σdr)+ join llm_calls 的 completion。**609 请求全 join(0 未匹配)**:

| 请求类型 | n | 输出tok | draft | accepted | **命中率(acc/draft)** | acc/输出 |
|---|---|---|---|---|---|---|
| RESEARCH_SUMMARY_TEXT | 378 | 116434 | 165858 | 53021 | **32%** | 46% |
| FINAL_REPORT_MARKDOWN | 47 | 46413 | 57860 | 33022 | **57%** | 71% |
| RESEARCH_SUMMARY_OUTLINE | 81 | 8603 | 14671 | 5761 | 39% | 67% |
| QUERY_PLAN_JSON | 65 | 3373 | 4206 | 1811 | 43% | 54% |
| SUPERVISOR_DECISION | 22 | 3558 | 4935 | 1630 | 33% | 46% |

**关系钉死**:`accepted ≤ draft` 且 `accepted < 实际输出` 在**每一个请求**上都成立(输出 = accepted + 步数,accepted 只是"照抄白赚"的那部分)。抽样最大 report:输出10000/draft11459/acc8127/1876步 → acc<输出。命中率 report(57%)≫ summary(32%),因 report 照搬多。物证:`traj_v2.<rank>` + [measure_per_request.py](../../modify-code-runs/parallel-gen/measure_per_request.py)。

## 四、机制分析

**report 赢**:单流阶段、GPU 空 → 并发分节填空闲算力;GPU 越多越赚(TP2 1.56×→TP4 ~3×)。report 源(各 summary)天然不同话题、可干净分。

**summary 亏**:阶段本已 8-9 路并发、GPU 饱和 → 分节多产的 token 只变排队;而且**分节输出膨胀 ~1.59×**(3 节各自铺垫/重叠)。TP2→TP4 把 summary 段代价从 +28% 降到 +14%(加卡吸收了"饱和"那一半),但另一半是**内容膨胀**,加卡去不掉。

**⚠️ 膨胀数纠正**:v1 报的 **1.94× 系"大纲后 0.8s 窗口聚簇"在并发下误纳别子查询分节 → 高估**;不依赖聚簇的全局平均是 **~1.59×(分节)/1.78×(含废弃大纲)**。

## 五、prompt 去冗余:能不能减 summary/report 的冗余?

**能。** 完整成对 A/B(同输入、同大纲,唯一变量=prompt,彻底排除 run 间噪声):

| | 样本 | 旧 tok/input(均/中位) | 新 tok/input | **减少率(均/中位)** | 减少/变长 |
|---|---|---|---|---|---|
| **summary** | 20 子查询 / 53 节 | 385 / 324 | 298 / 274 | **−18.7% / −16.5%** | 16 减 / 4 长 |
| **report** | 3 篇 / 6 节 | 1041 / 780 | 693 / 540 | **−28.9% / −30.8%** | 3 减 / 0 长 |

**结论**:report 减得狠且稳(−29%,3/3 全减);summary 温和(−18.7%,16/20 减,4 个本就紧凑的段略长——去冗余不硬压已经紧的段)。物证 [bloat_experiment_result.json](../../modify-code-runs/parallel-gen/bloat_experiment_result.json)、脚本 [bloat_experiment.py](../../modify-code-runs/parallel-gen/bloat_experiment.py)。

**逐段真实改前/改后**(直观看砍了什么)见 [real_output_before_after.md](../../modify-code-runs/parallel-gen/real_output_before_after.md):砍的是"**以下几个方面 1.2.3. 铺垫 + 编号骨架 + 重复复述数字**"这三类注水,**事实与 inline 引用全留**——是去冗余非删内容。样例:summary 节 202→82(去 5 条编号列表)、report「人数规模」750→502(删掉把数字又复述一遍的"发展趋势"段)。

**⚠️ 但"能减冗余"≠"e2e 能变快"**:e2e 层面被 ~20% run 间变异淹没、测不出(见 §二)。而且——

### 为什么减了冗余,summary 并行还是变慢?

去冗余减的只是"每节输出 token",但 summary 并行比单次多花的时间有**四块,去冗余只碰第 4 块**:

| # | 多出来的开销 | 去冗余碰得到? |
|---|---|---|
| 1 大纲调用 | 每子查询多一次请求(~106 tok,还废弃) | ❌ |
| 2 请求数翻倍 | 1 → 1大纲+N分节,砸进已饱和队列 | ❌ 结构没变 |
| 3 分节重复 prefill | 每节各自 prefill 它的源子集 | ❌ |
| 4 分节输出 decode | N 段总输出 | ✅ 只减这块 −18.7% |
| 背景 | summary 阶段 GPU 已 8-9 路饱和 | ❌ |

上面四块开销结构性存在。但"summary 并行到底亏不亏"要用**受控实验**答,不能只看被 run 变异污染的 e2e。

#### 受控实验修正(⚠️ 推翻早先"summary 并行净亏"的判断)

**A. 受控并发 replay**(18 真子查询,并发 8,同输入,唯一变量=prompt/拆不拆,TP4):

| 臂 | 墙钟 | 总请求 | 总输出tok |
|---|---|---|---|
| 单次 | 16.9s | 18 | 4310 |
| 旧-并行 | 10.6s | 64 | 6783 |
| 新-并行 | **9.3s** | 65 | 5348 |

→ 并发吞吐下 **并行比单次快**(0.55×),**新 prompt 比旧快 12%**。

**B. 逐子查询隔离计时**(18 个各自单独计时,揭示 batch 均值藏起来的规律):
- **去冗余(新 vs 旧并行):14/18 变快,中位 0.66×(新比旧快 ~34%)** —— 修改确实让并行提速,稳。
- **并行 vs 单次:中位 0.96×(≈平),9 快 9 慢** —— 且**强烈依赖 summary 长度**:长 summary(单次≥4s)并行大赚(0.26-0.34×,拆长生成砍关键路径),短 summary(单次≤1s)并行反亏(1.5-3.8×,大纲固定开销压死小生成),交叉点约 2-3s(≈大纲那~1s 开销量级)。

**修正后的结论(替代早先的"summary 一律别并行")**:
1. **去冗余 prompt 确实让 summary 并行提速**(逐条 −34%、batch −12%);
2. 早先"summary 并行净亏"是**基于 e2e 的 interval-merge(会高估并行段跨度)+ TP2 饱和假设**得出的,**在 TP4/够并发/受控条件下站不住**——TP4 batch 8 未饱和,有空闲吃下并行的短请求;
3. **并行该不该开取决于 summary 长度**:长的开(省一半多)、短的别开(大纲开销倒亏)→ **正确做法是按预估长度/源数门控,不是一刀切**;
4. 物证:[summary_replay_result.json](../../modify-code-runs/parallel-gen/summary_replay_result.json)、[per_subquery_result.json](../../modify-code-runs/parallel-gen/per_subquery_result.json)、脚本 [summary_replay.py](../../modify-code-runs/parallel-gen/summary_replay.py) / [per_subquery.py](../../modify-code-runs/parallel-gen/per_subquery.py)。

> **⚠️ §二 的"summary 段 interval-merge +14%"据此重判**:那是"并行 summary 请求铺得更散、活跃跨度更长"的指标伪影,不代表吞吐更慢;真实吞吐见本节受控 replay(并行更快)。

## 六、suffix 命中率:分节 vs 不分节(⚠️ 仅存档,不写入论文 — 用户决定)

`/metrics` 差分实测(5 summary / 2 report):summary 分节 23.6%→22.9%(平)、report 分节 48.4%→**38.9%(降 10pp)**。原因:分节+"详细分析+inline引用"逼模型改写扩写 → 非逐字 → 命中率掉。**含义(仅记录):分节(并行)与 suffix 命中互相打架**;report 并行变快靠"填空闲 GPU"不是 suffix。要靠 provenance 抬命中率得走"生成不变、只收窄草稿源"的 drafter-scoping(未做)。物证 [accept_result.json](../../modify-code-runs/parallel-gen/accept_result.json)。

## 七、附录

- **四类调用输入→输出示例**:见 v1 §附录A / 全量 [outline_section_io_examples.md](../../modify-code-runs/parallel-gen/outline_section_io_examples.md)。
- **SUFFIX_TRAJ 用法**:server 起时带 `SUFFIX_TRAJ=<路径>`([vllm 的 suffix_decoding.py](../../anaconda3/envs/lmcache/lib/python3.12/site-packages/vllm/v1/spec_decode/suffix_decoding.py) 插桩),逐请求逐步写 `{r,acc,dr,ctx,t}`,每 TP rank 一文件(内容同);按 `r` 聚合得逐请求 acc/draft,`r.rsplit("-",1)[0]` = llm_calls 的 req_id 可 join completion。

## 八、结论

1. **report 开大纲(稳赚);summary 大纲按长度门控**(长的开省一半多、短的别开因大纲开销倒亏),而非一刀切关——受控 replay + 逐子查询已修正早先"summary 净亏"(那是 e2e interval-merge 伪影,详见 §五)。
2. **加速本质**:report 靠填空闲 GPU(加卡放大),summary 亏在内容膨胀(加卡只治饱和)。**并行只在空闲阶段有效、饱和阶段反噬**——对论文是硬约束(不能当无条件红利)。
3. **acc 恒 < 输出**(逐请求真实数据坐实);命中率 report 57% / summary 32%。
4. **未闭环**:report 并行版质量判官(锁 −25% 速度)、prompt 去冗余的 e2e 效应(需 10+ 题)。

## 物证索引

| 物证 | 路径 |
|---|---|
| 逐请求 traj | `../../modify-code-runs/parallel-gen/traj_v2.<rank>` |
| 逐请求测量脚本 | [measure_per_request.py](../../modify-code-runs/parallel-gen/measure_per_request.py) |
| 膨胀 A/B | [bloat_result.txt](../../modify-code-runs/parallel-gen/bloat_result.txt) |
| 命中率(仅记录) | [accept_result.json](../../modify-code-runs/parallel-gen/accept_result.json) |
| 审查报告(prompt 改动) | [review-report-section-focus.md](../../modify-code-runs/parallel-gen/review-report-section-focus.md) |
| 调用示例 | [outline_section_io_examples.md](../../modify-code-runs/parallel-gen/outline_section_io_examples.md) |
| run status(TP2/TP4/TP4新/TP4traj) | `../../modify-code-runs/parallel-gen/e2e3*status.txt` |
