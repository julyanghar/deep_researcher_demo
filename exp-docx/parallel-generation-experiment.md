# 实验报告:大纲驱动的并行生成(summary / report)

> **⚠️ 本文为 v1,已被 [parallel-generation-experiment-v2.md](parallel-generation-experiment-v2.md) 取代**(v2 增补:去冗余 prompt、逐请求 acc/draft/输出、膨胀数 1.94×→1.59× 纠正、run 间变异边界)。v1 保留存档。

> **一句话**:给 summary 和 report 生成各加一个 env 开关,把"一次整块生成"改成"**先列大纲 → 每节指定源 → 各节并行 → 拼接**"。e2e A/B(local + Ceager,TP2 与 TP4 各 3 配置 × 3 题)结论:**report 并行大赚(TP4 报告段 2.79×、e2e −25%),summary 并行净亏(内容膨胀 1.94×)。最优 = report 开、summary 关。**
> **代码**:[../deep_researcher_demo/agents.py](../deep_researcher_demo/agents.py)、[../deep_researcher_demo/schemas.py](../deep_researcher_demo/schemas.py);**物证**:[../../modify-code-runs/parallel-gen/](../../modify-code-runs/parallel-gen/)。

---

## 一、改了什么(代码索引)

纯新增、双 env 门控、默认关=字节不变;KV-reuse 模式下自动禁用。

| 位置 | 作用 | 代码 |
|---|---|---|
| `Outline`/`OutlineSection` | 大纲数据模型(节标题 + source_ids) | [schemas.py:28](../deep_researcher_demo/schemas.py#L28-L40) |
| `_env_flag` | 调用时读 env(进程内设置生效) | [agents.py:43](../deep_researcher_demo/agents.py#L43-L48) |
| **`outline_then_parallel`** | 共享机制:大纲→并行分节→拼接(summary/report 共用) | [agents.py:146](../deep_researcher_demo/agents.py#L146-L221) |
| `_SUMMARY_OUTLINE_SYSTEM` | summary 大纲提示 | [agents.py:223](../deep_researcher_demo/agents.py#L223-L229) |
| `summarize_results` 分支 | `SUMMARY_PARALLEL` 开关入口 | [agents.py:457](../deep_researcher_demo/agents.py#L457-L474) |
| `FinalWriter._OUTLINE_SYSTEM` / `_SECTION_SYSTEM` | report 大纲/分节提示 | [agents.py:656](../deep_researcher_demo/agents.py#L656-L669) |
| `FinalWriter.write` 分支 | `REPORT_PARALLEL` 开关入口 | [agents.py:711](../deep_researcher_demo/agents.py#L711-L712) |
| `FinalWriter._write_parallel` | report 并行 + 拼 References | [agents.py:764](../deep_researcher_demo/agents.py#L764-L794) |

**机制**([agents.py:146](../deep_researcher_demo/agents.py#L146-L221)):① 一次 `call_json` 出 `Outline`(节+source_ids);② `asyncio.gather` 各节,每节只喂**它分到的源子集**;③ `\n\n` 拼接,加标题(summary `### `、report `## `)。大纲空/解析失败 → 回退单次生成(不崩)。

---

## 二、e2e A/B 结果(local + Ceager,3 配置 × 3 题)

配置:`off`=无大纲 / `rep`=只 report 大纲 / `both`=summary+report 都大纲。生效核对全 OK(off 零大纲、rep 3 个 report 大纲、both 63 summary + 3 report 大纲)。

### e2e 墙钟(status 实测)

| 配置 | TP2 | TP4 |
|---|---|---|
| off | 945s | 879s |
| **rep** | 823s(−13%) | **659s(−25%)** ⭐最优 |
| both | 1103s(+17%) | 772s(−12%) |

### 阶段拆(interval-merge)+ 关键比值

| | TP2 | TP4 |
|---|---|---|
| **report 段加速**(off→rep) | 202→129s = **1.56×** | 251→90s = **2.79×** |
| **summary 段代价**(off→both) | 404→516s = **+28%** | 297→338s = **+14%** |
| both vs baseline(e2e) | +17% 更慢 | −12% 更快 |
| 检索(非LLM,固定) | ~245s | ~257s |

> 物证:[e2e3_status.txt](../../modify-code-runs/parallel-gen/e2e3_status.txt)(TP2)、[e2e3_tp4_status.txt](../../modify-code-runs/parallel-gen/e2e3_tp4_status.txt)(TP4);逐调用计时在各配置的 `eval/results/drbench/{p3,p4}{off,rep,both}/q*/llm_calls.jsonl`。

---

## 三、机制分析:为什么 report 赢、summary 亏

### report 并行 = 赚,GPU 越多越赚

report 是**单流阶段**(每题一次调用),GPU 空。拆成 N 节并发,直接填空闲算力 → TP2 报告段 1.56×,**TP4 翻到 2.79×**(卡多一倍,并发铺得更开)。逐题:单次 67s/题 → 并行 43s/题(TP2)。report 的源(各 summary)天然不同话题、可干净分。

### summary 并行 = 亏,根子是**内容膨胀**不是算力

**决定性数字(每子查询)**:

| | token | 相对单次 |
|---|---|---|
| 单次 summary | 565 | 1.00× |
| **并行拼接后 summary** | **1096** | **1.94×** |
| + 废弃大纲(纯浪费) | 106 | — |
| 并行总 decode 量 | 1202 | **2.13×** |

一个子查询被切 ~3.2 节、每节 ~277 token,各自重开背景、内容重叠 → 拼起来近翻倍。**这坨膨胀砸在已 8-9 路并发、GPU 饱和的 summary 阶段** → 排队变长 → 墙钟涨。

**TP2 vs TP4 坐实了"膨胀 ≠ 饱和"**:summary 段代价 TP2 **+28%** → TP4 **+14%**,砍了一半——**一半是"饱和排队"(加卡吸收)**,**另一半 +14% 是"内容膨胀"(加卡去不掉,内容层面)**。所以 summary 并行"又慢(饱和)又长(膨胀)又可能更差(重复)",且加卡只治一半。

> 逐子查询归因物证 + 一个"丢源"实例见 [summary_split_example.md](../../modify-code-runs/parallel-gen/summary_split_example.md):真例子里大纲用了源 [1,2,3,5]、**整篇丢弃 [4,6]**(信息损失)。

### ⚠️ 数据纠正 + prompt 去冗余(2026-07-08)

**纠正**:上面 summary 膨胀 **1.94×** 系"大纲结束后 0.8s 窗口聚簇"在 8-9 路并发下**误纳了别的子查询的分节 → 高估**。不依赖聚簇的**全局平均**:分节总 token 56.5k ÷ 63 子查询 = 899/子查询 = **~1.59×(分节)/ 1.78×(含废弃大纲)**。**真实膨胀 ~1.59×,不是 1.94×**(上表 1.94×/2.13× 按此下修)。

**prompt 去冗余(已实现,[agents.py](../deep_researcher_demo/agents.py))**:给每个分节 prompt 加**兄弟节清单 `<outline>` + 专注指令**(只写本节/不铺垫/不重复/简洁),summary 分节措辞由"答整个子查询"改 terse,report 分节"detailed"改"concise"。**受控 A/B(同 5 子查询、同大纲,唯一变量 prompt)**:分节输出 353 → **267 tok/子查询(−24%)**,输出更干净(去"以下几个方面:1.2.3."铺垫、平铺同要点)。审查报告 [review-report-section-focus.md](../../modify-code-runs/parallel-gen/review-report-section-focus.md)、物证 [bloat_result.txt](../../modify-code-runs/parallel-gen/bloat_result.txt)。**真实 e2e 影响(summary 净亏缩没缩、report 快没快)待新 prompt 重跑 TP4 确认。**

### suffix 命中率:分节 vs 不分节(⚠️ 仅存档记录,不写入论文 — 用户决定)

`/metrics` 差分实测(空载 Ceager,`spec_decode_num_{draft,accepted}_tokens_total`,5 summary / 2 report 单次跑):

| 生成 | draft | accepted | 命中率 |
|---|---|---|---|
| summary 不分节 | 2131 | 502 | **23.6%** |
| summary 分节 | 2904 | 666 | **22.9%** |
| report 不分节 | 5828 | 2819 | **48.4%** |
| report 分节 | 4272 | 1662 | **38.9%** |
| (summary 大纲JSON本身) | 814 | 362 | 44.5% |

**结论:分节没提命中率——summary 平(23.6→22.9),report 反而降 ~10pp(48.4→38.9)。** 原因:分节 + "detailed/analytical + inline 引用" 逼模型**改写扩写** findings(=那 1.94× 膨胀),而改写=非逐字=不可抄 → 命中率掉。

**机制含义(仅记录)**:**分节(并行)与 suffix 命中是互相打架的**——report 并行**变快靠的是并行填空闲 GPU,不是 suffix,甚至 suffix 效率还降了**。真想靠 provenance 抬命中率得走"生成不变、只收窄草稿源"的另一条路(drafter-scoping,未实现),而非分节改写。样本小(单次跑),report 10pp 降幅方向明确。物证:[accept_result.json](../../modify-code-runs/parallel-gen/accept_result.json)、脚本 [measure_accept.py](../../modify-code-runs/parallel-gen/measure_accept.py)。

---

## 四、现用 prompt + 冗余诊断 + 待调整

### 现在实际在用的 prompt

- **summary 大纲**([agents.py:223](../deep_researcher_demo/agents.py#L223-L229)):"Split into **2-5 short, non-overlapping sections** …"
- **summary 分节** = 复用 `summary_instruction`([agents.py:439](../deep_researcher_demo/agents.py#L439-L441)):"Compress … **for the current sub-query** …"
- **report 大纲**([agents.py:656](../deep_researcher_demo/agents.py#L656-L661)):"Split into clear, **non-overlapping sections** …"
- **report 分节**([agents.py:662](../deep_researcher_demo/agents.py#L662-L669)):"write the body of the section … **Be detailed and analytical** …"
- **分节 user 消息**([agents.py:176](../deep_researcher_demo/agents.py#L176-L182)):只含 `<question>` + `<section_to_write>标题` + `<sources>本节分到的源`。

### 膨胀从哪来(prompt 层面)

1. **各节彼此失明**:分节 prompt 只给自己的标题+源,**不知道别的节写什么** → 每节重开 preamble、相邻节撞车。大纲那句 "non-overlapping" 只约束规划、约束不到生成([agents.py:176](../deep_researcher_demo/agents.py#L176-L182) 的 user 消息里没有兄弟节信息)。
2. **措辞鼓励啰嗦**:summary 分节仍 "Compress … for the current **sub-query**"(每节去答整个子查询);report 分节 "**Be detailed and analytical**"(每节都往详细写,×N)。
3. **无长度预算**:两个分节 prompt 都没说"简短/别铺垫/别复述问题"。

### 待调整(直击膨胀,尚未实现)

- **让每节知道兄弟节**:分节 user 消息塞全大纲清单 + system 加 "write ONLY <section_to_write>, don't cover other sections, no intro/preamble/restating the question"。
- summary 分节改 "Extract ONLY this section's aspect, be terse, no preamble"。
- report 分节保留 "detailed" 但加 "no filler, don't overlap other sections"。
- 预期:把 1.94× 压回近 1×,summary 净亏可能转正,report 输出更短→prefill 更轻→e2e 再省;质量更干净。
- **注**:106 token 废弃大纲是两步法固有的,prompt 调不掉(除非把大纲合进第一节)。

---

## 五、待办(未闭环)

- [ ] **质量判官**:report 并行版输出膨胀到 2-3× token(8 节 vs 1 篇),`−25%` 速度**必须过 kimi-k2.5 盲判**确认不掉质量才算数(现只验了"3/3 出报告没崩")。
- [ ] **prompt 去冗余**(§四)+ 重跑一次量膨胀掉多少。
- [ ] 样本从 3 题扩到 10 题,坐稳 report 段比值(3 题噪声 ~±40s)。

---

## 六、物证索引

| 物证 | 路径 |
|---|---|
| 代码审查报告(改动/验收/A5核对) | [review-report.md](../../modify-code-runs/parallel-gen/review-report.md) |
| summary 拆分实例(含丢源) | [summary_split_example.md](../../modify-code-runs/parallel-gen/summary_split_example.md) |
| TP2 run status | [e2e3_status.txt](../../modify-code-runs/parallel-gen/e2e3_status.txt) |
| TP4 run status | [e2e3_tp4_status.txt](../../modify-code-runs/parallel-gen/e2e3_tp4_status.txt) |
| 逐调用计时 | `eval/results/drbench/{p3,p4}{off,rep,both}/q*/llm_calls.jsonl` |
| 代码基线快照 | [../../modify-code-runs/parallel-gen/baseline/](../../modify-code-runs/parallel-gen/baseline/) |

## 七、结论

1. **report 开大纲、summary 关大纲**——两个平台一致,TP4 更明显(rep 659s 严格最优)。
2. **加速本质**:report 靠"填空闲 GPU"(加卡放大,2.79×);summary 亏在"内容膨胀近翻倍"(加卡只稀释饱和、动不了膨胀)。这是"并行只在**空闲**阶段有效、**饱和**阶段反噬"的实测。
3. 对论文:这批数据是 [suffix-spec-decode/docs/explore-idea/paper-skeleton.md](suffix-spec-decode/docs/explore-idea/paper-skeleton.md) 里"角色②并行"的实证——**并行加速真实存在但强烈依赖阶段的 GPU 空闲度与内容可分性**,而非无条件收益。

---

## 附录 A:四类调用 输入→输出 实例(真数据 + Ceager 现跑)

> 源块为可读性截断(源 snippet ~150字、findings ~200字);**system 全文、输出全文**(输出正好展示分节的啰嗦/引用形态)。全量未截断版:[../../modify-code-runs/parallel-gen/outline_section_io_examples.md](../../modify-code-runs/parallel-gen/outline_section_io_examples.md)。

### ① summary 大纲(system=`_SUMMARY_OUTLINE_SYSTEM`)

**system**:`Plan a short outline for compressing the search results into a digest for the sub-query. Split the answer into 2-5 short, non-overlapping sections. For each section give a concise title and the ids of the search results it should use … Return only JSON {"sections":[{"title","source_ids"}]}`

**user**:
```
<question>
如何定义和识别中国社会中的中产阶层，并基于可获得的数据分析其收入、资产和消费特征？
</question>

<search_results>
[1] Query: 中国中产阶层消费特征和趋势
Title: [PDF] 中国新中产圈层白皮书 - 胡润百富
URL: https://res.hurun.cn/...
Snippet: 购买力，带动社会的消费潮流…新中产人群寻求"休闲享受"…消费更趋向理性…
[2] Title: 郁方中国的中产阶级与他们的消费文化 …
[3] Title: 中产阶级的消费水平和消费方式 …
[4] Title: McKinsey 中产阶级重塑中国消费市场 …
[5] Title: 中国中产阶层崛起及其对经济的影响 …
[6] Title: 城市化与中产阶层成长 …
</search_results>
```

**输出**:
```json
{"sections": [
  {"title": "中产阶层的消费特征与趋势", "source_ids": [1, 2, 3]},
  {"title": "中产阶层的经济影响与规模预测", "source_ids": [5, 6]}
]}
```
> 注:大纲用了 [1,2,3,5,6],**丢了源 [4] McKinsey**——分节固有的丢源问题(见 §三)。

### ② summary 分节(system=`summary_instruction`;本节"消费特征与趋势"只喂源 [1,2,3])

**system**:`Compress the provided search results for the current sub-query. Only Extract key information that is relevant to the overall research question. Write plain text only.`

**user**:`<question>…同上…</question>` + `<section_to_write>中产阶层的消费特征与趋势</section_to_write>` + `<search_results>[1][2][3] 三篇(只本节分到的)</search_results>`

**输出**(纯文本,无引用):
```
中产阶层的消费特征与趋势显示，他们作为社会消费的主体力量，消费观念日益理性，注重"休闲享受"的消费方式，消费结构中休闲娱乐占比最高，其次是金融保险和理财领域。中产阶层的消费行为不仅影响社会消费潮流，也推动了消费升级。此外，中产阶级在家庭耐用品拥有率、住房和汽车消费方面表现出与其他社会阶层的显著差异。消费分层趋势明显，不同群体的消费水平和方式逐渐分化。
```

### ③ report 大纲(system=`_OUTLINE_SYSTEM`)

**system**:`Plan an outline for a research report that answers the question using ONLY the findings. Split it into clear, non-overlapping sections. For each section give a title and the ids of the findings it should draw from … Return only JSON`

**user**:
```
<question>
中国中产阶层有哪些显著特点？
</question>

<findings>
[1] sources: https://res.hurun.cn/...
**中国中产阶层的定义与识别：** …中产阶层多为受薪白领…麦肯锡定义家庭年收入6万—22.9万…
[2] sources: https://res.hurun.cn/...
…人数规模：2017年3.9亿…2025年突破4亿…
[3] sources: https://baike.baidu.com/...
…2023年约4亿人，2030预计超8亿…
[4] sources: http://www.sociology2010.cass.cn/...
…教育：大专及以上学历，重视子女教育投入…
</findings>
```

**输出**:
```json
{"sections": [
  {"title": "中国中产阶层的定义与识别", "source_ids": [1]},
  {"title": "中国中产阶层的人数规模与发展趋势", "source_ids": [2, 3]},
  {"title": "中国中产阶层的教育特征与家庭投入", "source_ids": [4]}
]}
```

### ④ report 分节(system=`_SECTION_SYSTEM`;本节"定义与识别"只喂 finding [1])

**system**:`You are writing ONE section of a research report. Using ONLY the provided findings, write the body of the section named in <section_to_write> in Markdown. Be detailed and analytical. Do NOT repeat the section title as a heading and do NOT add a references list. Cite sources inline … Use the same language as the question.`

**user**:`<question>中国中产阶层有哪些显著特点？</question>` + `<section_to_write>中国中产阶层的定义与识别</section_to_write>` + `<findings>[1] 只本节分到的那条</findings>`

**输出**(markdown + inline [url] 引用,明显比源啰嗦——膨胀来源):
```
中国中产阶层的定义与识别涉及多个维度，包括收入水平、职业特征和社会地位。通常而言，中产阶层是指在社会结构中处于中间位置的群体，其成员多为受薪白领，拥有稳定的职业和收入来源 [https://res.hurun.cn/...]。从收入角度来看，麦肯锡的研究将中产阶层定义为家庭年收入在6万至22.9万元人民币之间的群体…[https://res.hurun.cn/...]。然而，胡润百富…新中产人群的平均家庭年收入为65万元…[https://res.hurun.cn/...]…（一条 finding 被扩写成数倍长度）
```
> ④ 直观展示膨胀:一条 ~200 字的 finding,被"detailed and analytical + 每句加引用"扩写成数倍长的段落——这就是 report/summary 分节输出膨胀的 prompt 级来源(见 §四 待调整)。
