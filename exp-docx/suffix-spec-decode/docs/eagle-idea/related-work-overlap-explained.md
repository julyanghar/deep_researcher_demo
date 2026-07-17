# 与我们重叠的现有工作：每篇讲透（零基础版）

> **缘起**：论文叙事稿的相关工作节（[once-generated-paper-narrative.md](once-generated-paper-narrative.md) §6）是给论文用的**压缩划界**——每篇一句话。本文是它的**展开底稿**：对每个近邻，用零基础讲法说清它到底做了什么、和我们哪条主张撞、撞到什么深度、停在哪里。写 rebuttal、被审稿人问"你和 X 有什么区别"时，答案都在这里。
> **读法**：先读"三个问题 + 三把尺子"（§0–§1，这是判断一切重叠的坐标系），再按需跳到 A/B/C/D 四组；每篇的格式固定：**做了什么 → 关键数字 → 与我们重叠 → 停在哪 → 划界句**。C 组只讲重叠角度，训练配方细节在 [eagle3-continued-training-references.md](../../../eagle-spec-decode/eagle3-continued-training-references.md)（分工不重复）。
> **数字口径**：各论文数字来自 2026-07-11 的四路对抗核查（一手全文/源码核出）；我们自己的数字出处见 [Once Generated.md](Once%20Generated.md)。

---

## TL;DR：一张地图

我们的主张 = **在 LLM 调用层，对 deep research 流水线做内容级 trace 刻画，并据此在写（decode）读（prefill）两侧联合加速**。近邻们各占一角、无人进入交集：

| 论文 | 对象 | 层 | 测/做了什么 | 撞我们哪条 | 停在哪 |
|---|---|---|---|---|---|
| TraceLab | coding agent | 系统层 | 真·trace 级刻画（4,300 会话） | "trace 级 agent 刻画"这件事本身 | 内容脱敏丢弃，物理上测不了照抄 |
| Agentic AI Workload Char. | ReAct agent | 系统层 | decode 主导、上下文增长 | 我们的两个单项结论 | 非 DR、无内容层 |
| Agent Memory | 记忆 agent | 系统层 | 逐调用成本/能耗 | "trace 级刻画"第三家 | 非 DR、无内容层 |
| Agentic Search in the Wild | **DR 式负载** | **搜索日志层** | 词项重叠 CTAR≈54%+轮次演化 | 最危险：DR 对象上的内容重叠测量 | 看不见 LLM 侧 |
| DeepResearch-ReportEval | **DR** | 报告质量层 | 报告段落间冗余（LLM 判） | "DR 冗余"这个词 | 非 trace、非逐字、非 serving |
| Byte-Exact Dedup in RAG | RAG prompt | 内容层 | 字节级 chunk 重复 | 逐字口径 | 单轮 RAG、无输出→输入方向 |
| AgentInfer | **DR** | 系统层 | KV 命中 92%→15% | DR 对象上的系统测量 | 零星 profiling、无内容层 |
| Parrot / Preble / Mooncake | 多 agent/混合 | prefix-token 层 | 前缀冗余 94–99% / 85–97% | "跨调用冗余"直觉 | prefix 级≠内容级、motivation 级 |
| SuffixDecoding | agentic 混合 | 机制 | 后缀树投机 5.3× | 写侧机制本身 | 只报接受率、无刻画无路由 |
| LLMA | RAG | 机制 | 输出照抄检索→2× | "照抄可加速"原始观察 | 单轮 RAG、无 agent 流水线 |
| SpecDec++ / BanditSpec | 通用 | 机制 | 学"发多长/选哪个 drafter" | 路由/下注这件事 | 信号=运行时反馈，非内容结构 |
| Aurora / OSD / 2604.26779 / PayPal | 流量/RL/商务 agent | 训练 | （继续）训 drafter | 特化训练这件事 | 无一在 agent 真实轨迹×post-hoc EAGLE 交集 |
| Plato | 多 agent 单query | 机制 | 骨架图并行+前缀 KV 复用 | 读侧+调用图利用 | 位置相关、自陈无法重排 |
| RelayCaching | 多 agent | 机制 | decode KV→下游 prefill | **读侧机制叙事（抢跑）** | 无刻画、无质量验证、无写侧耦合 |
| KVCOMM / CacheBlend / DroidSpeak | 多 agent/RAG | 机制 | 非前缀 KV 复用 | 读侧机制家族 | 复用率是方法产出非负载测量 |

---

## §0 判断"撞没撞"的三个问题

任何一篇看起来相似的论文，先问三件事——**桥句：撞不撞，看三件事：测的是谁（对象）、在哪一层测（层）、用什么尺子测（度量）。**

1. **对象**：coding agent、ReAct 单 agent、记忆系统、RAG、还是 deep research 多智能体流水线？对象不同，负载性质（照抄率、调用图形态）就不同。
2. **层**：同一个系统可以在四层被测量——**搜索日志层**（只看查询和检索结果，看不见模型）、**LLM 调用层**（每次调用的输入输出全文，我们在这里）、**报告质量层**（只看最终产物好不好）、**系统层**（时延/token 数/缓存命中率，不看文本内容）。层不同，结论不能互相替代。
3. **度量**：字符/token 级的逐字重叠？词项集合的重叠？语义相似度？还是缓存命中率、投机接受率这类**间接**信号？尺子不同，测出来的是不同的东西。

我们占的位置：**DR 对象 × LLM 调用层 × 逐字内容度量 × 与 serving 成本联动**。下面每篇都用这三个问题定位。

## §1 三把尺子手把手（一个例子贯穿）

沿用教程里的编辑部例子（[once-generated-explained.md](once-generated-explained.md)）：

- 资料 **A**（记者小结，23 字）：`全球电动车销量达1700万辆，但增速明显放缓。`
- 报告摘句 **S**（24 字）：`据研究，全球电动车销量达1700万辆，行业放缓。`
- 证据文档 **E**（23 字）：`全球电动车销量达1700万辆，同比增长25%。`

**尺子① coverage / density（Grusky 2018，摘要"抽取性"度量的祖师爷）**。在 S 里贪心找"逐字出现在 A 中的连续片段"（教学口径：字级、只计 ≥2 字；Grusky 原版是词级）：找到 `全球电动车销量达1700万辆，`（15 字）和 `放缓。`（3 字）。**coverage** = 被片段覆盖的比例 = (15+3)/24 = **75.0%**；**density** = 片段长度平方和/总长 = (15²+3²)/24 = **9.75**——平方项使 density 对"长段整搬"敏感：同样 75% 覆盖，若由 9 个 2 字碎片凑成，density 只有 36/24=1.5。**这正是我们"@8 好看但 @30 塌方、字符覆盖率 r=−0.23"口径教训的度量论亲戚**：coverage 像我们的 Coverage@8，density 像我们的"可收割量"直觉。我们没有发明这把尺子，我们是第一个拿它（的思想）去量 DR 流水线逐调用负载的。

**尺子② CTAR（Agentic Search in the Wild，2026）**。问的是另一个方向的问题：agent **新发出的查询词**，有多大比例来自**此前检索到的证据文本**？玩具例：历史查询用过 {电动车, 销量}，新查询 `电动车 增长 补贴 政策` 中新引入的词 = {增长, 补贴, 政策}，其中出现在 E 里的只有 `增长`（E 有"同比增长"）→ CTAR = 1/3 ≈ **33%**（CMU 真实测得均值 54%）。注意方向：CTAR 测的是 **证据→查询**（词级、集合口径），我们测的是 **输入→输出**（字符/token 级、连续段口径）——同在"内容重叠"名下，测的是两件不同的事。

**尺子③ 接受率 ≠ 照抄率**。投机解码的接受率（acceptance rate）常被当成"内容重复度"的代理，但它混入了机制因素：要先"对上暗号"（结尾匹配）才有草稿可发，树的冷热、草稿深度、并发都影响它。同一批报告调用，我们测得**照抄率 90.7%（文本性质）但 suffix 接受率只有 61.2%（机制产出）**——差出来的 30 个点就是机制损耗。所以"SuffixDecoding 报了接受率"不等于"它刻画了负载"（B1 详述）。

---

## A 组：刻画线的近邻（撞"首个刻画"主张）

### A1 TraceLab（arXiv:2606.30560，2026-06，UW）——方法论最像的一篇

**做了什么**：作者自己日常使用 Claude Code 和 Codex 十个月（2025-09~2026-06），录下 **~4,300 个会话、35.7 万次 LLM 调用、43.2 万次工具调用**的完整 trace，做了 coding agent 负载的系统层刻画：长自治循环、长上下文短输出、工具调用重尾分布、prefix cache 命中率高但不完美、人机交替的间隙节奏。trace 和分析代码公开。
**与我们重叠**：这是"对 agent 负载做 trace 级刻画并发布"这件事的直接先例——方法论（真实使用、逐调用、发数据集）和我们几乎同构；它的"长上下文、prefix cache 不完美"也与我们的负载常识一致。
**停在哪**：两条。①对象是 coding，不是 deep research，且论文自己把"其他 agent 负载"列为 future work；②**关键**——它明确写明脱敏时"丢弃原始消息与工具输入输出文本，只保留字符数"：**文本没了，照抄率、逐字冗余在它的数据上物理上不可测**。通俗讲：trace 是他们自己十个月的真实工作记录，里面有真代码、真文件路径、真对话，直接公开会泄露隐私——所以发布前把每条消息的**正文删掉**，只留下"这条消息 1,532 字"这样的元数据。对他们的研究问题（时延、上下文多长、调了几次工具、缓存命中）够用——这些只需要长度、时间戳和次数；但照抄率问的是"输出里这句话是否逐字出现在输入里"，回答它必须**把两段原文摆在一起逐字对**——原文已经销毁，"物理上不可测"指的就是：不是他们没算，而是任何人拿着这份公开数据想替他们算也算不出来。这也点破了内容层刻画稀缺的结构性原因：要么像我们一样自己跑流水线（数据自产、无隐私问题），要么就得公开敏感原文——大多数人选了脱敏，内容层于是一直空着。
**划界句**：TraceLab characterizes coding-agent traces at the systems level and explicitly drops message text, precluding any content-level (verbatim-copy) analysis; we characterize a deep-research pipeline at the content level.

### A2 Agentic AI Workload Characteristics（arXiv:2605.26297，2026-05）——抢走了我们两个"单项"

**做了什么**：对 ReAct 式 agent 在 5 个 benchmark（ADE-Bench、DABStep、GAIA、SWE-bench Pro、Terminal-Bench 2.0）上做端到端 tracing：token 构成（思考/消息/工具调用）、prefill/decode 时间占比、context cache 命中、轮数与上下文累积。两个与我们直接相关的结论：**有效缓存下 agentic 执行是 decode 主导**；**上下文随轮次累积**（首轮数万 token、后续每轮只增数百）。
**与我们重叠**：我们叙事里"decode 占 95%""上下文增长"这两个单项，它在泛 agentic 对象上已经测过、发表过——**这两项不能再当我们的新颖性卖点**，只能作为"在 DR 对象上的复核+更极端的数值"出现，并引用它互证。
**停在哪**：五个 benchmark 里没有 deep research 流水线（GAIA 只有孤立检索调用）；"复用"只按 cache 命中口径测，没有任何文本级测量。
**划界句**：Prior ReAct-agent characterization already established decode dominance and context accumulation at the systems level; our contribution is the content layer (per-call verbatim copy, cross-call redundancy) on a deep-research pipeline, where these systems-level facts are corroborated but not the claim.

### A3 Agent Memory（arXiv:2606.06448，2026-06）——第三家 trace 刻画

**做了什么**：对 10 个记忆增强 agent 系统（Mem0、Letta、HippoRAG v2、GraphRAG 等）做逐调用测量：调用类型、起止时间、prompt/completion token、逐阶段能耗与成本。
**与我们重叠**：把"trace 级 agent 刻画"的先例从两家变成三家——**"首个 trace 级 agent 刻画"这种不带对象限定的话彻底不能说**。
**停在哪**：对象是记忆系统、指标全在系统层（API/硬件/成本三层），无 DR、无内容。
**划界句**：与 A1 合并一句即可（"prior trace studies cover coding/ReAct/memory agents at the systems level only"）。

### A4 Agentic Search in the Wild（arXiv:2601.17617，2026-01，CMU）——最危险的近邻 ⚠️

**做了什么**：分析 DeepResearchGym（deep research 式 agentic 搜索 API）的 **14.44M 真实搜索请求、3.97M 会话**。提出 **CTAR**（§1 尺子②）：新引入查询词有多大比例出现在此前累积检索到的证据文档里——**均值 54%**，并给出随历史步数衰减的曲线（两种上下文口径 C_k^agg / C_k^last）。这已经是【DR 相邻对象 × token 级内容重叠 × 随轮次演化】三样占齐。
**与我们重叠**：它证明了"DR 式负载中内容在轮次间逐字回流"这个大方向**有人测过了**——如果我们的措辞是"首个对 deep research 的内容重叠测量"，这篇当场击穿。它的 CTAR 衰减曲线和我们的"多轮放大照抄"讲的是同一族现象（内容随轮次流动）。
**停在哪**：论文自认"agent 以程序方式消费检索结果，不留可观察痕迹"——**它完全看不见 LLM 侧**：没有逐调用输入输出全文、没有输出→输入照抄率、没有 decode/prefill、没有任何 serving 指标；公开数据集只有查询文本+时间偏移等 6 个字段。方向也相反：它测"证据→下一次查询"（词级集合），我们测"输入→输出"（字符连续段）。
**划界句**：Agentic Search in the Wild measures term adoption from evidence into subsequent queries at the search-log layer, explicitly blind to the LLM side; we measure verbatim output-to-input copying at the LLM-call layer, jointly with serving cost structure. 两篇是**互补的两半**（日志侧 + 模型侧），论文里应作为互证引用，而非竞品。

### A5 Understanding DeepResearch via Reports / ReportEval（arXiv:2510.07861，2025-10）——同对象、同"冗余"一词、不同层

**做了什么**：用 LLM-as-judge 对 4 个商业 deep research 系统的 100 份报告打分，维度含质量/事实性/**冗余**——这里的冗余指**最终报告内部**段落间的主题重复与复述。
**与我们重叠**：它是唯一在标题层面"测过 DR 冗余"的工作——审稿人若只看关键词会以为撞了。
**停在哪**：层完全不同：它在**产物质量层**（报告好不好读），我们在 **serving trace 层**（生成过程中内容怎么流动、系统为此付了多少钱）；它的度量是 LLM 主观判断，不是逐字统计。
**划界句**：ReportEval scores redundancy as a quality attribute of final reports via LLM judges; we measure verbatim redundancy as a serving-layer property of the generation process.

### A6 Byte-Exact Deduplication in RAG（arXiv:2605.09611，2026-05）——同口径、不同对象

**做了什么**：对 RAG 拼装的 prompt 做 chunk 级**字节精确**冗余测量，覆盖三类场景（BEIR、企业语料、WildChat 多轮），给出"三种冗余机制"的实证分析。
**与我们重叠**：逐字/字节级精确口径与我们一致——证明"内容级精确测量"这条方法路线成立。
**停在哪**：对象是单轮 RAG 的 prompt 内部（检索 chunk 之间重复），没有多智能体流水线、没有"输出被照抄进后续输入"的方向性、没有轮次放大。
**划界句**：Byte-exact redundancy has been measured within single-turn RAG prompts; we extend exact-match measurement to the output→input direction across a multi-agent pipeline.

### A7 AgentInfer（arXiv:2512.18337，2025-12，华为）——DR 对象上的系统层单点

**做了什么**：在真·deep research 工作流（OpenPangu-DeepDiver，Ascend NPU）上做系统 profiling：KV cache 命中率从短上下文 92% 塌到 32K+ 的 **15%**；搜索结果占最终轨迹上下文 50%+；量化悖论（TPS +45% 但端到端反慢 70%）。
**与我们重叠**：证明"在 DR 对象上测 serving 指标"有先例——连"DR 上的 KV 命中率"都不能当我们的新发现。
**停在哪**：三张表级别的零星 motivation profiling，无刻画章节、无内容层。
**划界句**：与 A2 合并进"单项系统指标均有先例"一句。

### A8 度量传统：Grusky 2018 与 copy coverage/density（arXiv:2510.00508，2025-10）

**做了什么**：Grusky 等 2018 年为摘要研究定义 coverage/density（§1 尺子①），此后 Goyal 2022、Zhang 2023 等用它测得"LLM 摘要比人类摘要更抽取"。2510.00508 把它精化为 copy coverage κ / copy density δ，在**单轮 RAG QA** 上画了逐样本分布——目的是用"多抄少编"干预幻觉。
**与我们重叠**：**度量本身不是我们发明的**——"逐调用照抄率+分布"这个形式，2510.00508 在单轮 RAG 上已经画过。
**停在哪**：单文档/单轮、无 pipeline、无跨调用、无轮次、无 serving 联动；目的分别是摘要研究和幻觉干预，不是负载刻画。
**划界句**：Copy-coverage metrics originate in summarization research; we are the first to apply this family to characterize per-call workload of a production deep-research pipeline and to connect it to serving cost.

### A9 motivation 级三家：Parrot、Preble、Mooncake（合并讲）

**做了什么**：Parrot（OSDI'24）在 motivation 里测得跨请求 **94%** 前缀 token 可复用、MetaGPT 多智能体因反复携带对话历史冗余约 **99%**；Preble 测五类负载 prompt 比输出长 37–2494×、**85–97%** token 跨请求共享；Mooncake 公开 23K 条 Kimi 生产 trace（内容以 512-token 块哈希表示），Conversation 场景 prefix 命中 ~40%。
**与我们重叠**：这些是"跨调用冗余巨大"直觉的最早定量来源——尤其 Parrot 的 99% 常被引用，审稿人可能问"冗余不是早就知道了吗"。
**停在哪**：全部是 **prefix-token 层**：只统计"整段前缀逐字相同"（缓存视角），测不出部分照抄、乱序复用、输出→输入方向；且都是两三个数字的 motivation，不是刻画研究；块哈希（Mooncake）只能表达整块相同。**prefix 冗余是输入之间的重复，我们的照抄是输出对输入的重复——方向和粒度都不同。**
**划界句**：Prefix-level redundancy (94–99%) motivated prompt-caching systems; it measures repetition among inputs at cache granularity, whereas we measure verbatim copying from inputs into outputs at character/token granularity.

---

## B 组：写侧（投机/路由）的近邻

### B1 SuffixDecoding（arXiv:2411.04975，Snowflake）——我们写侧机制的直接来源

**做了什么**：提出用后缀树（对 prompt + 历史输出建索引）做 model-free 投机解码，在 agentic 负载上最高 **5.3×**、AgenticSQL 流水线吞吐 **2.9×**。它的立论观察正是"agentic 输出自重复"。
**与我们重叠**：写侧主臂机制**就是它**（vLLM 的 suffix 实现），"agentic 输出可作自己的草稿源"这个观察也是它先说的——我们不能认领机制，也不能认领这个定性观察。
**停在哪**：①它对"自重复"的呈现只有接受率和一处定性示例——**接受率≠照抄率**（§1 尺子③：机制产出混入了树状态/深度/并发因素），负载本身的内容画像它没有做；②无按调用路由、无深度经济学（我们的五常数模型给出 0.62ms/位、盈亏线 2.4%）、无 DR 对象、无与 prefill 侧的耦合。
**划界句**：SuffixDecoding supplies the mechanism and the qualitative observation of agentic self-repetition; we supply the workload characterization that explains when it pays (61.2% vs 28.8% acceptance from a 90.7% vs 53.8% copy-rate split), a cost model that prices draft depth, and its coupling with prefill-side reuse.

### B2 LLMA（arXiv:2304.04487，2023，微软）——"照抄可加速"的原始观察

**做了什么**：最早指出 RAG 输出大量 span 逐字来自检索文档，用"复制并校验"拿到 ~2× 无损加速。
**与我们重叠**：我们特征1（输出照抄输入）的定性内核，它在单轮 RAG 上三年前就发现了。
**停在哪**：单轮 RAG、单次调用；无多智能体流水线、无 output-becomes-input 链、无量化画像（它是机制论文附带观察）。
**划界句**：LLMA observed verbatim copying in single-turn RAG; deep research amplifies this into a pipeline-wide recycling structure (90.7% report-level copying, growing with rounds), which we quantify and exploit on both sides.

### B3 SpecDec++ / BanditSpec / HedgeSpec——"学着调投机"赛道

**做了什么**：SpecDec++（2405.19715）在 draft hidden state 上训 acceptance 预测头动态定草稿长度（2.04–2.26×）；BanditSpec（2505.15141）训练-free bandit 在线选投机超参/方法；HedgeSpec（2510.20064）full-information 在线选 drafter。
**与我们重叠**："按情况选策略/定深度"这件事被它们占了——我们不能说"首个自适应投机调度"。
**停在哪**：它们的信号是**运行时接受反馈**（试了才知道），我们的信号是**负载内容结构**（重叠率，生成前可从内容/先验得知）；它们面向通用负载，无 DR、无刻画驱动。二者正交可叠加。
**划界句**：SpecDec++-style controllers adapt from runtime acceptance feedback; our routing derives from measured content structure of the workload—the two signals are complementary.

---

## C 组：特化训练的近邻（只讲重叠角度；配方细节见[续训参考](../../../eagle-spec-decode/eagle3-continued-training-references.md)）

### C1 Aurora（arXiv:2602.06932，Together AI）

**重叠**：字面意义的"在已训好的 EAGLE-3 上继续训练"它做了（trained-from-static 臂，2.63→2.99，+1.25× 吞吐）——**"没人续训过 EAGLE-3"绝对不能说**。
**停在哪**：训练流量是 40k 混合域 prompt（GSM8K/Spider/代码/金融/闲聊）模拟 serving，**不是 agent 轨迹**；目的是在线跟漂移，不是域特化研究。

### C2 OSD（ICML'24）与 DistillSpec（ICLR'24）

**重叠**："用目标分布的数据（继续）训 drafter 提接受率"的范式先例（OSD：100–200 请求见效、α 0.28→0.76；DistillSpec：+10–45%）。
**停在哪**：训练对象都是**独立小 draft model**，不是 EAGLE 式 hidden-state head；无 agent 场景。

### C3 arXiv:2604.26779 §3.3

**重叠**：在 RL rollout **自身轨迹**上在线更新 EAGLE-3 head——"自身轨迹×EAGLE-3"两角占齐；且它的消融（对齐初始化后在线更新零增益 1.77 vs 1.78；错配时 1.51→1.63）给了我们"特化的价值=错配大小"的定价框架。
**停在哪**：数学 RL（DAPO），不是 agent 工作流；在线组件是可选项非研究主体。

### C4 PayPal（arXiv:2604.19767）与 ISSTA'26（arXiv:2604.26469）

**重叠**：PayPal 在生产商务 agent 上用了 EAGLE-3（training-free，接受率 25–36%）；ISSTA'26 用 SpecForge 在 SWE-Gym（软工 agent 数据）上从头训过 EAGLE-3 drafter。"agent + EAGLE-3"的组合有人碰过。
**停在哪**：PayPal 零训练、并把"在商务数据上微调 EAGLE3"**明写为 future work**（坑空着的直接证据）；ISSTA'26 是补权重的工具性动作（Llama-70B 无官方 head）非研究贡献，且是从头训、发现反而利好 model-free 方法。
**C 组合并划界句**：Prior work continues training drafters on serving traffic (Aurora, OSD) or RL rollouts (2604.26779), or deploys EAGLE-3 on agents training-free (PayPal, which lists fine-tuning as future work); specializing a post-hoc EAGLE-3 head on an agent's own trajectories remains unexplored—we report it with pre-registered criteria either way.

---

## D 组：读侧（KV 复用）的近邻

### D1 Plato（COLM 2025，arXiv:2402.12280）——最近的整体竞品

**做了什么**：用 LLM 建骨架依赖图，对单条 query 的多段生成做语义级并行 + KV 复用 + 异构模型选择（+68% 吞吐、KV 复用降开销 75%）。
**与我们重叠**："利用 agent 输出结构做并行+KV 复用"的整体思路——三件套里它占了两件的"名字"。
**停在哪**：它的 KV 复用**严格是位置相关的前缀复用**——原文自陈把所有前序输出按固定顺序累加成单调增长前缀（P→P+A→P+A+B…），并明确写"若只选择性纳入依赖，KV 值依赖 token 位置…系统将不得不重新 prefill、生成全新 KV"（§4.2.2，原文铁证已逐句核实，见 [paper-skeleton.md](../explore-idea/paper-skeleton.md) §四.2–4.3）。即 **Plato 做不到乱序、做不到取子集**——而 DR 的 writer 恰恰要按任意顺序取小结子集。它也不做 token 级投机、单 query 内部而非跨请求。
**划界句**：Plato reuses KV strictly as a monotonically growing fixed-order prefix and states that selective/reordered reuse would force full re-prefill; our content-addressed reuse is position-independent across agents and requests, and we add token-level speculative harvesting on the decode side.

### D2 RelayCaching（arXiv:2603.13289，2026-03 preprint）——读侧机制叙事的抢跑者 ⚠️

**做了什么**：正是"上游 agent decode 产生的 KV 传给下游 prefill 复用"（AutoGen/MetaGPT 场景，AIME/coding 实验）。
**与我们重叠**：decode-to-prefill 这个**机制叙事被它先说了**——我们读侧的"新机制"帽子戴不上。
**停在哪**：无负载刻画（它断言冗余、不测量）、无质量验证（我们有 40 题盲评无害+更强终裁在排期）、无与写侧的税对冲耦合、非 DR 对象；截至核查未见正式 venue。
**划界句**：RelayCaching proposes the decode-to-prefill mechanism narrative; our contribution is characterization-driven deployment (quantifying reusable volume), quality validation, and the TTFT-tax coupling with decode-side acceleration—plus the boundary condition (crossover ≈4–6k) it does not report.

### D3 KVCOMM（NeurIPS'25）、CacheBlend（EuroSys'25 best paper）、DroidSpeak（NSDI'26）

**做了什么**：非前缀 KV 复用家族——KVCOMM 用 anchor 对齐做跨 agent 复用（**>70% 复用率**、5-agent TTFT 430→55ms）；CacheBlend 选择性重算 ~15% token 换 TTFT 2.2–3.3×（位置无关复用的机制来源，也是我们自己的工作线）；DroidSpeak 做跨模型分层复用。
**与我们重叠**："位置无关 KV 复用"机制本身——我们不认领机制新颖性（明写站在 CacheBlend 肩上）。
**停在哪**：全在 **prefill 侧跨请求**（谁的 prompt 和谁的 prompt 重）；"decode 产物→下游 prefill"的方向、DR 负载、与投机的耦合都没有。**特别注意 KVCOMM 的 >70% 不是负载刻画**——那是它的方法在自建 benchmark 流水线上达成的复用率（方法产出），不是对某真实负载"有多少可复用"的测量；我们的正向口径测量（下游 prefill 中来自上游 decode 的占比）`[待测]` 才是刻画。
**划界句**：We inherit position-independent reuse mechanics from CacheBlend and apply them in the decode-to-prefill direction on a characterized DR workload; reported reuse rates in prior work are method outputs on synthetic pipelines, not workload measurements.

### D4 前缀缓存家族（RadixAttention/SGLang、Hydragen）

一句话即可：它们把"逐字相同的**前缀**"缓存住（命中 50–99%），位置相关；是 infra 引用对象，不是竞品——我们的复用恰好覆盖它们覆盖不了的"换了位置的相同内容"。

---

## 追问区（预判的坑）

**Q1：CTAR 54% 和我们的照抄率 90.7% 是一回事吗？能直接比吗？**
不能。三处不同：层（搜索日志 vs LLM 调用）、方向（证据→下一次查询 vs 输入→输出）、口径（词集合 vs 字符连续段）。玩具例里同一次会话 CTAR=33% 而报告照抄=62.9%（教程手算），两个数字互不约束。论文里两者是互证（"内容在轮次间回流"的两个侧面），不是竞争。

**Q2：Parrot 都测出 99% 冗余了，我们的刻画还新在哪？**
99% 是 MetaGPT 输入间的 prefix 冗余——"这轮 prompt 和上轮 prompt 开头一样"，缓存视角一个数字。它既看不见输出→输入的方向（recycling 的本质），也看不见非前缀位置的复用（DR writer 乱序取小结），更没有分布/形态/演化。把 94–99% 引成"冗余早已知"反而帮我们立论：知道冗余大，但没人测过它的**内容结构**。

**Q3：SuffixDecoding 已经在 agentic 负载上拿到 5.3× 了，写侧我们还剩什么？**
机制归它，账本归我们：它回答"能加速"，我们回答**为什么、何时、多深**——61.2% vs 28.8% 的分裂由 90.7% vs 53.8% 的照抄结构解释（刻画）；0.62ms/位+盈亏线 2.4% 决定发多深（经济学）；接受率 4.9% 的草稿在并发下 0.75×（何时该关）。另有独立系统发现（锁竞争 +28%）和与读侧的耦合。

**Q4：为什么说 KVCOMM 的 ">70%" 不算刻画？**
它是"我的方法能复用掉 70%"（方法在自建流水线上的产出），不是"这个负载有 70% 可复用"（负载性质的测量）。类比：投机接受率之于照抄率（§1 尺子③）——机制产出 ≠ 文本性质。我们的正向口径 `[待测]` 测的才是后者。

**Q5：这么多近邻，交集主张会不会太"组合式"？**
风险真实存在（评审已提示"新颖性押交集"是最后防线）。护栏有二：①交集不是拼贴而有因果链——刻画的两轴分别**决定**路由必要性与字面机制可行性，税对冲**要求**两侧同框；②每个单项都以"继承+超越"姿态明写（继承 Grusky 度量/SuffixDecoding 机制/CacheBlend 复用，超越各自的对象与联动缺失）。

## 术语表

| 术语 | 一句话 |
|---|---|
| 系统层 / 内容层 | 测时延、token 数、缓存命中（不看文本）vs 测文本本身的重叠、照抄、冗余 |
| 搜索日志层 / LLM 调用层 / 报告质量层 | 只看查询与检索结果 / 看每次调用的输入输出全文 / 只看最终报告好坏 |
| coverage / density | Grusky 2018：被逐字片段覆盖的比例 / 片段长度平方和÷总长（对长段敏感） |
| CTAR | 新引入查询词中来自此前检索证据的比例（词级集合口径，2601.17617） |
| prefix 冗余 | 不同请求的 prompt 有多长的**开头**逐字相同（缓存视角，Parrot/Preble/Mooncake） |
| 接受率 vs 照抄率 | 机制产出（受树状态/深度/并发影响）vs 文本性质（只由内容决定）；90.7% vs 61.2% |
| 方法产出 vs 负载测量 | "我的系统复用掉了 X%"（KVCOMM 70%）vs "这个负载有 X% 可复用"（刻画） |
| 前缀复用 vs 位置无关复用 | 只认"开头一致"（RadixAttention/Plato）vs 内容寻址、换位置照样用（CacheBlend 系/我们） |
| trained-from-static | 从已训好的 speculator 初始化继续训练（Aurora 的实验臂名） |

## 相关链接

- 论文用的压缩划界：[once-generated-paper-narrative.md](once-generated-paper-narrative.md) §6；"首个"安全措辞：同文 §1 P3
- 研究骨架（新颖性收窄 ⭐ 两条）：[Once Generated.md](Once%20Generated.md)
- 特化训练配方细节：[eagle3-continued-training-references.md](../../../eagle-spec-decode/eagle3-continued-training-references.md)
- Plato 原文铁证逐句核实：[paper-skeleton.md](../explore-idea/paper-skeleton.md) §四.2–4.3
- 调研原始材料：[survey.md](survey.md)（本目录）
