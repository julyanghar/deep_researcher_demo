# Once Generated, Many Times Served: Characterizing and Exploiting Verbatim Content Recycling in Deep Research Agent Inference

论点: Deep research agent 推理负载的定义性特征是"逐字内容再循环"——同一段 token 被生成一次、随后在下游调用中被反复作为输入 prefill、又被大面积逐字照抄进输出——本文首次做 trace 级系统刻画（照抄率/冗余口径/decode 占比/段类型异质性），并据此构建一个 recycling-aware 的 prefill+decode 联合加速系统：decode 侧用覆盖率路由的混合投机（suffix 树打照抄段、本域特化 EAGLE-3 头打非照抄残差、hidden-state 覆盖率判决器选臂），prefill 侧用 decode-to-prefill KV 复用对冲投机的 TTFT 税，端到端在质量无害前提下把 deep research 延迟近乎腰斩。

## DR特点
- 特征1【输出大面积逐字照抄输入】：REPORT 调用 Coverage@8 均值 90.7%（100% 调用 >50%，阈值提到 ≥30 字连续片段仍覆盖 74.5%，最长单段照抄 694 字）；SUMMARY 均值 53.8% 且双峰分布。这是全文地基：解释了为什么复制类投机（suffix）在 report 段接受率 61.2%、replay 1.76x，而通用 draft 模型停在 ~27% 天花板（token 级长段 cov24 仅 0.4%——'能抄的都在 prompt 里，不在语料里'）。一手数字齐全（prompt-overlap-analysis-v2）。
- 特征2【冗余是逐字复制而非语义改写】：同题 summary 两两配对逐字重合 out_contain8 均值 22.9%（37.3% 配对硬照抄 >30%），但语义口径 out_jac5 仅 7.7%。这条支撑两个设计决策：(a) 文本级/KV 级精确匹配复用可行、不需要 embedding 检索式近似复用；(b) 分节改写会破坏复用（report 分节使 suffix 命中 48.4%→38.9%），所以加速要在 serving 层做而不是靠改写 prompt。一手数字齐全（researcher-redundancy-v2）。
- 特征3【output-becomes-input 数据流 + 多轮放大】：researcher 的 decode 产出（summary）成为 writer 的 prefill 输入；v1 单轮→v2 多轮迭代使 REPORT 照抄加重（均值 86.9%→90.7%，>90% 占比 36%→71%），多轮互抄边际 ×1.65。这是 decode-to-prefill KV reuse 的结构性依据（上游 decode 时 KV 已经算过一遍，下游 prefill 是纯浪费）。注意：现有测量是'输出反向查 prompt'口径，正向口径——下游 prefill token 中有多大比例逐字来自某次上游 decode 产出、其中多少满足 KV 可复用条件（chunk 对齐、[:-1] 坑）——需补测量，这是 KV reuse 那一节的动机数字。
- 特征4【decode 主导、长报告段主导 e2e】：decode 占 GPU 计算 95%，40 题 e2e 中 report 段占 4584/9293s ≈ 一半。直接推论：优化重心必须在 decode 侧（这就是为什么 KV reuse 端到端只值 −8~10%、只能当配菜），而 report 段 2.29-2.37x 的投机收益能撬动 e2e −47.5%。一手数字齐全（claude-docx/11 + e2e-3config-40q）。另外文献盘点确认 deep research 无公开 trace 级刻画（TraceLab 只做 coding agent），'搜索等待 vs prefill vs decode' 的 e2e 时间分解也需补一张正式的分解图。
- 特征5【段类型间 + 段内双层异质性】：接受率 report 59.5% / query_plan 39.4% / summary 26.7% 是段间分裂；段内更狠——SUMMARY 逐题照抄率 8.0%–88.2%、Coverage 双峰、81% 的 summary 步零接受但尾部段仍 5–9% > 盈亏线 2.4%。段间异质性用免费的 tag 路由就能吃掉（这是'训判决模块'曾被否决的原因），段内双峰才是 hidden-state 覆盖率判决器的生存空间——但'per-call 路由显著优于 per-tag 路由'的 oracle 上界需补测量，测不出就砍。
- 特征6【通用 draft head 败于文体而非语言，特化必要】：现成 eagle3 双 head 实测接受率 10.7–20.3%，全面输给 suffix（26–32%/57%），20 题无一翻盘；corr(中文占比, 接受率)=−0.60 证明败因是 deep-research 文体（引用密集、检索片段拼接）而非中英文。这个诚实的负结果正是'必须在本 agent 轨迹上特化训练 EAGLE-3'的动机——不是'特化更好'的泛泛之谈，而是'现成的在这个负载上定量地不行'。一手数字齐全（eagle-idea.md）。
- 特征7【并发扇出下劣质草稿付真钱】：3-5 个 researcher 并行 + 分节并行输出膨胀 1.59×，batch=1 时免费的垃圾草稿在并发下把 SUM e2e 打到 0.75x（ngram 4.9% 接受率案例）；五常数税模型（graph 24.0 + 混合边界税 8.1 + 草稿费 5.8+0.62/位，七配置残差 0）给出投机的精确成本账。这条把判决模块从'锦上添花'升格为'并发 serving 下的止损保险'，且税模型本身是刻画贡献的一部分（含锁竞争发现 + enforce-eager 修复 e2e +28%）。一手数字齐全（fixed-tax-conclusions.md）。

## 工作量映射
论文四段式：§刻画（贡献1，故事的承重墙）→ §decode 侧（贡献2=主菜）→ §prefill 侧（贡献3=配菜）→ §e2e 系统评估。三件已有工作量的位置：【(2) 特化 EAGLE-3 = 主菜的核心增量】。逻辑链：刻画显示照抄段 suffix 已近吃干（report 2.29x）、但非照抄段是硬骨头（summary 0.87–0.96x 且任何调参翻不正，已证）；现成 eagle3 头被文体打死（特征6），所以在本 agent 轨迹上特化训练 head 是被负结果逼出来的必然动作，目标是把 summary 段从 0.93x 翻正——这是三件里唯一 related work 无直接命中的位置（'agent-trace-trained draft head'确凿搜不到已发表工作），也是目前唯一还没动手的主菜。【(1) 判决模块 = 主菜的路由层，但必须重新定位】。原动机'选投机策略'已被'tag 免费'判死——段类型路由零训练成本就能做。它的合法生存空间只剩特征5的段内双峰：同是 summary 调用，照抄率 8%–88%，tag 看不见。重定位为：hidden state → 本次调用覆盖率预测 → 三臂选择（suffix / 特化 head / 关投机）+ 下注深度（盈亏线 2.4% 的经济学）。这个信号来源（预测输入输出覆盖率）在 SpecDec++/BanditSpec 赛道里确凿没人做过，是增量但有区分度。前置条件：先跑零成本 oracle 模拟证明 per-call 混合上界显著高于 per-tag（D2 思路），上界不到 5% 就诚实降级为刻画章节的一个分析。【(3) decode-to-prefill KV reuse = 配菜/infra】。三个原因不能当主菜：decode 占 95% 罩死 e2e 收益（−8~10%）；RelayCaching (2603.13289) 已抢跑机制叙事；小档 crossover 4-6k 有净亏区。但它在故事里有三个不可替代的角色：(a) 特征3 的直接消费方，刻画→优化的闭环最干净；(b) 精确对冲投机 enforce-eager 的 TTFT +14% 税——'decode 加速付 prefill 税、prefill 复用把税缴回来'是系统章节最漂亮的一句话；(c) 已有 40 题质量无害证据（SW 30.3%/C 28.2% vs B 24.8%）+ 生产 LMCache 路径实现，vs RelayCaching 可用'质量验证 + 刻画驱动 + 与 decode 侧联合'三点区分。整体身份：这是一篇 MLSys/OSDI 风格的 workload characterization + guided optimizations 论文，最强的已兑现数字（e2e −47.5% 近腰斩、五常数税模型残差 0、锁竞争 +28%）全部挂在刻画和系统章节下，三件工作量是刻画长出来的枝而不是各自为战的三根柱子。

## 缺口
- P0：特化 EAGLE-3 head 实训（主菜目前 0 行代码/0 个 checkpoint）。用本 agent 轨迹（online100_v2 的 1912 次调用 + 增采）训 Qwen3-32B 的 EAGLE-3 头，验收线明确：summary 段接受率 >27%（超过 suffix 的 26–32% 区间下沿才有混合价值），e2e 把 summary 从 0.93x 翻正到 ≥1.05x。训不出来整个主菜塌方，所以必须最先做、留退路（退路=判决模块只在 suffix/off 两臂间路由）。
- P0：判决模块的生死判——零成本 oracle 模拟。用已有 609 请求 traj + 逐段接受率数据，离线算 per-call oracle 混合（每次调用事后选最优臂）相对 per-tag 路由的上界增益。>5% 才立项训 LoRA/MLP；否则降级为刻画分析，论文改讲'tag 路由已捕获绝大部分异质性'这一诚实结论。这一步一天能出结果，必须在投入训练前做。
- P1：刻画章节的正向补测三件套（文献盘点确认全是空白，是'首次刻画'主张的弹药）：(a) 下游 prefill token 中来自上游 decode 产出的逐字占比 + KV 可复用率实测；(b) e2e 时间分解（搜索等待 / prefill / decode / 空转），对齐 TokenCake 的定性描述给定量版；(c) 上下文增长曲线与扇出统计。数据（online100_v2 + LLM_CALL_LOG）已在手，主要是分析工程。
- P1：三件合体的端到端评估。统一 server：suffix + 特化 head + 覆盖率路由 + blend KV reuse，对 Ceager 基线跑 40–100 题 DRBench，报 e2e 墙钟 + TTFT + 质量（kimi-k2.5 评委，qwen-flash 已证不可靠不能用）。含消融：每件单开/组合，证明可加性（尤其 KV reuse 的 TTFT 收益是否真能对冲 enforce-eager 的 +14%）。
- P1：泛化性防线。至少第二个 deep research 框架（GPT Researcher 或 open deep research）+ 换一个 base 模型复测照抄率/冗余率核心表，证明 recycling 是负载性质不是自家 prompt 模板 artifact。这是审稿人第一刀，必须挡。
- P2：KV reuse 收尾。(a) replay 固定搜索结果的 blend vs native 质量终裁（现有 Exp_M 欠功效，测不出≠无害）；(b) load 134ms 的 launch-bound 搬运批量化，把小档净亏区收窄、crossover 从 4-6k 拉低——否则审稿人会问为什么 crossover 以下不 fallback。
- P2：related work 查全防撞车。spec-decode-drafting 轴（REST/SAM-decoding/PLD/cascade）还没查（GADS 生死未判）；RelayCaching 是否已中会、BanditSpec/Not-a-Bandit 正式 venue 需确认；EAGLE-3.1 作为移动基线要在特化实验里对照。

## 弱点
- '为什么不改 prompt 而要改 serving'——自家去冗余 A/B 显示 prompt 工程直接砍掉 summary −18.7%/report −28.9% 的输出 token，审稿人会拿这个数字反杀：负载里的照抄一部分是可消除的浪费而非本质。需要正面回应：事实与引用全留（照抄的主体是必须转述的检索证据，去冗余砍的是骨架铺垫）、且分节改写实验证明消照抄会伤复用（48.4%→38.9%），serving 层利用比应用层消除更稳。这个回应要写进论文，不能等 rebuttal。
- 单系统单模型 artifact 风险：全部刻画数字来自一个自建 deep_researcher_demo + 一个模型的 100 题 trace。'首次系统刻画'的主张越大，这个样本越显薄（对比 Preble/TraceLab 的多负载覆盖）。不补第二框架/第二模型，characterization 贡献会被降级为 case study。
- 主菜还没训出来：特化 EAGLE-3 目前是纯承诺，而且有自家负结果压顶（现成 head 10–20%）和移动基线（EAGLE-3.1 长上下文 acceptance 2x）。如果自训 head 也翻不过 suffix，decode 侧就只剩'suffix + 一个 flag 修复'，技术浓度撑不起主菜。
- 三件各自被抢跑/挤占，合体必要性是唯一护城河：RelayCaching 占了 decode-to-prefill 机制叙事（且窗口在收窄），SpecDec++ 占了 acceptance 预测头，DistillSpec/OSD/Baseten 工业教程占了域训 draft head。审稿人若认为三件可以拆开分别引用已有工作，'联合系统'就退化为工程集成。必须用刻画数字证明三件耦合（例：投机的 TTFT 税恰好被 KV reuse 缴回、路由信号恰好来自照抄率刻画）。
- 判决模块与自家'tag 免费'否决结论的内在张力：论文里同时出现'我们否决过朴素判决模块'和'我们训了判决模块'，讲不好就是自相矛盾。唯一出路是 oracle 上界数字说话；若上界小，诚实砍掉比硬塞进去安全。
- KV reuse 的'无害'主张证据不牢：Exp_M 新 run 上界塌 4.5 分、欠功效，40 题样本也不大；有 2026 负结果 preprint（judge 任务复用伤质量）在场，审稿人会要求更强的质量裁决（replay 固定检索 + 更大样本）。同时 e2e 只 −8~10%，收益与风险比会被质疑。
- 已知现象的边际新颖性：LLMA 已证 RAG 输出照抄检索文档，SuffixDecoding 已证 agentic 输出自重复。审稿人会问'你的刻画比这两篇多了什么'——答案必须是量化的负载级画像（双峰分布、逐字 vs 语义口径分离、多轮放大、五常数税、per-段接受率分裂）而非'照抄存在'这个定性事实本身。
- 最强 e2e 数字的归属：−47.5% 近腰斩里,大头来自现成 suffix 方法 + enforce-eager 一个参数 + report 分节并发（收益来自并发非命中）。系统评审会追问'你们自己的三个组件贡献了几个百分点'，消融表必须诚实拆开，拆开后如果三件合计增量小，故事重心就得进一步向刻画倾斜。

---

# EchoServe: Exploiting Cross-Call Content Recurrence for End-to-End Serving of Deep Research Agents

论点: Deep research agent 的负载有一条可量化的系统性质——内容在调用图中逐字复现流动（上游 decode 出的 token 既成为下游 prefill 的输入、又成为下游 decode 的模板，report 段照抄面积 90.7%、suffix 命中率 report 61% vs summary 29%）——因此可以用一个统一的"复现度预测"信号，在 prefill 侧门控 decode-to-prefill KV 复用、在 decode 侧路由 suffix/特化 draft head 并决定下注深度，构成第一个面向 deep research 负载、同时打 prefill+decode 两侧的 recurrence-aware serving 系统（该 cross-layer 切口经调研在顶会尚无先例）。

## DR特点
- 【逐字照抄是主要内容流动形式】REPORT 调用 Coverage@8 均值 90.7%/中位 93.1%，阈值提到 ≥30 字连续片段仍 74.5%；SUMMARY 均值 53.8%。这是'上游输出→下游模板'成立的直接证据，也是 suffix 投机 report 段 2.29-2.37x 的因果解释（一手：prompt-overlap-analysis-v2）。
- 【复现是 verbatim 不是 paraphrase】同题 summary 配对逐字口径 out_contain8 均值 22.9%，语义口径 out_jac5 仅 7.7%（>30% 只占 0.4%）——冗余是复制不是改写。这条决定了'字面机制'（后缀树、内容哈希 KV blend）适用而语义缓存不适用，是选型的地基；反面证据也吻合：分节改写使 report suffix 命中率 48.4%→38.9%，改写破坏字面复用（一手：researcher-redundancy-v2 + parallel-generation-experiment-v2）。
- 【调用图角色异质但角色内方差巨大】逐段接受率 report 59.5%/query_plan 39.4%/summary 26.7%——tag 是免费的粗信号（这正是朴素判决模块被否决的原因）；但 SUMMARY 内部每题照抄率 8.0%–88.2%、分布双峰（80-90% 桶 542 次最大），tag 分不开同一角色内的赢家和输家——这是学习型覆盖率预测器唯一站得住的立足点，且已有初步收益证词：生成前预测器路由可把 summary 段 0.93x→1.05x（一手：paper-skeleton + spec-decode 值得做清单）。
- 【decode 主宰端到端】decode 占 GPU 计算 95%，40 题 vanilla 9293s 里 report 段占 4584s；KV blend 大档 prefill −51~67% 但 e2e 只 −8~10%。这条决定了主菜必须在 decode 侧、prefill 侧 KV reuse 只能当配菜，与文献（arXiv 2605.26297 缓存后 decode-dominated）互证（一手：claude-docx/11 + e2e-3config-40q）。
- 【多轮迭代放大照抄】v1 单轮→v2 多轮 REPORT 照抄均值 86.9%→90.7%、>90% 占比 36%→71%；多轮互抄放大 ×1.65。说明 agent 越'deep'（轮次越多）复现性质越强、本系统收益越大——这是把系统绑死在 deep research 而非泛 agent 上的论据（一手：prompt-overlap-analysis-v2）。
- 【扇出子代理间存在检索与文本冗余，但只对 prefill 侧有用】同 researcher 配对逐字重合 34.6%、14.6% 配对共享 URL、prompt snippet 重合 27.6%——这是跨 sibling KV 复用的机会；但必须诚实标注：decode 侧的兄弟互抄增强已被判死（token 级兄弟 cov24 仅 0.69%，接受率对兄弟数全平），此特点只能喂 prefill 侧叙事（一手：researcher-redundancy-v2 + idea-verdicts #4）。
- 【本负载无公开 trace 级刻画】文献盘点确认 deep research 至今没有逐调用 timing/照抄率/decode 占比/扇出统计的公开 characterization（TraceLab 只做 coding agent）——我们的 online100_v2（1912 次调用）刻画本身就是论文 §2 的独立贡献；但'搜索等待 vs prefill vs decode 的墙钟分解'一项目前缺失，标注需补测量。
- 【现成通用 draft head 在本负载失效】现成 eagle3 双 head 接受率 10.7-20.3%，逐题最好成绩低于 suffix 单题最差（20 题无一翻盘），corr(中文占比,接受率)=−0.60 说明败因是文体而非语言——这条被证伪的事实转正为特化训练的 motivation：本负载的文体分布偏离通用训练分布，必须在本域轨迹上训（一手：eagle-idea.md）。

## 工作量映射
主菜 = decode 侧的复现度路由投机（suffix 为主力 + 判决模块做细粒度路由），理由是所有已兑现的大数字都在这里：e2e 三配置 −47.5%、enforce-eager +28%、report 段 2.37x、五常数成本模型给下注经济学定价（0.62ms/位、盈亏线 2.4%）。三件计划中工作量的嵌入方式——【1. 判决模块】是系统的连接组织（统一信号），但必须重新立论：不能做 per-call-type 路由（tag 免费、已被自家证据否决），要做的是角色内细粒度覆盖率预测——SUMMARY 双峰分布（每题 8%-88%）是它唯一的生存空间，职责有三：(a) 把 summary 段从 0.93x 拉到 1.05x+（初步证词已有）；(b) 按预测覆盖率决定下注深度（高覆盖段敢发深注，报告尾部 p10-12 段 5-9% > 盈亏线 2.4%）；(c) 同一信号门控 prefill 侧 KV reuse（预测复用量低 + prompt < crossover 4-6k 则关掉，避免小档 +32~60% 净亏）——第 (c) 条是'统一信号'主张的承重点，也是它区别于 SpecDec++/BanditSpec（都只管 decode 侧）的新颖性来源。【2. 特化 EAGLE-3】是填缝的配菜：suffix 只吃照抄内容，summary 81% 的步零接受是 suffix 的结构性盲区；现成 head 证伪（文体失配，corr −0.60）恰好构成'必须本域特化'的 motivation；它作为 routed pool 里的低覆盖臂存在（预测覆盖低 → 走特化 head，高 → 走 suffix），go/no-go 线是 summary 段接受率从现成的 ~13% 抬过 suffix 的 27%——高风险，故事要设计成它失败也不塌（退化为 suffix 单臂 + 路由开/关）。【3. decode-to-prefill KV reuse】是完整性配菜：它把'同一复现性质打两侧'的主张补齐（没有它故事就只是又一篇投机解码论文），实现已完成（blend_store_generated）、质量无害已证（40 题 kimi 判）、大档 prefill −51~67%，但 e2e 只 −8~10% 且 RelayCaching preprint 已抢跑机制叙事——所以论文里它的卖点不是数字而是(i)与统一信号的门控耦合、(ii)位置无关内容哈希对 Plato 前缀单调式的可区分性，绝不能把它写成主承重。

## 缺口
- 【P0 判决模块本体】训练 hidden-state/prompt-feature → 覆盖率预测器并跑通三个消费场景（summary 路由、下注深度、KV reuse 门控）的收益；预检必须先立'简单基线打不过'——用 tag+prompt长度+轮次的 logistic 基线对比，赢不了则整个统一信号叙事死（自家证据已判过一次'tag 免费'）；先用已有 1912 调用 traj 做零成本离线仿真定上界再决定训练投入。
- 【P0 合体 e2e】三组件（suffix+enforce-eager+路由 / 特化或现成投机 / KV reuse 门控开）在 40-100 题 DRBench 上的端到端墙钟 + ablation（各去一件），质量用 kimi-k2.5 判无损——目前没有任何一个合体数字，而这是'统一系统'故事的封面数字。
- 【P1 特化 EAGLE-3 训练】本域轨迹（online100_v2 及扩采）训 head，milestone：summary 段接受率 >27%（对齐 suffix）才进 routed pool，否则降级为负结果对照写进分析；须对照移动基线 EAGLE-3.1（长上下文 acceptance 2x）。
- 【P1 KV reuse 接入 agent 真管线】blend_store_generated 从单测接到 deep researcher 全链路（注意 generated[:-1] 末位坑），加 crossover 门控；blend vs native 质量裁决需 replay 固定搜索结果消检索噪声；顺手补 load 134ms launch-bound 的批量化 kernel（可把小档翻正，直接扩大可用面）。
- 【P1 characterization 补洞】搜索等待 vs prefill vs decode 的墙钟三分解（文献空白⑤）、子代理间实际 KV 可共享率实测（空白⑥）、上下文增长曲线——这是 §2 measurement study 成为独立贡献的最后三块板。
- 【P2 并发口径】现有大量数字是 batch=1 replay；并发洪水税已证真钱（ngram 垃圾草稿把 SUM 打到 0.75x），系统论文必须给 batch>1 下的吞吐/延迟曲线，且门控在并发下的价值可能反而更大（是机会不只是负担）。
- 【P2 泛化与对比】至少一个公开 agent 框架（GPT Researcher / open deep research）+ 至少第二个 base model 复现照抄率与加速；对比臂：纯 SuffixDecoding、CacheBlend/KVCOMM、（若可复现）RelayCaching；补查 GADS 新颖性欠的 spec-decode-drafting 轴（REST/SAM-decoding/PLD/cascade）。

## 弱点
- 统一信号可能被拆穿为包装：三个组件各有更便宜的门控（tag 免费、prompt 长度阈值、suffix 默认参已最优），且自家历史证据判过'训判决模块不值得'——若学习型预测器打不过 tag+长度的 logistic 基线，'统一'骨架整体坍塌，论文退化为三件不相干工作的缝合（A+B+C stapling 是系统审稿人最常见的拒稿理由）。
- KV reuse 一角先天疲软：e2e 只 −8~10%（decode 占 95%），机制叙事又被 RelayCaching（2603.13289）抢跑，若其间中会则该组件从'顶会空缺'跌为'增量跟随'；靠门控耦合与位置无关性区分的论证链较细。
- 特化 EAGLE-3 是未做的高风险臂：需要接受率 ~13%→>27% 的 3x 提升才有资格入 pool，现成 head 已全灭；若失败，'低覆盖段怎么办'没有答案，系统只剩 suffix 一种 drafter，与 SuffixDecoding+缓存的既有组合区分度骤降。
- 最大单项加速（enforce-eager +28%）本质是 vLLM 驱动锁工程修复而非方法；report 分节的 −25.4% 收益来自并发而非命中率（命中率反降 61.2%→46.2%）——审稿人拆账后'复现性质驱动收益'的因果叙事会缩水，须诚实分账并把锁发现写成独立 systems finding 而非混入方法收益。
- 外部效度：全部一手数字来自单一自建 demo、单 base model（Qwen 系）、中文混合语料（corr −0.60 显示文体敏感）；90.7% 照抄率可能被本 agent 的 prompt 设计放大，公开框架上若照抄率减半整个 motivation 打折。
- 质量无损的证据欠功效：Exp_M null 是测不出而非证伪、上界塌 4.5 分未归因、评委曾有漂移史（qwen-flash 0.236）——审稿人可攻'加速以质量为代价未被排除'，需要更大功效或 replay 固定检索的受控质量实验。
- 口径风险：多数投机数字是 batch=1 冷树 replay，serving 论文的标准口径是并发吞吐/尾延迟；热树污染假 1.912x 的前科说明测量纪律必须写进论文，否则数字可信度被质疑。

---

# Reading the Copy Signal: Coverage-Predictive Routing and Trajectory-Specialized Drafting for Deep Research Agent Serving

论点: Deep research agent 的每次生成调用在"逐字照搬"与"自由改写"两个 regime 之间剧烈摆动（类间 report 90.7% vs summary 53.8%，类内 8%–88% 双峰），而这个 regime 可以在生成前从 target model 自身的 hidden state 读出——因此我们训练连续覆盖率预测器做 per-call 投机策略路由与下注深度控制，为照搬 regime 配后缀树投机+decode-to-prefill KV 复用、为改写 regime 在 agent 自身轨迹上特化训练 draft head（现成 head 已实测全面失效，特化是必需而非锦上添花），构成第一个 regime 感知的学习路由 agent 推理加速系统。

## DR特点
- 特点1【输出大面积逐字照搬输入】：SUMMARY 调用 Coverage@8 均值 53.8%/中位 62.4%，REPORT 均值 90.7% 且 100% 调用 >50%、≥30 字连续片段仍覆盖 74.5%（一手：prompt-overlap-analysis-v2）。这是 copy regime 存在的地基，支撑 suffix/检索式投机在 report 段 2.29-2.37x 的收益来源。
- 特点2【同一调用类型内覆盖率剧烈方差+双峰】：SUMMARY 每题均值 8.0%–88.2%，分布双峰（80-90% 桶 542 次最大）；summary 81% 的步零接受但整体仍 28.8% 命中（一手：prompt-overlap-analysis-v2 + spec-decode traj）。这是'tag 免费但不够用'的正面回应——tag 只能给类型先验，per-call 连续覆盖率必须预测，是学习型判决模块的生存理由。
- 特点3【agent 轨迹文体使现成 draft head 失效】：现成 eagle3 双 head 接受率 10.7-20.3%，全面输给 suffix（26-57%），20 题无一翻盘，败因是文体（corr=−0.60）而非语言（一手：eagle-idea.md）。这是'必须特化训练'的直接证据，把一个负结果转成主菜二的 motivation。
- 特点4【output-becomes-input 的链式结构 + decode 主导】：上游 summary 的 decode 输出逐字流入下游 report 的 prefill（report 照抄 13432 字符/次）；decode 占 GPU 计算 95%，blend prefill 大档 −51~67% 但 e2e 仅 −8~10%（一手：claude-docx/11 + overlap 分析）。同时支撑 KV reuse 组件的可行性和它的配菜定位。
- 特点5【多轮迭代加深照搬】：v1 单轮→v2 多轮迭代，REPORT 照抄均值 86.9%→90.7%，>90% 占比 36%→71%（一手：v1/v2 对比）。说明 agent 越深、copy regime 红利越大，给'方法随 agent 复杂度增值'的成长性论述。
- 特点6【并行扇出使路由有真实经济代价】：分节并行输出膨胀 1.59×，并发下垃圾草稿把 e2e SUM 打到 0.75x，分节改写使 report 命中率 −10pp（一手：parallel-generation-experiment-v2 + spec-tax）。门控/路由只有在并发场景才值钱，这是预测器的经济学舞台。
- 特点7【hidden state 可预测本次调用覆盖率】：整个主菜一的核心假设，目前零证据——需补测量（1912 调用的离线回归可预测性实验，R²/Spearman vs tag-only 与 P0 探针基线）。
- 特点8【deep research 负载无公开 trace 级刻画】：文献确认 TraceLab 只覆盖 coding agent，DR 的内容重叠率/decode 占比/上下文增长无人测（survey 推断的空白）。使论文的 workload characterization 部分（1912 调用画像）本身可认领为贡献。

## 工作量映射
主菜一（判决模块，升级为"连续覆盖率预测器"）：不再做内部已否决的"tag 二分类替代品"，而是从 target hidden state（prompt 末态或前 32-64 token 的 decode 态）回归预测本次调用的连续覆盖率（Coverage@8），承担三个 tag 拿不到的职能——(a) 面向无 tag 的通用 serving 层：云端推理服务看不到 agent 框架的调用类型标签，hidden state 是 serving 层免协作可得的唯一信号；(b) 类内连续下注：一手数据显示同为 SUMMARY 的调用覆盖率从 8.0% 摆到 88.2%（双峰分布），tag 只给类型先验给不了 per-call 值，而连续预测值直接接上下注经济学（0.62ms/位草稿费、2.4% 盈亏线）决定草稿深度 k，把"路由"细化成"下注额度"；(c) 并发门控：ngram 4.9% 接受率的垃圾草稿在并发下把 e2e 打到 0.75x（batch=1 免费、并发付真钱），预测器是内部已选定的 P0 门控（题型先验+前 32-64 token 探针）的学习化推广，论文里 P0 探针就是它必须打赢的非学习基线。主菜二（特化 EAGLE-3）：suffix 只吃"照搬 regime"（report 57-61% 命中），在"改写 regime"（summary 28.8% 命中、0.87-0.96x 且调参翻不正）是死区；现成 eagle3 head 已实测全面证伪（10.7-20.3%，逐题最好 16.7% 低于 suffix 最差 23.5%，败因 corr(文体)=−0.60 是文体非语言）——这个负结果不是包袱而是"必须在 agent 自身轨迹上特化训练"的直接论据，验收线明确：summary 接受率从 ~11% 提到 >27%（3×）才翻盘。两个 drafter 各占一个 regime（suffix=copy、特化 head=paraphrase），预测器在顶层按预测覆盖率路由并定深度，合成"regime-aware 学习路由投机"这一个闭环故事，这是 NeurIPS 叙事的心脏。配菜（decode-to-prefill KV reuse）：decode 占 GPU 计算 95% 的一手事实一石二鸟——既解释为什么 KV 复用只能当配菜（e2e 仅 −8~10%），又反过来为"主菜必须打 decode 侧"正名；工程已完成（blend_store_generated，质量 40 题判无害），论文里作为 prefill 侧补全出现，与 RelayCaching/Plato 用"位置无关内容哈希+生产 LMCache 路径+质量无害证据"区分，不单独认领机制新颖性。

## 缺口
- P1（生死实验）：hidden state→覆盖率的离线可预测性。用 online100_v2 的 1912 调用，取 prompt 末 hidden state / 前 32-64 token decode hidden states 训 LoRA/MLP 回归 Coverage@8，报 R²/Spearman 与校准曲线；必须同时跑三个基线：tag-only、P0 探针（前 32-64 token 实测命中率外推）、tag+长度等浅特征。学习版打不赢 P0 探针则整个主菜一死，这个实验一周内可出结果，最先做。
- P2（主菜二的生死实验）：在本 agent 轨迹上特化训练 EAGLE-3 head。收集轨迹训练数据（现有 100 题×1912 调用可能不够，需扩采），验收线=summary 段接受率 >27%（超 suffix 单题最差 23.5% 与整体 28.8%）且 replay summary 段 ≥1.05x；同时报 vs EAGLE-3.1 新基线。若 81% 零接受步源于本质熵高而非 head 不适配，此路不通，需尽早止损判据。
- P3：路由闭环 e2e。预测器接入 serving（两 server 或 per-request 切换），对 Ceager 双冠军基线（SUM 1.00x/REP 2.29x）量增量，必须含并发场景（batch>1）证明门控省下洪水税的真钱——目前全部 e2e 数字都是 batch=1。
- P4（NeurIPS 硬门槛）：泛化。≥2 个其他 agent 负载（GPT Researcher / coding agent / AgenticSQL）+ ≥2 个 base model，证明覆盖率可预测性和特化配方（非具体权重）可迁移；顺带补上'DR workload 刻画'在第二个框架上的复现，回应单框架过拟合质疑。
- P5：KV reuse 与投机同开的联合 e2e（目前两件从未合体测过），加 blend vs native 的质量裁决（replay 固定搜索结果消掉检索噪声）。
- P6：消融与 novelty 收尾。连续覆盖率 vs 二分类 vs tag-only 消融；预测粒度（整调用 vs 分段）消融；补查 spec-decode-drafting 轴（REST/SAM-decoding/PLD/cascade）和 BanditSpec/HedgeSpec 的 venue 确认，钉死相关工作定位。

## 弱点
- 自家证据反咬：内部判决已写明'训判决模块 LoRA/MLP 不值得，tag 免费'。若 P1 实验里学习版对 tag+P0 探针的增量只有 1-2pp，审稿人（和作者自己）都会看出学习模块是为发论文而学习。防线只有两条：无 tag 通用 serving 场景讲得足够硬（需引证 serving 层确实拿不到应用 tag），或连续下注深度拿出探针拿不到的显著钱。
- 主菜二是期货：'现成 head 不行'只证明必要性不证明可行性。summary 81% 步零接受可能意味着内容本质不可预测（新检索结果的事实性内容），特化训练也救不了；一旦 P2 达不到 27% 验收线，故事只剩'预测器路由 suffix 开/关'，单薄到撑不起 NeurIPS。且 EAGLE-3.1 已把长上下文 acceptance 提 2x，基线在移动。
- 单框架单模型过拟合：全部一手数字来自一个 demo 系统。去冗余 prompt A/B 显示 summary −18.7%/report −28.9% 输出可被 prompt 工程直接砍掉——审稿人会问：你加速的是 workload 的本质还是你家 prompt 模板的啰嗦？必须正面回应（照搬是引用忠实性的内生需求，砍掉的只是铺垫）并用 P4 的跨框架数据压制。
- 赛道拥挤下的增量感：SpecDec++/BanditSpec/HedgeSpec 占'学习选投机策略'，DistillSpec/OSD/OmniDraft 占'在自有分布上训 drafter'且 Baseten/NVIDIA 已做成工业教程，RelayCaching 抢跑 decode-to-prefill KV。可辩护的新颖性只剩'信号来源=覆盖率预测'+'copy/paraphrase 双 regime 画像'，一旦 regime 洞察被认定为观察而非方法，论文被降级为场景组装。
- 收益天花板算术：suffix+enforce-eager 零学习已拿走 e2e −47.5%，summary 死区调参翻不正、KV reuse e2e 仅 −8~10%，学习组件的可兑现剩余空间可能只有 10-20%。'一套预测器+一次 head 训练换 15%'在系统口味的审稿人那里性价比存疑，必须把叙事重心压在'学习方法的可迁移 insight'而非绝对加速数字上。
- 会议错位风险：NeurIPS/ICLR 审稿人要普适的学习 science（为什么 hidden state 编码了未来的照搬行为？这和模型记忆/复制机制的什么理论连接？），如果论文只交付'预测器→路由→加速'的工程闭环，最可能的 meta-review 是'solid work, better fit for MLSys'。需要在 regime 可读性上做至少一节机理分析（如逐层探针、注意力对 prompt 的 copy head 证据）。

---

