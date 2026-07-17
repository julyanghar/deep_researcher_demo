# 故事1: Once Generated, Many Times Served: Characterizing and Exploiting Verbatim Content Recycling in Deep Research Agent Inference  score=6
## fatal_flaws
- 封面数字归属造假风险：e2e −47.5% 近腰斩里，大头=现成 SuffixDecoding + enforce-eager 一个 flag + report 分节（且分节收益来自并发、命中率反而 61.2%→46.2% 下降）。三个自研组件（特化 head/判决器/KV reuse）目前对 e2e 的已兑现贡献是 0。消融表诚实拆开后，审稿人第一问就是'去掉你们的三件，数字掉几个点'——按现有证据答案接近'不掉'。这会把论文从 system contribution 打回 deployment experience report。
- 刻画主指标与收益因果断裂（自家数据已证）：故事拿 Coverage@8 当地基解释一切，但 per-request-speedup-variance.md 白纸黑字写着字符级 Coverage@8 对 summary 加速 r=−0.23（反向），只有 token 级可收割量能预测（打平点 15-20%）。SUMMARY 的 53.8% 在 @30 口径下坍缩到 9.5%（最长片段均值仅 51 字）——所谓照抄一半是 8 字符碎片，与投机可利用性弱相关。'刻画 → guided optimization' 的因果链在占调用数 95% 的 SUMMARY 段上是断的，必须换 harvestable-token 口径重铸，否则 §2 和 §3 之间是装订不是推导。
- hidden-state 覆盖率判决器已被自家 oracle 上界预先罩死：完美路由只能把 summary 段 0.93x→1.05x（summary 段占 e2e 约 1/3，折合 e2e 约 +4%），大概率低于故事自设的 5% 立项线——而且这是 oracle，学习版兑现一半就只剩 e2e +2%。这条枝几乎注定要砍，砍完 decode 侧只剩 suffix+flag+期货 head。
- 特化 EAGLE-3 验收线高于自家预注册判据：故事写 '>27% 才有混合价值'，但 eagle3-domain-training-plan.md 自己预注册的是 P0 15%/P1 20%/P2 35%，且诚实注记'没有找到一篇现成 head→域内续训的干净公开案例'；旁证（OSD +0.1~0.65 绝对值、Draft-OPD +23% 相对）覆盖不了从 ~11% 到 27% 的 +16pp 确定性。主菜成功率 <50%，退路（suffix/off 两臂路由）又恰好撞上上一条的 oracle 罩顶。
- 首次刻画的边际新颖性：LLMA(2023) 已证 RAG 输出照抄检索文档，SuffixDecoding 已证 agentic 输出自重复，Preble 已做共享 prompt workload study。'首次 DR trace 刻画'的增量必须靠双峰/逐字vs语义分离/多轮放大/五常数税这些二阶画像撑，而这些目前全来自单一自建 demo + 单模型 + 100 题——不补第二框架就是 case study，补了才够 measurement study 的格。
- decode 占 95% 的口径未在并发下验证：该数字来自 batch=1 blend 实验；3-5 researcher 并发 + 分节 228 调用时 prefill 与 decode 重叠，时间构成会变。全文所有加速数字是 batch=1 冷树 replay，而 MLSys/OSDI 的 serving 论文标准口径是并发吞吐/尾延迟——热树假 1.912x 的前科说明这不是吹毛求疵。
- 质量无害主张欠功效：Exp_M 新 run 上界塌 4.5 分未归因、null=测不出≠证伪，40 题样本小，且 2026 已有'复用伤质量'负结果 preprint 在场。KV reuse 拿 e2e −8~10% 的收益扛质量风险质询，收益风险比难看。
## salvageable
- characterization-first 的四段式骨架（刻画→decode→prefill→e2e）是三个故事里唯一抗塌结构：任何一件期货失败，刻画+suffix+锁发现+税模型+KV 配菜仍组得成一篇——这个结构本身应保留为融合方案的底盘。
- 标题 'Once Generated, Many Times Served' 是三个标题里最好的：一句话钉死负载性质，且不预支任何未兑现的方法主张。
- 正向口径补测三件套清单（下游 prefill 来自上游 decode 的占比+KV 可复用率、e2e 墙钟三分解、上下文增长/扇出）全是文献确认的空白且数据在手，是'首次刻画'主张最便宜的弹药。
- '为什么在 serving 层做而非改 prompt'的防线论证（照抄主体=必须转述的检索证据；分节改写伤命中 −10pp；去冗余砍的是骨架铺垫）——这是全场唯一正面接住'prompt 工程反杀'的论述，必须进正文。
- 锁竞争发现+enforce-eager 修复（+28%）与五常数税模型（7 配置残差 0）作为独立 systems finding 是 MLSys 口味的硬通货，诚实分账后反而加分。
- 预设砍线文化（oracle 上界 <5% 就砍、诚实降级为分析）是审稿人视角最可信的写法，融合方案应继承。

# 故事2: EchoServe: Exploiting Cross-Call Content Recurrence for End-to-End Serving of Deep Research Agents  score=4
## fatal_flaws
- '统一信号'是伪统一，一拆即穿：prefill 侧门控需要的是输入侧复用量（prompt 里多少能从 cache 命中），decode 侧路由需要的是输出侧照抄率（将生成的 token 多少可 draft）——这是两个不同的预测目标，两者相关性从未测过，用'复现度'一个词捆绑是营销不是技术。更难堪的是自家数据显示 KV reuse 的正确门控信号是 prompt 长度（crossover 4-6k 是 launch-bound 固定开销决定的，与复用量无关）——一个 if 语句替代'统一信号'在 prefill 侧的全部职能。承重点 (c) 塌了，'统一'骨架就只剩三件不相干工作的缝合，而 A+B+C stapling 正是系统审稿人最常用的拒稿理由（故事自认了，但没有解法）。
- 三单点串联的最脆结构：统一信号死→骨架塌（自认）；特化 head 死→低覆盖臂无答案（自认）；KV reuse 不接入真管线→'两侧'不成立。三个故障点任何一个触发整个故事降级，而三个的单独成功率都不高（预测器要打赢 tag+长度 logistic、head 要 ~13%→27%、KV 接入含 [:-1] 坑的全链路工程）。一个投稿周期内三关全过的概率我评 <15%。
- 覆盖率预测信号被自家数据反证（与故事3 同罪）：Coverage@8 对 summary 加速 r=−0.23，'复现度预测'若以字符覆盖率为目标就是在预测一个与收益反向的量；若改用 token 级可收割量为目标，则 oracle 上界（summary 段 0.93→1.05）又只值 e2e ~4%。信号叙事两头堵。
- 系统身份与组件成色不匹配：EchoServe 作为系统论文，已兑现数字全来自现成组件（suffix）+一个 vLLM flag；架构新颖性主张'cross-layer 顶会空缺'依赖 survey 未确认项（RelayCaching venue 未核、Teola venue 未核），窗口随时被 2026 扎堆 preprint（TokenDance/LRAgent/QKVShare）关掉。空缺是真的，但空缺不等于你的填法成立。
- 并发口径缺失对本故事伤害最大：故事把'门控在并发下才值钱'（洪水税 0.75x）当预测器的经济学舞台，但全部 e2e 证据是 batch=1——即系统的核心卖点恰好建在唯一没测过的口径上。
- TTFT 税对冲叙事（decode 投机付 +14%、prefill 复用缴回）是全文最漂亮的一句话，但两件从未合体实测，目前只是算术推演；且对小档（<4-6k）调用是双重净亏（blend +32~60% 再叠 eager +14%），对冲只在大 prompt 上成立，论文里必须带边界条件说，否则一个 microbenchmark 就能戳破。
## salvageable
- cross-layer 定位本身（同一负载性质在 prefill+decode 两侧的利用，顶会无先例）是 survey 确认的最有价值切口——但正确写法是'一个性质、两种机制'（性质统一，机制各自诚实），不是'一个信号、统一门控'。
- TTFT 税对冲：enforce-eager +14% 恰好被 blend 大档 −51~67% 缴回，是论证'两件必须合体、拆开引用已有工作不等价'的最强武器——值一个合体实测，成本一两天。
- sibling 冗余的诚实分流（decode 侧互抄判死 cov24 0.69%、冗余只喂 prefill/检索侧 14.6% 共享 URL）是三个故事里最干净的负特征运用，防止审稿人拿 idea#4 的尸体反杀。
- '多轮越深、复现越强'（86.9%→90.7%、>90% 桶 36%→71%）的成长性论据，把系统绑定 deep research 而非泛 agent，是 motivation 章节的好料。
- vs Plato 的可区分性表述（位置无关内容哈希 vs 前缀单调累加，原文铁证已核）应原样搬进融合稿的 related work。

# 故事3: Reading the Copy Signal: Coverage-Predictive Routing and Trajectory-Specialized Drafting for Deep Research Agent Serving  score=3
## fatal_flaws
- 核心信号已被自家数据双重反证，且故事对证据存在误引：论点声称'生成前预测器路由可把 summary 段 0.93x→1.05x（初步证词已有）'——查证 suffix-spec-decode/docs/summary.md:78，这是完美路由 oracle 上界，不是任何已兑现收益；同一行还写着字符级 Coverage@8 对 summary 加速 r=−0.23（反向）。即：论文心脏（回归预测 Coverage@8 → 路由）的训练目标是一个与收益负相关的量，而换成正确目标（token 级可收割量）后 oracle 天花板折合 e2e 仅约 +4%。这不是'需补测量'，这是'已测过且答案不利'。
- 与自家判决体系全面冲突：'训判决模块 LoRA/MLP 不值得，tag 免费'是内部已定案的否决；idea#1 早 miss 门控（precision 0.14-0.31≈基线）、idea#2 ctx/长度门控（四分位持平）两个行为侧信号门控已死；共同教训'热全局树+题内并发已把聪明调度空间提前吃掉'是对一切聪明路由的先验判决。且免费基线极强：首中→后中 73.6% vs 首否→10.7% 意味着前 32-64 token 运行时探针（实测外推）几乎就是可部署的强预测器，学习版要赢的是它而不是 tag——生成前 vs 探针期的差异只值调用长度的 5-20%。论文里同时出现'我们判死过判决模块'和'我们训了判决模块'，rebuttal 无解。
- 两个主菜全是期货且串联：主菜一（预测器）被 oracle 罩死+目标变量反向；主菜二（特化 head）验收线 27% 高于自家训练计划预注册的 P1 20%，成功率 <50%。两主菜联合存活率 <25%，故事自认'失败只剩 suffix 开/关路由，单薄到撑不起 NeurIPS'——把一篇论文押在轮盘上。
- 会议错位是结构性的：NeurIPS/ICLR 要普适 learning science（hidden state 为何编码未来照搬？与复制机制/induction head 的理论连接？），故事只有工程闭环+单负载绑定（还把绑定当卖点），meta-review 大概率是 'solid work, better fit for MLSys'；转投 MLSys 则全部数字是 batch=1 replay、无并发口径，系统审稿人同样拒。两头不靠，且补机理分析（逐层探针/copy head 证据）是又一整块未列入计划的工作量。
- regime 二分的理论包装超出证据：双峰的主要方差来源可能是题目而非调用（每题均值 8-88% 说明题间方差巨大），题内 vs 题间分解没做——若 regime 由题决定，'per-call 预测'退化为'同题首调用外推'，又一个免费信号杀死学习模块。
- '无 tag 通用 serving'动机在自家实验设置里是伪需求：自家全栈可见 tag（tag 路由零成本正是判死判决模块的理由），要讲云端 multi-tenant 无标签场景需要换一套实验设置来演示，这是计划外的新工作量，不做这条动机就是纸糊的。
## salvageable
- 下注深度经济学（单位草稿费 0.62ms/位、盈亏线 2.4%、报告尾部段 5-9% 敢深注）是税模型（残差 0）到策略的最干净闭环——把'路由选臂'降维成'给 suffix 配深度 k'，无需学习、有现成证据，应作为融合稿 decode 侧的分析亮点。
- copy/paraphrase regime 画像（双峰、题级加速 0.71x-2.56x、检索型 vs 分析型分裂）作为刻画章节的观察极有价值——作为观察，不作为方法。
- '字符覆盖率好看但不预测收益、token 级可收割量才预测'这个口径教训本身就是刻画贡献的一部分（对后来者是避坑指南），值得单独一小节。
- 特化 head 的预注册判据设计（P0/P1/P2 分档+held-out 20 题+止损判据）是三个故事里最规范的实验设计纪律，融合稿应照抄这个纪律而非故事3 拍脑袋的 27% 单线。
- eagle3-domain-training-plan.md 的数据盘点（harvest 12,373 条、去重 3-5K、题库可扩产 500 题≈+9K/20h）证明特化训练不是零起点，作为 timebox 实验的可行性底稿保留。

# 融合建议
最强组合 = 故事1 的骨架 + 故事2 的旗帜 + 故事3 的经济学，同时把三个故事共同的'学习型判决器'幻想按自家已有数据当场处决。具体拼法和理由如下。

【底盘：故事1 的 characterization-first 四段式，投 MLSys】为什么：三个故事里只有它的主承重（刻画）是证据已在手 80% 的贡献，且结构抗塌——任何期货组件失败都不塌楼。会议选 MLSys 而非 NeurIPS/OSDI：measurement+guided-optimization+工程发现（锁竞争 +28%、五常数残差 0）正好是 MLSys 的口味谱，NeurIPS 要的 learning science 拿不出，OSDI 要的系统架构浓度不够。标题沿用 Once Generated, Many Times Served。

【第一处修正（最重要）：把刻画主指标从 Coverage@8 换成 harvestable tokens】自家数据已证字符级 Coverage@8 对 summary 加速 r=−0.23（反向），token 级连续可收割量才与加速单调对应（打平点 15-20%，harvest 定律 r=0.90）。三个故事都把 Coverage@8 当地基，这是全场最大的未爆弹——审稿人只要复算一次相关性就能引爆。正确做法：§2 刻画以'可收割量'为一等公民指标，Coverage@8 降为内容画像的辅助口径，并把'字符口径好看≠可收割'（SUMMARY @8=53.8% 但 @30=9.5%、最长片段仅 51 字 vs REPORT @30 仍 74.5%、最长 694 字）写成一个独立的口径教训小节——这本身就是刻画的增量贡献，也顺手解释了 suffix 命中率 report 61% vs summary 29% 的两级分化。

【旗帜：借故事2 的 cross-layer 定位，但换写法】'同一负载性质（verbatim recycling）在 decode 侧（suffix 投机+深度下注）与 prefill 侧（decode-to-prefill KV blend）的两种利用，构成第一个 DR-aware 的 prefill+decode 联合 serving'——性质统一、机制各自诚实，绝不用'统一信号'这种一拆即穿的包装。两件耦合的必要性用 TTFT 税对冲钉死：enforce-eager 为 decode 换来 +28% 但付 TTFT +14%，blend 大档 prefill −51~67% 恰好缴回这笔税——这句话是回应'A+B stapling'质疑的唯一硬通货，所以合体实测（suffix+eager+blend 同开，40 题）必须做、且要带小档边界条件（<4-6k 双重净亏，用长度 if 门控，坦白这就是个 if）。

【decode 侧策略层：用故事3 的下注经济学替换所有学习组件】判决器处理方式：先用已在手的数据把 oracle 上界写进论文——完美 per-call 路由仅 summary 段 0.93x→1.05x、折合 e2e 约 +4%，据此公开宣判'学习路由在本负载 batch=1 下被上界罩死'，这是一个对 SpecDec++/BanditSpec 赛道有引用价值的负结果，比硬训一个 LoRA 诚实且省一个月。留一个活口：并发场景的洪水税（ngram 垃圾草稿把 SUM 打到 0.75x）不在该上界内，若 batch>1 重算 oracle 后门控价值 >5% 再上非学习的三层免费信号栈（tag 先验 → prompt 长度门 KV → 前 32-64 token 探针外推，73.6%/10.7% 的可复现性已证）。决策的'聪明'全部收敛到下注深度：按段位与盈亏线 2.4% 连续调 k，这是税模型直通策略的漂亮闭环，零训练成本。

【特化 EAGLE-3：降为 timebox 4 周的预注册实验，成败都是内容】按 eagle3-domain-training-plan.md 原计划执行（SpecForge warm-start AngelSlim、只训 summary 臂、held-out 20 题），判据用计划自己的 P0 15%/P1 20%/P2 35%，论文预注册写明：过 P2（35%>suffix）→ 混合投机成为 decode 侧第二臂，故事升级；只到 P1 → 写成'域适配可移动接受率但不足以翻盘 suffix'的定量负结果+数据量下界分析（与现成 head 全灭、corr=−0.60 文体归因串成完整的 drafter 选型指南）。这样它从'塌了故事就塌'的主菜变成'怎么着都有一节'的实验，且 agent-trace-trained head 无已发表工作的空缺照样占住。

【一个投稿周期的排期（按生死顺序）】第 1 周：并发口径下重算路由 oracle（判决器最终生死，数据在手纯分析）+ 正向口径三件套启动（下游 prefill 来自上游 decode 的占比/KV 可复用率、墙钟三分解、扇出与上下文增长——全是文献空白且纯分析工程）。第 1-2 周：合体 e2e（suffix+eager+report 分节+KV blend 长度门控）40 题 + 消融诚实拆账——这是封面数字，不依赖任何期货。第 2-6 周：batch>1 并发吞吐/尾延迟曲线（serving 论文硬门槛，也是洪水税门控唯一能兑现价值的口径）。第 3-8 周：GPT Researcher（最便宜的第二框架）复测照抄率/冗余率核心表——单框架 artifact 是所有审稿人的第一刀，这是唯一挡法。并行 timebox：特化 head 4 周。质量侧：replay 固定检索的 blend vs native 终裁 + kimi 评委加大样本，把'无害'从欠功效 null 加固成有功效结论。放弃项：机理分析（copy head 探针）、无 tag multi-tenant 场景、GADS 主张（除非 spec-decode-drafting 轴查完确认不撞）——都是好题目但不属于这个周期。

【为什么这个组合最强】它把已兑现的最强资产（−47.5%、+28%、残差 0、90.7%/74.5% 照抄画像、质量无害证据）全部放在承重位，把两个高风险期货（预测器、特化 head）分别处决和降级成'成败都有内容'的预注册实验，把三个故事各自最好的论证（抗塌骨架、税对冲耦合、下注经济学）拼在一起，且每一处修正都有自家一手数据背书——审稿人能攻的面从'期货兑现不了'收缩到'单框架泛化'和'增量新颖性'两个可用工作量正面回应的点。预计成稿是一篇 MLSys borderline-accept 档的论文：measurement 部分首创性够、系统部分数字硬、诚实负结果多到成为风格。

# 最终DR特点清单
- 【已测】输出对输入的大面积逐字照抄，且按调用角色硬分层：REPORT Coverage@8 均值 90.7%/中位 93.1%（100% 调用 >50%，≥30 字连续片段口径仍 74.5%，最长单段 694 字）；SUMMARY 均值 53.8% 且双峰（80-90% 桶 542 次最大）。n=1812+100，prompt-overlap-analysis-v2。需第二框架/第二模型复测防 prompt 模板 artifact，否则会被降级为 case study。
- 【已测，且是关键口径教训】SUMMARY 的照抄是碎片化短片段、REPORT 是长段整搬：SUMMARY @8=53.8% 但 @30 仅 9.5%、最长片段均值 51 字；REPORT @30 仍 74.5%、最长片段均值 246 字。字符覆盖率不预测投机收益（Coverage@8 对 summary 加速 r=−0.23 反向），token 级连续可收割量才单调对应（打平点 15-20%，harvest 定律 r=0.90）。这条应作为刻画主指标的选型依据写进论文。
- 【已测】跨调用冗余是逐字复制而非语义改写：同题 summary 18975 配对 out_contain8 均值 22.9%（37.3% 配对 >30%）vs 语义口径 out_jac5 仅 7.7%（>30% 仅 0.4%）；同 researcher 配对 34.6% vs 不同 21.7%。决定字面机制（后缀树/内容哈希 KV）适用、语义缓存不必要。researcher-redundancy-v2。
- 【反向口径已测/正向口径需补测】output-becomes-input 链式数据流：上游 summary 的 decode 产出成为下游 report 的 prefill 输入，REPORT 每次生成平均 13432 字符逐字来自 prompt（max 31122）。缺正向口径：下游 prefill token 中逐字来自某次上游 decode 的占比、其中满足 KV 可复用条件（chunk 对齐、generated[:-1]）的比例——这是 KV reuse 章节动机的缺板，数据在手纯分析可补。
- 【已测，归因需谨慎】多轮迭代放大照抄：v1 单轮→v2 多轮 REPORT 照抄均值 86.9%→90.7%、>90% 占比 36%→71%，多轮互抄边际 ×1.65；SUMMARY 基本不变。仅一对系统版本对比，轮次与其他改动可能混杂，论文表述应留余地。
- 【已测 batch=1/并发口径需补测】decode 主导：decode 占 GPU 计算 ~95%（blend 实验口径），report 段占 e2e 近半（4584/9293s）；直接推论=优化重心在 decode 侧、prefill KV 复用 e2e 仅 −8~10% 只能当配菜。需补：搜索等待/prefill/decode 墙钟三分解（文献空白）与 batch>1 下的时间构成。
- 【段间已测/题内外方差分解需补测】接受率与照抄率的段类型硬分裂+类内巨方差：suffix 命中 report 61.2% vs summary 28.8%；逐段接受率 59.5/39.4/26.7%；SUMMARY 每题均值 8.0%-88.2%、题级加速比 0.71x-2.56x（检索型加速、分析型拖慢）。需补题内 vs 题间方差分解——它决定 per-call 信号是否被'同题外推'免费替代。
- 【已测】现成通用 draft head 因文体（非语言）在本负载失效：eagle3 双 head 接受率 10.7-20.3%，逐题最好 16.7% 低于 suffix 单题最差 23.5%，20 题无一翻盘；corr(中文占比,接受率)=−0.60 排除语言归因。特化训练必要性成立（可行性另证，预注册 P0 15%/P1 20%/P2 35%）。eagle-idea.md。
- 【已测 batch=1 与并发对照/系统级曲线需补测】并发下劣质草稿付真钱：batch=1 免费的垃圾草稿（ngram 4.9% 接受率）在并发下把 SUM e2e 打到 0.75x；每步成本可加分解为五常数模型（graph 24.0+eager 0.5+管线 1.9+混合边界税 8.1+激活 5.8+0.62/位，7 配置残差 0），其中混合边界税主体是 cuLaunchKernel 驱动锁等待 4.5ms/步，enforce-eager 一参消除（e2e 中位 +28%，代价 TTFT +14%）。
- 【已测，负特征】兄弟 summary 的 decode 侧互抄不可利用：token 级兄弟 cov24 仅 0.69%，接受率对兄弟数全平（29.2%→29.5%）；sibling 冗余真实存在但在检索/prefill 侧（14.6% 配对共享 URL、每题共享 URL 占比均值 16.7%、prompt snippet 重合 27.6%）。划定跨 sibling 复用只能做 prefill 不能做 draft 源的边界。
- 【已测】照抄部分可被应用层消除但消除伤复用：去冗余 prompt A/B 使 summary 输出 −18.7%、report −28.9%（事实与引用全留，砍的是铺垫骨架）；report 分节改写使 suffix 命中 48.4%→38.9%（−10pp）但靠并发仍净赚 −25.4%。这是'serving 层利用 vs 应用层消除'取舍的直接证据，必须写进正文抢答 prompt 工程反杀。
- 【文献已有，用于定位】外部互证：agentic 输出自重复（SuffixDecoding 5.3x）、RAG 输出照抄检索文档（LLMA 2x）、缓存后 decode-dominated（arXiv 2605.26297）、prompt≫output 且 85-97% 跨请求共享（Preble）、多 agent ≈15x chat token（Anthropic）。论文增量=DR 负载上这些性质的定量画像与相互耦合，不认领定性现象本身。
- 【检索推断的空白，投稿前需复查】deep research 负载无公开 trace 级系统刻画：TraceLab 仅覆盖 coding agent，内容重叠率/decode 占比/上下文增长曲线/扇出统计均无人测。'首次'主张成立的前提是补齐正向口径+墙钟分解+第二框架，且投稿前再查一轮 2026 新 preprint 防抢跑。