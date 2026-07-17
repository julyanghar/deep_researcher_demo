# Deep Research Agent 的负载特点 × 三件工作量：顶会故事线综合判决

> **缘起**：用户问——手里的工作量只有三件（LoRA/MLP 判决模块、场景特化 EAGLE-3 训练、decode-to-prefill KV reuse），怎么和 deep research agent 本身的特点绑成一个顶会故事。
> **方法**：10 个 agent 的多路调研（4 路盘本地一手证据 + 2 路外部文献 + 3 个角度各写一版故事线 + 1 个对抗评审），主笔综合。原始材料见文末存档。
> **一句话答案**：DR agent 的定义性系统特征是**逐字内容再循环（verbatim content recycling）**——同一段 token 被生成一次，随后既作为下游 prefill 的输入（→KV reuse 的地基）、又作为下游 decode 的模板（→suffix 投机的地基）。推荐故事 = **characterization-first、投 MLSys**；三件工作量里 KV reuse 和特化 EAGLE-3 各有明确位置，但**学习型判决模块被自家已有数据三重反证，建议处决为负结果 + 用下注经济学替代**。

---

## 一、DR agent 可以讲的特点清单（按证据状态）

普通 chat 是"一次问答"，RAG 是"一次检索一次生成"；deep research agent 的本质区别在于**内容在一张调用图里流动**：检索片段 → researcher summary → writer report，每一跳都是逐字搬运为主。由此长出下面这些可量化的系统特征。

### 已有一手数字的（论文可以直接落数）

| # | 特点 | 关键数字 | 出处 |
|---|---|---|---|
| 1 | **输出大面积逐字照抄输入，且按角色硬分层** | REPORT Coverage@8 均值 90.7%（100% 调用 >50%、@30 口径仍 74.5%、最长单段 694 字）；SUMMARY 均值 53.8% 且双峰 | [prompt-overlap-analysis-v2](../../../prompt-overlap-analysis-v2/summary.md) |
| 2 | **照抄形态分两种：长段整搬 vs 碎片**（⭐ 口径教训） | SUMMARY @8=53.8% 但 @30 仅 9.5%、最长片段均值 51 字；REPORT @30 仍 74.5%、最长 246 字。**字符覆盖率不预测投机收益（r=−0.23 反向），token 级可收割量才预测（harvest 定律 r=0.90）** | [per-request-speedup-variance.md](data-source/per-request-speedup-variance.md) |
| 3 | **复现的"量"分调用异质、"形态"一旦发生则逐字**（2026-07-11 修正：有的 call 逐字照抄、有的语义改写——量的异质性正是按重叠率路由的动机；形态逐字则决定字面机制适用） | 量：见 #1/#7 的谱系与双峰；形态：同题 summary 配对逐字口径 22.9% vs 语义口径仅 7.7% → 后缀树/内容哈希 KV 接得住，语义缓存不必要 | [researcher-redundancy-v2](../../../researcher-redundancy-v2/redundancy_summary.md) |
| 4 | **output-becomes-input 链式数据流** | REPORT 每次生成平均 13432 字符逐字来自 prompt（max 31122），prompt 主体 = 上游 decode 产出 | 同 #1（反向口径） |
| 5 | **多轮迭代放大照抄** | v1→v2：REPORT 照抄 86.9%→90.7%、>90% 占比 36%→71%；多轮互抄 ×1.65 → agent 越 deep 本方法红利越大 | 同 #1 |
| 6 | **decode 主导端到端** | decode 占 GPU 计算 ~95%；report 段占 e2e 近半（4584/9293s）→ 优化重心必须在 decode 侧，prefill KV 复用只值 e2e −8~10% | LMCache claude-docx/11 + [e2e-3config-40q](../e2e-3config-40q.md) |
| 7 | **接受率按段类型硬分裂 + 类内巨方差** | suffix 命中 report 61.2% vs summary 28.8%；SUMMARY 逐题照抄率 8.0%–88.2%、题级加速 0.71x–2.56x | traj + per-request 分析 |
| 8 | **现成通用 draft head 因文体（非语言）失效** | eagle3 双 head 10.7–20.3%，20 题无一翻盘 suffix；corr(中文占比,接受率)=−0.60 排除语言归因 → **特化训练的必要性由负结果直接给出** | [eagle-idea](../eagle-idea/eagle-idea.md) |
| 9 | **并发下劣质草稿付真钱** | batch=1 免费的垃圾草稿（ngram 4.9% 接受率）并发下把 SUM e2e 打到 0.75x；五常数税模型 7 配置残差 0；锁竞争 + enforce-eager e2e +28%（代价 TTFT +14%） | [fixed-tax-conclusions](../examine-spec-tax/fixed-tax-conclusions.md) |
| 10 | **兄弟互抄不可利用（负特征，划边界）** | token 级兄弟 cov24 仅 0.69%、接受率对兄弟数全平；sibling 冗余真实存在但只在检索/prefill 侧（14.6% 配对共享 URL） | [idea-verdicts](idea-verdicts-and-standing.md) #4 |
| 11 | **照抄可被应用层消除，但消除伤复用** | 去冗余 prompt 砍 summary −18.7%/report −28.9% 输出；但分节改写使 suffix 命中 48.4%→38.9% → "serving 层利用 vs 应用层消除"的取舍有一手证据，可正面抢答 prompt 工程反杀 | [parallel-generation-experiment-v2](../../../parallel-generation-experiment-v2.md) |

### 需补测量的（都是纯分析工程，数据在手）

- **正向口径**：下游 prefill token 中逐字来自上游 decode 的占比 + 其中满足 KV 可复用条件（chunk 对齐、`generated[:-1]`）的比例——这是 KV reuse 章节 motivation 的缺板。
- **e2e 墙钟三分解**：搜索等待 / prefill / decode（文献确认空白，TokenCake 只有定性）。
- **题内 vs 题间方差分解**：决定 per-call 信号是否被"同题首调用外推"免费替代。
- **并发口径**：全部投机数字目前是 batch=1 replay，serving 论文标准口径是并发吞吐/尾延迟。

### 文献互证与空白（survey 确认）

- 互证：agentic 输出自重复（SuffixDecoding 5.3x）、RAG 输出照抄检索文档（LLMA 2x）、缓存后 decode-dominated（arXiv 2605.26297）、prompt≫output 且 85–97% 跨请求共享（Preble）、多 agent ≈15x chat token（Anthropic）。
- **空白 = 机会**（2026-07-11 对抗核查后收窄）：deep research 负载**在 LLM 调用层的内容级 trace 刻画**仍无人做；但"首个"须钉四限定——trace 级 agent 刻画已有 TraceLab（coding）/2605.26297（ReAct）/2606.06448（记忆）三家；搜索日志层已有 2601.17617 测 CTAR≈54%+轮次演化；DR 上 KV 命中率已有 AgentInfer 2512.18337。安全措辞见 [Once Generated.md](../eagle-idea/Once%20Generated.md) 缺口末条与 [once-generated-paper-narrative.md](../eagle-idea/once-generated-paper-narrative.md) §6。窗口在快速关闭（半年五篇近邻），建议尽快挂 arXiv。

---

## 二、三件工作量的映射判决

### (3) decode-to-prefill KV reuse —— 配菜，但有一个不可替代的角色

- **对应特点**：#4 output-becomes-input（上游 decode 时 KV 已算过一遍，下游 prefill 是纯浪费）。
- **资产已在手**：blend_store_generated 已实现（LMCache）、质量无害 40 题已证（SW 30.3%/C 28.2% vs 不复用 B 24.8%，kimi-k2.5 判）、大档 prefill −51~67%。
- **为什么只能当配菜**：decode 占 95% 罩死 e2e 收益（−8~10%）；且 **RelayCaching（arXiv 2603.13289，2026-03 preprint）已抢跑机制叙事**（同样是上游 agent decode KV 给下游 prefill），窗口在收窄。
- **不可替代的角色 = TTFT 税对冲**：enforce-eager 给 decode 换来 +28% 但付 TTFT +14%；blend 大档 prefill −51~67% 恰好缴回这笔税。**"decode 加速付 prefill 税、prefill 复用把税缴回来"是回应 A+B stapling 质疑的唯一硬通货**——所以 suffix+eager+blend 的合体实测必须做（目前两件从未合体测过），且要带小档边界条件（<4–6k token 双重净亏，用长度 if 门控，坦白这就是个 if）。

### (2) 场景特化 EAGLE-3 —— 降为 timebox 4 周的预注册实验，成败都有一节

- **对应特点**：#7 段类型分裂（summary 是 suffix 的结构性盲区，81% 步零接受）+ #8 现成 head 文体失效。负结果直接给出"必须在本 agent 轨迹上特化"的必要性——这不是"特化更好"的泛泛之谈，而是"现成的在这个负载上定量地不行"。
- **新颖性**：survey 确凿搜不到"agent-trace-trained draft head"已发表工作（DistillSpec/OSD/OmniDraft 占通用位置，Baseten/NVIDIA 是工业教程非论文），空缺是真的。
- **正确姿势**：按 [eagle3-domain-training-plan.md](../../../eagle-spec-decode/eagle3-domain-training-plan.md) 的预注册判据执行（P0 15% / P1 20% / P2 35%，held-out 20 题，3 epoch <15% 止损）。**过 P2 → decode 侧第二臂、故事升级；只到 P1 → 写成"域适配可移动接受率但不足以翻盘 suffix"的定量负结果 + 数据量下界分析**，与现成 head 全灭、corr=−0.60 文体归因串成完整的 drafter 选型指南。这样它从"塌了故事就塌"的主菜变成"怎么着都有一节"的实验。
- ⚠️ 注意移动基线：EAGLE-3.1（2026-05）长上下文 acceptance 提 2x，实验必须对照。

### (1) LoRA/MLP 判决模块 —— 被自家数据三重反证，建议处决为负结果 + 下注经济学替代

三重反证（全部有一手出处，已核实）：

1. **训练目标反向**：它要预测的字符级 Coverage@8 与 summary 实际加速 **r=−0.23（负相关）**；真正预测收益的是 token 级可收割量（[per-request-speedup-variance.md](data-source/per-request-speedup-variance.md)）。
2. **上界罩死**：即便换对目标，**完美 per-call 路由的 oracle 上界只有 summary 段 0.93x→1.05x，折合 e2e 约 +4%**（[summary.md:78](../summary.md)）；学习版兑现一半只剩 +2%。
3. **免费基线太强**：tag 免费（内部已判）；前 32–64 token 运行时探针的可复现性 73.6% vs 10.7%，几乎就是可部署的强预测器。论文里同时出现"我们判死过判决模块"和"我们训了判决模块"，rebuttal 无解。

**替代方案（保留"判决"的智力贡献，不押在训练上）**：
- (a) 把 oracle 上界写进论文，公开宣判"学习路由在本负载 batch=1 下被上界罩死"——这是对 SpecDec++/BanditSpec 赛道有引用价值的**负结果**，比硬训一个 LoRA 诚实且省一个月。
- (b) 决策的"聪明"全部收敛到**下注深度经济学**：按段位与盈亏线 2.4%（草稿费 0.62ms/位）连续调草稿深度 k——税模型（残差 0）直通策略的最干净闭环，零训练成本。
- (c) **留一个活口**：并发场景的洪水税（0.75x）不在该上界内。第 1 周先在并发口径下重算 oracle，若门控价值 >5% 再上**非学习**的三层免费信号栈（tag 先验 → prompt 长度门 KV → 探针外推）。

---

## 三、推荐故事线（融合方案）

**标题**：*Once Generated, Many Times Served: Characterizing and Exploiting Verbatim Content Recycling in Deep Research Agent Inference*
**投递**：MLSys（measurement + guided optimization + 工程发现正好是它的口味谱；NeurIPS 要 learning science 拿不出，OSDI 要架构浓度不够）。

**四段式骨架**（抗塌结构：任何期货组件失败，刻画+suffix+锁发现+税模型+KV 配菜仍组得成一篇）：

1. **§2 刻画（主承重，证据已在手 80%）**：DR 负载的 recycling 画像。⭐ **主指标必须从 Coverage@8 换成 harvestable tokens**（字符口径 r=−0.23 是全场最大未爆弹，审稿人复算一次相关性就能引爆）；"字符口径好看 ≠ 可收割"本身写成独立的口径教训小节，顺手解释 report 61% vs summary 29% 的两级分化。
2. **§3 decode 侧**：suffix + enforce-eager（锁竞争发现是 MLSys 硬通货）+ 下注深度经济学 + 特化 head 预注册实验（成败都有内容）+ 学习路由的上界负结果。
3. **§4 prefill 侧**：decode-to-prefill KV blend（位置无关内容哈希 vs Plato 前缀单调累加的铁证区分保留原样）+ TTFT 税对冲 + 长度门控。
4. **§5 e2e**：合体实测 + 诚实消融拆账 + 并发口径 + 质量裁决（kimi-k2.5，qwen-flash 已证不可靠）。

**一个投稿周期的排期（按生死顺序）**：
- 第 1 周：并发口径重算路由 oracle（判决器最终生死，纯分析）+ 正向口径三件套启动（全是文献空白且数据在手）。
- 第 1–2 周：合体 e2e（suffix+eager+report 分节+KV blend 长度门控）40 题 + 消融诚实拆账——封面数字，不依赖任何期货。
- 第 2–6 周：batch>1 并发吞吐/尾延迟曲线（serving 论文硬门槛）。
- 第 3–8 周：GPT Researcher（最便宜的第二框架）复测照抄率/冗余率核心表——**单框架 artifact 是所有审稿人的第一刀，这是唯一挡法**。
- 并行 timebox：特化 head 4 周（预注册判据）。
- 质量侧：replay 固定检索的 blend vs native 终裁 + 加大样本，把"无害"从欠功效 null 加固成有功效结论。
- **放弃项**（好题目但不属于这个周期）：copy-head 机理探针、无 tag multi-tenant 场景、GADS 主张（除非 spec-decode-drafting 轴查完确认不撞）。

**投稿前必查**：RelayCaching 是否已中会；BanditSpec/HedgeSpec 正式 venue；2026 新 preprint（TokenDance/LRAgent/QKVShare 已在扎堆）再扫一轮。

---

## 四、三个候选故事存档（对抗评审打分）

| 故事 | 一句话 | 评分 | 死因/保留 |
|---|---|---|---|
| A. Once Generated, Many Times Served（characterization-first） | 首次 DR trace 刻画 + recycling-aware 联合加速 | **6/10** | 骨架最抗塌 → **当底盘**；死穴=封面数字归属（−47.5% 大头来自现成组件）须诚实拆账 |
| B. EchoServe（统一系统） | 一个"复现度"信号统一门控 prefill+decode | 4/10 | "统一信号"一拆即穿（prefill 门控其实是 prompt 长度 if）；**cross-layer 旗帜和 TTFT 税对冲保留** |
| C. Reading the Copy Signal（学习方法） | hidden state 预测覆盖率做路由 | 3/10 | 训练目标与收益反向 + oracle 罩死 + 会议错位；**下注经济学和预注册纪律保留** |

原始材料：三版完整故事线与评审全文在 `/tmp/claude-1008/-home-yilin/e831ffb1-4994-4b74-9fe2-5e8dc3cc8916/scratchpad/`（stories.md / critique.md / survey.md / kvreuse.md，会话结束会清）；本文档为长期存档版。

---

## 相关链接

- 论文骨架（provenance 版，本文的前身）：[paper-skeleton.md](paper-skeleton.md)
- 各 idea 判决：[idea-verdicts-and-standing.md](idea-verdicts-and-standing.md)
- EAGLE-3 特化训练计划（预注册判据）：[eagle3-domain-training-plan.md](../../../eagle-spec-decode/eagle3-domain-training-plan.md)
- 税账本与经济学：[../examine-spec-tax/fixed-tax-conclusions.md](../examine-spec-tax/fixed-tax-conclusions.md)
