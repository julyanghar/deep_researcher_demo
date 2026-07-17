# KV Reuse 的故事：自洽性、创新性检讨与 TTFT 预算账

> **缘起**：M·ΔV 方案中 M（supervisor 决策注意力）没有"上游"可借——supervisor 是第一个消费 summary 的 agent。讨论出的自洽故事是"supervisor 当显著性源（full prefill）、writer 当复用消费者"。用户两问：①这样创新性是否不够？②按已有实验数据，这样做整体 TTFT 优化是否有限？
> **数据**：SW 臂逐调用时延日志 `/home/yilin/tmp/logs/llm_calls_SW.jsonl`（Exp_test_supervisor，40 题，922 次调用，字段含 tag/prompt_tokens/elapsed；`ttft_s` 未填全空——见 §五要补的测量）+ LMCache blend 实测（claude-docx/11）+ Exp_M 80 题（[results_expm_80q_final.md](../Exp_M/results_expm_80q_final.md)）。
> **两句话答案**：**①创新性确实单薄，但有一条升级路（首读全价 + 显著性标注 KV 存储）能把故事从"一条边的技巧"变成"流水线不变量"；②TTFT 优化不只有限，是结构性地小——整个 decode-to-prefill 复用面只占 prefill 计算的 ~9.3%，supervisor 独占其中 2/3 且中位 prompt 低于 blend 盈亏线。TTFT 的真钱在 researcher 侧的检索片段复用（90% 的 prefill 面），不在 summary 复用。**

---

## 一、TTFT 预算账（先算账，账定故事）

### 1.1 一手数据：40 题 SW 臂的 prefill 构成（时间口径，非 token 近似）

`ttft_s` 字段全空，但日志有 `start_ts/end_ts`。方法：先按区间重叠重建每次调用的**并发度**，取 166 条**独跑**调用（无排队污染；含全部 94 次 supervisor 和 28 次 writer——它们本来就串行）做最小二乘拟合：

```
elapsed = −0.054s + 23.77ms×completion_tokens + 0.143ms×prompt_tokens    (R² = 0.999)
```

两个独立交叉验证：decode 系数 23.8ms/token 与五常数税模型的 graph 基线 24.0ms 吻合；固定项 ≈0；二次项无改善（**该长度段 prefill 线性**，≈7K tok/s）。将拟合的 prefill 时间函数应用到全部 922 次调用：

| 调用类型 | 次数 | 次/题 | prompt 中位 | **prefill 时间占比** | **单次 prefill 中位** |
|---|--:|--:|--:|--:|--:|
| RESEARCH_SUMMARY_TEXT（researcher 读检索片段） | 579 | 14.5 | 5,374 | **90.1%** | 0.85s |
| SUPERVISOR_DECISION_JSON（读 summary 决策） | 94 | 2.4 | 1,844 | **6.1%** | **0.26s**（p90 0.80s） |
| FINAL_REPORT_MARKDOWN（writer 读 summary 写报告） | 28 | 0.7 | 3,448 | **3.2%** | 0.49s（p90 1.19s） |
| QUERY_PLAN / INITIAL_QUESTIONS | 221 | 5.5 | ~100 | 0.6% | ~0.02s |

因为 prefill 实测线性、固定项≈0，**时间占比与 token 占比数值一致**——token 近似在本长度段被证实成立。全 run 串行等效：prefill 总计 544s（13.6s/题）vs decode 4,339s（108s/题），prefill 占模型时间 11.3%。

| 调用类型 | 次数 | 次/题 | prompt 中位 | p90 | **prompt token 占比** |
|---|--:|--:|--:|--:|--:|
| RESEARCH_SUMMARY_TEXT（researcher 读检索片段写小结） | 579 | 14.5 | 5,374 | 11,004 | **90.1%** |
| SUPERVISOR_DECISION_JSON（读 summary 做决策） | 94 | 2.4 | **1,844** | 5,574 | **6.1%** |
| FINAL_REPORT_MARKDOWN（writer 读 summary 写报告） | 28 | 0.7 | 3,448 | 8,298 | **3.2%** |
| QUERY_PLAN / INITIAL_QUESTIONS | 221 | 5.5 | ~100 | — | 0.6% |

**decode-to-prefill KV reuse 的全部作用面 = supervisor + writer = 9.3% 的 prefill 计算量**（而 prefill 本身只占 GPU 计算 ~5%）。

### 1.2 三个直接结论

**结论 1：用户的担心被数据证实，而且更狠。** supervisor 占复用面的 2/3（6.1% vs writer 3.2%）。若 supervisor 走 full prefill（v1 故事），复用面只剩 writer 的 ~3.2%。换算成时间：writer 每题 0.7 次 × 单次 prefill 0.49s × blend 大档折扣 ~60% ≈ **每题省 ~0.2s，而 e2e 每题 ~230s——万分之八**；"系统级 TTFT 改善"只发生在每题 ~0.7 次调用上（23 次调用中）。**abstract 里 "cuts TTFT by X%" 若指望 decode-to-prefill 这条线，X 撑不起来。**

**结论 2：更扎心——supervisor 复用在本 demo 形态下本来就多半不该开。** supervisor decide 中位 prompt 仅 1,844 token，**低于 blend 的 4–6K crossover（小档净亏 +32~60%）**；p90 也才 5,574、刚过线。换成时间单位更直观：**supervisor 单次 prefill 中位只有 0.26s**，而 blend 小档的固定装载开销实测就有 ~134ms+——就算复用命中率 100%，单次最多省一百多毫秒，还要冒决策质量风险。即：就算解决了"没有上游 M"，**大多数 decide 调用开 blend 也是亏或近乎白干**。"supervisor 怎么复用"在 TTFT 意义上大半是个伪问题——长度门控已经替我们回答了：多数时候不该开。
（⚠ 口径注意：这是 Exp_test_supervisor 的 40 题非并行 demo；论文目标的多轮 800 题 run 中 supervisor 上下文会随轮次变大，正式下结论前须在目标 run 上重跑同款聚合——见 §五。）

**结论 3：TTFT 的真钱不在这，在 researcher 侧。** researcher 读检索片段占 **90.1%** 的 prefill 计算，其中跨 researcher 的片段逐字重合率 27.6%（snip_contain8）、14.6% 配对共享 URL——**检索片段的跨请求 KV 复用**（CacheBlend 经典场景，不是 decode-to-prefill）的可作用面 ≈ 90% × 15~27% ≈ **13~24% 的 prefill**，比 supervisor+writer 加起来大一倍以上，且这些调用中位 5.4K token、恰在 crossover 之上。**如果论文要在 abstract 主张 TTFT 改善，应该把这块纳入 prefill 侧设计**；它与 M·ΔV 正交（复用的是检索内容而非生成内容，无决策保真问题）。

### 1.3 对故事的直接含义

M·ΔV 的价值主张必须从"省 TTFT"**换轨到"质量保真"**：它是让复用在多智能体流水线里**不伤下游**的机制（decision/report fidelity），不是让系统更快的机制。TTFT 数字由"检索片段复用 + writer 大档复用"去挣，M·ΔV 负责"挣钱时不出质量事故"。这个分工在 abstract 里也自然：反派本来就是"cross-request KV reuse perturbs downstream decisions"——**M·ΔV 是那个反派的解药，解药的 KPI 是质量不是速度。**

---

## 二、v1 故事（显著性源）为什么创新性单薄

v1 = supervisor full prefill 产出决策注意力 M，writer 复用时用 M×ΔV 选位重算。自洽，但薄在三处：

1. **机制骨架是别人的**：选择性重算是 CacheBlend 的（它用 ΔKV 几何信号），我们只换了信号来源（上游注意力）。"换信号"是增量，尤其当——
2. **自家数据显示新信号没赢**：Exp_M 80 题干净对比里 M×ΔV − ΔV 全部不显著（差 ≤0.03、CI 全跨 0）。哪怕是欠功效的 null，审稿人也会问"信号更贵、没更好，为什么要它"。
3. **只作用于一条边**：supervisor→writer 单边单向，一次性。一条边上的技巧撑不起"机制贡献"的段位。

---

## 三、升级路：从"一条边的技巧"到"流水线不变量"

### 3.1 核心升级：首读全价 + 显著性标注的 KV 存储（salience-annotated KV store）

把规则从"supervisor 永远 full prefill"改成一条对**任意内容、任意 agent**成立的不变量：

> **任何内容的第一次被读是全价精确的，并顺手产出该内容的显著性标注；此后所有 re-read 走复用，由累积的显著性标注指导选位修复。**

- KV 缓存条目升级为 **KV + 显著性元数据**：每个消费过这段内容的 agent，把自己读它时的注意力（免费副产品）累积写回缓存条目。
- **supervisor 的"无上游"问题在多轮里自己消失**：轮 1 decide 首读新 summary（全价，产出 M）；轮 2+ decide re-read 旧 summary 时，**用自己轮 1 的 M 指导复用**——不需要"上游"，需要的是"上一次读"。writer 读时已有 supervisor（可能多轮）的累积标注。
- **消费越多、制导越准**：报告修订、追问、多轮迭代都在给同一份内容投显著性票——这是多智能体流水线独有的性质，单模型 serving 结构上不存在。

创新性定位（vs 近邻）：RelayCaching 传 KV 不传语义元数据；KVCOMM 的 anchor 是几何对齐；CacheBlend 的信号是本地 ΔKV；**"跨 agent 累积的注意力作为缓存条目的一等元数据"没有先例**（2026-07-11 调研口径）。且它把 v1 的"supervisor 特殊论"消掉了——没有特殊角色，只有"首读者"和"再读者"，规则对全图统一。

### 3.2 配套升级：复用面扩到检索片段（TTFT 的钱从这来）

researcher 读的检索片段占 90% prefill 面。同一 URL/片段被多个 researcher 读时：首读全价（产出标注），re-read 复用。这把 §一.3 的 13~24% 复用面纳入同一条不变量——**故事统一（还是首读全价），数字变大（TTFT 有的报了）**。检索内容无决策保真之忧，是低风险高覆盖的那一半；生成内容（summary）复用是高风险低覆盖的另一半，由显著性标注护住质量。两半合起来，prefill 侧才是完整设计。

### 3.3 与 decode 侧的统一（论文级叙事）

论文主线是"按内容结构路由"：decode 侧路由信号 = copy rate（这段内容会被怎么写），prefill 侧制导信号 = salience（这段内容曾被怎么读）。**一个是内容的未来消费预测，一个是内容的历史消费记录**——对称、都免费、都来自流水线本身。这句对称性是"两侧是一个系统"在机制层的呼应（税对冲是它在性能层的呼应）。

---

## 四、推荐的故事版本（合并后）

1. **不变量**：first-read-exact——首读全价并产出显著性标注；re-read 复用并按标注修复。
2. **prefill 侧双面**：检索片段复用（大面、低风险、挣 TTFT）+ 生成内容复用（小面、高风险、靠显著性标注保质量）。
3. **M·ΔV 的身份**：显著性标注指导下的选位修复 = "decision-preserving" 的实现件；KPI 是质量无损（40 题盲评已有正证据），TTFT 是顺带。
4. **反派改写**（abstract 已埋好）：cross-request KV reuse perturbs downstream decisions——我们的解不是"不复用"（太贵）也不是"盲复用"（伤质量），而是"首读产证、再读凭证"。

---

## 四b、若坚持 supervisor+writer 两处都复用：创新点的三层写法（2026-07-16 追问回填）

前提切换：不谈 TTFT 占比，就在 supervisor decide 和 writer 做 KV reuse。创新点**不押"复用机制"**（CacheBlend/RelayCaching 的地盘），**押"消费者"**：

**第一层（观察，承重墙）：复用风险是消费者角色依赖的。** 同一批 summary、同一套复用机制，誊写型消费者（writer，输出=忠实转述）鲁棒，决策型消费者（supervisor decide，输出=继续搜什么/停不停）敏感。空地：先前工作把消费者当同质、只报聚合分；没人在同一条流水线内按角色拆分过复用后果。措辞：*quality risk of approximate KV reuse is consumer-dependent: harmless to transcription-type consumers but perturbs decision-type consumers.*
⚠ 证据张力要诚实处理：支持侧=Exp_M supervisor 强制全复用塌上界 4.5 分 + 外部 judge 任务负结果 preprint；反对侧=Exp_test_supervisor SW 臂（只 supervisor 复用）在 **report 指标**上无害（30.3 vs 24.8）。两者口径不同（决策质量 vs 报告质量），立住需干净实验：同 run 分别开/关两类消费者复用 × 两套指标（检索覆盖/答案 F1 vs 报告分）——**该实验同时就是 Exp_M 修信噪比的实验**。

**第二层（机制）：显著性生命周期（salience-annotated KV）。** 修复制导信号随消费历史升温：冷启动=几何信号 ΔKV（CacheBlend 式）；每被读一次，消费者注意力 M 免费写回缓存条目；supervisor 轮 k 用自己前几轮的 M（**自历史——no-upstream-M 的正解**），writer 用累积 M。缓存条目=KV+消费历史元数据，无先例（7/11 调研）。首读用几何信号是生命周期的冷启动阶段，不是缺陷。
⚠ 唯一必须翻案的一层：M×ΔV vs ΔV 目前 null（欠功效）；修信噪比后仍平则降级（M 决定"修多少"而非"修哪里"）或诚实负结果。

**第三层（策略）：决策感知的修复预算。** 复用不是开/关，是每调用的重算预算，按消费者敏感度定价：决策型高预算（旧 14 分支：**75% 选择性重算追平 full-prefill 上界 ~35 分**）、誊写型低预算（writer 复用 40 题无害）。与 decode 侧"copy rate 定草稿深度"对称：decode 按内容的未来消费方式下注，prefill 按消费者身份拨款。最扎实的一层，不依赖 M 赢 ΔV。

**退路阶梯**：M 翻案失败 → 第二层降级，第一+三层照常成立（ΔV 选位 + 角色定预算，创新点仍是 consumer-aware）。**最怕的写法** = "我们用注意力选位重算"这种单点机制主张——正撞 CacheBlend 且被自家 null 顶着。

## 四c、创新点再拔三档：原语/目标函数/抽象（2026-07-16 追问回填；用户判 §四b 三层栈太 trivial）

§四b 的三层是"策略+标注"级——审稿人看到 knob 不是 primitive。顶会级要给新原语、新目标函数或新抽象之一。三档组合（互锁）：

**A. 有证书的近似推理（certified bounded-drift reuse）**。关键再发现：**M·ΔV 不是启发式，是注意力输出漂移的一阶展开 W=Σⱼaⱼ·Δvⱼ**——该量已在两数据集全量观察过（见 [[kvcomm-attn-weighted-reuse-error]] 那轮分析）。升级：①新目标=给定预算 B 选重算集 S 最小化一阶证书 ‖Σ_{j∉S}aⱼΔvⱼ‖，目标可分→按 aⱼ‖Δvⱼ‖ 贪心即最优（有最优性结构的 knapsack）；②新运行时对象=每次复用产出漂移证书，超阈自动升预算/回退全算；③supervisor 冷启动降级为细节：a 用早层探针/上游 M 当先验，证书事后兜底。MLSys 双菜：一阶界+贪心最优性证明（原理）+ 证书驱动自适应（系统）。

**B. 质量控制搬到决策空间（decision-space quality control）**。现有复用工作全在 KV/表征空间控保真；新 stance："控制下游**决策**变没变，不是 KV 像不像"。机制近乎免费：决策 token 的 **logit margin** 在线可得——margin 高放心复用、低则升预算。margin-gated reuse = 复用系统首次有"这次近似伤没伤到事"的自我诊断；与 A 组合（margin=证书的零阶验证）。

**C. 论文伞：speculation 是 recycling 负载的普适原语**。speculate→verify→repair 按经济学定价：decode 侧=草稿/target 前向精确验证/拒收/草稿深度定价；prefill 侧=复用 KV/证书+margin 有界验证/选择性重算/knapsack 定价。全文="一个原语在两侧实例化，同一套成本模型定价"——五常数、下注经济学、M·ΔV、税对冲全收进伞。⚠ 措辞红线：prefill 侧验证非无损，必须写 bounded/calibrated。

**要立的新支柱**：①一阶界推导写严（Taylor 余项、softmax 重归一化——审稿人第一戳）；②证书校准实验（证书大小 vs 实际决策翻转率标定曲线——这是新的生死实验，取代 M×ΔV 对比）；③近邻必查：Cache-to-Cache（C2C 直接 KV 语义通信）、SpecPrefill（名字撞车）、DroidSpeak 的"KV 即通信"话术。

**与 §四b 的关系**：三层栈不作废，降为 C 伞下的策略层（消费者敏感度=证书阈值的先验来源）；"不考虑证据"只免了性能证据，机制地基已有一半（W=Σa×Δv 已测、选位管线能跑、margin 白拿）。

## 五、风险与必须补的测量

| 事项 | 为什么 | 怎么做 |
|---|---|---|
| **M×ΔV 的 null 未翻案** | 上界塌 4.5 分、欠功效（Exp_M §4）；故事再好，机制得先证明比纯 ΔV 强 | 按 Exp_M 建议 A：supervisor 不强制全复用重跑，拉回上界后再测 M×ΔV vs ΔV——**这恰好就是新故事的设定（首读全价），一箭双雕** |
| **`ttft_s` 全空** | 本次账已用"独跑调用回归"补上时间口径（R²=0.999，decode 系数与五常数模型交叉验证吻合），但论文正式评估要报**直测** TTFT | 修 llm.py 计时（流式首 token 打点），或 server 侧记录 prefill 完成时刻 |
| **口径外推** | 1.8K 中位 prompt 是 40 题非并行 demo 的形态 | 在 800 题目标 run 上重跑本文件 §1.1 的聚合（同一脚本一行命令） |
| **检索片段复用的实际命中率** | 27.6%/14.6% 是文本口径，KV 可复用还要 chunk 对齐 | 正向口径测量（原刻画排期里已有此项，扩到 researcher prompt） |
| **显著性写回的工程成本** | 标注要随 KV 存取，怕热路径变慢 | 注意力聚合在 decode 已有计算里旁路提取（Aurora 已示范 serving 侧攒 hidden state 的做法可参照） |

---

## 附：数据出处

- 逐调用日志：`/home/yilin/tmp/logs/llm_calls_SW.jsonl`（922 条，40 题 SW 臂；聚合脚本内联于本文 §一.1，可对 800 题 run 复用）
- blend crossover 与大小档：LMCache `claude-docx/11`（大档 −51~67%，小档 +32~60%，crossover 4–6K）
- 片段/URL 重合：[researcher-redundancy-v2](../researcher-redundancy-v2/redundancy_summary.md)（snip_contain8 27.6%、共享 URL 14.6%）
- M×ΔV null 与上界塌陷：[results_expm_80q_final.md](../Exp_M/results_expm_80q_final.md)
- 质量无害正证据：[Exp_test_supervisor_result.md](../TRASH/Exp_test_supervisor/Exp_test_supervisor_result.md)（SW 30.3 / C 28.2 / B 24.8，kimi-k2.5）
