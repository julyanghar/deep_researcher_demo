# Paper 骨架:Provenance-Routed Speculative Decoding

> **定位**:承接 [provenance-decoding-from-zero.md](provenance-decoding-from-zero.md) 的"一张图三个角色"叙事,但**重心已按文献综述纠偏**——③(provenance 路由投机解码)+ characterization 主承重,①②作 infra,Plato 单列区分。综述重叠分析见本文 §四;各 idea 判决见 [idea-verdicts-and-standing.md](idea-verdicts-and-standing.md)。
> **数字口径**:`[填]` = 待测/你补;其余为已实测(census 生产口径),来源见各文档。

---

## 一、重心与承重次序(为什么这么排)

综述结论:我们原"三位一体加速框架"的主承重(①KV复用、②并行)被 **Plato(COLM 2025)** 几乎整包吃掉(它一张骨架依赖图同时做并行+KV复用+异构模型);唯一空地是 **③ provenance 路由的 token 级投机解码**——但③恰是我们自己数据说最脆的(接受率被内容天花板锁死)。解法 = **换承重结构**:

| 层 | 内容 | 角色 | 处理 |
|---|---|---|---|
| **主承重** | provenance 路由投机解码(**经济学赢法**:高 provenance 结构化段敢发深注,**不碰"破天花板"**) | ③ | 核心机制贡献 |
| **主承重** | 多 agent deep-research 输出的 **provenance/冗余结构画像**(含诚实负结果) | — | 核心实证贡献 |
| **infra(vs Plato 可区分)** | **位置无关、跨 agent** 的 KV 复用(乱序小结 blend 进报告 prompt) | ① | 引用你 CacheBlend/KVCOMM;差异化点在此 |
| **infra(引用不认领)** | 依赖感知并行分节解码 | ② | 明写站在 Plato/DoT 肩上 |

**为什么这样排能同时躲两个坑**:
- 躲 **desk-reject "This is Plato"**:①②不认领为贡献;而①因"位置无关"vs Plato 仍可区分(§四.2 铁证)。
- 躲 **"用你自己数据打你"**:③框成"敢发深注的经济学"(数据支持:报告命中 59.5%、尾部 p10-12 仍 5-9% > 盈亏线 2.4%、当前"下注太浅"),**绝不主张"换源破 29% 天花板"**(那条被 idea #4 判死)。

---

## 二、Abstract(草稿)

> 多智能体 deep-research 智能体生成的长报告,内容**大量重组自它自己的中间产物**:报告每一节由少数几份 researcher summary 写成,每份 summary 又摘自几页检索结果。我们指出:**现有服务栈在解码时丢掉了这层 provenance 结构,把每个输出 token 当作从零生成。**
> **(实证贡献)** 我们量化了一条多 agent deep-research 流水线输出的 provenance/冗余结构——报告照搬面积 91%、字符级覆盖 ~`[填]`%、但 token 级长段 cov24 仅 ~0.4%、逐段接受率**报告 59.5% / query_plan 39.4% / 小结 26.7%**——揭示一个**尖锐分裂**:结构化段落近乎逐字复用出处,论述段落是模型原创。这个分裂**给一切基于复用的加速划了硬上界**,也解释了为何朴素全语料投机解码停在 ~27% 接受率。
> **(机制贡献)** 据此提出 **provenance-routed speculative decoding**:把每段输出的草稿**限定到它的出处来源**,从而在高 provenance 的结构化段落上**敢发更深的草稿**(单位置经济学 0.62ms/位、盈亏线 2.4% 支持深注)。我们把它与**位置无关的跨 agent KV 复用**(把某份 summary 的 KV 直接复用进报告 prompt,而非重算)和**依赖感知的并行分节解码**组合。
> **(结果)** 在 `[benchmark]` 上取得报告段 2.29×、端到端 `[填]`×,质量不降(kimi-k2.5 盲判)。
> **(诚实负结果)** 论述段落的接受率天花板由内容决定,provenance 路由**破不了**——本文厘清了复用式加速**在哪适用、在哪不适用**。

---

## 三、Intro 五段骨架(每段标落哪个数)

- **P1 设定 + 瓶颈**:deep research = 记者→小结→主编→报告;decode + prefill 两头重。落数:`e2e 墙钟拆分 小结 32.8% / 报告 27.8% / 检索 31.8%`、`报告 prompt ~8.5万字符 ≈ 2万 token`。
- **P2 洞察 + 反派**:输出是**沿 provenance 图重组**的,不是从零生成;**服务栈把这张图丢了**(一句话立反派)。
- **P3 Characterization(贡献1)**:量结构 → **结构化/论述分裂** → 划上界 + 解释朴素投机为何停在 27%。**把负结果写成力量**。落数:接受率 26.7%、匹配 ≥97%、每步草稿均 2.9(66% 仅 1-2 token)、token cov24≈0.4%、报告 59.5% / query_plan 39.4%。
- **P4 Method(贡献2)**:provenance-routed spec decode(经济学:高 provenance 段敢发深注)+ 组合位置无关跨 agent KV 复用 + 依赖感知并行。**明写 infra 站在 Plato / RadixAttention / CacheBlend 肩上,我们加的是 ③ 和跨 agent 位置无关那块**。
- **P5 结果 + 区分预告**:`[填]` + 一句 "unlike Plato (semantic-level parallel, **prefix-only / position-dependent** KV reuse, single-query), we add token-level provenance-routed speculation and position-independent cross-agent reuse."

---

## 四、Related Work:必引 + 必区分

### 4.1 必引必区分表

| 工作 | venue | 撞我们哪 | 必须说的区分点 |
|---|---|---|---|
| **Plato** | COLM 2025 | ①+②(一张图→并行+KV复用+异构模型) | 语义级并行、**前缀/位置相关**KV复用、单 query;我们加 token 级 provenance 投机 + **位置无关跨 agent** 复用(§4.2 铁证) |
| **SoT / SoT-R** | ICLR 2024 | ② + "是否并行"门控 | 全并行伤连贯;我们依赖感知 + 投机是 token 级正交 |
| **APAR / PASTA / Multiverse** | arXiv / ICML'25 / NeurIPS'25 | ② 学习式并行 | 需微调;我们检索式草稿、核心是投机命中而非语义分块 |
| **DoT / GoT** | WWW'25 / AAAI'24 | ② 依赖图/图推理 | 面向推理质量非解码延迟 |
| **RadixAttention / Hydragen** | NeurIPS'24 / ICML'24 | ① 共享**前缀**复用 | 位置相关,只吃逐字相同前缀;引用为 infra |
| **CacheBlend / LMCache / KVCOMM** | 你的旧工作 | ① 位置无关复用机制 | 机制是我们自己的;本文新意=provenance 指定 blend 哪块 + 跨 agent 应用 |
| **AgentWrite / LangChain ODR / Anthropic** | ICLR'25 / blog | 反向当**论据** | 朴素并行伤连贯(-6%)、"write in one-shot" → 佐证需 provenance 感知 |
| **suffix decoding / PLD** | — | ③ 的退化基线 | 全语料查=朴素;我们 provenance 路由是其改进(实验主轴) |

### 4.2 Plato 区分段(草稿,已核实原文)

> **Plato [COLM 2025, arXiv:2402.12280]** 与我们最近:它同样用 LLM 建骨架依赖图做语义感知并行解码,并做 KV 复用与异构模型选择(报告 vs 自回归 +68% 吞吐、KV 复用降开销 75%)。三点关键区别:
> **(i)** Plato 是**语义级并行**,**不做 token 级投机解码**;我们**正交地**在其上加 provenance 路由的投机草稿。
> **(ii)** Plato 的 KV 复用**严格是位置相关的前缀复用**——它显式"leverag[es] the prefix caching capabilities of modern LLM inference engines",把共享内容"plac[ed] ... at the prefix",并把**所有前序节点输出按固定顺序累加成单调增长的前缀**(P→P+A∗→P+A∗+B∗→…)。论文自陈:若只选择性纳入依赖,"**KV cache values depend on tokens' positions ... Since B∗ and C∗ would appear at different positions ... the system would need to perform additional prefill operations, generating entirely new KV caches**"(§4.2.2)——即 Plato **无法重排、无法只取依赖子集**,否则退化为全量重算。我们做**位置无关、跨 agent** 的复用(把早先生成的 summary 的 KV 直接 blend 进报告 prompt,靠 RoPE 偏移 + 选择性重算修正),**正是 Plato 明言做不到的那件事**。
> **(iii)** Plato 在**单条 query 内部**分解;我们的 provenance **跨多个 agent、多次请求**,且报告写作要求 supervisor **按任意顺序取不同 summary 子集**——这从根上违反 Plato 的"固定前缀累加"前提,使其 KV 策略在本设定不适用。

### 4.3 核实证据盒(Plato §4.2.2 原文,备查)

来源:arXiv:2402.12280 PDF §4.2.2 "KV cache Reuse Optimization"(本地抽取 `/home/yilin/tmp/plato.txt` L318-362)。关键原文:

- L324-325:"maximizes KV cache reuse ... by **leveraging the prefix caching capabilities** of modern LLM inference engines"
- L330-331:"By **placing shared contents (1,2,3) at the prefix** of the prompt, we ensure these elements' corresponding KV cache are cached and reused across nodes"
- L338-350:"**using all previous node results ... is more efficient than only including dependent nodes' results**" → 前缀单调累加 P+A∗+B∗+C∗,每输出只 prefill 一次
- **L353-356(铁证)**:"this approach would significantly reduce KV cache reuse, as **KV cache values depend on tokens' positions in the sequence. Since B∗ and C∗ would appear at different positions ..., the system would need to perform additional prefill operations, generating entirely new KV caches.**"

**判读**:Plato 明确**意识到并绕开**位置相关性——靠"永不重排、全量累加固定前缀"来保住前缀复用。代价 = 每个后续节点 prompt 必须扛上全部前序输出(prompt 单调变长)。它**没有** RoPE 偏移 / 选择性重算 / 位置无关复用(通篇无 CacheBlend 类机制)。→ 我们的位置无关跨 agent 复用是**vs Plato 的确凿差异化**(机制credit 归你 CacheBlend/KVCOMM 旧工作,新意在 provenance 选择 + 跨 agent 应用)。

---

## 五、我们的差异化机制清单(vs 每个近邻,一句话)

1. **vs Plato**:+ token 级 provenance 投机;+ 位置无关跨 agent KV 复用(它做不到);跨请求而非单 query。
2. **vs SoT 家族(APAR/PASTA/Multiverse)**:不微调、检索式草稿;核心是投机命中经济学,不是语义分块。
3. **vs RadixAttention/Hydragen**:它们逐字前缀复用,我们乱序/跨请求复用。
4. **vs suffix/PLD**:它们全语料查(=我们朴素基线),我们 provenance 限定草稿源 → 敢发深注。
5. **实证独有**:多 agent deep-research 的 provenance 画像 + 一手负结果(论述段天花板、并行伤连贯),别人没有。

---

## 相关链接

- 从零讲通本 idea(原理/故事):[provenance-decoding-from-zero.md](provenance-decoding-from-zero.md)
- 机制账 / 下注经济学 / 五常数:[proposal-mechanism-acceptance-uplift.md](proposal-mechanism-acceptance-uplift.md)
- idea 判决(含 #4 兄弟增强判死 = ③ 风险来源):[idea-verdicts-and-standing.md](idea-verdicts-and-standing.md)
- copy-speculation 蓝图 / 资产盘点:[research-directions-copy-speculation.md](research-directions-copy-speculation.md)
- decode 每步计算构成:[../examine-spec-tax/decode-step-compute-anatomy.md](../examine-spec-tax/decode-step-compute-anatomy.md)
