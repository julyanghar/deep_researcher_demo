# EAGLE / MTP 零基础讲解:不查表、听"脑电波"的猜题人

> 缘起:suffix 投机解码的根因分析钉死了它的物种极限——它是"只会背这份 prompt 的复读机",summary 里 81% 的步模型在说**自己的话**,书里没有原文,复读机干瞪眼纯交税(见 [why-high-overlap-no-speedup.md](suffix-spec-decode/docs/why-high-overlap-no-speedup.md))。优化第二轮把 **EAGLE3 混合**列为最对症方案(见 [optimization-round2-analysis.md §3.7](suffix-spec-decode/docs/optimization-round2-analysis.md))。本文零基础讲清:EAGLE / MTP 是什么、为什么它们能猜中"模型自己的话"、代价是什么、和 suffix 怎么互补。
> **读法**:§一~§四 顺序读(每节只引入一个想法);§五 对照表可直接跳看;§六 是落地清单。投机解码本身的"猜+验"框架见 [18 号文档](18-vllm-suffix-decoding.md),此处只用一句话带过。

## TL;DR

**suffix 的猜题人是一本字典(prompt 原文),EAGLE 的猜题人是一个学徒小网络——它不读字典,它把听诊器贴在大模型脑门上,读大模型每一步的内部状态(hidden state)来猜接下来几个词。** 所以模型说自己的话时它照样能猜中(典型每步 3~5 个),这正是 suffix 结构性够不着的那 81%。MTP 则是"模型出厂自带的多预测头"(DeepSeek-V3 那种),思路同源、集成更深,但要基座预训练时就长好——Qwen3 没有,所以我们能走的是 EAGLE 后装。

---

## 一、回顾一句话 + 本文唯一的问题

投机解码 = 先猜一串草稿,让大模型一次前向把整串验证,对的全收——**验证一串和生成一个,成本几乎一样**(都只读一遍权重)。这个框架里,猜题人可以是任何东西,唯一的要求:**猜得快(比大模型走一步便宜得多)、猜得准(不然白验证)**。

suffix 的猜题人是查表:"输出结尾这串,上次在 prompt 里出现时后面跟什么"。它的知识 = 这份 prompt。本文的问题:**有没有一种猜题人,知识 = 大模型本身的说话习惯,从而在"模型说自己的话"时也能猜中?**

## 二、先看笨办法为什么不行:直接找个小模型来猜

最直观的想法:找个同系列的小模型(比如 0.5B)当猜题人——它也会说话,猜"模型自己的话"应该行?这就是最早的 draft model 投机(vLLM 的 `method="draft_model"`)。实践里它不够好,三个原因:

1. **口径不齐**:小模型是独立训练的,和 32B 的"说话习惯"对不上——32B 想说"综上所述",0.5B 猜"总的来说",意思对了但 token 不对,验证照样拒。接受率上不去。
2. **它得从头读题**:小模型自己也要跑一遍完整的 prompt(prefill)、自己维护一份 KV cache——几万 token 的 prompt,这笔开销不小。
3. **它不便宜**:0.5B 每猜一个 token 也要跑一次完整前向。

问题出在:小模型是个**外人**,它对这道题的全部理解要靠自己从零建立。

## 三、EAGLE 的核心想法:别从零猜,大模型的"脑电波"里已经有答案

关键观察:大模型每生成一个 token,它最后一层的 **hidden state**(一个几千维的向量)是它对"当前局面 + 接下来想说什么"的完整浓缩——**下一个词的信息本来就在里面**(logits 就是拿它乘个矩阵得到的)。

EAGLE 的做法:训练一个**很小的头**(约一层 transformer 的量级,几亿参数),它的输入不是白纸,而是:

> **大模型当前步的 hidden state + 已经出的 token** → 猜下一个 token(以及下一步的 hidden state,再喂回自己,连猜多步)

打个比方:draft model 是"隔壁班同学替你答题"(他得自己重新读题);EAGLE 是"**贴着你脑门的听诊器学徒**"——你(32B)读题、思考都做完了,学徒只负责从你的脑电波里把你**接下来要说的几个词**提前念出来。它不需要理解题目,只需要学会"从这个人的脑电波解码他的下一句话"——这个任务比"理解上下文"容易几个数量级,所以一个小头就够,而且因为读的是**这个模型自己的**内部状态,口径天然对齐。

**为什么准**:学徒的训练数据就是"大模型的 hidden state → 大模型实际说的下一个词",专门拟合这一个模型的习惯。典型效果(温度 0):**每步接受 3~5 个 token**——不管模型是在抄原文还是在自由发挥。对照我们的场景:suffix 在 summary 的零接受步(81%)上颗粒无收,EAGLE 在这些步上照样 3~5 个。

**EAGLE 还自带"并行多候选"**:它一次猜的不是一条链,而是一棵**token 树**(每个位置留几个备选分支),用 tree attention 一次前向验证整棵树——我们在 suffix 里讨论过、但 vLLM 没给 suffix 接线的"并行验证",在 EAGLE 这里是标配。版本演进一句话:EAGLE-1 固定树;EAGLE-2 按置信度动态长树;**EAGLE-3** 改进训练方式(训练时直接模拟多步推理)+ 融合多层特征,接受长度进一步提高——现在社区默认推荐 EAGLE-3。

## 四、MTP 是什么:出厂自带的多预测头

**MTP(Multi-Token Prediction)**是同一思想的"原生版":模型在**预训练时**就额外长了几个预测头,一次前向不但预测 t+1,还顺带预测 t+2、t+3……(DeepSeek-V3 就带这个)。推理时,这些头的输出天然就是草稿。

和 EAGLE 的区别一句话:**EAGLE 是后装的外挂**(任何现成模型都能加训一个头),**MTP 是出厂配置**(要基座预训练时就设计进去,后装不了)。效果上 MTP 集成更深、开销更小;但 Qwen3 没有原生 MTP,所以对我们:**只能走 EAGLE 路线**(找社区训好的 Qwen3 EAGLE3 头,或用 SpecForge/AngelSlim 这类工具链自己训)。

## 五、EAGLE vs suffix:不是替代,是互补

| | suffix(现状) | EAGLE3 |
|---|---|---|
| 猜题人的知识 | **这份 prompt 的原文**(+历史输出) | **这个模型的说话习惯**(训练学来) |
| 擅长 | 逐字照搬:一口能吞 20+ token(整段掏) | 任意文本:稳定每步 3~5 个,**转述/自由发挥也行** |
| 死穴 | 模型说自己的话 → 颗粒无收(81% 的 summary 步) | 超长逐字照抄时吃不到 suffix 那种大口 |
| 猜的成本 | CPU 查表,GPU 零开销 | GPU 前向内多跑一个小头(便宜但非零)+ 显存放一个小网络 |
| 异步调度 | **被迫关闭**(草稿是采样后 CPU 产物)→ ~18% 税的主要来源 | **兼容**(草稿产在 GPU 前向内部)→ 这半边税直接拿回来 |
| 并行多候选 | C++ 有但 vLLM 没接线 | 树草稿 + tree attention 是标配 |
| 准备成本 | 零(开箱即用) | 需要训好的头(核实社区权重,没有就得训) |
| 我们场景预期 | report 1.76x / summary ~1.0x(端到端) | summary 有望真正翻正;report 未必比 suffix 强(抄长段时 3~5 个/步 < suffix 的整段掏) |

最后一行就是"互补"的含义,也是**按阶段分池**方案的依据:report 池继续 suffix(零成本、已验证),summary 池换 EAGLE(吃转述步)。vLLM 里投机方法是 server 级配置(`speculative-config`),一台 server 只能选一种——分池天然绕开这个限制。

## 六、落地清单(按顺序)

1. **核实权重**:找 Qwen3-32B 的 EAGLE3 head(SpecForge/AngelSlim 等社区工具链有 Qwen3 系列的发布,32B 是否在列需要核实;没有就要用这些工具链自训,成本约几天单机)。
2. **replay 对比**:同 80 条 workload、冷启动、batch=1,`speculative-config: {"method":"eagle3", ...}` vs suffix vs 不开——直接复用现有 replay 脚本和口径(长度归一,避开三大测量坑)。
3. **端到端**:v2 同款多轮画像负载跑一遍(suffix 的端到端优势里有全局树跨请求命中,EAGLE 没有这块,必须端到端才公平)。
4. **决策**:若 summary 段 EAGLE 明显翻正而 report 段不输太多 → 评估分池 vs 全换的运维成本。

## 术语表

| 术语 | 一句话 |
|---|---|
| draft model | 独立小模型当猜题人(最早的投机方案,口径不齐、要重读题) |
| hidden state | 大模型每步最后一层的内部向量,"接下来想说什么"的浓缩 |
| EAGLE | 后装小头,吃大模型 hidden state 连猜多步;-2 动态树,-3 改进训练+多层特征 |
| MTP | 预训练时自带的多 token 预测头(DeepSeek-V3);Qwen3 无 |
| token 树 / tree attention | 一次猜多分支、一次前向验证整棵树 |
| 接受长度 | 一步验证后平均收下几个草稿 token(EAGLE3 典型 3~5) |

## 相关

- 为什么需要它(suffix 的 81% 天花板):[why-high-overlap-no-speedup.md](suffix-spec-decode/docs/why-high-overlap-no-speedup.md)
- 方案定位与优先级:[optimization-round2-analysis.md](suffix-spec-decode/docs/optimization-round2-analysis.md) §3.7、[spec-optimization-six-questions.md](suffix-spec-decode/docs/spec-optimization-six-questions.md)
- 投机解码框架与 suffix 全链路:[18-vllm-suffix-decoding.md](18-vllm-suffix-decoding.md);后缀树本体:[19-suffix-tree-explained.md](19-suffix-tree-explained.md)
