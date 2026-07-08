# 周报:suffix 投机解码 × deep research agent(截至 2026-07-02)

> 一周工作线:**照搬现象统计 → 隔离 replay 实测 → token 级机制分析(模拟器)→ 同题冗余分析 → v2 端到端上线判决 → 优化路线评估**。本文是导师汇报版总纲,每节末尾挂详细文档。

---

## 0. 一页结论

1. **部署判决(已可上线)**:demo 全局开 suffix 投机解码、默认参数,v2 端到端 10 题双臂对照:**整题墙钟 −29%(57.6→41.2 min/题)**,report 段 x2.14、summary 段 x1.00 不亏、QUERY_PLAN 白赚 x1.26。
2. **照搬现象坐实**:1912 条生产调用统计,report 字符级 Coverage@8 均值 **91%**,summary **54%**——但换成机制真正使用的 **token 口径**后差距拉开一个量级:≥8 token 连续照搬 report 65.6% vs summary **12.9%**,最长连续段中位 73.5 vs **14 个 token**——同是"重叠大",形状是整页复印 vs 碎纸屑。
3. **决定加速的是"token 级连续可收割占比"(harvest),不是重叠面积、不是任务类型**:80 条逐条实证,harvest 与加速比相关 r=ρ=**0.90**,打平点 ~15-20%;字符级 Coverage@8 无预测力甚至组内反向(−0.23,其中一半是语言混淆——它与"每 token 字符数"相关 +0.88,几乎在测语言)。
4. **summary 冷树口径略亏的根因已机制级钉死**(逐 token 模拟器):就算验证零成本,summary 的重叠形状理想上界也只有 **1.33x**,交完 ~18% 固定税天花板 ≈1.09x——**调参救不了**(6 配置 sweep 证实,默认即最优)。
5. **端到端为什么翻盘**:同题 18 条 summary 的**输出**彼此逐字重合(union 覆盖 31.7%,比被自己 prompt 覆盖还高),相继流过同一 server 喂饱全局树,接受率 26.4%→42.5%。**方法论结论:评估投机解码对 agent 负载的价值必须端到端测**——隔离 replay 低估一个量级(每题 +3.5% vs 墙钟 −29%)。
6. **优化路线已评估分级**:引用共设计(L3)实测否决(吞吐 +49% 但输出膨胀 +72% 吃光倒贴);多候选并行验证经两层 oracle 实验证实无肉(score 单候选已拿到 97-100%;树状投机的无限宽上界也仅 1.33→1.50x);下一步最值得的是 proposer 小改包与 EAGLE3 混合对比。

---

## 1. 背景与研究问题

deep research agent 的两类重负载调用——RESEARCH_SUMMARY(每题 ~18 次)与 FINAL_REPORT(每题 1 次,decode 大头)——输出疑似大量照搬输入 prompt。由此三个递进问题:**①到底抄多少、怎么抄?②开 vLLM 内置 suffix 投机解码能快多少?③为什么快/不快,怎么部署?**

suffix 投机解码原理一句话:把 prompt(和历史输出)建成后缀树,拿输出结尾去树里查"上次这串后面跟什么",把后续整段当草稿、目标模型一次前向验证——猜中省 decode 步数,无损(输出逐 token 一致)。机制详解:[18 号文档](../../18-vllm-suffix-decoding.md)(调用链+参数+手算例)、[19 号教程](../../19-suffix-tree-explained.md)(后缀树零基础)。

## 2. 本周完成的实验/分析矩阵

| # | 工作 | 规模/口径 | 回答什么 | 产出 |
|---|---|---|---|---|
| A | 照搬现象统计 | 1912 条生产调用,字符级 SAM | 抄多少、什么形状 | [prompt-overlap-analysis-v2](../../prompt-overlap-analysis-v2/summary.md) |
| B | 隔离 replay 主实验 | 80 条真实 prompt,冷树/batch=1/串行,6 配置 sweep | 开 suffix 快多少、参数敏感性 | [summary.md](summary.md) |
| C | 重叠↔加速逐条实证 | 同 80 条,token 级指标 + 配对加速比 | 什么指标预测加速 | [overlap-replay.md](overlap-replay.md)、[per-request-speedup-variance.md](per-request-speedup-variance.md) |
| D | 机制模拟器 | 80 条逐 token 重放(理想上界+trace) | 为什么快/不快、上界多高 | [why-high-overlap-no-speedup.md](why-high-overlap-no-speedup.md)、[sim-suffix-explained.md](sim-suffix-explained.md) |
| E | 同题冗余分析 | v2 每题 18 条 summary 的输入/输出两两比对 | 同题请求间有没有互抄的料 | [redundancy_summary.md](../../researcher-redundancy-v2/redundancy_summary.md) |
| F | v2 端到端判决(L0/L3) | 10 题 × 2 配置完整跑 + 引用共设计第三臂 | 真实负载赚不赚、prompt 共设计行不行 | [l0-l3-v2-results.md](l0-l3-v2-results.md) |
| G | 优化路线两轮评估 | L0~L4 + 六问 + oracle 实验 | 下一步做什么、不做什么 | [optimization-round2-analysis.md](optimization-round2-analysis.md)、[spec-optimization-six-questions.md](spec-optimization-six-questions.md) |

## 3. 发现一:照搬面积大,但两类调用的"形状"截然不同——且必须用 token 尺子量(A+C)

**字符级第一眼**(1912 条生产调用,分析 A):

| | SUMMARY(n=1812)| REPORT(n=100)|
|---|--:|--:|
| Coverage@8(≥8 字逐字片段覆盖率)| 均值 53.8% / 中位 62.4% | 均值 **90.7%** / 中位 93.1% |
| 阈值提到 @30 后 | **崩到 9.5%** | 仍有 **74.5%** |
| 最长逐字段(中位)| **41 字** | **226 字** |
| 照搬段数(均值)| 38 段小碎片 | 170 段长段 |

但字符只是"第一眼"——**suffix 的匹配、验证、收益单位全是 token,真正有影响的是 token 级连续重叠**;字符口径还会系统性失真(它与"每 token 字符数"相关 +0.88,几乎在测语言,见 §5)。换成 token 尺子(80 条 replay 子集,Qwen tokenizer;与录制口径的一致性已验证:字符 cov8 54.6% vs 54.5%):

| | SUMMARY | REPORT |
|---|--:|--:|
| token Coverage@8(≥8 token 连续照搬)| **12.9%** | **65.6%** |
| token Coverage@24 | **0.4%** | **44.7%** |
| 最长连续照搬段(token,中位)| **14** | **73.5** |
| harvest(锁定后可白捡 token 占比,中位)| **14.3%** | **62.6%** |

关键读法:**summary 字符级"54% 重叠"换到 token 口径只剩 12.9%,≥24 token 几乎归零——字符面积把 suffix 能吃到的肉高估了约一个量级**。形状差异的本质是 token 级的:summary 最长连续段中位仅 14 个 token(锁定费 ~3 个就吃掉两成),report 是 73.5 个;summary 的 harvest 中位 14.3%,正好落在打平线(15-20%)之下。**这组 token 形状数字预言了后面所有结果**——字符口径为什么不可用,§5 有专门拆解。

## 4. 发现二:隔离 replay——report 稳赚,summary 冷树略亏且调参无用(B)

全新 server 冷树、batch=1 串行、长度归一 char/s 口径:

| 任务 | 加速 | 备注 |
|---|--:|---|
| FINAL_REPORT(20)| **x1.76~1.84** | 对参数不敏感;关全局树仅 1.84→1.76(它靠 prompt 树就够)|
| RESEARCH_SUMMARY(60)| **x0.955** | 扫了 prob∈{0.05,0.1,0.3,0.5}、factor=0.5、关全局树共 6 配置,**无一翻正,默认即最优** |

两个重要副产品:①**接受率≠加速**(关全局树接受率 50% 反而 0.872x——省时间靠接受总量不靠命中率);②**固定税 ≈18%**:开 suffix 即关 async scheduling + 每步查树,几乎无照搬的请求恒 ~0.81x,与命中无关、调参消不掉。

## 5. 发现三:机制级解释——决定加速的是 harvest,不是重叠面积(C+D)

**指标**:harvest = "前面已连续照搬 ≥3 token、且下一个还在延续"的位置占比(锁定后可白捡的 token,静态可算,与模拟器实际接受相关 0.99)。

**逐条实证(80 条,长度归一)**:harvest 与加速比 Pearson=Spearman=**0.90**,剂量-反应单调(harvest 4%→0.79x,…,71%→2.17x),**打平点 ~15-20%**;summary 的 harvest 中位仅 14.3%,在打平线之下——**≤20% 的 42 条中位 0.86x,>20% 的 18 条中位 1.07x,过线就翻正**;harvest 44~52% 的 summary(检索/罗列型)跑出 1.4~1.5x,比最低的 report 还快——**决定因素是 harvest 不是任务标签**。

**反直觉发现**:字符级 Coverage@8 对 summary 加速是 **−0.23 的反向相关**。拆出两个原因(同一根源:**门槛单位错位**):
①**字符的"8"多半落在锁定费之下**。一段连续照搬的价值 ≈ max(0, 段长 − 锁定费 ~3 token):8~30 字符的片段在英文里只有 2~6 个 token,大量短于锁定费、价值趋零——**注意并非"碎片一律无价值"**:token 口径下**各阈值**的连续重叠都与加速显著正相关(tok cov8:summary +0.56 / report +0.89 / 合并 0.90,tok cov4 也有 +0.51/0.85),因为 ≥8 token(≈30+ 字符)的段早已越过锁定费、每段净赚 ~5 个以上。失真的是字符口径的度量,不是"短段"这个概念本身。
②**语言混淆**——字符 cov8 与"每 token 字符数"相关 +0.88(英文长术语轻松凑 8 字符;中文 8 字 = 4~6 token,同样字符数价值高得多——正是①的门槛错位随语言变化),它几乎在测语言;控制语言后,中文半区 cov8 与加速恢复 +0.42 正相关,harvest 则跨语言稳健。

**机制根因(逐 token 模拟器,80 条全量重放)**:suffix 每段照搬要交"锁定费"(~3 token 对上暗号才能出草稿)、草稿预算=已匹配长度(雪球要滚几步)、微改写当场斩断草稿。碎片式照搬每段都交锁定费、雪球滚不大。最干净的对照:**i53 与 i61 字符级 cov8 几乎相同(85.9% vs 85.0%),harvest 9.5% vs 71.1%,实测 0.78x vs 1.88x**。全量校准:summary 就算验证零成本,理想上界也只有 **1.33x**(report 2.62x),交完 18% 税天花板 ≈1.09x——"调参救不了"钉死在机制上限,与实测 0.955x 自洽。

## 6. 发现四:同题冗余——输入不共享,输出高度重合(E)

v2 每题 18 条 summary 两两比对:**输入**几乎不共享(来源 URL Jaccard 中位 **0**);但**输出**逐字重合显著——两两 22.9%(同 researcher 34.6%),每条 summary 被"同题更早生成的 summary"逐字覆盖(union)**31.7%,比被自己 prompt 覆盖(27.8%)还高**。原因:不同来源讲同一主题,产出相同实体/数字/术语/句式。**这为端到端翻盘埋下伏笔:suffix 的全局树缓存的正是输出。**

## 7. 发现五(最重要):v2 端到端判决——全开净赚,评估必须端到端(F)

三个实验递进(E1 冷树 replay → E2 端到端 → E3 引用共设计):

| | E1:v2 replay(冷树)| E2:端到端(10 题×2 臂)|
|---|--:|--:|
| SUMMARY | x0.81 | **x1.00(不亏)** |
| REPORT | x1.87 | **x2.14**(输出归一 x1.94)|
| 每题净效果 | 仅 +3.5% | **墙钟 x1.40(−29%)**,decode 总量 x1.16 |

**翻盘机制(证据链)**:E2 里同题 18 条 summary 相继流过同一 server,发现四的"输出间重合"喂饱全局树 → summary 段接受率 **26.4%→42.5%**(report 段两臂一致,59%/56%,证明变化只来自树内容);墙钟收益(1.40)大于 decode 收益(1.16)是因为 report 是每题唯一的串行段,省的 73s/题 1:1 落墙钟。

**E3(L3 引用共设计)否决**:prompt 要求 summary 逐字引用,机制确实生效(summary 吞吐 x1.49),但输出膨胀 +72% 把收益吃光倒贴(decode 总时间 +15%),质量 pairwise 平手,还有 1/10 题因膨胀撑爆 report 上下文。

**方法论结论(愿意单独强调)**:隔离 replay 丢失了 agent 负载的内部结构(同题输出相继生成、彼此重合),对本负载**低估一个量级**(+3.5% vs −29%)。**评估投机解码对 agent 工作负载的价值,必须端到端测。**

## 8. 发现六:候选选择算法已近最优——并行投机收益有限,方向被排除(G)

一个自然的优化猜想:proposer 每步会枚举多份候选草稿(各 match_len 一份、本地/全局树各一份),但只把 score 最高的一份交给验证——**只验一份是不是浪费?能不能并行验证多份候选(树状投机),命中哪份用哪份?** 我们用模拟器装"事后诸葛亮"(oracle)把这个方向的上界量了出来,两层:

| 实验 | 做法 | 结果 |
|---|---|---|
| **选择-oracle** | 每步把全部 match_len 候选**都**拿去验证,取命中最好的 | **score 单候选已拿到 97~100% 的可拿收益**:i53 仅 95→103 个 token(+1~2pp),被更优候选打败的步只有 0~3%(i10 为 0)|
| **树-oracle** | 分叉处所有支路无限宽全验证(树状投机的理论天花板:验证无损,任意有限宽度 k 的命中集合都是无限宽的子集——真实续写只要在树中任一路径就必然命中,"选错支路"的损失被修到零;仍保留预算 m×factor)| summary 理想上界仅 **1.33→1.50x**(步数多省 11.3%);i61 3.32→3.67x |

**为什么无肉**:summary 零接受步(i53 占 81%)的本质是"模型在说自己的话"——prompt 树里**根本不存在**正确续写,**所有候选都错,不是选错了候选**。而且现有算法的目标函数本身就对:score = Σ 累积置信度,数学上恰好是"期望接受 token 数"的估计,max score 本来就是期望收益最大化。树-oracle 那 0.17x 的名义空间,再叠上真实树宽(2~3 路而非无限)、18% 固定税、并发下草稿 token 抢算力三笔折扣,**配不上所需的引擎级改造**(vLLM 需先支持 tree attention 验证)。

**意义**:干净地排除了一整类看似诱人的算法优化方向(改选择算法、并行/树状投机)。剩余的优化空间只在两处——**换草稿来源**(EAGLE3 等学习型 drafter,不依赖照搬,直接打"模型不在抄"的根因)和**降固定税**——这正是 §10 优先级表的依据。详见 [optimization-round2-analysis.md](optimization-round2-analysis.md)(上界阶梯与 oracle 方法)、[spec-optimization-six-questions.md](spec-optimization-six-questions.md)(六个优化思路的完整判定)。

## 9. 测量方法论:踩过并修掉的四个坑

1. **热树污染**:全局树跨请求记忆,背过题的 server 重测同类请求假显 1.9x(真实 0.917x)→ 配置对比必须全新 server 冷树;
2. **并发缩水**:投机收益 batch=1 最大(闲算力白捡验证),并发大了草稿与请求抢算力 → 单机口径是上界;
3. **流式 delta ≠ token**:spec decode 一步多 token 打包成一个流式块,按块数算 tps 假低 → 改用配对聚合 char/s;
4. **长度归一勘误**:temp0 两次运行输出长度不同(中位差 7%、P90 33%),decode 时间直接比会把"这次恰好写长"算进快慢——单条引用必须长度归一(极端例 i23:直接比 0.34x vs 归一 2.56x);桶级结论两口径一致。

## 10. 部署建议与下一步

**部署(已定)**:全局开 suffix、默认参数;**别关全局树**(端到端收益核心来源之一);更高并发档位上线前小流量验一遍。

**下一步优先级**(两轮优化评估的收口,含已排除项):

| 优先 | 动作 | 依据 | 成本 |
|---|---|---|---|
| 1 | proposer 小改包:draft 级 score 地板、全局树 margin、propose 增量化、max_tree_depth 扫描 | 修"垃圾草稿挤位"+降 18% 税的 CPU 半边;全是 arctic 侧小改可逐项 A/B | 小 |
| 2 | EAGLE3 混合对比 | summary 81% 零接受步是查表法结构性够不着的(树-oracle 上界 1.50x 也翻不过),学习型 drafter 不依赖照搬——唯一直接打根因的方案 | 中(核实 Qwen3-32B 权重+一天实验)|
| 3 | 题级路由/双池 | harvest r=0.90 可预测,但上界 12pp 且端到端下 summary 已不亏 | 中 |
| ✗ 排除 | 多候选并行验证 | oracle 实验:score 单候选已拿到 97-100% 可拿收益(被打败步数 2-3%) | — |
| ✗ 排除 | L3 引用共设计(当前形态)、summary 调参 | E3 实测否决;sweep 证明 0.1 即最优 | — |

## 11. 产物索引

- **文档**(本目录):[summary.md](summary.md)(replay 主实验)/ [overlap-replay.md](overlap-replay.md) / [per-request-speedup-variance.md](per-request-speedup-variance.md) / [why-high-overlap-no-speedup.md](why-high-overlap-no-speedup.md)(机制+80 条全量表)/ [l0-l3-v2-results.md](l0-l3-v2-results.md)(端到端判决)/ [optimization-round2-analysis.md](optimization-round2-analysis.md) + [spec-optimization-six-questions.md](spec-optimization-six-questions.md)(优化评估)/ [sim-suffix-explained.md](sim-suffix-explained.md)(模拟器教程)/ [questions-log.md](questions-log.md)(Q1~Q22 全程问答索引)
- **机制文档**:[18-vllm-suffix-decoding.md](../../18-vllm-suffix-decoding.md)、[19-suffix-tree-explained.md](../../19-suffix-tree-explained.md)
- **数据**:[../data/](../data/)(workload/res_*/per_call_overlap*.csv);**脚本与物证**:`~/modify-code-runs/suffix-spec-decode/`(replay/端到端日志、mechanism-sim 模拟器与 trace)
