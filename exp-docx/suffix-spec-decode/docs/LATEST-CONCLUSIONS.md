# suffix 投机解码:最新准确结论(唯一权威,2026-07 定稿)

> **本文档地位**:这是 suffix 投机在本 DR agent 上的**唯一最新结论**。docs/ 下其余文档是**研究演进过程的存档**,含大量已被后续实测**推翻的早期结论**(尤其 7/6 spec-tax 战役之前的预言)。**任何加速比数字,以本文档为准;和本文档冲突的旧文档结论一律作废。**
>
> 立此文档的原因:早期文档把"冷树 replay 的悲观数字"当成了 summary 的最终结论,又留了一批被自己推翻的预言,导致反复误读。本文档用一张"口径地图"根治。

## 一、口径地图(误读的根源:四个正交轴)

任何 suffix 加速比数字,**必须同时钉死这四个轴**才有意义。把不同轴的数字混用,就是之前所有误读的来源:

| 轴 | 取值 | 影响 |
|---|---|---|
| **数据来源** | 隔离 replay(录制 prompt 原样重发) / 真实 e2e(demo 完整跑) | replay 系统性**过度悲观** |
| **树状态** | 冷树(全新 server) / 热树(同题相继焐热) | 但热树对接受率贡献实测≈0(见下) |
| **并发** | batch=1 严格串行 / 真实并发(题内并行,summary 峰值 9 路) | 并发**摊薄固定税**,是 summary 翻正主因之一 |
| **执行模式** | 默认**混合 graph+eager**(交 8.1ms 边界税) / `--enforce-eager` 全 eager(消税) | 换 eager 是 summary 翻正另一主因 |

## 二、summary 段准确画像(最容易被误读)

**冷树 replay·batch=1·默认混合模式**(悲观口径):
- 全 sweep 区间 0.87–0.96x,默认点 **0.955x**(v1)、**0.81x**(v2)。
- 内部双峰:token harvest >20% 的题中位 **1.07x**(检索型赚)、≤20% 的中位 **0.86x**(分析型亏),打平点 ~15–20%。

**真实 e2e·热树·并发·Ceager(enforce-eager)**(生产口径,以此为准):
- summary 段 **x1.00–1.07**(L0-V2 x1.00、40q 逐题中位 1.04x/聚合 1.07x,25胜13平2负)。**不亏,打平到轻赚。**
- 双峰大幅收窄:分析型题从冷树"重亏 0.71x"软化成 e2e"轻亏,最差 0.95x"。

**⚠️ 作废的旧结论**:"summary 净亏/拖后腿/双峰打平附近"——那是**冷树 replay 口径**,真实 e2e 已翻正。评估 summary **必须端到端测**,replay 数只能看形状不能当结论。

## 三、report 段准确画像

- 冷树 replay:**1.76x**(关全局树,最贴生产)/1.84x(带预热);v2 **1.87x**。
- 真实 e2e:更猛,**x2.14**(L0-V2)、单调用 tok/s 从 41→96 翻倍(40q)。
- 逐请求命中率(e2e 40q):**61.2%**(vs summary 28.8%)。
- 机制:report 大段照抄自己 prompt 里的 summary(单段中位 226 字),suffix 一次命中整段。
- **余粮**:`max_spec_factor=4` 是双涨甜点(report replay +18.6%),但当前生产 config 还是 factor=1,report 还有一截没吃进生产。

## 四、整体 e2e(以此判断"值不值得开")

- **全开 suffix(Ceager)**:e2e 墙钟 **−29%**(40q A→B −29.6%;L0-V2 −29%)。report 段砍半主导,summary 段 −9%(也是省的,不是亏)。
- **叠加 report 分节**(并行写报告):再 −25.4%,两步合计 **−47.5%(近腰斩)**,RACE 质量无损(Overall +0.004,噪声内)。收益来自**并发**不是命中(命中反从 61%→46%,墙钟却更低)。
- **值得开**:净收益看长报告占比,report 是 decode 大头,对整个 pipeline 划算。

## 五、关键口径事实(反复搞错的)

1. **suffix 不是硬约束 enforce-eager**——它因草稿长度每步变化过不了 uniform-decode 检查,被判进 **PIECEWISE 混合模式**(FFN 在 graph、注意力 eager),用不了 FULL cudagraph;但默认是**混合 graph+eager**,不是纯 eager。`--enforce-eager` 是 7/6 后**人为加的优化**,消掉混合模式独有的 8.1ms/步边界税(主体是驱动锁等待 4.5ms)。换 eager 后 replay summary 从 0.83x→1.00x、report→2.29x。
2. **baseline vanilla = FULL cudagraph**。
3. **五常数税模型**:每步 = graph 24.0 + eager 0.5 + spec管线 1.9 + 混合边界税 8.1 + 草稿费(激活 5.8 + 0.62/位置)。eager 相对 cudagraph 只贵 +0.5ms/步(极小),所以 e2e −29% 压倒性是投机效应,不是执行模式混淆。
4. **理想上界**(零税零成本模拟):summary 也只有 **1.33x**、report **2.62x**——"summary 调参救不了"是机制上限,不是配置问题。

## 六、已作废的早期结论(读旧文档时警惕)

| 作废结论 | 出处(旧) | 被谁推翻 |
|---|---|---|
| "真实并发下 suffix 收益缩水甚至转负" | optimization-proposal.md:36(7/2前预言) | 同文档:94 自我修订 + l0-l3-v2-results.md 实测(题内 9 路并发,summary x1.00 没转负) |
| "e2e 翻身靠热树/兄弟树抬高接受率(26%→42%)" | 早期叙事 | l0-l3-v2-results.md:45:干净轨迹口径 e2e summary 接受率仅 28.3%≈冷树,热树贡献 Δ+0.3pp≈0。真因=并发摊薄税+换eager消边界税 |
| "summary 净亏/双峰打平附近"(当最终结论) | summary.md:10 冷树数 | 真实 e2e summary x1.00–1.07(已翻正) |
| "suffix 硬约束/永远 enforce-eager" | (口头理解) | mixed-mode-tax-explained.md:14:实为默认混合模式,eager 是后加优化 |
| 字符级 Coverage@8 预测 summary 加速 | v2 头条指标 | variance.md:36:字符 cov 对 summary r=−0.23(负相关!),只有 token harvest 有预测力(r=0.55) |

## 七、最新实测文档索引(可信,按主题)

- **e2e 三配置 40 题**(投机 + report 分节,−47.5%):[e2e-3config-40q.md](e2e-3config-40q.md)
- **L0-L3 v2 端到端**(summary x1.00 翻正、热树祛魅、L3 否决):[l0-l3-v2-results.md](l0-l3-v2-results.md)
- **五常数税模型 / enforce-eager 真相**:[examine-spec-tax/suffix-vs-native-cost-anatomy.md](examine-spec-tax/suffix-vs-native-cost-anatomy.md)、[examine-spec-tax/mixed-mode-tax-explained.md](examine-spec-tax/mixed-mode-tax-explained.md)、[examine-spec-tax/replay-vs-e2e-reversal.md](examine-spec-tax/replay-vs-e2e-reversal.md)
- **单请求方差 / summary 双峰 / token harvest 预测**:[explore-idea/data-source/per-request-speedup-variance.md](explore-idea/data-source/per-request-speedup-variance.md)
- **理想上界模拟**:[spec-decode/sim-suffix-explained.md](spec-decode/sim-suffix-explained.md)

## 八、与 phase2(EAGLE 域训)的关系

phase2(exp-docx/paper-submission)是**另一套实验**:EAGLE-3 域训 vs suffix,评测**全是 summary 段单独 replay**(冷树口径)。那里 suffix DRGym summary 0.976× / DRBench 1.16× 是 **summary 段 replay** 数,和本文档的 e2e 口径不可直接比。EAGLE 各臂用 cudagraph、suffix 单独用 enforce-eager——**口径见 [phase2 汇总](../../eagle-spec-decode/phase2-results-summary.md)**。router 的定位:summary 段换 EAGLE(补 suffix 冷树短板)、report/其余保持 suffix(它的强项),真实 e2e 净账正在测(exp-docx 外 modify-code-runs/e2e-drgym-router)。
