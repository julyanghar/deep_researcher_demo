# blend-router-e2e 最终实验报告(DRBench 100 + DRGym 200 + DRQA 200,TP4)

> 2026-07-21 完结。方案:[experiment-plan.md](experiment-plan.md);分阶段/语言归因原始数据:/home/yilin/tmp/blend_e2e/{final100.txt,stage_analysis.txt,gym_analysis.txt,race100_summary.txt}。
> 代码状态:LMCache `1ff8935e`(含实验中修复并提交的三重栈集成崩溃);deep_researcher_demo 审阅+补节机制(终审 DONE,REPORT_REVIEW 默认开)。
> **A 臂** = 纯 vanilla vLLM(无投机、report 单次生成、原始 prompt、cudagraph);**B 臂** = attn-guided blend(RATIOS=0.2)+ router 投机(summary→eagle3 域训头,其余→suffix,锁 eager)+ report 分节+审阅+补节。局内 3 迭代×3 researchers×3 subqueries、题间并行 1、local 缓存、temperature 0。

## 一、总结论(TL;DR)

| 口径 | 结果 |
|---|---|
| TTFT(supervisor decide,干净归因 KV 复用) | **−759ms(−41~55%)**,n=152 配对,CI 不跨零 |
| e2e[含非LLM],100 题 | **−40.1%(1.67×)** CI95=[−115.1,−81.2]s |
| e2e[仅LLM 并集],100 题 | **−44.7%(1.81×)**;英文负载(gym)达 **−57.0%(2.33×)** |
| report 质量(RACE,100 题配对) | **B 显著更优:+0.0250** CI95=[+0.0150,+0.0350],B 优 75/100 |
| 三重栈稳定性 | 140 题(100+40)零崩溃、attn-guided 修正失败 0、writer 零误用 blend |
| **质量全量口径(2026-07-26 补)** | **报告质量涨、事实召回跌**:RACE +0.025、DRGym quality +0.190;但 DRGym KPR **−0.041**、DRQA 子答案正确率 **−5.5pp**(详见 §八) |

与先例 ~2.2× 的出入已定位:**主因是中文**(eagle3 英文域训头在中文 summary 上净减速);英文负载 2.14×/2.33× 达到并超过预期。

## 二、TTFT(干净归因 blend KV 复用)

- supervisor decide 配对(n=152):**−759ms**,CI95=[−836,−681]ms;中文题 −472ms、英文题 −1011ms(英文 A 基线上下文构成不同);
- blend 核实(server 日志,全程标签 AttnGuidedPrefill):drbench B 全程 supervisor 窗口复用命中(40 题批 64/64;60 题批同口径),gym 52/52;**writer/report 调用窗口真实命中 0**(40 题批 0/370、gym 0/332;时间窗口径见 [analyze_blend_e2e.py](analyze_blend_e2e.py) window_hits 注释——终点余量会把下一题 warmup 命中误归,已修正);
- 复用量级:40 题批 reuse 358k token(强制重算仅 739);"summary 存 + supervisor 复用 + writer 不碰"逐调用成立。

## 三、e2e(双口径 + 阶段/语言归因)

### 100 题主表

| 口径 | A | B | Δ |
|---|---|---|---|
| e2e[含非LLM]/题 | 244.5s* | 146.5s* | **−98.0s(−40.1%)** CI95=[−115.1,−81.2]s |
| e2e[仅LLM 并集]/题 | 222.9s | 123.3s | **−99.6s(−44.7%)** CI95=[−116.3,−83.0]s |
| 非 LLM 地板/题(搜索回放/检索/编排) | 21.6s | 23.1s | 两臂共同成本 |
| 同轮数子集(56 题,干净系统效应) | — | — | −31.9% |

*由 Δ 与 Δ% 反推的全量均值;逐题配对为准。

### 阶段拆解(40 题批实测,窗口口径)

| 阶段 | A | B | Δ% |
|---|---|---|---|
| research(plan+summary 并发窗口) | 110.8s | 96.7s | −12.7%(中文;英文 gym −48.4%) |
| decide(串行和) | 9.6s | 6.0s | −37.0% |
| **report(outline→节→审阅→补节)** | **122.0s** | **41.1s** | **−66.3%**(gym −67.8%,与语言无关) |

**"分节-审阅-补节吃时间"的假设不成立**:report 阶段贡献了总节省的绝大部分(A 的单次报告生成高达 122s,占 e2e 51%);B 的审阅+补节新增 ~11s,被分节并行收益淹没。

### 语言归因(三重验证闭合)

| 证据 | 中文 | 英文 |
|---|---|---|
| drbench 内 summary decode B/A(50/50 题) | **0.89(净减速)** | **1.18(净加速)** |
| gym 英文 40 题 e2e | — | **−53.3%(2.14×)/仅LLM −57.0%(2.33×)** |
| drbench 同轮数子集 e2e | −28.0% | −35.2% |

结论:eagle3 域训头(英文 drgym 轨迹训练)在中文 summary 上草稿接受率塌掉,draft+verify 变净税,把中文 research 段的投机收益吃平;英文负载达到先例预期。次要因素:router 锁 enforce-eager(先例 suffix 为混合非硬 eager,e2e+28% 教训)、题内 9 路并发下投机 verify 税。
**改进方向**:中文负载 summary 路由改 suffix(或 eagle3 补中文域训);解除 router 的硬 eager 依赖。

## 四、report 质量(RACE,gemini-3.5-flash,一把尺子)

### drbench 100 题配对(核心结论)

| | A | B | B−A |
|---|---|---|---|
| **Overall(100 题)** | 0.3716 | **0.3966** | **+0.0250** CI95=[+0.0150,+0.0350],**B 优 75/100** |
| 中文 50 | 0.3627 | 0.3869 | +0.0242 CI=[+0.0066,+0.0419] |
| 英文 50 | 0.3804 | 0.4063 | +0.0259 CI=[+0.0165,+0.0356] |

分维度(40 题批):Insight +0.04 最大、Comprehensiveness/IF +0.02、Readability 持平——与审阅+补节(引言/结论/结构重排)的作用方向一致;attn-guided 近似 KV 未见质量代价。

- **评委自一致性**(同批双打 5 组:a40/b40/a60/gymA/gymB):聚合自漂移 0.001~0.005(RACE)/0.005~0.167(gym quality),效应 +0.025 远超 RACE 自漂移,结论可信;
- b_blend60 第二遍因 gemini 月度上限未跑(已有 5 组双打作证,冗余验证省略;如需补跑提额后一条命令即可);
- 报告正文 fim_pad 泄漏 0/100(B);sumOL=0;审阅降级率 2/100(outline JSON 失败走设计内回退)。

### gym 英文(DRGym 自带口径)

| 指标 | A | B |
|---|---|---|
| quality(两遍) | 5.521 / 5.354 | **5.737 / 5.742**(两遍 B 均高;B 自漂 0.005) |
| kpr(要点召回) | 0.4386 | 0.3942(B 略低:轨迹更短覆盖要点少,如实记录) |
| citation_recall | 0.991 | 0.953 |
| citation_precision | 弃用(scorer 无逐题容错,429 即全崩;B 得 0.0 为伪值) |

> **与 DRGym 论文(arXiv:2505.19253)数值的口径对照(勿直接对表)**:论文 judge=gpt-4.1-mini、0-100 报数、quality 只有 Clarity+Insightfulness 两维、检索用官方 ClueWeb22/FineWeb 沙箱语料;我们 judge=gemini、0-10 报数、六维平均(Support/Clarity 拖低 overall)、检索用自建 DuckDuckGo 网页缓存。**换算到论文口径(×10)后我们 quality ≈ 55(A)/57(B)、KPR ≈ 39-44、citation recall 95-99——与论文中开源 baseline 同档**(OpenDeepSearch 59.2/47.0、HF-DeepSearch 57.5/48.0、KPR 42.8;商业系统 85-89 是另一档)。看似"很低"是量表未×10 + 对标对象 + 维度集/评委/语料四重口径差,非异常。

## 五、机制核实与运行质量(140 题全程)

- 三重栈(blend+attn-guided+router)零崩溃、`ATTN_GUIDED job failed`=0、路由零错配(冒烟逐请求验证);
- 审阅+补节实际行为:审阅 98/100 题恰 1 次(2 题设计内降级),补节 ~2.1 个/题(典型引言+结论);真实审阅质量目检:引言开头、结论收尾、中间逻辑序;
- 搜索缓存零 miss(local 模式,两臂同缓存);A 臂零 outline/review 调用(单次生成口径核实)。

## 六、实验中发现并修复的问题(已提交 LMCache `1ff8935e`)

**三重栈集成崩溃(blend × 投机解码首开暴露)**:eagle3 drafter 的 attention 层走同一 KV 包装器拨动 blend 逐层节拍,且 LMCache 上游把 drafter 层计入 kv_shape(65 层),与 blend 栈 64 层假设错位 → StopIteration/IndexError。修复=全链路目标层隔离(manager kv_shape 不加 drafter + register 过滤 + save/load 早退);0.5B blend-only 回归无感。物证:[blend-spec-layer-filter/](../../../modify-code-runs/blend-spec-layer-filter/blueprint.md)。
另:gym scorer 两处小修(quality/kpr 信号量 env 可调 DRGYM_SEM,适配 gemini 限速)。

## 七、数据与日志索引

- 生成:`eval/results/drbench/{a_van40,a_van60,b_blend40,b_blend60}/q*/`(合并视图 {a_van100,b_blend100} 符号链接)、`eval/benchmarks/results/drgym/{a_van40_gym,b_blend40_gym}/`;
- 打分:各 tag 下 `race/race_result.txt` + raw_results.jsonl;gym `summary.json`;
- 分析物证:/home/yilin/tmp/blend_e2e/(final100.txt、stage_analysis.txt、gym_analysis.txt、race100_summary.txt、language_split.txt、final_analysis.{txt,json});
- server/driver 日志:/home/yilin/tmp/logs/(server_{a,b,a_gym,b_gym}.log、drbench_{a,b,a60,b60}.log、drgym_{a,b}.log、race_*.log、gym_score*.log);
- suffix 轨迹:/home/yilin/tmp/blend_e2e/traj_b*(SUFFIX_TRAJ 纪律);作废:a_van40_tp2、smoke_b~b4。

## 八、全量三 benchmark 质量评测(2026-07-26 补齐)

> 数据规模翻倍后的完整口径:**DRBench 100 + DRGym 200 + DRQA(DeepSearchQA) 200**,双臂各跑满、零缺题零空报告。
> 评委分层(用户定):**已被 3.5-Flash 打过的不重打**;新增数据用 **gemini-3.1-flash-lite**(中途短暂用过 2.5-flash 的部分按"已测不重测"沿用)。
> **合并方式(2026-07-26 用户定:暂时采用跨评委平均)**:绝对分数与配对差**均按题数加权合并**。
> ⚠ 注意其代价:3.5-Flash 打分整体比 3.1-Lite 低约 1.3 分,所以合并后的**绝对值**混入了评委偏移,只适合本报告内部横比,**不可与任何单一评委口径或论文数值对表**;**配对差 B−A 不受影响**(每题内评委恒定),仍是可信的效应量。

### 8.1 合并总表

| Benchmark | 指标 | 规模 | A(vanilla) | B(blend+router) | **B−A(合并)** | 95% CI | 逐题胜负 |
|---|---|---|---|---|---|---|---|
| DRBench | RACE overall | 100 | 0.3716 | **0.3966** | **+0.0250** | [+0.0150,+0.0350] | B 优 75/100 |
| DRGym | quality | **200** | 6.430 | **6.620** | **+0.190** | 160 题段 [+0.040,+0.240] | 160 题段 B 78 胜/57 负/25 平 |
| DRGym | KPR | **200** | **0.566** | 0.525 | **−0.041** | 160 题段 [−0.059,−0.020] | 160 题段 B 45 胜/88 负/27 平 |
| DRQA | 全对率 | 200 | 28.0% | 23.5% | **−4.5pp** | [−11.0,+1.5] | B 16 胜/25 负/159 平 |
| DRQA | 子答案正确率 | 200 | 39.0% | 33.5% | **−5.5pp** | [−11.0,−0.2] | — |

### 8.2 分评委明细(供核对,勿跨行比绝对值)

| Benchmark | 评委 | 题数 | A | B | B−A |
|---|---|---|---|---|---|
| DRGym quality | gemini-3.5-flash(旧尺) | 40 | 5.354 | 5.742 | +0.388 |
| DRGym quality | gemini-3.1-flash-lite | 160 | 6.699 | 6.840 | +0.141 |
| DRGym KPR | gemini-3.5-flash(旧尺) | 40 | 0.4386 | 0.3942 | −0.044 |
| DRGym KPR | gemini-3.1-flash-lite | 160 | 0.598 | 0.5583 | −0.040 |
| DRQA F1 / precision / recall | gemini-3.1-flash-lite | 200 | 34.70 / 34.53 / 38.98 | 29.88 / 31.00 / 33.48 | −4.8 / −3.5 / −5.5 |

### 8.3 结论:B 写得更好,但记得更少

两类指标**方向相反,且都不是噪声**:

- **报告质量类 B 显著更优**:DRBench 的 RACE(+0.025,CI 不含 0)与 DRGym 的 quality(+0.141,CI 不含 0)同向,且来自两个独立 benchmark、两把不同的尺子;
- **事实召回类 B 显著更差**:DRGym 的 KPR −0.040(CI 不含 0,逐题 88 负 vs 45 胜)、DRQA 子答案正确率 −5.5pp(CI 上界 −0.2)。DRQA 全对率 −4.5pp 的 CI 含 0,只能算方向一致、不显著。

**与机制自洽**:复用的是别的调用算出来的 KV,近似误差最容易落在**具体事实**(数字、实体、要点)上,而行文组织、结构、可读性受影响小——这与 §四 观察到的"Insight/结构维度涨、Readability 持平"是同一现象的两面。**这是本轮最重要的新结论:此前 100 题只测了 RACE,看不到召回侧的代价。**

### 8.4 两个必须记住的口径缺陷

1. **DRQA 的"两遍自一致性"是失效的**:两遍 200 题**逐题判定完全相同**(200/200),因为评委调用 `temperature=0`,第二遍只是确定性复制,不构成独立重打。要做真自一致性须调温或换 seed(未擅自改,待定)。
2. **DRGym 200 题是混合评委**:40 题(3.5-Flash)+ 160 题(3.1-Lite,含沿用 2.5-Flash 的部分)。按用户决定**暂时跨评委平均**绝对分(6.430/6.620、0.566/0.525),但该绝对值含评委偏移(两把尺差约 1.3 分),**只可本报告内部横比**;**配对差 +0.190 / −0.041 不受此影响**,CI 只能由 160 题段给出。若后续要统一口径,需用同一评委重打全部 200 题。

### 8.5 打分链路上修掉的三个真 bug(否则数据不可信)

| bug | 症状 | 修复 |
|---|---|---|
| `AUTORATER_MAX_TOKENS=1500` 对思考模型太小 | gemini-3-flash-preview 的 JSON 在 Explanation 中间截断,DRQA 报错率 **68–70%**,且 A/B 落在**不同子集**上完全不可比 | 提到 8000;并改用非思考模型 |
| gym 打分器只在全部完成后返回 | 中途 429 即丢掉**全部**已完成结果(曾丢 37 题、92 题) | 增量落盘 + 连续失败早停;`run_drgym.score()` 逐指标落盘(原来从不写文件,内建续跑机制形同虚设) |
| `score_drqa.py` 无早停且重试 3 次 | 配额耗尽时 200 题×4 遍×3 重试 = **2400 次请求全打在 429 上**,并污染结果文件(续跑会跳过) | 连续 429 早停 + **失败行不落盘** |

配额教训:Gemini 日限 **按模型** 计(10k/天),且 **429 响应本身也扣次数**——所以在日配额模型下**重试是帮倒忙**(一次失败变多次扣费),最终把重试降到 1 次、并发降到 3,限流才消失。
