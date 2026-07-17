# 40 题 e2e:三配置对照(vanilla / 投机 / 投机+report分节)

> **口径**:local 模式、**40 题**、题间串行(concurrency 1)、`REPORT_MODE=detailed_cited`、temp=0、TP4、GPU 0-3。B/C 的 Ceager server 全程带 `SUFFIX_TRAJ`(逐请求 acc/dr)。
> **动机**:20 题([e2e-4config-20q.md](e2e-4config-20q.md))推出"最优臂 = 投机 + 只 report 分节 + summary 不分节",但那次 c3/c4 是 summary+report 都并行、没单独隔离过"只 report 分节"。本实验在 40 题上干净隔离它,并用中间量拆出**投机单独贡献 vs report 分节增量**。

## 三个配置

| 标签 | server | 生成 | 代码 |
|---|---|---|---|
| **A** c1_van40 | vanilla(无投机·cuda-graph) | 非并行 | 新 prompt(cac4bab) |
| **B** c2_spec40 | Ceager(suffix+enforce-eager) | 非并行 | 同 |
| **C** c3_reppar40 | Ceager(同 B) | **只 report 分节**(`REPORT_PARALLEL=1`,summary 不分节) | 同 |

生效核对(全过):A `speculative_config=None`+`sumOL=0 repOL=0`;B `suffix+eager`+`sumOL=0 repOL=0`;**C `suffix+eager`+`sumOL=0 repOL=40`**(report 分了 40 个大纲、summary 一个没分 → 精确坐实"只 report 分节");每配置 40/40 报告齐;B/C `SUFFIX_CFG` 断言确认 traj 生效。

## 一、阶段拆(40 题合计,秒)

| 配置 | e2e | summary段 | report段 | plan/sup | 非LLM检索 |
|---|---|---|---|---|---|
| A vanilla | 9293 | 2957 | 4584 | 686 | 1066 |
| **B 投机非并行** | **6540** | 2691 | 2209 | 546 | 1094 |
| **C 投机+report分节** | **4879** | 2414 | 941 | 456 | 1065 |

(e2e = 每题 [首call起, 末call止] 跨度之和;阶段段 = 该阶段 tag 的区间并集;非LLM = 跨度 − 全 LLM 区间并集。非LLM ~1066-1094s 三配置稳定 = 检索与配置无关,健全性检查通过。)

## 二、三个 delta

**① 投机效应 A→B(⚠️ 含 eager + traj 混淆)**
- e2e **9293→6540 = −29.6%**;**几乎全靠 report 段 −52%**(4584→2209),summary 段仅 −9%(2957→2691)。
- → **投机价值 ∝ 照搬内容**:report 照抄命中 **61.2%** 大赚,summary 改写命中 **28.8%** 沾不上(见 §四)。

**② ★report 分节增量 B→C(干净:同 server、同执行模式、都带 traj)**
- e2e **6540→4879 = −25.4%**;**report 段再 −57%**(2209→941),summary 段基本不动(2691→2414,−10%,因为 summary 没并行)。
- report 分节把单条大报告拆成"大纲 + 40×N 段并发写"(report 调用 40→228),**靠并发把 report 段墙钟砍一半以上**。
- → **这是本实验最干净的一条:report 分节在投机之上再省 25%,且 summary 零代价(没碰它)。**

**③ 总提升 A→C**:e2e **9293→4879 = −47.5%**,端到端近乎腰斩。

## 三、逐 tag:wallclock均 + token/s(三配置)

| tag | A 均wall / tok·s | B 均wall / tok·s | C 均wall / tok·s |
|---|---|---|---|
| RESEARCH_SUMMARY_TEXT(分析) | 27.8s / 25.8 | 26.6s / 28.1 | 24.3s / 30.8 |
| FINAL_REPORT_MARKDOWN(照抄) | 114.6s / 41.2 | **55.2s / 95.7** | 15.4s / 57.7 |
| FINAL_REPORT_OUTLINE_JSON | — | — | 6.2s / 69.9 |
| QUERY_PLAN_JSON | 1.9s / 42.9 | 1.3s / 67.4 | 1.1s / 94.8 |

- **report 单调用**:A→B 投机让 tok/s 翻倍(41→96)、墙钟减半(114→55);C 因拆成短段(228 次)单调用 tok/s 回落(58)但**靠并发让 report 段总墙钟继续砍**(§一 report 段 2209→941)。
- **summary 单调用**:三配置几乎不变(27→24s、26→31 tok/s)——投机对改写没用、也没并行,符合预期。

## 四、逐 tag suffix 命中率(B/C,traj 真实逐请求)

| tag | B 命中率 | C 命中率 |
|---|---|---|
| **FINAL_REPORT_MARKDOWN(照抄)** | **61.2%** | 46.2% |
| RESEARCH_SUMMARY_TEXT(分析) | 28.8% | 32.4% |
| QUERY_PLAN_JSON | 39.2% | 50.6% |
| SUPERVISOR_DECISION_JSON | 33.1% | 36.1% |
| INITIAL_RESEARCH_QUESTIONS_JSON | 25.5% | 71.7% |

- **照抄 vs 改写分化坐实**:report **61.2%** ≫ summary **28.8%**(B)——和 token/s(report 96 vs summary 28)、和 eagle3 对比(eagle3 summary 10.7%/report 17.4%,[eagle-idea.md](eagle-idea.md))全对上。
- **report 分节反而降 report 命中(61.2%→46.2%)**:拆成短段后每段照抄上下文变小、逐字复制机会减少;**但它仍靠并发净赚**(命中降、墙钟却更低)——说明 report 分节的收益来自**并发**,不是命中。

## 五、逐题 e2e(40 题,秒)

| q | A_van | B_spec | C_rep | C/B | q | A_van | B_spec | C_rep | C/B |
|--|--|--|--|--|--|--|--|--|--|
| 1 | 258 | 204 | 179 | 0.88 | 21 | 130 | 189 | 83 | 0.44 |
| 2 | 319 | 308 | 231 | 0.75 | 22 | 154 | 107 | 121 | 1.13 |
| 3 | 386 | 259 | 114 | 0.44 | 23 | 385 | 119 | 168 | 1.42 |
| 4 | 165 | 110 | 64 | 0.58 | 24 | 131 | 82 | 67 | 0.82 |
| 5 | 421 | 269 | 190 | 0.71 | 25 | 276 | 179 | 175 | 0.98 |
| 6 | 265 | 128 | 88 | 0.69 | 26 | 270 | 191 | 157 | 0.82 |
| 7 | 400 | 223 | 162 | 0.73 | 27 | 205 | 196 | 189 | 0.96 |
| 8 | 200 | 114 | 90 | 0.79 | 28 | 286 | 252 | 147 | 0.59 |
| 9 | 160 | 126 | 146 | 1.16 | 29 | 280 | 232 | 191 | 0.82 |
| 10 | 390 | 296 | 185 | 0.62 | 30 | 187 | 107 | 114 | 1.07 |
| 11 | 151 | 96 | 62 | 0.64 | 31 | 164 | 94 | 73 | 0.78 |
| 12 | 151 | 90 | 72 | 0.81 | 32 | 197 | 173 | 118 | 0.68 |
| 13 | 297 | 240 | 206 | 0.86 | 33 | 150 | 92 | 60 | 0.65 |
| 14 | 158 | 215 | 168 | 0.78 | 34 | 236 | 158 | 124 | 0.78 |
| 15 | 172 | 158 | 90 | 0.57 | 35 | 250 | 215 | 80 | 0.38 |
| 16 | 210 | 115 | 77 | 0.67 | 36 | 290 | 122 | 160 | 1.31 |
| 17 | 144 | 167 | 67 | 0.40 | 37 | 386 | 170 | 101 | 0.60 |
| 18 | 169 | 145 | 122 | 0.84 | 38 | 232 | 125 | 81 | 0.65 |
| 19 | 114 | 76 | 49 | 0.64 | 39 | 211 | 155 | 114 | 0.74 |
| 20 | 106 | 75 | 56 | 0.75 | 40 | 236 | 170 | 138 | 0.81 |
| | | | | | **合计** | **9293** | **6540** | **4879** | |

**逐题读法**:C(最优臂)vs B **33/40 题更快**、中位 C/B ~0.74。少数 C>B(q9/q22/q23/q30/q36)是 report 轻/summary 重的题——并发填不满、大纲那一跳反而多花时间;但绝大多数题 report 分节净赚。

## 六、结论

1. **投机 = 大赢(−29.6%)**,靠 report(照抄命中 61%);summary(改写命中 29%)沾不上——**投机价值 ∝ 照搬内容**。
2. **★report 分节在投机之上再赚 25.4%(干净隔离)**,report 段 −57%,**summary 零代价**(没并行)——**"投机 + 只 report 分节 + summary 不分节"这条最优臂,40 题上确证成立**。
3. **两步叠加端到端 −47.5%(近腰斩)**。
4. **report 分节的收益来自并发,不是命中**(命中反从 61%→46%,墙钟却更低)。

## 六b、报告质量验证(2026-07-11 补,RACE 打分)

对同 40 题、同 reference、同裁判(kimi-k2.5,经 judge_adapter)跑 DRBench RACE(每篇 LLM 清洗 + 1 次合并打分;分数为相对 reference 的比值,0.5=打平):

| 配置 | Comprehensiveness | Insight | Instr. Following | Readability | **Overall** |
|---|--:|--:|--:|--:|--:|
| c2 投机不分节 | 0.3533 | 0.3309 | 0.4072 | 0.3952 | **0.3657** |
| **c3 投机+report分节** | 0.3658 | 0.3463 | 0.3991 | 0.3799 | **0.3700** |

**结论:report 分节质量无损**(Overall +0.004,噪声内;全面性/洞察略升、循令/可读性略降,均在方差内)——**最优臂"投机+只 report 分节"拿 −47.5% 端到端提速,报告质量不掉**。物证:`eval/results/drbench/{c2_spec40,c3_reppar40}/race/race_result.txt`、日志 `~/modify-code-runs/race-4way/`。(online100/v2 的交集打分输入已备好 `raw_data/*_i40.jsonl`,暂缓评估;续跑时重跑 `run_race4.sh` 即可——已完成的 tag 有 raw_results.jsonl 会自动跳过。)

## 七、诚实边界

- **A→B 混了三件事**:投机 + 执行模式(vanilla=cuda-graph、Ceager=enforce-eager)+ traj 写盘 overhead(仅 B/C 有 traj,A 没有)。所以 −29.6% 是"投机+eager+traj"合力,非纯投机;方向与 report/summary 分化稳健。
- **B→C 干净**:同 server、同执行模式、都带 traj → −25.4% 是 report 分节的干净增量。
- summary 全程不分节(20 题已坐实 summary 分节亏),本实验只验 report 分节。

## 八、为什么少数题 C 慢过 B(追问回填)

**先厘清**:C 慢过 B 的 5 题(q9/q22/q23/q30/q36)对 **A(vanilla)全部仍更快** —— "C 一直快过 A" 成立,反常的只是 C 偶尔慢过 B。

拆这 5 题的段(B vs C):

| q | Δsummary段 | Δreport段 | sumTok B→C |
|---|---|---|---|
| q9 | **+26s** | −12s | 7108→11915(+68%) |
| q22 | **+29s** | −22s | 4553→8395(+84%) |
| q23 | **+57s** | −22s | 7944→16941(+113%) |
| q30 | **+31s** | −27s | 4379→8463(+93%) |
| q36 | **+29s** | +2s | 9860→13721(+39%) |
| **合计** | **+172s** | **−82s** | |

**结论:慢的是 summary 段,不是 report 分节。**
1. **report 分节在这 5 题也照样有效**(Δreport 合计 **−82s**,4/5 题 report 段仍降;大纲那跳仅 3-10s)。
2. **拖慢的是 summary 段**(Δsummary 合计 **+172s**,盖过 report 的 −82s)。而 summary 段 B/C 走**完全相同的非并行代码**(C 只分 report)——不是并行问题。
3. **根因 = summary 阶段 run 间 token 方差**:C 这几题恰好多生成 40-113% 的 summary token(agent 每遍子查询拟几条、summary 写多长会变,temp=0 也架不住 TP 非确定性级联到"研究几轮"的决策)。B、C 是两遍独立 run、不共享子查询,故单题会抖。

**含义**:report 分节机制**始终有效**;逐题 C vs B 的翻车是 summary 工作量的 run 间噪声(即之前诊断的"~20% run 间噪声"来源)。**聚合(40 题合计 B→C −25.4%)把噪声平均掉,是稳的**;要逐题干净比得锁死同一套子查询(本实验未锁)。

## 物证

- 一条龙脚本 + status:`~/modify-code-runs/e2e-3config-40q/run_bc_traj.sh`、`status_bc.txt`、`status_40q.txt`(A)
- 逐调用计时:`eval/results/drbench/{c1_van40,c2_spec40,c3_reppar40}/q*/llm_calls.jsonl`
- 逐请求 acc/dr:`~/modify-code-runs/e2e-3config-40q/traj_bc.<pid>`(4 worker,2.16M 行)
- 分析脚本 + 原始输出:`~/modify-code-runs/e2e-3config-40q/analyze_40q.py`、`analysis_out.txt`
- server 日志:`server_vanilla.log`(A)、`server_ceager_traj.log`(B/C,含 SUFFIX_CFG 断言)
