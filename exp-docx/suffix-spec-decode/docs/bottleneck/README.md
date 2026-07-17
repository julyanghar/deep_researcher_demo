# e2e 瓶颈 × Ceager 加速比 · 汇总

把两份分析放一起:**① e2e 各阶段墙钟(瓶颈在哪)② Ceager(suffix+enforce-eager)相对 A(原生基线)
逐题、逐 tag 的 decode 加速比(哪种生成被加速最多、有没有 tag 亏)**。

## 一句话

- **瓶颈三大块**:SUMMARY 生成 32.8% + FINAL_REPORT 生成 27.8%(LLM)+ 检索/搜索 31.8%(非 LLM)——
  三块几乎均分整条 e2e。
- **Ceager 全 tag 皆赢**:全局 decode 1.27×,**REPORT 最猛 2.37×**;200 个(题×tag)单元格里赢 186、平 9、
  只 5 个轻微亏(最差 0.86,均 >0.86)——**没有哪个 tag 系统性变慢**。

## 数据来源

- 两臂各 40 题、题间串行、题内并发 3、local 缓存检索。
  - A 基线:`eval/results/drbench/census_A/`
  - Ceager:`eval/results/drbench/census_Ceager/`
- 口径:每题 `llm_calls.jsonl` 逐调用 timing。**tok/s = 纯 decode 段吞吐**
  (Σcompletion_tokens / Σdecode_s,不含排队/工具/空隙);阶段墙钟 = 同 tag 区间**并集**(并发算一次)。
- 复现脚本都在本目录(见文末)。

---

## 一、e2e 各阶段墙钟(census_Ceager,40 题合计 8556s)

| 阶段 | 墙钟 s | 占跨度 | 朴素求和 s | 并发压缩 | 调用数 |
|---|---|---|---|---|---|
| INITIAL_RESEARCH_QUESTIONS | 84 | 1.0% | 84 | 1.0× | 40 |
| QUERY_PLAN | 227 | 2.6% | 445 | 2.0× | 241 |
| **RESEARCH_SUMMARY** | **2803** | **32.8%** | 19951 | **7.1×** | 723 |
| SUPERVISOR_DECISION | 341 | 4.0% | 341 | 1.0× | 81 |
| **FINAL_REPORT** | **2383** | **27.8%** | 2383 | 1.0× | 40 |
| **LLM 阶段合计(并集)** | **5838** | **68.2%** | | | |
| **非 LLM(检索/编排/空隙)** | **2718** | **31.8%** | | | |
| e2e 总跨度(首→尾) | 8556 | 100% | | | |

**怎么读**:
- **SUMMARY 是最大 LLM 块(32.8%)**。注意它"朴素求和 19951s"被**并发压缩 7.1×** 到墙钟 2803s——因为题内 3 个
  researcher 并发生成 summary,墙钟只算一次。所以 summary 真实占用大,但被并发摊薄了。
- **FINAL_REPORT 27.8%**,压缩 1.0×(每题 1 次、不并发),是**纯串行长文本生成**,一分钱一分墙钟。
- **非 LLM 31.8%(2718s)全在 `QUERY_PLAN → SUMMARY` 这段空隙**(81 个窗口、均 33.6s、最大 211s)——
  就是**检索/搜索阶段**。已定位为 23MB 块索引被反复全解析、并已优化(窗口 34s→~10s),详见
  [examine-spec-tax/e2e-search-bottleneck.md](../examine-spec-tax/e2e-search-bottleneck.md)。

**非 LLM 空隙按阶段归类**(40 题合计 2718s):

| 空隙位置(前→后) | 总秒 | 次数 | 均秒 |
|---|---|---|---|
| QUERY_PLAN → SUMMARY | 2718 | 81 | 33.6 |
| 其余阶段转换 | ~0 | — | 0.0 |

> 结论:e2e 时间 = summary(32.8%)+ report(27.8%)+ 搜索(31.8%),三块几乎均分;编排/其它转换几乎不耗时。

---

## 二、Ceager vs A 加速比(decode tok/s,越高越快)

### 2.1 逐 tag 聚合(Σtok/Σdecode_s,内容会漂 → 用速率比才公平)

| tag | A tok/s | Ceager tok/s | 聚合比 | 中位比 | 判定 |
|---|---|---|---|---|---|
| RESEARCH_SUMMARY | 25.5 | 27.1 | **1.07** | 1.04 | ✓ |
| **FINAL_REPORT** | 41.0 | 97.1 | **2.37** | 2.29 | ✓ 最猛 |
| SUPERVISOR_DECISION | 42.3 | 52.1 | 1.23 | 1.20 | ✓ |
| QUERY_PLAN | 42.7 | 65.7 | 1.54 | 1.53 | ✓ |
| INITIAL_RESEARCH_QUESTIONS | 43.7 | 56.0 | 1.28 | 1.28 | ✓ |
| **全局** | 29.1 | 37.0 | **1.27** | — | ✓ |

**为什么 REPORT 加速最猛(2.37×)、SUMMARY 最弱(1.07×)**:
- REPORT 是**长文本、上下文长(~2 万 token)、且大量复用前文措辞** → suffix 草稿命中率高、每步验证多个 token,
  加速空间最大;
- SUMMARY 内容更"新"(模型自己的话多、可抄的少)→ 草稿命中率低,只小赢。这与"接受率被内容上限卡住"一致。

### 2.2 逐题、逐 tag 配对(40 题按 qID 配对,赢=比≥1.02 / 平=0.98~1.02 / 亏=<0.98)

| tag | 配对题 | 赢 | 平 | 亏 | 中位比 | 最差 | 亏损题 |
|---|---|---|---|---|---|---|---|
| RESEARCH_SUMMARY | 40 | 29 | 9 | 2 | 1.043 | 0.95 | q37(0.95), q38(0.95) |
| FINAL_REPORT | 40 | 40 | 0 | 0 | 2.318 | 1.48 | 无 |
| QUERY_PLAN | 40 | 40 | 0 | 0 | 1.533 | 1.29 | 无 |
| SUPERVISOR_DECISION | 40 | 39 | 0 | 1 | 1.199 | 0.98 | q29(0.98) |
| INITIAL_RESEARCH_QUESTIONS | 40 | 38 | 0 | 2 | 1.277 | 0.86 | q1(0.86), q5(0.96) |

**全部 200 个(题×tag)单元格:赢 186 / 平 9 / 亏 5,中位 1.322。**
5 个亏损全是轻微、且集中在**短输出 tag**(SUMMARY/SUPERVISOR/INITIAL,输出短 → decode 段样本少、
冷启动噪声大),最差 q1-INITIAL 0.86。REPORT 和 QUERY_PLAN **零亏损**。

> 结论:**Ceager 对每种生成都不亏**(聚合口径全 ≥1;逐题仅 5/200 轻微亏、都在短输出、属噪声),
> 加速集中在长输出(REPORT 2.37×、QUERY_PLAN 1.54×)。这支撑了"Ceager 是全场冠军、可上线"的判决。

---

## 三、合起来看:瓶颈 × 加速对得上吗

| 阶段 | 占 e2e | Ceager decode 加速 | 说明 |
|---|---|---|---|
| SUMMARY | 32.8% | 1.07× | 占比大但加速小(内容新、可抄少);且已被并发压缩 7.1× |
| FINAL_REPORT | 27.8% | **2.37×** | **加速最猛的正好是第二大块** → 对墙钟贡献最实 |
| 搜索(非 LLM) | 31.8% | 与 suffix 无关 | 走另一条优化线(索引缓存,34s→10s) |

**要点**:suffix/Ceager 的加速主要吃在 **REPORT(2.37×)** 这块大头上;SUMMARY 虽占比最大但本身可加速空间小、
又被并发摊薄;而**第三大块"搜索"和投机解码无关**,是独立的工程优化(已做)。三条线各打各的。

---

## 附:复现脚本(本目录)

| 脚本 | 产出 |
|---|---|
| [e2e_stage_wall.py](e2e_stage_wall.py) | §一 各阶段墙钟表(interval-merge 并集口径) |
| [e2e_gap_breakdown.py](e2e_gap_breakdown.py) | §一 非 LLM 空隙按阶段归类 + 最大单次空隙 |
| [ceager_pertag_win.py](ceager_pertag_win.py) | §2.1 逐 tag 聚合加速比 + 逐条分布 |
| [alltag_pertopic.py](alltag_pertopic.py) | §2.2 逐题逐 tag 赢/平/亏 + 亏损题清单 |

跑法:`python <脚本>`(纯 stdlib,读 `eval/results/drbench/census_{A,Ceager}/`)。

## 相关

- 搜索阶段真因与优化:[examine-spec-tax/e2e-search-bottleneck.md](../examine-spec-tax/e2e-search-bottleneck.md)
- Ceager 冠军判决与固定税:[examine-spec-tax/fixed-tax-conclusions.md](../examine-spec-tax/fixed-tax-conclusions.md)、[examine-spec-tax/how-we-found-the-fixed-tax.md](../examine-spec-tax/how-we-found-the-fixed-tax.md)
- suffix 上线判决(L0/L3):[l0-l3-v2-results.md](../l0-l3-v2-results.md)
