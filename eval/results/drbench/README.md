# eval/results/drbench/ 运行族索引

DRBench 各次运行,按"实验族 + 臂"分目录。臂的语义以 `exp-docx/suffix-spec-decode/docs/` 为准
(相对本目录:`../../../exp-docx/suffix-spec-decode/docs/`)。

## 一、tax_*_e2e —— 固定税拆解实验(五常数模型)· 各 10 题

拆 suffix 每步比原生多花的钱,分"固定税(每步都交)+ 按量税(每多一个草稿位)"。
权威:[fixed-tax-conclusions.md](../../../exp-docx/suffix-spec-decode/docs/examine-spec-tax/fixed-tax-conclusions.md)、
[how-we-found-the-fixed-tax.md](../../../exp-docx/suffix-spec-decode/docs/examine-spec-tax/how-we-found-the-fixed-tax.md)。

| 目录 | 臂 | 含义 |
|---|---|---|
| `tax_A_e2e` | A | 原生、不投机、async 开(基线,每步 ~24ms) |
| `tax_B_e2e` | B | 原生 + `--no-async-scheduling`(A−B = 纯异步损失) |
| `tax_C_e2e` | C | suffix 投机解码(每步 ~42ms;C−A = 全部 suffix 税) |
| `tax_D_e2e` | D | 通用 spec 管线 / GPU drafter(每步 ~31ms;D−A = 固定机制税) |
| `tax_E_e2e` | E | 通用 spec + no-async(D−E = 投机时的异步效应) |

## 二、census_* —— census 40 题(最全口径)

| 目录 | 臂 | 含义 |
|---|---|---|
| `census_A` | A | 原生基线,40 题 |
| `census_Ceager` | Ceager | **suffix + 全程 enforce-eager**(全场冠军:端到端墙钟 1.28×、REPORT 2.37×,唯一代价 TTFT +14%)。**本会话 e2e 墙钟/搜索瓶颈分析用的就是它** |
| `census_Ceager.bad_07052102` | — | 一次失败/中断的备份,可删 |

相关:[e2e-search-bottleneck.md](../../../exp-docx/suffix-spec-decode/docs/examine-spec-tax/e2e-search-bottleneck.md)(用 census_Ceager 拆的搜索阶段)。

## 三、e2e5_* —— e2e 小规模对照(各 10 题)

同一批题上跑不同臂,比端到端墙钟 / 各阶段吞吐。"eager" = 加 enforce-eager 消 graph↔eager 边界。

| 目录 | 臂 | 含义 |
|---|---|---|
| `e2e5_A` | A | 原生基线 |
| `e2e5_C` | C | suffix(混合模式) |
| `e2e5_Ceager` | Ceager | suffix + enforce-eager(慢的是"混合"不是 eager,eager 反而更快) |
| `e2e5_CeagerK8pad` | CeagerK8pad | Ceager + k=8 + padding(**两刀合一,被否**:pad 洪水把接受率打到 13%) |
| `e2e5_D` | D | 通用 spec 管线 |
| `e2e5_Deager` | Deager | D + enforce-eager |
| `e2e5_G` | G | padding 臂(G−G0 = padding 价值;全 eager 下 pad 是负资产) |

## 四、l0v2_* / l3v2_* —— suffix 上线判决(各 10 题)

权威:[l0-l3-v2-results.md](../../../exp-docx/suffix-spec-decode/docs/l0-l3-v2-results.md)。

| 目录 | 含义 |
|---|---|
| `l0v2_baseline` | L0:不开 suffix 的基线(v2 多轮画像) |
| `l0v2_suffix` | L0:全局开 suffix(**采用**:墙钟 −29%、decode +16%、summary 不亏) |
| `l3v2_quote` | L3:逐字引用共设计 `SUMMARY_QUOTE=1`(**否决**:吞吐 +49% 但输出膨胀 +72% 吃光收益) |

## 五、online100_v2 —— 在线 100 题

`online100_v2`(100 题,~88M):联网真实检索的大规模运行(v2 多轮画像的在线口径)。

## 其它

- `TRASH/`:已弃结果,可删。
- `tax_A_e2e.old_07041916`:tax_A 的旧备份,可删。
