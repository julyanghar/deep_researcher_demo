# eval/results/ 实验结果总索引

deep_researcher_demo 所有跑出来的实验产物都在这里。**本目录整体在 .gitignore 里、不入库**(体积 ~3.5G),
这些 README 只是给本机/后续查阅用的地图。权威的实验设计与结论在 `exp-docx/suffix-spec-decode/docs/`。

## 顶层目录一览

| 目录 | 体积 | 是什么 | 详细 |
|---|---|---|---|
| `drbench/` | ~300M | DRBench(Deep Research Bench)各次运行的输出,按"运行族"分子目录(census/e2e5/tax/l0-l3/online100) | [drbench/README.md](drbench/README.md) |
| `search_cache/` | ~3.1G | **检索缓存**:每题联网抓的网页 + 切块 embedding 索引(record 时建、replay 时读)。本仓库检索/提速工作的数据底座 | [search_cache/README.md](search_cache/README.md) |
| `saved/` | ~170M | 更早期阶段的归档运行(blend/native 对比、real50、deepsearchqa50 等,多为 6 月上旬) | [saved/README.md](saved/README.md) |
| `emb_test/` | ~460K | embedding 检索的小规模冒烟测试输出 | — |
| `search_cache_test/` | 小 | 搜索缓存机制的测试输出 | — |

## 每个"题目录"里长什么样

drbench 下每个运行族按题分 `q1/ q2/ …`,每题目录固定三件套:

| 文件 | 内容 |
|---|---|
| `llm_calls.jsonl` | **逐次 LLM 调用的 timing**(每行一次调用:tag、start_ts/end_ts、elapsed_s、completion_tokens、decode_s…)。e2e 墙钟/decode 吞吐都从这里算 |
| `harvest.jsonl` | 每次 LLM 调用的**完整存档**(messages / content / token_ids / store_generated_kv 等),供 KV 复用分析 |
| `report.md` | 该题最终生成的研究报告 |

> 口径提醒(踩过的坑):`llm_calls.jsonl` 里 tok/s 是**纯 decode 段**吞吐;墙钟含检索/编排/空隙。
> 题间**串行**(各题时间跨度不重叠),所以题内 LLM 空转 = 本题在做 CPU 侧检索,不是跨题排队。

## 命名约定

- 后缀 `.bad_<时间>` / `.old_<时间>`:失败或被替换的旧运行备份,可删。
- `TRASH/`:已弃结果。
- 运行族名 = 实验族 + 臂,例如 `census_Ceager` = census 40 题的 Ceager 臂。各臂含义见 [drbench/README.md](drbench/README.md)。
