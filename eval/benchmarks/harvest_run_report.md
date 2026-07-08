# 全量 benchmark 生成 + 调用链 harvest · 审查报告(交接闸)

> 任务:用现有 benchmark runner 在 DeepResearchBench + DeepResearchGym 上**全量跑、只生成报告(不评估)**,每题留**详细调用链 harvest** + **逐调用 prefill/decode 计时**,供以后离线分析/打分。
> 走 `/modify-code` 收敛循环;物证:`~/modify-code-runs/benchmark-harvest-gen/`(blueprint.md + run-1.log + run_full.log + 基线)。
> 一句话:**两个 benchmark 各 100 题全部生成完成、0 失败;每题报告 + 5 类调用完整 harvest + 逐调用计时齐全;全程 vanilla 32B,未做任何评分。**

---

## 一、改了什么(代码)
只新增 1 个 helper + 给 2 个 runner 各加一个 `--harvest` 开关,**不动 demo 核心、不动两 benchmark 源码**。

### 1. 新增 [harvest_gen.py](harvest_gen.py)
`gen_one()` —— **每题起一个 demo CLI 子进程**(`python -m deep_researcher_demo --quiet --output report.md "<question>"`),给子进程设独立 env:
- `REPORT_MODE=detailed_cited`(详细+带 inline 引用的报告)
- `DRIFT_HARVEST=<out>/q<id>/harvest.jsonl` + `DRIFT_HARVEST_TAGS=`(全链 5 项)
- `LLM_CALL_LOG=<out>/q<id>/llm_calls.jsonl`(逐调用计时)
- `MODEL=Qwen3-32B`、`OPENAI_BASE_URL=localhost:30000/v1`、`SEARCH_CACHE=record`、`SEARCH_CACHE_SAMPLE_ID=<id>`、`LLM_TIMEOUT=900`
- `--output q<id>/report.md`
`run_batch()` —— 子进程并发池(Semaphore)+ 断点续(report.md 已存非空则跳)。
> **为什么子进程**:harvest 路径 `DRIFT_HARVEST` 是全局(llm.py import 期读),进程内并发跑多题会串;每题独立子进程天然分题不串(沿用 exp_writer 的老规矩)。

### 2. [run_drgym.py](../DeepResearchGym/run_drgym.py) / [run_drbench.py](../DeepResearchBench/run_drbench.py) 加 `--harvest`
`generate_harvest()`:走 harvest_gen 子进程路径(而非原 in-process generate),跑完把 report.md **落成 scorer 格式**(DRGym `<tag>/<id>.a`+`.q`;DRBench `raw_data/<tag>.jsonl`),便于以后 `--mode score` 直接打分。
> 默认 generate(不带 --harvest)路径不变;此开关只为"留 harvest"。

### 前置(运行环境)
vanilla 32B vLLM:`config_no_lmcache.yaml` + `--tensor-parallel-size 2`(GPU1,2)+ `VLLM_RESP_TIMING=1`(回 prefill_s/decode_s)。**无 lmcache/blend**。

---

## 二、验收证据(/modify-code,样例 q346541 逐条钉)
| 验收 | 证据地址 | 结果 |
|---|---|---|
| A1 harvest 含全 5 tag | run-1.log A1 段 / `drgym/harvtest/q346541/harvest.jsonl` | ✅ INITIAL1/QUERY_PLAN9/SUMMARY27/DECISION3/REPORT1 |
| A2 报告分章节+引用 | `harvtest/q346541/report.md` | ✅ 21KB / 18 章节 / 127 inline 引用 / References |
| A3 scorer 格式就绪 | `harvtest/346541.a`+`.q` | ✅ |
| A4 断点续 | run-1.log A4 段 | ✅ 再跑同题 → skip |
| A5 search record 生效 | `eval/results/search_cache/q346541` | ✅ 页被存 |
| A6 不评估 | harvtest 目录无 evaluation_results_* | ✅ |
| A7 逐调用 prefill/decode 计时 | `harvtest/q346541/llm_calls.jsonl` | ✅ 41 调用**全带** prefill_s/decode_s/ttft_s/queued_s/inference_s + 阶段 tag |

---

## 三、全量运行结果(DRBench 100 + DRGym 100)
| | report.md | harvest.jsonl | llm_calls.jsonl | scorer 格式 | 失败/空 |
|---|---|---|---|---|---|
| **DRBench** | 100/100 | 100 | 100 | `raw_data/full100.jsonl`(100,0 空)| **0** |
| **DRGym** | 100/100 | 100 | 100 | 100×`.a`+`.q` | **0** |

- 耗时:DRBench `00:03→04:36`、DRGym `04:36→08:13`(vanilla 32B / TP2)。产物 71M + 64M。
- 全程 **只 generate、未调任何 scorer**;`SEARCH_CACHE=record`(页已存进 `eval/results/search_cache/`,后续 citation/复现可用)。

---

## 四、各阶段 prefill / decode 耗时

### 4.1 逐调用均值(200 题、6179 次调用,秒)
| 阶段(tag) | 调用数 | prefill | decode | ttft |
|---|--:|--:|--:|--:|
| INITIAL_RESEARCH_QUESTIONS_JSON | 200 | 0.21 | 5.5 | 0.38 |
| QUERY_PLAN_JSON | 1332 | 0.17 | 2.8 | 0.37 |
| RESEARCH_SUMMARY_TEXT | 3996 | 2.98 | 24.9 | 6.12 |
| SUPERVISOR_DECISION_JSON | 451 | 1.86 | 8.5 | 2.48 |
| **FINAL_REPORT_MARKDOWN** | 200 | 3.15 | **236.1** | 3.37 |

- **decode 占总 LLM 时间 92%**;最终长报告单次 decode 均值 236s(detailed_cited 报告长,这也是把 `LLM_TIMEOUT` 提到 900s 的原因)。summary 因搜索结果长 → prefill ~3s、ttft ~6s。

### 4.2 每题各阶段:总调用 + 调用/题 + prefill/decode(合计 = prefill+decode 之和,秒)
**DRBench(100 题)**
| 阶段 | 总调用 | 调用/题 | prefill/题 | decode/题 | 合计/题 |
|---|--:|--:|--:|--:|--:|
| INITIAL_QUESTIONS | 100 | 1.0 | 0.2 | 6.2 | 6.4 |
| QUERY_PLAN | 747 | 7.5 | 1.3 | 21.5 | 22.9 |
| **RESEARCH_SUMMARY** | **2241** | 22.4 | 69.5 | 541.7 | **611.2** |
| SUPERVISOR_DECISION | 255 | 2.5 | 5.0 | 23.5 | 28.5 |
| FINAL_REPORT | 100 | 1.0 | 3.5 | 245.9 | 249.4 |
| **合计** | **3443** | 34.4 | | | **918.3** |

**DRGym(100 题)**
| 阶段 | 总调用 | 调用/题 | prefill/题 | decode/题 | 合计/题 |
|---|--:|--:|--:|--:|--:|
| INITIAL_QUESTIONS | 100 | 1.0 | 0.2 | 4.7 | 5.0 |
| QUERY_PLAN | 585 | 5.8 | 0.9 | 16.3 | 17.3 |
| **RESEARCH_SUMMARY** | **1755** | 17.6 | 49.5 | 452.8 | **502.3** |
| SUPERVISOR_DECISION | 196 | 2.0 | 3.4 | 14.8 | 18.2 |
| FINAL_REPORT | 100 | 1.0 | 2.8 | 226.4 | 229.2 |
| **合计** | **2736** | 27.4 | | | **771.9** |

- 调用次数:INITIAL/FINAL 每题各 1 次(固定);QUERY_PLAN ≈ researcher 数(7.5 / 5.8);RESEARCH_SUMMARY 最多(= researcher × sub-query);DECISION ≈ 轮数(2–2.5)。

### 4.3 并行重叠修正:相加 vs 真实墙钟(并集)
⚠️ 4.2 的"合计/题"是各调用耗时**直接相加**,但 **RESEARCH_SUMMARY / QUERY_PLAN 是多 researcher 并行的,时间重叠**,直接加会**高估墙钟**。下表"相加"用 elapsed(start→end,含排队);"并集墙钟"= 把该阶段调用的 `[start,end]` 区间求并集(合并重叠)后的总长——**这才是该阶段真正占用的墙钟**。

| 阶段 | DRBench 相加 | **DRBench 墙钟** | DRGym 相加 | **DRGym 墙钟** | 重叠压缩 |
|---|--:|--:|--:|--:|--:|
| INITIAL_QUESTIONS | 6.5 | 6.5 | 5.3 | 5.3 | 1.0x(单调用)|
| QUERY_PLAN | 25.0 | 9.2 | 18.2 | 6.6 | ~2.7x(并行)|
| **RESEARCH_SUMMARY** | 699.8 | **177.8** | 539.9 | **122.3** | **3.9–4.4x(并行)** |
| SUPERVISOR_DECISION | 30.8 | 30.8 | 18.7 | 18.7 | 1.0x(顺序)|
| FINAL_REPORT | 249.7 | 249.7 | 229.4 | 229.4 | 1.0x(单调用)|
| **整题** | 1011.9 | **474.1** | 811.5 | **382.2** | 2.1x |

- **RESEARCH_SUMMARY 真实墙钟只有 ~178s/122s**(不是相加的 700/540s)——~22 个 researcher 并行被 vLLM 批处理压缩 ~4x。"按相加它占 65%"是高估,**墙钟口径下它退居第二**。
- **真正的墙钟大头是 FINAL_REPORT**:DRBench 250s = 整题 474s 的 **53%**;DRGym 229s = 382s 的 **60%**。因为它是**单次长 decode、无并行可掩盖**(batch=1)。
- 各阶段并集相加 ≈ 整题并集 → **阶段之间基本顺序**(多轮 research → 写报告),只有**阶段内部**(summary、query_plan)并行。
- **结论:要压每题墙钟,头号是缩短最终报告 decode(占一半多),其次才是 summary**;summary 调用最多、算力最大,但并行已把它的墙钟摊到第二位。原理见 [vLLM 并行 decode 内部机制](../../../LMCache/claude-docx/17-vllm-decode-batching-internals.md)。

> 逐题逐调用明细在各 `q<id>/llm_calls.jsonl`(字段:tag/prompt_tokens/completion_tokens/prefill_s/decode_s/ttft_s/queued_s/inference_s/start_ts/end_ts/elapsed_s)。

---

## 五、产物布局 + 后续怎么用
```
eval/benchmarks/results/
  drbench/full100/q<id>/{report.md, harvest.jsonl, llm_calls.jsonl}   + DRBench raw_data/full100.jsonl
  drgym/full100/q<id>/{report.md, harvest.jsonl, llm_calls.jsonl}     + <id>.a / <id>.q
```
- **以后要打分**(本次没做):`eval/DeepResearchBench/run_drbench.py --tag full100 --mode score`(RACE+FACT)、`eval/DeepResearchGym/run_drgym.py --tag full100 --mode score`(KPR/Quality/Citation)。
- harvest.jsonl = 每题完整调用链(每条调用的完整 messages + content),用于离线分析(漂移/复用/重叠等)。

---

## 六、边界 / 需要你拍板
1. **整体可置 DONE 吗?**(代码 A1–A7 + 全量 200 题产物 + 计时)
2. **32B server 还开着**(TP2 GPU1,2):留着→可直接跑 `--mode score`;否则停掉释放 GPU。
3. 样例 `harvtest/` 目录可清。
4. 已知:citation_precision 若以后要用,本次 record 缓存已存页 → 可核到(不像之前 live 模式)。
