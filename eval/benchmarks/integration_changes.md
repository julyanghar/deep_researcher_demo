# 整合 DeepResearchBench + DeepResearchGym 评估 · 改动 & 审查包

> 目的:让 deep_researcher_demo 成为这两个 benchmark 上的被测系统(在它们原题库上跑 demo 报告,用各自 gold + scorer 打分,评委指到 demo 的 DashScope kimi)。
> 跑法见 [README.md](README.md);口径分析见 `/home/yilin/LMCache/exp-docx/research_benchmarks/`;modify-code 物证在 `~/modify-code-runs/integrate-drbench-drgym/`。

---

## 一、总体架构(以胶水为主,只动 1 处核心)
```
DRBench/DRGym 原题库  ──►  demo 工作流(REPORT_MODE=detailed_cited)  ──►  详细+带 inline 引用 的报告
                                                                              │
                                          ┌───────────────────────────────────┤ 落成各 scorer 要的格式
                                          ▼                                   ▼
                           DRBench raw_data/<tag>.jsonl              DRGym reports/<tag>/<id>.{a,q}
                                          │                                   │
                                   RACE + FACT(子进程)              KPR/Quality/Citation(import 调用)
                                          │                                   │
                                          └──────────► summary.json ◄──────────┘
                          评委统一 = DashScope kimi(judge_adapter 把 OpenAI 兼容 env 指过去)
```
只有 **agents.py 的 writer 报告模式**算动核心;其余是新增 runner/adapter,不改两个 benchmark 的源码。

---

## 二、逐文件改动(带代码引用)

### 1. [../../deep_researcher_demo/agents.py](../../deep_researcher_demo/agents.py)(+52/-1)——加 detailed_cited 报告模式
- 新增开关 `REPORT_MODE`([agents.py:26](../../deep_researcher_demo/agents.py#L26)):默认 `answer`(deepsearchqa 短答案式,**KV 复用路径完全不变**);`detailed_cited` 走新分支。
- `FinalWriter._SYSTEM_DETAILED`([agents.py:516](../../deep_researcher_demo/agents.py#L516)):要"分章节长报告 + 每个事实带 inline `[URL]` 引用(只引 findings 里给的源)+ 末尾 References"。
- `FinalWriter.write()`([agents.py:548](../../deep_researcher_demo/agents.py#L548)):**实时读 env**(`os.getenv("REPORT_MODE", REPORT_MODE)`,避免 import 期固化)分流;新签名加 `summary_sources`。
- `FinalWriter._write_detailed_cited()`([agents.py:573](../../deep_researcher_demo/agents.py#L573)):逐条 finding 拼上它的来源 URL(去重)、不走 `join_reusable_segments`(KV 复用只在默认模式)。

### 2. [../../deep_researcher_demo/workflow.py](../../deep_researcher_demo/workflow.py)(+11/-1)——把来源 URL 透传给 writer
- 新增 `summary_sources: list[list[str]]`,与 `summaries` 一一对齐累积(每轮 `round_results` 的 URL)。
- `write()` 调用处传 `summary_sources=summary_sources`。
- 为什么:writer 要有每条 summary 的来源 URL 才引得出 inline 引用。

### 3. [judge_adapter.py](judge_adapter.py)(新增)——把 scorer 接到 DashScope + 缓存核引用
- `judge_settings()`/`scorer_env()`:从 demo `.env` 读 `JUDGE_*`,设 `OPENAI_API_KEY`/`OPENAI_BASE_URL`/`RACE_MODEL`/`FACT_MODEL` 指到 DashScope kimi(两 benchmark 的 OpenAI 兼容 client 自动认这些 env,无需改源码)。
- `apply_dashscope_env()`:在 **import DRGym scorer 之前**调(DRGym 模块级 `client=AsyncOpenAI()` import 期读 env)。
- `build_cache_url_index()`/`cache_fetch()`:扫 demo 的 search 缓存(`pages_index`→`.txt`),给 citation 核验用,替代 crawl4ai。

### 4. [run_drgym.py](../DeepResearchGym/run_drgym.py)(新增)——跑 DRGym + KPR/Quality/Citation
- `--mode generate`:复用 `eval.run_deepsearchqa.GenerationRunner.build_workflow` 跑题(喂**原始 query**,不套 deepsearchqa 的 `## Final Answer` wrapper),写 `reports/<tag>/<id>.{a,q}`。
- `--mode score`:`apply_dashscope_env()` 后 import 各 scorer,**直接调 `evaluate_folder_async(tag, model, reports_root[, key_point_dir])`**(路径是参数,硬编码只在它们 `__main__`,故无需改 DRGym)。
- citation_precision:注入 crawl4ai stub(免装)+ monkey-patch `crawl_urls` 用 demo 缓存。
- `_mean_quality`/`_mean_kpr`:从各 scorer 返回的逐题 dict 算均值(quality 5-6 维各 0-10;kpr=Supported 占比)。

### 5. [run_drbench.py](../DeepResearchBench/run_drbench.py)(新增)——跑 DRBench + RACE/FACT
- `--mode generate`:同上,写 `DRBench/data/test_data/raw_data/<tag>.jsonl`(id/prompt/article)。
- `--mode score`:`scorer_env()` 子进程调 `deepresearch_bench_race.py`(RACE)与 `utils.{extract,deduplicate,scrape,validate,stat}`(FACT)。
- `_collect_race`:从 `race_result.txt` 读 Overall+4 维(兜底 raw_results.jsonl 的 `overall_score`)。
- FACT 的 scrape 走 DRBench 自带 Jina(`JINA_API_KEY`)。

---

## 三、验收 & 审查包(score 半边,样例报告,无需 GPU)
| 验收 | 证据地址 | 结果 |
|---|---|---|
| A1 detailed_cited 装配 | `~/modify-code-runs/integrate-drbench-drgym/run-1.log` stub 测段 | ✅ system 含 INLINE/References、user 含 findings+sources、URL 去重 |
| A2 DRGym Quality | `results/drgym/sampletest/summary.json` | ✅ overall **4.5**(6维) |
| A3 DRGym KPR | 同上 | ✅ **0.667** |
| A4 DRGym Citation 精/召 | 同上 | ✅ recall **0.857** / precision **0.0**(样例URL非缓存,坏路径给0) |
| A5 DRBench RACE | `results/drbench/sampletest/race/race_result.txt` | ✅ Overall **0.2001** + 4维(样例短弱 vs 强参考,低分合理)|
| A6 DRBench FACT | `results/drbench/sampletest/fact/fact_result.txt` | ✅ 全跑通(Jina):total_citations 3 / valid_rate 0.0(样例URL编的,正确) |
| A7 README | [README.md](README.md) | ✅ |
| **A8 generate 真 32B 端到端** | `results/drgym/native32b/{346541.a,summary.json}` | ✅ 见下 |

**A8(真 Qwen3-32B,原生 vLLM TP4 GPU0,3,4,5)**:demo 生成 `346541.a` = **13581 字 / 17 章节 / 19 个 inline `[http]` 引用 / 含 References**(detailed_cited + URL 透传端到端成立);真报告打分 **Quality 5.33 / KPR 1.0(全覆盖15关键点)/ citation_recall 0.576 / citation_precision 0.0**。
> A8 期间修了 2 个 bug(纳入 diff):① runner 类名 `GenerationRunner→EvalRunner`;② detailed_cited 长报告(max_tokens 1万)在默认 120s 超时下 ReadTimeout → runner generate 加 `os.environ.setdefault("LLM_TIMEOUT","900")`。

> 区分度自检:样例报告短弱/URL 假 → RACE 0.20、Quality 4.5、citation/FACT 给低/0;真 32B 报告 → Quality 5.33、KPR 1.0 → 好坏路径分得开,信号可信。

---

## 四、边界 / TODO
- **A8**:跑真 demo 报告需原生 32B vLLM(TP4)。当前 GPU 被他人占,按用户要求**等空闲再起**(哨兵盯着);同一 runner 切后端只需改 `OPENAI_BASE_URL/MODEL`。
- **citation 核验抓网页**:DRGym citation 已复用 demo 缓存(免 crawl4ai);DRBench FACT 用 Jina。若要 FACT 也复用缓存,可在 scrape 前预填 `deduplicated.jsonl` 的 `url_content`(scrape 会跳过已有项)——选配。
- **样例产物**:`sampletest`(DRGym/DRBench)是验证用样例,可清。
