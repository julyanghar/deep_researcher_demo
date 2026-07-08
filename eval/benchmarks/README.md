# benchmarks · 把 DeepResearchBench + DeepResearchGym 评估整合进 demo

让 deep_researcher_demo 成为这两个 benchmark 上的**被测系统**:在它们的原题库上跑 demo 的研究报告,
再用各自自带的 gold + scorer 打分,评委统一指到 demo 的 **DashScope JUDGE(kimi)**。

| benchmark | 题库 | 指标 |
|---|---|---|
| **DeepResearchGym** | Researchy 真实查询(自带 gold 关键点)| KPR / Quality(5维)/ Citation 精·召 |
| **DeepResearchBench** | 100 专家题(自带 criteria + 参考报告)| RACE(报告质量)/ FACT(引用核验)|

> 背景/口径详见 `/home/yilin/LMCache/exp-docx/research_benchmarks/`(三份分析报告)。

## 自包含目录(不再依赖外部 repo)
三个评测协议各有自包含目录,评测代码 + 数据都在目录内,**不跳到外部 repo**:
- `eval/DeepResearchBench/`:`run_drbench.py` + `deepresearch_bench_race.py` + `utils/` + `prompt/` + `data/`(题集/criteria/参考报告)
- `eval/DeepResearchGym/`:`run_drgym.py` + 4 个 `eval_*_async.py` + `queries/` + `key_point/`
- `eval/deepresearchqa/`:`run_deepsearchqa.py` + `judge.py` + `scoring.py`;题集 `data/deepsearchqa.jsonl`(本地优先,缺则从 HF 下载落盘)
- 共享生成层(三协议共用,保留在 `eval/benchmarks/`):`harvest_gen.py`(子进程跑 demo)、`judge_adapter.py`(评委 env + 缓存引用核)。
- 默认指向本目录;如需指回外部 repo 可设 `DRBENCH_DIR` / `DRGYM_DIR`。

## 前置
- **报告模式**:benchmark 评测要用"详细+带引用"的报告 → 设 env **`REPORT_MODE=detailed_cited`**(runner 已自动设)。默认 `answer`(deepsearchqa 短答案式)不受影响。
- **generate 阶段需 vLLM 服务器**(demo 的 LLM,见 `eval/EVAL_GUIDE.md` 起 server)。
- **score 阶段只需 DashScope 评委**:从 `.env` 读 `JUDGE_API_KEY` / `JUDGE_BASE_URL` / `JUDGE_MODEL`(judge_adapter 自动把 `OPENAI_API_KEY`/`OPENAI_BASE_URL`/`RACE_MODEL`/`FACT_MODEL` 指过去,无需改 benchmark 源码)。
- 解释器:`/home/yilin/anaconda3/envs/gpt-deep/bin/python`。

## 跑法
两个 runner 都分 `--mode generate|score|all`(generate 需 server,score 不需)。

```bash
cd /home/yilin/deep_researcher_demo
PY=/home/yilin/anaconda3/envs/gpt-deep/bin/python

# DeepResearchGym(先小样本)—— runner 已移入自包含目录 eval/DeepResearchGym/
$PY eval/DeepResearchGym/run_drgym.py --tag demo --mode all --n 100 \
    --metrics quality,kpr,citation_recall,citation_precision --concurrency 2
#  → 报告: eval/benchmarks/results/drgym/demo/<id>.{a,q}
#  → 汇总: eval/benchmarks/results/drgym/demo/summary.json

# DeepResearchBench(100 题,可 --only-lang en/zh 减半)—— 自包含目录 eval/DeepResearchBench/
$PY eval/DeepResearchBench/run_drbench.py --tag demo --mode all --n 100 \
    --metrics race,fact --concurrency 2 --workers 4
#  → 报告: eval/DeepResearchBench/data/test_data/raw_data/demo.jsonl(本地 vendored 数据)
#  → 汇总: eval/benchmarks/results/drbench/demo/summary.json(含 race 的 Overall+4维)
```

只重打分(报告已生成):`--mode score`。

## 产物
- `results/<bench>/<tag>/summary.json`:各指标汇总数字。
- DRGym:`<id>.a`(报告)/`<id>.q`(query)/ `evaluation_results_*.json`(逐题明细)。
- DRBench:`race/{raw_results.jsonl,race_result.txt}`、`fact/{extracted,deduplicated,scraped,validated}.jsonl + fact_result.txt`。

## 指标口径
- **Quality**:5–6 维各 0–10(Clarity/Depth/Breadth/Support/Insight…),取均值。
- **KPR**:报告 Supports 的 gold 关键点占比(每题对 `key_point/<id>_aggregated.json` 逐点判 Supports/Omits/Contradicts)。
- **Citation recall**:报告里有源支撑的 claim 占比(不抓网页)。
- **Citation precision**:抽(claim,引用URL)→ 取网页正文 → LLM 判该网页是否支持 claim(full/partial/no=1/0.5/0)→ 均值。
- **RACE**:对参考报告比着打,4 维加权,Overall=target/(target+reference)。
- **FACT**:抽(事实,URL)→ 去重 → 抓网页 → 验证支撑 → Citation Accuracy / Effective Citations。

## 引用核验的抓网页(重要)
- **Citation precision(DRGym)**:已改为**复用 demo 的 search 缓存**(`pages_index`→`.txt`)核引用,免装 crawl4ai、确定可复现(`judge_adapter.build_cache_url_index/cache_fetch`)。报告引的 URL 不在缓存里 → 算 no_support。
  - ⚠ **要让它有意义,generate 时设 `SEARCH_CACHE=record`**:demo 默认 live 搜索**不写缓存** → 报告引的页缓存里没有 → citation_precision 偏 0(实测真 32B 报告就是 0)。record 模式会把访问过的页存进缓存,citation_precision 才查得到正文。
- **FACT 的 scrape(DRBench)**:DRBench 自带 scrape 走 **Jina(`JINA_API_KEY`)**。**当前未配 JINA_API_KEY** → scrape 步取不到正文、validate 多判 no_support。
  - 选项:① 配 `JINA_API_KEY`;② **TODO**:仿 DRGym 在 scrape 前用 demo 缓存预填 `deduplicated.jsonl` 的 `url_content`(scrape 会跳过已有 url_content 的项)。
  - FACT 的 **LLM 步骤(extract/validate)走已验证的 DashScope 路径**,只有抓网页这一环受 Jina 限制。

## 验证状态(2026-06-29,样例报告,无需 GPU 的 score 半边)
- DRGym 四指标端到端跑通:Quality 4.5 / KPR 0.667 / citation_recall 0.857 / citation_precision 0.0(样例 URL 非真实缓存,符合预期)。
- DRBench RACE 端到端:Overall 0.2001 + 4 维(样例报告短弱、vs 强参考,低分合理),runner 自动收集到分数。
- generate 半边:代码复用 `GenerationRunner`、喂原始问题、`REPORT_MODE=detailed_cited`;真实跑需起 vLLM 服务器。
