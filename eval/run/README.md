# eval/run —— 实跑脚本(online + Tavily 等)

本目录放**端到端实跑**的脚本与步骤说明(区别于 `eval/benchmarks/` 的打分 runner)。

## run_online_tavily.sh —— DeepResearchBench × online × Tavily
在 DeepResearchBench 题库上跑 demo:**online 模式 + Tavily 搜索**,出详细报告 + 块级 embedding 缓存。

```bash
bash eval/run/run_online_tavily.sh [题数N=3] [起始index=1]
# 例:前 3 题
bash eval/run/run_online_tavily.sh 3 1
```

### 前置
- 本地 vLLM 在跑(默认 `OPENAI_BASE_URL=http://localhost:30000/v1`,served `Qwen3-32B`)。
- `TAVILY_API_KEY` 已设(Tavily 搜索 key)。脚本未内置 key,运行前 `export TAVILY_API_KEY=...`。
- `.env` 里 `JUDGE_API_KEY`/`JUDGE_BASE_URL`(DashScope)—— online 建块索引的 **embedding**(text-embedding-v3)用它。

### 关键环境(脚本里已设)
| env | 值 | 作用 |
|---|---|---|
| `RESEARCH_MODE` | online | 缓存 record + cache_relevance(块级 embedding 检索)|
| `SEARCH_PROVIDER` | tavily | 搜索引擎=Tavily(同时返回 url+正文,**不走 crawl4ai**)|
| `SEARCH_BENCHMARK` | drbench | 缓存按 `<root>/drbench/q<id>/` 分类 |
| `REPORT_MODE` | detailed_cited | 详细、带 inline 引用的报告 |
| `MAX_CONCURRENCY` | **1** | **串行,避 embedding 限流(见下)** |

### ⚠️ 为什么串行 / 低并发(踩坑记录)
首次用 **3 题并发 + 每题 MAX_CONCURRENCY=3** 跑 → DashScope embedding **每分钟限额被冲爆**:
`openai.RateLimitError: 429 insufficient_quota`(online 每个子查询都要 embed 建块索引,9 路并发瞬间超额)。
**单次 embedding 调用其实正常 → 是限流不是额度耗尽。** 改 **串行 + `MAX_CONCURRENCY=1`** 后顺畅。
> 想恢复高并发:给 `deep_researcher_demo/relevance.py` 的 `_embed` 加 429 重试/退避(代码改动,走 /modify-code),或换更高额度的 embedding key。

### 产物
**实验结果** `eval/benchmarks/results/tavily_online/q<id>/`:
- `report.md`(detailed_cited 报告)、`harvest.jsonl`(5 类调用全链)、`llm_calls.jsonl`(逐调用 prefill/decode 计时)、`run.log`。

**缓存结果**(online 建的)`eval/results/search_cache/drbench/q<id>/`:
- `chunks.jsonl`(块级 embedding 索引 {url,ci,text,emb(1024)})、`pages/<hash>.txt`(Tavily 正文)、`pages_index.json`、`search_cache.json`。

### 看结果
```bash
ls -la eval/benchmarks/results/tavily_online/q*/report.md
for q in 1 2 3; do echo "q$q: chunks=$(wc -l <eval/results/search_cache/drbench/q$q/chunks.jsonl 2>/dev/null) pages=$(ls eval/results/search_cache/drbench/q$q/pages 2>/dev/null|wc -l)"; done
```

## run_drbench_online_100.sh —— DeepResearchBench 100 题 × online × Tavily(批量)
全量 100 题,**复用 `run_drbench.py --harvest`**(并发池 + 断点续 + raw_data 汇总 + per-题 harvest/计时)。

```bash
export TAVILY_API_KEY=tvly-dev-...
bash eval/run/run_drbench_online_100.sh [题数N=100] [题级并发=3] [tag=online100]
# 例:全量 100、并发 3
bash eval/run/run_drbench_online_100.sh 100 3 online100
```
- **断点续**:中断后**重跑同命令** → `report.md` 已存的题自动 skip,只补没跑完的。
- env 见脚本(online + tavily + benchmark=drbench + MAX_RESULTS=6 + 每题内部 MAX_CONCURRENCY(见脚本))。题级并发由第 2 个参数控制。
- 产物:报告/harvest/计时在 `eval/results/drbench/<tag>/q<id>/`;缓存在 `eval/results/search_cache/drbench/q<id>/`。
- **风险**:Tavily dev key 额度(100 题 ~400-600 次搜索可能超)→ 失败题进 `q<id>/error.txt`、可断点续;真超了改 `SEARCH_PROVIDER=duckduckgo` 续跑剩余题。embedding 限流已有 429 退避重试兜底。

## 相关
- 搜索机制/调用链:`deep_researcher_demo/claude-docx/search-call-chain.md`、`research-walkthrough.md`、`research-modes-and-search.md`。
- 打分(RACE/FACT 等):`eval/benchmarks/`。
