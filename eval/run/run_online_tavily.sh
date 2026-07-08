#!/bin/bash
# 在 DeepResearchBench 题库上跑 deep_researcher_demo:online 模式 + Tavily 搜索,
# 出 detailed_cited 报告 + 块级 embedding 缓存 + harvest + 逐调用计时。
#
# 用法:  bash eval/run/run_online_tavily.sh [题数N=3] [起始index=1]
# 关键:  **串行 + MAX_CONCURRENCY=1**,避免并发把 DashScope embedding 每分钟限额冲爆(429)。
set -u
cd "$(dirname "$0")/../.." || exit 1          # → 仓库根 deep_researcher_demo/

N="${1:-3}"; START="${2:-1}"
PY="${PY:-/home/yilin/anaconda3/envs/gpt-deep/bin/python}"
OUT="eval/benchmarks/results/tavily_online"
QUERY_FILE="eval/DeepResearchBench/data/prompt_data/query.jsonl"
HARV="INITIAL_RESEARCH_QUESTIONS_JSON,QUERY_PLAN_JSON,RESEARCH_SUMMARY_TEXT,SUPERVISOR_DECISION_JSON,FINAL_REPORT_MARKDOWN"

# --- 凭据 / 模式 ---
set -a; source .env 2>/dev/null; set +a       # JUDGE_API_KEY/JUDGE_BASE_URL → DashScope embedding
: "${TAVILY_API_KEY:?需设 TAVILY_API_KEY(Tavily 搜索 key)}"
export RESEARCH_MODE=online                    # → 缓存 record + cache_relevance(块级 embedding 检索)
export SEARCH_PROVIDER=tavily                  # 搜索引擎=Tavily(同时返回 url+正文,不走 crawl4ai)
export SEARCH_BENCHMARK=drbench                 # 缓存按 <root>/drbench/q<id>/ 分类
export SEARCH_CACHE_DIR=eval/results/search_cache
export REPORT_MODE=detailed_cited              # 详细、带 inline 引用
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:30000/v1}" OPENAI_API_KEY=dummy MODEL="${MODEL:-Qwen3-32B}"
export MAX_ITERATIONS=1 MAX_QUERIES_PER_RESEARCHER=2 MAX_CONCURRENCY=1 MAX_RESULTS=10 LLM_TIMEOUT=900

echo "===== START N=$N from index $START $(date) ====="
for ((i=START; i<START+N; i++)); do
  row=$($PY -c "import json;rows=[json.loads(l) for l in open('$QUERY_FILE') if l.strip()];r=rows[$((i-1))];print(str(r['id'])+chr(9)+r['prompt'])")
  id="${row%%	*}"; prompt="${row#*	}"
  d="$OUT/q$id"; rm -rf "$d" "eval/results/search_cache/drbench/q$id"; mkdir -p "$d"
  echo "===== q$id START $(date) ====="
  SEARCH_CACHE_SAMPLE_ID="$id" \
  DRIFT_HARVEST="$d/harvest.jsonl" DRIFT_HARVEST_TAGS="$HARV" \
  LLM_CALL_LOG="$d/llm_calls.jsonl" \
  timeout 1200 $PY -m deep_researcher_demo --quiet --output "$d/report.md" "$prompt" > "$d/run.log" 2>&1
  echo "q$id done rc=$? report=$(wc -c <"$d/report.md" 2>/dev/null)字 chunks=$(wc -l <eval/results/search_cache/drbench/q$id/chunks.jsonl 2>/dev/null)"
done
echo "===== ALL DONE $(date) ====="
