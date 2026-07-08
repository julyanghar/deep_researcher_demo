#!/bin/bash
# DeepResearchBench 批量跑:online 模式 + Tavily,出 detailed_cited 报告 + harvest + 计时 + 块级缓存。
# 复用 eval/DeepResearchBench/run_drbench.py --harvest(并发池 + 断点续 + raw_data 汇总)。
#
# 用法:  bash eval/run/run_drbench_online_100.sh [题数N=100] [题级并发=3] [tag=online100]
# 断点续:中断后重跑同命令 → report.md 已存的题自动 skip。
set -u
cd "$(dirname "$0")/../.." || exit 1          # → 仓库根 deep_researcher_demo/

N="${1:-100}"; CONC="${2:-3}"; TAG="${3:-online100}"
PY="${PY:-/home/yilin/anaconda3/envs/gpt-deep/bin/python}"

# --- 凭据 / 模式(子进程 harvest_gen 会继承这些)---
set -a; source .env 2>/dev/null; set +a       # JUDGE_API_KEY/JUDGE_BASE_URL → DashScope embedding
: "${TAVILY_API_KEY:?需先 export TAVILY_API_KEY(Tavily 搜索 key)}"
# 备用 key(主 key 额度超了 export TAVILY_API_KEY=<其一> 重跑同命令断点续):
#   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
#   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
#   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
export RESEARCH_MODE=online                    # → record + cache_relevance(块级 embedding 检索)
export SEARCH_PROVIDER=tavily                  # 搜索引擎=Tavily(同时返回 url+正文)
export SEARCH_BENCHMARK=drbench                 # 缓存按 <root>/drbench/q<id>/ 分类
export MAX_RESULTS=6                            # 每子查询最多 6 个 url
# 每题内部低负载(题级并发已 =CONC,内部别再放大 embedding/搜索 burst):
export MAX_ITERATIONS=3 MAX_QUERIES_PER_RESEARCHER=3 MAX_CONCURRENCY=3

echo "===== DRBench online+tavily | N=$N concurrency=$CONC tag=$TAG | $(date) ====="
$PY eval/DeepResearchBench/run_drbench.py --tag "$TAG" --mode generate --harvest \
    --n "$N" --concurrency "$CONC" --model Qwen3-32B --base-url http://localhost:30000/v1
echo "===== DONE $(date) ====="
