#!/bin/bash
# DRQA record 预跑:200 题联网建搜索缓存(Tavily),7 块换 key,SKIP_FINAL_REPORT 省报告尾。
# 不进对比计时;A/B 两臂之后用 RESEARCH_MODE=local 回放本缓存。
# 用法: ./run_drqa_record.sh  (A server 需在 30000 端口)
set -e
KEYS_FILE=/home/yilin/tmp/blend_e2e/tavily_keys.txt
mapfile -t KEYS < <(grep "^tvly" "$KEYS_FILE")
[ ${#KEYS[@]} -ge 7 ] || { echo "健康 key 不足 7 把" >&2; exit 1; }
cd /home/yilin/deep_researcher_demo
source /home/yilin/tmp/blend_e2e/drqa_embed.env

# 7 块:6×29 + 1×26 = 200
OFFSETS=(0 29 58 87 116 145 174)
NS=(29 29 29 29 29 29 26)
for i in "${!OFFSETS[@]}"; do
  off=${OFFSETS[$i]}; n=${NS[$i]}; key=${KEYS[$i]}
  echo "=== record 块 $((i+1))/7: q${off}..q$((off+n-1)) key=...${key: -8} ==="
  env RESEARCH_MODE=online SEARCH_BENCHMARK=drqa \
    SEARCH_PROVIDER=tavily TAVILY_API_KEY="$key" \
    SKIP_FINAL_REPORT=1 \
    MAX_ITERATIONS=3 MAX_FOLLOWUPS=3 MAX_QUERIES_PER_RESEARCHER=3 MAX_CONCURRENCY=3 \
    OPENAI_BASE_URL=http://localhost:30000/v1 MODEL=Qwen3-32B \
    /home/yilin/anaconda3/envs/gpt-deep/bin/python eval/deepresearchqa/run_drqa.py \
    --tag drqa_record --n "$n" --offset "$off" --concurrency 1 \
    --base-url http://localhost:30000/v1
  # 块后体检:该块缓存目录数应=n;Tavily 错误(配额/4xx)出现则停
  got=$(ls -d eval/results/search_cache/drqa/q* 2>/dev/null | wc -l)
  echo "  块 $((i+1)) 完成,累计缓存题目录: $got"
done
echo RECORD_ALL_DONE
