#!/bin/bash
# 双臂 driver:用法 ./run_arm.sh a|b [N] [TAG]
# A 臂:原始 prompt、无路由、report 单次(detailed_cited)
# B 臂:blend prompt(writer 置空)、PROPOSER_ROUTING、report 分节
set -e
ARM=${1:?用法: run_arm.sh a|b [N] [TAG] [OFFSET]}
N=${2:-40}
if [ "$ARM" = a ]; then DEFTAG=a_van40; else DEFTAG=b_blend40; fi
TAG=${3:-$DEFTAG}
OFFSET=${4:-0}   # 剩余60题: run_arm.sh a 100 a_van60 40 → q41..q100
cd /home/yilin/deep_researcher_demo
COMMON=(
  RESEARCH_MODE=local SEARCH_BENCHMARK=drbench
  MAX_ITERATIONS=3 MAX_FOLLOWUPS=3 MAX_QUERIES_PER_RESEARCHER=3 MAX_CONCURRENCY=3
  REPORT_MODE=detailed_cited
  OPENAI_BASE_URL=http://localhost:30000/v1 MODEL=Qwen3-32B
)
if [ "$ARM" = b ]; then
  EXTRA=(
    REPORT_PARALLEL=1
    REPORT_REVIEW=1      # 分节后审阅+补节(排序/补引言结论),用户拍板进 B 臂
    PROPOSER_ROUTING=1   # 只 RESEARCH_SUMMARY_TEXT→eagle3 域训头,其余→suffix(llm.py 默认即此)
    KV_REUSE_SEPARATOR='<|fim_pad|>'
    FINAL_KV_REUSE_SEPARATOR=   # 空=writer 全量 prefill(getenv 空串不回退,已核 config.py:109)
    KV_REUSE_TOKENIZER=/data/yilin/huggingface/Qwen3-32B
  )
else
  EXTRA=()
fi
exec env "${COMMON[@]}" "${EXTRA[@]}" \
  /home/yilin/anaconda3/envs/gpt-deep/bin/python eval/DeepResearchBench/run_drbench.py \
  --tag "$TAG" --mode generate --harvest --n "$N" --offset "$OFFSET" --concurrency 1 \
  --base-url http://localhost:30000/v1
