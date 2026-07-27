#!/bin/bash
# DRGym 英文 40 题对照:用法 ./run_arm_gym.sh a|b [N] [TAG]
# 配置与 run_arm.sh 完全一致,仅题库换 drgym(英文)
set -e
ARM=${1:?用法: run_arm_gym.sh a|b [N] [TAG] [QUERIES]}
N=${2:-40}
if [ "$ARM" = a ]; then DEFTAG=a_van40_gym; else DEFTAG=b_blend40_gym; fi
TAG=${3:-$DEFTAG}
QUERIES=${4:-}   # 可选题库文件(如 gym160_cached_notrun.jsonl);缺省用 run_drgym 默认题库
cd /home/yilin/deep_researcher_demo
COMMON=(
  RESEARCH_MODE=local SEARCH_BENCHMARK=drgym
  MAX_ITERATIONS=3 MAX_FOLLOWUPS=3 MAX_QUERIES_PER_RESEARCHER=3 MAX_CONCURRENCY=3
  REPORT_MODE=detailed_cited LLM_TIMEOUT=900
  OPENAI_BASE_URL=http://localhost:30000/v1 MODEL=Qwen3-32B
)
if [ "$ARM" = b ]; then
  EXTRA=(
    REPORT_PARALLEL=1
    REPORT_REVIEW=1
    PROPOSER_ROUTING=1
    KV_REUSE_SEPARATOR='<|fim_pad|>'
    FINAL_KV_REUSE_SEPARATOR=
    KV_REUSE_TOKENIZER=/data/yilin/huggingface/Qwen3-32B
  )
else
  EXTRA=()
fi
QARG=()
[ -n "$QUERIES" ] && QARG=(--queries "$QUERIES")
exec env "${COMMON[@]}" "${EXTRA[@]}" \
  /home/yilin/anaconda3/envs/gpt-deep/bin/python eval/DeepResearchGym/run_drgym.py \
  --tag "$TAG" --mode generate --harvest --n "$N" --concurrency 1 \
  --base-url http://localhost:30000/v1 "${QARG[@]}"
