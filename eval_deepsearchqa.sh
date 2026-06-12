#!/usr/bin/env bash
set -euo pipefail

# DeepSearchQA evaluation launcher.
#
# The Python eval command loads /home/yilin/deep_researcher_demo/.env by default.
# This script does not source .env; it only builds command-line arguments.
#
# Common usage:
#   ./eval_deepsearchqa.sh
#   LIMIT=100 OUTPUT_DIR=eval/results/deepsearchqa_100 ./eval_deepsearchqa.sh
#   MODE=generate LIMIT=50 ./eval_deepsearchqa.sh
#   MODE=score LIMIT=50 RESUME=0 OVERWRITE=1 ./eval_deepsearchqa.sh
#   MODE=all LIMIT=50 ./eval_deepsearchqa.sh
#   RESUME=1 OVERWRITE=0 ./eval_deepsearchqa.sh
#   ./eval_deepsearchqa.sh --category Science --judge-model your-local-judge-model
#
# Script environment variables:
#   OUTPUT_DIR:
#     Directory for eval outputs. The command writes reports.jsonl,
#     predictions.jsonl, metrics.json, and failures.jsonl here.
#     Default: eval/results/deepsearchqa_10
#
#   MODE:
#     Eval phase to run. Use generate to only write reports.jsonl, score to
#     rate existing reports.jsonl, or all to run both phases.
#     Default: all
#
#   LIMIT:
#     Number of DeepSearchQA examples to evaluate. Set LIMIT="" to run the
#     whole eval split.
#     Default: 50
#
#   START:
#     Dataset index to start from when IDS is not set.
#     Default: 0
#
#   IDS:
#     Comma-separated dataset sample ids/indexes to run, for example IDS=4,10,23.
#     When set, IDS takes priority over START and LIMIT.
#     Default: empty
#
#   CATEGORY:
#     Optional problem_category filter, for example CATEGORY=Science.
#     Default: empty
#
#   SAMPLE_CONCURRENCY:
#     Number of benchmark samples to generate/score concurrently.
#     Keep this low if your local LLM endpoint has limited throughput.
#     Default: 1
#
#   RESUME:
#     If 1, skip reports that already succeeded in reports.jsonl, then rerun
#     the unified scoring phase.
#     Default: 0
#
#   OVERWRITE:
#     If 1, delete previous output files in OUTPUT_DIR before running.
#     Must not be 1 together with RESUME=1.
#     Default: 1
#
#   QUIET:
#     If 1, hide deep researcher internal progress logs. Benchmark-level
#     progress still prints.
#     Default: 1
#
#   ENV_FILE:
#     Optional env file path passed to --env-file. If empty, Python loads the
#     repo-level .env by default.
#     Default: empty
#
#   JUDGE_MODEL_ARG:
#     Optional explicit --judge-model value. If empty, the eval code resolves
#     judge model as JUDGE_MODEL from .env/env, then deepseek-v3.2.
#     Default: empty
#
#   JUDGE_BASE_URL_ARG:
#     Optional explicit --judge-base-url value for the autorater endpoint.
#     If empty, JUDGE_BASE_URL from .env/env is used, defaulting to DashScope
#     China OpenAI-compatible mode.
#     Default: empty
#
#   JUDGE_API_KEY_ARG:
#     Optional explicit --judge-api-key value for the autorater endpoint.
#     If empty, JUDGE_API_KEY from .env/env is used. Required for MODE=score/all.
#     Default: empty
#
#   MODEL_OVERRIDE:
#     Optional --model value. Overrides supervisor/researcher/summary/final
#     model roles for the researcher workflow.
#     Default: empty
#
#   BASE_URL:
#     Optional --base-url value for an OpenAI-compatible endpoint. If empty,
#     OPENAI_BASE_URL from .env/env is used.
#     Default: empty
#
#   SEARCH_PROVIDER:
#     Optional --search-provider override. Supported values: duckduckgo, tavily.
#     Tavily requires TAVILY_API_KEY in .env/env.
#     Default: empty
#
#   MAX_ITERATIONS:
#     Optional --max-iterations override for the deep researcher loop.
#     Default: empty, meaning use .env/env config.
#
#   MAX_FOLLOWUPS:
#     Optional --max-followups override; max follow-up questions supervisor can
#     request per iteration.
#     Default: empty, meaning use .env/env config.
#
#   MAX_QUERIES_PER_RESEARCHER:
#     Optional --max-queries-per-researcher override; max search queries each
#     researcher can generate from one research sub-question.
#     Default: empty, meaning use .env/env config.
#
#   MAX_CONCURRENCY:
#     Optional --max-concurrency override inside one research workflow.
#     This is different from SAMPLE_CONCURRENCY, which is benchmark sample-level.
#     Default: empty, meaning use .env/env config.
#
#   MAX_RESULTS:
#     Optional --max-results override; max search results per generated query.
#     Default: empty, meaning use .env/env config.
#
#   FETCH_WEBPAGES / MAX_CONTENT_CHARS / FETCH_TIMEOUT / FETCH_CONCURRENCY:
#     DuckDuckGo page-fetch settings are read by Python from .env/env. Export
#     them before running this script if you want to override .env.
#
# Any extra arguments passed to this script are appended to the Python command
# and can override the defaults above, for example:
#   ./eval_deepsearchqa.sh --limit 5 --category Geography

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-eval/results/deepsearchqa_10_test_lmcache}"
MODE="${MODE:-generate}"
# score all generate
LIMIT="${LIMIT:-10}"
START="${START:-0}"
IDS="${IDS:-}"
CATEGORY="${CATEGORY:-}"
SAMPLE_CONCURRENCY="${SAMPLE_CONCURRENCY:-1}"
RESUME="${RESUME:-0}"
OVERWRITE="${OVERWRITE:-1}"
QUIET="${QUIET:-1}"
ENV_FILE="${ENV_FILE:-}"
JUDGE_MODEL_ARG="${JUDGE_MODEL_ARG:-}"
JUDGE_BASE_URL_ARG="${JUDGE_BASE_URL_ARG:-}"
JUDGE_API_KEY_ARG="${JUDGE_API_KEY_ARG:-}"
MODEL_OVERRIDE="${MODEL_OVERRIDE:-}"
BASE_URL="${BASE_URL:-}"
SEARCH_PROVIDER="${SEARCH_PROVIDER:-}"
MAX_ITERATIONS="${MAX_ITERATIONS:-}"
MAX_FOLLOWUPS="${MAX_FOLLOWUPS:-}"
MAX_QUERIES_PER_RESEARCHER="${MAX_QUERIES_PER_RESEARCHER:-}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-}"
MAX_RESULTS="${MAX_RESULTS:-}"

if [[ "$RESUME" == "1" && "$OVERWRITE" == "1" ]]; then
  echo "RESUME=1 and OVERWRITE=1 are mutually exclusive. Use OVERWRITE=0 with RESUME=1." >&2
  exit 2
fi

cmd=(
  python3 -m eval.run_deepsearchqa
  --mode "$MODE"
  --output-dir "$OUTPUT_DIR"
  --start "$START"
  --sample-concurrency "$SAMPLE_CONCURRENCY"
)

if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit "$LIMIT")
fi
if [[ -n "$IDS" ]]; then
  cmd+=(--ids "$IDS")
fi
if [[ -n "$CATEGORY" ]]; then
  cmd+=(--category "$CATEGORY")
fi
if [[ "$RESUME" == "1" ]]; then
  cmd+=(--resume)
elif [[ "$OVERWRITE" == "1" ]]; then
  cmd+=(--overwrite)
fi
if [[ "$QUIET" == "1" ]]; then
  cmd+=(--quiet)
fi
if [[ -n "$ENV_FILE" ]]; then
  cmd+=(--env-file "$ENV_FILE")
fi
if [[ -n "$JUDGE_MODEL_ARG" ]]; then
  cmd+=(--judge-model "$JUDGE_MODEL_ARG")
fi
if [[ -n "$JUDGE_BASE_URL_ARG" ]]; then
  cmd+=(--judge-base-url "$JUDGE_BASE_URL_ARG")
fi
if [[ -n "$JUDGE_API_KEY_ARG" ]]; then
  cmd+=(--judge-api-key "$JUDGE_API_KEY_ARG")
fi
if [[ -n "$MODEL_OVERRIDE" ]]; then
  cmd+=(--model "$MODEL_OVERRIDE")
fi
if [[ -n "$BASE_URL" ]]; then
  cmd+=(--base-url "$BASE_URL")
fi
if [[ -n "$SEARCH_PROVIDER" ]]; then
  cmd+=(--search-provider "$SEARCH_PROVIDER")
fi
if [[ -n "$MAX_ITERATIONS" ]]; then
  cmd+=(--max-iterations "$MAX_ITERATIONS")
fi
if [[ -n "$MAX_FOLLOWUPS" ]]; then
  cmd+=(--max-followups "$MAX_FOLLOWUPS")
fi
if [[ -n "$MAX_QUERIES_PER_RESEARCHER" ]]; then
  cmd+=(--max-queries-per-researcher "$MAX_QUERIES_PER_RESEARCHER")
fi
if [[ -n "$MAX_CONCURRENCY" ]]; then
  cmd+=(--max-concurrency "$MAX_CONCURRENCY")
fi
if [[ -n "$MAX_RESULTS" ]]; then
  cmd+=(--max-results "$MAX_RESULTS")
fi

cmd+=("$@")

display_cmd=("${cmd[@]}")
for index in "${!display_cmd[@]}"; do
  if [[ "${display_cmd[$index]}" == "--judge-api-key" && $((index + 1)) -lt ${#display_cmd[@]} ]]; then
    display_cmd[$((index + 1))]="<redacted>"
  fi
done

printf '[eval_command]'
printf ' %q' "${display_cmd[@]}"
printf '\n'

"${cmd[@]}"
