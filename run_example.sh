#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "$#" -eq 0 ]]; then
  set -- "What are the main tradeoffs of local LLM inference?"
fi

python3 -m deep_researcher_demo "$@"
