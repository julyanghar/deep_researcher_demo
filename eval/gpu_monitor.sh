#!/usr/bin/env bash
# GPU 占用监测器:10s 采样全部 GPU 的利用率/显存/功耗,追加到 CSV。
# 配合 A/B 实验使用,事后可与 reports.jsonl 的 ISO 时间戳对齐分析。
#
# 用法:
#   ./gpu_monitor.sh start [interval_seconds]   # 启动(默认 10s 采样)
#   ./gpu_monitor.sh stop                       # 停止
#   ./gpu_monitor.sh status                     # 查看状态与最近采样
#   ./gpu_monitor.sh marker <text>              # 追加阶段标记行(如 modeA_start)
#
# 环境变量:
#   GPU_MONITOR_CSV  输出文件(默认 eval/results/ab_meta/gpu_monitor.csv)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV="${GPU_MONITOR_CSV:-$SCRIPT_DIR/results/ab_meta/gpu_monitor.csv}"
PID_FILE="${CSV}.pid"

cmd="${1:-status}"

is_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

case "$cmd" in
  start)
    interval="${2:-10}"
    if is_running; then
      echo "already running (pid $(cat "$PID_FILE")), csv: $CSV"
      exit 0
    fi
    mkdir -p "$(dirname "$CSV")"
    if [[ ! -s "$CSV" ]]; then
      echo "timestamp, gpu_index, utilization_gpu_pct, memory_used_mib, power_w" > "$CSV"
    fi
    echo "# MONITOR_START $(date -u +%Y-%m-%dT%H:%M:%SZ) interval=${interval}s" >> "$CSV"
    nohup nvidia-smi \
      --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
      --format=csv,noheader -l "$interval" >> "$CSV" 2>/dev/null &
    echo $! > "$PID_FILE"
    echo "started (pid $!), interval=${interval}s, csv: $CSV"
    ;;
  stop)
    if is_running; then
      kill "$(cat "$PID_FILE")" && rm -f "$PID_FILE"
      echo "# MONITOR_STOP $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$CSV"
      echo "stopped"
    else
      rm -f "$PID_FILE"
      # 兜底:清理任何游离的同类进程
      STRAY=$(pgrep -f "[n]vidia-smi.*query-gpu=timestamp" || true)
      if [[ -n "$STRAY" ]]; then kill $STRAY && echo "killed stray: $STRAY"; else echo "not running"; fi
    fi
    ;;
  status)
    if is_running; then
      echo "running (pid $(cat "$PID_FILE")), csv: $CSV ($(grep -vc '^#' "$CSV" 2>/dev/null || echo 0) rows)"
      tail -3 "$CSV"
    else
      echo "not running, csv: $CSV"
    fi
    ;;
  marker)
    shift || true
    text="${*:-mark}"
    echo "# PHASE_MARKER $(date -u +%Y-%m-%dT%H:%M:%SZ) ${text}" >> "$CSV"
    echo "marker appended: $text"
    ;;
  *)
    echo "Usage: $0 {start [interval]|stop|status|marker <text>}" >&2
    exit 2
    ;;
esac
