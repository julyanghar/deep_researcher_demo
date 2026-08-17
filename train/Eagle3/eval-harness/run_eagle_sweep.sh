#!/bin/bash
# eagle3 成本解剖扫描:k=1/2/5(斜率-截距)+ k3-eager(边界税)+ k3-timing(分段账本)。
# k=3 cudagraph 干净数已有(speedup_eagle3_redhat)。RedHat head,TP4/0-3,util0.90。
set -u
RUN=/home/yilin/modify-code-runs/eagle3-test
DEMO=/home/yilin/deep_researcher_demo
VLLM=/home/yilin/anaconda3/envs/lmcache/bin/vllm
PY=/home/yilin/anaconda3/envs/lmcache/bin/python
HEAD=/data/yilin/huggingface/Qwen3-32B-eagle3
ST=$RUN/sweep_status.txt
: > $ST
echo "== EAGLE SWEEP start $(date) ==" >> $ST

kill_srv(){ ps -eo pid,cmd|grep -iE 'vllm serve|EngineCore|VLLM::Worker'|grep -v grep|grep -v run_eagle_sweep|awk '{print $1}'|xargs -r kill -9 2>/dev/null; sleep 12; }

arm(){ local LABEL=$1 K=$2 EXTRA=$3 TIMING=$4
  kill_srv
  echo "-- launch $LABEL (k=$K extra='$EXTRA' timing=$TIMING) $(date)" >> $ST
  CUDA_VISIBLE_DEVICES=0,1,2,3 env VLLM_RESP_TIMING=1 ${TIMING:+EAGLE_TIMING=1} $VLLM serve /data/yilin/huggingface/Qwen3-32B \
    --served-model-name Qwen3-32B --tensor-parallel-size 4 --port 30000 \
    --max-model-len 40960 --gpu-memory-utilization 0.90 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --speculative-config "{\"method\":\"eagle3\",\"model\":\"$HEAD\",\"num_speculative_tokens\":$K}" \
    $EXTRA > $RUN/server_$LABEL.log 2>&1 &
  local ok=0
  for i in $(seq 1 150); do
    curl -s -m3 http://localhost:30000/v1/models 2>/dev/null|grep -q Qwen3-32B && { ok=1; break; }
    grep -qiE "EngineCore failed|ValueError|OutOfMemory|CUDA error" $RUN/server_$LABEL.log && break
    sleep 5
  done
  [ $ok -eq 1 ] || { echo "[$LABEL] NOT_READY $(date)" >> $ST; grep -iE "Error|OutOfMemory"; tail -3 $RUN/server_$LABEL.log >> $ST; exit 1; }
  grep -q "num_spec_tokens=$K" $RUN/server_$LABEL.log && echo "[$LABEL] ASSERT k=$K OK" >> $ST || echo "[$LABEL] !!ASSERT k FAIL" >> $ST
  if [ -n "$EXTRA" ]; then grep -q "enforce_eager=True" $RUN/server_$LABEL.log && echo "[$LABEL] ASSERT eager OK" >> $ST || echo "[$LABEL] !!ASSERT eager FAIL" >> $ST; fi
  ( cd $DEMO && $PY $RUN/measure_speedup.py 30000 $LABEL 1 > $RUN/speedup_$LABEL.log 2>&1 )
  local tps=$(grep -A2 "report 照抄" $RUN/speedup_$LABEL.log | grep "decode tok/s" | head -1)
  echo "[$LABEL] DONE $(date) $tps" >> $ST
  if [ -n "$TIMING" ]; then grep -c "EAGLE_TIMING n=" $RUN/server_$LABEL.log | xargs -I{} echo "[$LABEL] timing行={}" >> $ST; fi
}

arm k1       1 ""                ""
arm k2       2 ""                ""
arm k5       5 ""                ""
arm k3eager  3 "--enforce-eager" ""
arm k3timing 3 ""                "1"
kill_srv
echo "SWEEP_DONE $(date)" >> $ST
