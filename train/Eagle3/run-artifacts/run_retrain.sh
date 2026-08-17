#!/bin/bash
# 二期重训:TP4@12288 + 双裁剪;$1=preflight|full
cd /home/yilin/SpecForge
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.85
RUN=/home/yilin/modify-code-runs/eagle3-trim
if [ "$1" = "preflight" ]; then
  export MCTRIM_DBG=1 MCTRIM_SELFCHECK=1
  EXTRA="--max-num-steps 2"
  LOG=$RUN/retrain_preflight.log
  OUT=$RUN/preflight_out
else
  EXTRA=""
  LOG=$RUN/retrain.log
  OUT=/home/yilin/deep_researcher_demo/train/Eagle3/runs/qwen3-32b-summary-12k-trim-r1
fi
/home/yilin/anaconda3/envs/specforge/bin/torchrun --nproc_per_node=4 --master_port=29640 scripts/train_eagle3.py \
  --target-model-path /data/yilin/huggingface/Qwen3-32B \
  --ckpt-dir /data/yilin/huggingface/Qwen3-32B-eagle3-angelslim \
  --train-data-path $RUN/train_main12k_3a1078d2.jsonl \
  --chat-template qwen3-instruct \
  --target-model-backend sglang --tp-size 4 --sglang-mem-fraction-static 0.44 \
  --max-length 12288 --ttt-length 3 \
  --attention-backend flex_attention \
  --shard-target-output \
  --trim-loss-positions --trim-prompt-rows \
  --learning-rate 2e-5 --num-epochs 3 --warmup-ratio 0.05 --max-grad-norm 0.5 \
  --draft-accumulation-steps 16 --save-interval 120 --resume \
  $EXTRA \
  --cache-dir $RUN/cache_retrain12k_3a1078d2 \
  --output-dir $OUT > $LOG 2>&1
echo "RETRAIN_DONE_$1 exit=$?" >> $LOG
