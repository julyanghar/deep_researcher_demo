#!/bin/bash
export CUDA_VISIBLE_DEVICES=5,7
exec /home/yilin/anaconda3/envs/lmcache/bin/vllm serve /data/yilin/huggingface/Qwen3-32B \
  --served-model-name Qwen3-32B \
  --tensor-parallel-size 2 \
  --port 30001 \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.85 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{"method": "suffix"}'
