#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
exec env VLLM_RESP_TIMING=1 /home/yilin/anaconda3/envs/lmcache/bin/vllm serve /data/yilin/huggingface/Qwen3-32B \
  --served-model-name Qwen3-32B --tensor-parallel-size 4 --port 30000 \
  --max-model-len 40960 --gpu-memory-utilization 0.65 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{"method":"eagle3","model":"/data/yilin/huggingface/Qwen3-32B-eagle3-specforge","num_speculative_tokens":3}'
