#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
exec env VLLM_RESP_TIMING=1 /home/yilin/anaconda3/envs/lmcache/bin/vllm serve \
  --config /home/yilin/LMCache/server/vllm/Qwen3-32B/config_no_lmcache.yaml \
  --tensor-parallel-size 4 --gpu-memory-utilization 0.80
