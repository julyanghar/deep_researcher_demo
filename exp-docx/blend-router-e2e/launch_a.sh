#!/bin/bash
# A 臂 server:纯 vanilla vLLM(无 LMCache env、无投机、默认 cudagraph)
# 用法:CUDA_VISIBLE_DEVICES=1,3,4,5 ./launch_a.sh > /home/yilin/tmp/logs/server_a.log 2>&1 &
set -e
: "${CUDA_VISIBLE_DEVICES:?必须显式传 CUDA_VISIBLE_DEVICES(严禁含 GPU6)}"
case ",$CUDA_VISIBLE_DEVICES," in *,6,*) echo "拒绝:含 GPU6(训练卡)" >&2; exit 1;; esac
export PYTHONHASHSEED=0
export VLLM_RESP_TIMING=1
CONFIG=${CONFIG:-/home/yilin/LMCache/server/vllm/config_no_lmcache.yaml}
exec /home/yilin/anaconda3/envs/lmcache/bin/vllm serve --config "$CONFIG"
