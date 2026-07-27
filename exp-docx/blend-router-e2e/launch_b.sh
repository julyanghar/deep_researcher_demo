#!/bin/bash
# B 臂 server:LMCache blend(attn-guided)+ APC + router 投机
# 用法:CUDA_VISIBLE_DEVICES=1,3,4,5 ./launch_b.sh > /home/yilin/tmp/logs/server_b.log 2>&1 &
# 冒烟时前置 MCPROUTE_DBG=1;正式跑必须去掉(污染 TPOT)
set -e
: "${CUDA_VISIBLE_DEVICES:?必须显式传 CUDA_VISIBLE_DEVICES(严禁含 GPU6)}"
case ",$CUDA_VISIBLE_DEVICES," in *,6,*) echo "拒绝:含 GPU6(训练卡)" >&2; exit 1;; esac
export PYTHONHASHSEED=0
export VLLM_RESP_TIMING=1
export LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=true LMCACHE_MAX_LOCAL_CPU_SIZE=16
export LMCACHE_ENABLE_BLENDING=true LMCACHE_USE_LAYERWISE=true
export LMCACHE_SAVE_UNFULL_CHUNK=true LMCACHE_SAVE_DECODE_CACHE=false
export LMCACHE_BLEND_SPECIAL_STR='<|fim_pad|>'   # 必须与 driver 的 KV_REUSE_SEPARATOR 逐字节一致
export LMCACHE_BLEND_CHECK_LAYERS=1
export LMCACHE_BLEND_ATTN_GUIDED=true            # attention-guided 修正(用户拍板)
export LMCACHE_BLEND_RECOMPUTE_RATIOS=0.2        # attn-guided 语义 = decode 修正预算 B(用户拍板)
export VLLM_DEFAULT_CACHE_SALT=cacheblend        # APC 兼容硬前提
export SUFFIX_TRAJ=${SUFFIX_TRAJ:-/home/yilin/tmp/blend_e2e/traj_b}
CONFIG=${CONFIG:-/home/yilin/deep_researcher_demo/exp-docx/blend-router-e2e/config-b-blend-router.yaml}
exec /home/yilin/anaconda3/envs/lmcache/bin/vllm serve --config "$CONFIG"
