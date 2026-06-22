# Eval 运行指引(deep_researcher A/B + blend 实验)

> 一份"怎么传参"的实操 runbook。配套更偏背景的 [AB_EXPERIMENT.md](AB_EXPERIMENT.md)。
> 实验分两段:**①起 vLLM 服务(server)②跑 eval 客户端(client)**。两段各有各的 env。
> 黄金法则:**web search 一律走 replay 缓存(零网络),每轮用独立 RUN_ID 目录,量重叠收益时 sync 必须 OFF。**

---

## 0. 前置

- 服务端 conda env `lmcache`(vllm 0.18 + 可编辑 lmcache);客户端 conda env `gpt-deep`。
  **客户端必须用绝对路径 python**:`PATH=/home/yilin/anaconda3/envs/gpt-deep/bin:$PATH`(`conda activate` 在脚本里不切 PATH)。
- 模型:`/data/yilin/huggingface/{Qwen3-32B, Qwen3-30B-A3B-Instruct-2507}`。
- server 配置按模型分文件夹:`/home/yilin/LMCache/server/vllm/<model>/{config.yaml(blend), config_no_lmcache.yaml(native), start.sh}`,端口 30000、TP=4。
- **搜索缓存必须先建库一次**(见 §4),之后所有计时跑都 replay。
- 共享 GPU 机:起服务前确认 ≥4 卡空闲;**kill 前用 `ps -o user=` 验属主=yilin**,杀 TP worker 要用显式 pid(见 §6)。

---

## 1. 服务端(server)env —— 传给 `start.sh`

```bash
# blend(Mode A):
setsid bash -c 'cd /home/yilin/LMCache/server/vllm/Qwen3-32B && \
  CUDA_VISIBLE_DEVICES=3,4,5,6 \
  VLLM_RESP_TIMING=1 \            # 每请求在响应体回 prefill_s/decode_s(逐请求计时,必开)
  PYTHONHASHSEED=0 \             # 跨进程 content-hash 一致(blend 硬前提)
  bash start.sh lmcache > /home/yilin/tmp/<RUN_ID>/<tag>_serve.log 2>&1' < /dev/null &

# native(Mode B):同上但 `bash start.sh no_lmcache`,且不需要 blend 相关 env。
```

| server env | 作用 | 备注 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | 选卡(TP=4 需 4 张) | 工作集 `3,4,5,6` |
| `VLLM_RESP_TIMING=1` | 响应体回逐请求 `prefill_s/decode_s` | **逐请求计时必开** |
| `LMCACHE_BLEND_TIMER_SYNC` | =1 在 blend 每层插 `cudaDeviceSynchronize` | **只为量 load/compute 拆分时开**;量重叠收益/生产态**必须不设(OFF)**,否则会把重叠重新串行化 |
| `LMCACHE_BLEND_SPECIAL_STR` | blend 段分隔符 | start.sh 默认 `<\|fim_pad\|>`;**必须与客户端 `KV_REUSE_SEPARATOR` 一致** |
| `PYTHONHASHSEED=0` | content-hash 跨进程一致 | blend 硬前提 |

就绪探测:`grep -q "Application startup complete" <serve.log>`(32B/TP4 加载约 2-3 分钟)。

---

## 2. 客户端公共 env —— 传给 `eval_deepsearchqa.sh`

```bash
cd /home/yilin/deep_researcher_demo
<这里堆所有 env> \
PATH=/home/yilin/anaconda3/envs/gpt-deep/bin:$PATH \
bash eval_deepsearchqa.sh > /home/yilin/tmp/<RUN_ID>/<tag>_eval.log 2>&1
```

| 公共 env | 取值 | 作用 |
|---|---|---|
| `MODE` | `generate`(/`score`/`all`) | 只生成不判分用 generate |
| `MODEL_OVERRIDE` | `Qwen3-32B` / `Qwen3-30B-A3B-Instruct-2507` | 必须 = server 的 served-model-name |
| `LIMIT` | 整数(实验用 20) | 跑几题 |
| `SAMPLE_CONCURRENCY` | `1` | 样本串行 → 计时干净 |
| `MAX_FOLLOWUPS` `MAX_QUERIES_PER_RESEARCHER` | `2` `2` | 控每样本搜索量,两模式必须同参 |
| `LLM_TIMEOUT` | `300` | 单次 LLM 超时秒(大档长摘要要够大) |
| `LLM_CALL_LOG` | 路径 | **逐请求日志**(tag/req_id/prefill_s/prompt_tokens…),分析的主数据 |
| `QUIET=1 OVERWRITE=1` | | 安静 + 覆盖旧输出 |

---

## 3. 档位 env(big / small)= 控制"复用规模"(prompt 大小)

prompt 大小是**涌现**的:由累积的 researcher 摘要总量决定(摘要拼进 SUPERVISOR/FINAL 的 prompt)。三个旋钮:

| | 大档(big) | 小档(small) |
|---|---|---|
| `SUMMARY_DETAILED` | `1`(指令:detailed/保留所有事实) | 不设(指令:compress/只取关键) |
| `RESEARCH_SUMMARY_MAX_TOKENS` | `4000` | `800` |
| `MAX_ITERATIONS` | `2` | `1` |

实测 prompt:大档 FINAL ~5–20k token、小档 ~0.4–3k。详见 [ab_sync_0614_summary.md](ab_sync_0614_summary.md)。

---

## 4. 搜索缓存 env(必须 replay,零网络)

```bash
# 一次性建库(实时搜一遍,会打 DDG,注意限流;只需做一次):
SEARCH_CACHE=record LIMIT=20 OUTPUT_DIR=eval/results/_cache_build \
  MAX_FOLLOWUPS=2 MAX_QUERIES_PER_RESEARCHER=2 \
  PATH=/home/yilin/anaconda3/envs/gpt-deep/bin:$PATH bash eval_deepsearchqa.sh
```

| 搜索 env | 取值 | 作用 |
|---|---|---|
| `SEARCH_CACHE` | `off`(默认实时) / `record`(实时+落盘) / **`replay`(零网络回放)** | 计时跑一律 `replay` |
| `SEARCH_CACHE_FIX_N` | `3` | replay 时每题固定取前 N 篇文档(A/B 公平、变量固定) |
| `SEARCH_CACHE_DIR` | 默认 `eval/results/search_cache` | 缓存根;**跨 A/B 持久复用,别放进 per-run OUTPUT_DIR** |

> 验证零真实搜索:跑完查 reports.jsonl 的 `search_cache_miss`(全 0 = 全程命中缓存、零回退实时搜)。
> 冷池(没 record 过的题)replay 会回退实时搜并打 `search_cache_miss` 标记 → 必须先建库。

---

## 5. 模式 env(A blend / B native)

| | A = CacheBlend | B = 原生 prefix cache |
|---|---|---|
| server | `start.sh lmcache` | `start.sh no_lmcache` |
| `KV_REUSE_SEPARATOR` | `'<\|fim_pad\|>'`(与 server sep 一致) | **不设** |
| `KV_REUSE_TOKENIZER` | 模型目录(截断摘要也能复用所需) | 不设 |

- `KV_REUSE_SEPARATOR` 非空会自动:① 改 prompt 布局(摘要前置、问题后置,blend 只认从段 0 起的连续命中);
  ② 只对 researcher 摘要 + warmup 请求带 `kv_transfer_params={"lmcache.blend_store_generated":true}`(请求级 opt-in 存生成段)。
- `KV_REUSE_TOKENIZER`:截断摘要(撞 max_tokens)复用所需;每模型对应自己的 tokenizer 目录:
  32B=`/data/yilin/huggingface/Qwen3-32B`、30B=`/data/yilin/huggingface/Qwen3-30B-A3B-Instruct-2507`。

---

## 6. 每轮独立 RUN_ID 目录(铁律,防跨轮污染)

`LLM_CALL_LOG` 是**追加写**、`OUTPUT_DIR`/serve.log 重用会**覆盖** → 同名跨轮会混多轮记录、冲掉 blend 日志。
**每轮新建一个 RUN_ID**,三处都带上:

```
OUTPUT_DIR   = eval/results/<RUN_ID>/<model>_<grade>_<mode>
LLM_CALL_LOG = eval/results/ab_meta/<RUN_ID>/<tag>_calls.jsonl
serve.log    = /home/yilin/tmp/<RUN_ID>/<tag>_serve.log   （eval.log 同理）
```

**服务启停拆成单独命令**:launch 用 `setsid bash -c "..." &`;kill 用显式 pid(`nvidia-smi --query-compute-apps=pid` 取真实占用 pid + 反查 ppid，`ps -o user=` 验 yilin 再 `kill -9`)。**别用 `pkill -f "VLLM::Worker_TP"`——会匹配到 kill 命令自身而自杀**。重启前轮询等 GPU 显存 <3GB。

---

## 7. 完整可复制示例(32B,一轮 = 大档 blend)

```bash
# ① 起服务(blend, sync OFF = 生产/量重叠)
setsid bash -c 'cd /home/yilin/LMCache/server/vllm/Qwen3-32B && \
  CUDA_VISIBLE_DEVICES=3,4,5,6 VLLM_RESP_TIMING=1 PYTHONHASHSEED=0 \
  bash start.sh lmcache > /home/yilin/tmp/RUN_demo/32b_bigA_serve.log 2>&1' < /dev/null &
# 等 "Application startup complete"

# ② 跑 eval(大档 + blend + replay + RUN 目录)
cd /home/yilin/deep_researcher_demo
SUMMARY_DETAILED=1 RESEARCH_SUMMARY_MAX_TOKENS=4000 MAX_ITERATIONS=2 \
KV_REUSE_SEPARATOR='<|fim_pad|>' KV_REUSE_TOKENIZER=/data/yilin/huggingface/Qwen3-32B \
SEARCH_CACHE=replay SEARCH_CACHE_FIX_N=3 \
LLM_TIMEOUT=300 LIMIT=20 MAX_FOLLOWUPS=2 MAX_QUERIES_PER_RESEARCHER=2 SAMPLE_CONCURRENCY=1 \
MODE=generate QUIET=1 OVERWRITE=1 MODEL_OVERRIDE=Qwen3-32B \
OUTPUT_DIR=eval/results/RUN_demo/32b_big_blend \
LLM_CALL_LOG=eval/results/ab_meta/RUN_demo/32b_bigA_calls.jsonl \
PATH=/home/yilin/anaconda3/envs/gpt-deep/bin:$PATH \
bash eval_deepsearchqa.sh > /home/yilin/tmp/RUN_demo/32b_bigA_eval.log 2>&1
```

小档:把 §3 三个旋钮换成 small。native(B):server `no_lmcache`、去掉 `KV_REUSE_*`、OUTPUT_DIR/LLM_CALL_LOG 换 `_native`/`B`。

---

## 8. 分析口径(算 prefill A/B)

逐请求 prefill 用 `LLM_CALL_LOG` 的 `prefill_s`,**两条过滤缺一不可**:
1. **`max_tokens != 8`**:排掉 warmup_kv_prefix 预热请求(同 tag、max_tokens=8;native 不发 warmup → 不排会口径不齐)。
2. **server 日志 `LMCache hit tokens > 0`**(⋈ calls.jsonl `req_id`,服务端 Reqid 多实例后缀去掉即等):A 列只取真正命中 blend 的请求。

脚本范例:`/home/yilin/tmp/RUN_sync_0614/analyze.py`。
**量重叠收益必须 sync OFF + 同会话背靠背只切代码**(跨天有噪声;同代码跨天就有 ~12% 漂移)。
端到端延迟受 decode(~90%)+ 输出长度漂移混淆,只作趋势;干净归因看 prefill。
