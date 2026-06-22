# CacheBlend(decode-KV 复用)vs 原生 vLLM Prefix Cache — A/B 对比实验复现指南

> 对比 deep_researcher_demo 在两种推理后端下跑 DeepSearchQA(50 题)的
> **效率(端到端延迟)** 与 **报告质量(qwen-flash 自动判分)**。
>
> - **Mode A(CacheBlend)**:vLLM + LMCache blend + `blend_store_generated`
>   (decode 生成的 researcher 摘要 KV 在 supervisor 决策/写报告时复用)
> - **Mode B(原生)**:纯 vLLM + 自带 prefix caching(默认配置)

---

## 0. 前置条件

| 项 | 值 |
|---|---|
| 模型 | `/data/yilin/huggingface/Qwen3-30B-A3B-Instruct-2507`(MoE,TP=4) |
| 服务环境 | conda env `lmcache`(vLLM 0.18.0 + LMCache 可编辑安装于 `/home/yilin/LMCache`) |
| demo/eval 环境 | conda env `gpt-deep`(`deep-researcher-demo` 可编辑安装) |
| GPU | 4 张空闲卡(本实验用 3,4,5,6;先 `nvidia-smi` 确认空闲) |
| 判分 | DashScope `qwen-flash`(`.env` 里的 `JUDGE_API_KEY/JUDGE_MODEL`) |
| 搜索 | duckduckgo(`ddgs` 库,已内置节流+重试,见 §6 坑 1) |

⚠️ **两个环境注意点**:
1. `conda activate` 在某些非交互 shell 中不切换 PATH——运行 demo/eval 一律用
   绝对路径 `/home/yilin/anaconda3/envs/gpt-deep/bin/python`,或像下面的命令
   一样用 `PATH=...gpt-deep/bin:$PATH`。
2. blend 不再需要 `enforce-eager`,但这依赖 site-packages 里
   `vllm/v1/worker/gpu_worker.py` 的补丁(向 `VLLMModelTracker` 注册模型前
   解包 `CUDAGraphWrapper`;原件备份在 `/home/yilin/LMCache/vllm_patch_backup/`)。
   **重装 vLLM 会丢补丁**,需重打或在 `config.yaml` 临时加 `enforce-eager: true`。
3. 日志/临时输出统一写到 `/home/yilin/tmp/`(不要用 `/tmp/`,可能被系统清理)。
   本文档命令已按此约定。
4. **逐请求 prefill/decode 计时补丁**(site-packages,重装 vLLM 会丢):
   - `vllm/entrypoints/openai/chat_completion/protocol.py`:`ChatCompletionResponse`
     加了可选 `timing` 字段;
   - `vllm/entrypoints/openai/chat_completion/serving.py`:非流式响应构造处,从
     `final_res.metrics` 算 `prefill_s/decode_s/ttft_s/queued_s/inference_s` 填进
     `timing`(与 vLLM prefill/decode 直方图同源同公式)。
   - **开关**:服务端启动加 `VLLM_RESP_TIMING=1` 才回传(默认关,响应不变)。
   - 客户端 `deep_researcher_demo/llm.py` 已把响应里的 `timing` 和 `req_id`
     并入 `LLM_CALL_LOG` 的每行 → calls.jsonl 单文件即含"角色 + prefill/decode",
     **无需事后 join**。非流式即可,不依赖流式。
5. **截断摘要的 KV 复用**(`KV_REUSE_TOKENIZER`):摘要若**撞 `max_tokens` 被截断**
   (`finish_reason=length`),服务端只存了 `output[:-1]`(M-1 token),而 API content
   含 M token → 下游内容哈希 miss、该段**白存且不可复用**。修复:eval 命令带
   `KV_REUSE_TOKENIZER=<该模型目录>`(必须与服务端**同一份** tokenizer),客户端对截断的
   摘要改嵌 `decode(token_ids[:-1])` 与存的 M-1 对齐。两个模型的值(目录名 = served-model-name):
   - 32B dense:`KV_REUSE_TOKENIZER=/data/yilin/huggingface/Qwen3-32B`
   - 30B MoE :`KV_REUSE_TOKENIZER=/data/yilin/huggingface/Qwen3-30B-A3B-Instruct-2507`
   (两者其实同一份 Qwen3 tokenizer:vocab 151643、`<|fim_pad|>`=151662 原子 token。)自然 EOS 的
   摘要不受影响(content 已 = 存的)。**不设此 env → 截断摘要不可复用**(行为同旧版)。
   实测(cap=200 故意制造截断):复用 hit_tokens 1821→3217(×1.77)。
   > 更优做法仍是把 `RESEARCH_SUMMARY_MAX_TOKENS` 设高当安全阀、长度靠指令控,从根上
   > 让摘要自然 EOS;`KV_REUSE_TOKENIZER` 是"万一截断了也能复用"的兜底。
6. **每次重跑用独立 RUN 目录(防跨轮污染,铁律)**:`LLM_CALL_LOG` 是**追加写**、
   `OUTPUT_DIR`/server-log 重用会**覆盖**——同名跨轮会让 calls.jsonl 混多轮记录、server
   日志被冲掉(曾导致 prefill 均值被陈旧记录污染、big 档复用日志丢失)。所以**每轮取一个
   `RUN_ID`(如 `20260614_2`),所有输出进该轮专属新目录**:
   - eval 结果:`eval/results/<RUN_ID>/<model>_<grade>_<mode>/`(OUTPUT_DIR)
   - calls + metrics 快照:`eval/results/ab_meta/<RUN_ID>/<prefix>_calls.jsonl` 等(LLM_CALL_LOG)
   - server 日志:`/home/yilin/tmp/<RUN_ID>_<...>_serve.log`(独立名,blend 档的复用日志靠它)
   起跑前 `mkdir -p` 这两个新目录。分析时**只在该 RUN_ID 目录内** join(server 日志 hit ⋈
   calls.jsonl req_id),天然干净、不会跨轮串。

## 1. 启动 Mode A 服务(CacheBlend)

```bash
CUDA_VISIBLE_DEVICES=3,4,5,6 \
LMCACHE_SAVE_DECODE_CACHE=false \
LMCACHE_BLEND_SPECIAL_STR='<|fim_pad|>' \
bash /home/yilin/LMCache/server/vllm/start.sh lmcache > /home/yilin/tmp/vllm_evalA_serve.log 2>&1 &

# 就绪探测(30B 加载约 2-4 分钟)
until curl -sf localhost:30000/v1/models >/dev/null; do sleep 15; done; echo READY
```

要点:
- **decode-generated KV 的存储现在是「请求级 opt-in」,不再有
  `LMCACHE_BLEND_STORE_GENERATED` 这个 env**(已删)。demo 在 KV-reuse 模式下
  (`KV_REUSE_SEPARATOR` 非空)会自动只对 researcher 摘要 + warmup 请求带上
  `kv_transfer_params={"lmcache.blend_store_generated": true}`,decide/最终报告/JSON
  不带 → 不存。所以服务端启动**什么都不用设**,行为由请求决定。
- `start.sh lmcache` 读 `server/vllm/config.yaml`,其中已含
  `no-enable-prefix-caching: true`(blend 与 vLLM prefix cache 不兼容,
  **必须用 `no-` 前缀写法**,`enable-prefix-caching: false` 在 yaml 里不生效)。
- sep 用模型的原子特殊 token `<|fim_pad|>`(文本级复用的关键:任何上下文中
  分词不变)。**服务端与 demo 的 `KV_REUSE_SEPARATOR` 必须一致**。
- 每轮正式实验前重启服务,保证 LMCache CPU 缓存从空开始。

## 2. 启动 Mode B 服务(原生 vLLM)

```bash
# 先停 Mode A:
pkill -9 -f "vllm serve"; pkill -9 -f "VLLM::"; sleep 5

CUDA_VISIBLE_DEVICES=3,4,5,6 \
bash /home/yilin/LMCache/server/vllm/start.sh no_lmcache > /home/yilin/tmp/vllm_evalB_serve.log 2>&1 &
# 同样的就绪探测
```

`no_lmcache` 模式读 `config_no_lmcache.yaml`:无 kv-transfer、prefix caching
默认开启、默认编译模式(CUDA graphs)。

## 3. GPU 占用记录(共享机干扰对照)

整个实验期间持续记录全部 8 卡占用(10s 采样),便于把"邻卡高负载时段"
与样本延迟对齐分析:

```bash
F=/home/yilin/deep_researcher_demo/eval/results/ab_meta/gpu_monitor.csv
mkdir -p "$(dirname $F)"
echo "timestamp, gpu_index, utilization_gpu_pct, memory_used_mib, power_w" > $F
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
  --format=csv,noheader -l 10 >> $F &
# 阶段切换时手动追加 marker:
echo "# PHASE_MARKER $(date -u +%FT%TZ) modeB_start" >> $F
```

## 4. 跑评测(50 题 generate)

**Mode A**(注意 `KV_REUSE_SEPARATOR` 与服务端 sep 一致):

```bash
cd /home/yilin/deep_researcher_demo
KV_REUSE_SEPARATOR='<|fim_pad|>' \
MODE=generate LIMIT=10 MAX_FOLLOWUPS=2 MAX_QUERIES_PER_RESEARCHER=2 \
OUTPUT_DIR=eval/results/ab_blend_10 SAMPLE_CONCURRENCY=1 QUIET=1 OVERWRITE=1 \
PATH=/home/yilin/anaconda3/envs/gpt-deep/bin:$PATH \
bash eval_deepsearchqa.sh > /home/yilin/tmp/evalA_gen.log 2>&1
```

**Mode B**(唯一区别:不设 `KV_REUSE_SEPARATOR`、输出目录不同):

```bash
cd /home/yilin/deep_researcher_demo
MODE=generate LIMIT=50 MAX_FOLLOWUPS=2 MAX_QUERIES_PER_RESEARCHER=2 \
OUTPUT_DIR=eval/results/ab_native_50 SAMPLE_CONCURRENCY=1 QUIET=1 OVERWRITE=1 \
PATH=/home/yilin/anaconda3/envs/gpt-deep/bin:$PATH \
bash eval_deepsearchqa.sh > /home/yilin/tmp/evalB_gen.log 2>&1
```

约定:
- **两种模式输出目录严格分离**(`ab_blend_50` / `ab_native_50`),防止覆盖。
- `SAMPLE_CONCURRENCY=1`:样本串行,计时干净。
- `MAX_FOLLOWUPS=2 MAX_QUERIES_PER_RESEARCHER=2`:控制每样本搜索量
  (降低 DDG 限流风险),两模式必须同参。
- `KV_REUSE_SEPARATOR` 同时改变 prompt 布局:摘要段前置、问题后置 + 角色
  warmup(blend 的 lookup 只认"从段 0 起的连续命中")。Mode B 用原版布局。
- 每条 report 记录 `latency_seconds` 与起止时间戳(ISO),可与 GPU CSV 对齐。

## 5. 判分(qwen-flash)与对比

```bash
# 分别判分(OVERWRITE=1 在 score 模式只清判分产物,不动 reports.jsonl)
MODE=score LIMIT=50 OUTPUT_DIR=eval/results/ab_blend_50  OVERWRITE=1 RESUME=0 \
  PATH=/home/yilin/anaconda3/envs/gpt-deep/bin:$PATH bash eval_deepsearchqa.sh
MODE=score LIMIT=50 OUTPUT_DIR=eval/results/ab_native_50 OVERWRITE=1 RESUME=0 \
  PATH=/home/yilin/anaconda3/envs/gpt-deep/bin:$PATH bash eval_deepsearchqa.sh

# 汇总对比(延迟统计 + metrics)
/home/yilin/anaconda3/envs/gpt-deep/bin/python -m eval.compare_ab \
  eval/results/ab_blend_50 eval/results/ab_native_50
```

服务端复用验证(Mode A 日志):

```bash
grep -c "Queued final save" /home/yilin/tmp/vllm_evalA_serve.log     # 每个完成请求 1 次
grep "hit tokens" /home/yilin/tmp/vllm_evalA_serve.log | tail        # decide/写报告应高命中
```

## 6. 已知坑

1. **DDG 限流**:持续高频搜索会让本机被搜索后端封禁(表现为
   `ConnectTimeout`/全部样本快速失败)。`search.py` 已内置:进程级节流锁
   (`DDG_MIN_INTERVAL`,默认 1.5s/次)、4 次指数退避重试、"无结果"降级为
   空列表。若仍大面积失败:停跑等待 10-30 分钟封禁解除(可用一条搜索探活),
   失败样本可用 `RESUME=1 OVERWRITE=0` 补跑。
2. **blender 在 forward 之外跑 MoE 的三个 vLLM 0.18 兼容点**(均已修在
   LMCache adapter 里,升级 vLLM 时留意):MoE 层名计数器(包
   `set_forward_context`)、MoE workspace 锁定(blend 前临时 `unlock_workspace`)、
   `Qwen3MoeForCausalLM` 的 blender 模型映射。
3. **判分网络**:qwen-flash 走 DashScope,确保 `.env` 的 `JUDGE_API_KEY` 有效。
4. 别人占用的 GPU(0,2,6,7)负载会通过功耗/PCIe 影响结果,
   `ab_meta/gpu_monitor.csv` 用于事后排查异常延迟时段。

## 6.5 理想计时环境:web search 录制/回放缓存

为把端到端计时测准,需把 web search 的网络延迟/限流/结果漂移从变量里去掉。
`search.py` 的 `CachingSearchProvider` 提供 record/replay 缓存,三个 env 开关:

| env | 取值 | 作用 |
|---|---|---|
| `SEARCH_CACHE` | `off`(默认)/`record`/`replay` | off=照常实时搜;record=实时搜+按题落盘;replay=只从缓存回放,零网络 |
| `SEARCH_CACHE_FIX_N` | 整数(默认 0=全部) | replay 时每题只用 manifest 前 N 个文档(固定变量,建议 3) |
| `SEARCH_CACHE_DIR` | 路径(默认 `eval/results/search_cache`) | 缓存根目录,**跨 A/B 多次跑持久复用,别放进 per-run OUTPUT_DIR** |

缓存按**题目共享池**:键=`sample_id`,每题一个目录 `q<id>/`(`*.md` + `manifest.json`)。
replay 时不管 researcher 发什么 query 都返回该题前 N 个文档 → A/B 两模式即使 query
漂移也拿同一批文档,公平。

**标准流程**:

```bash
# 1) 建库:实时搜一遍(会打 DDG,注意限流),把文档按题存成 markdown
SEARCH_CACHE=record SEARCH_CACHE_DIR=eval/results/search_cache \
  MODE=generate LIMIT=10 OUTPUT_DIR=eval/results/_cache_build \
  PATH=/home/yilin/anaconda3/envs/gpt-deep/bin:$PATH bash eval_deepsearchqa.sh

# 2) 之后所有计时跑都用 replay(零网络,每题固定 3 文档),Mode A / Mode B 同一缓存
SEARCH_CACHE=replay SEARCH_CACHE_FIX_N=3 SEARCH_CACHE_DIR=eval/results/search_cache \
  KV_REUSE_SEPARATOR='<|fim_pad|>' MODE=generate LIMIT=10 \
  OUTPUT_DIR=eval/results/ab_blend_10 ... bash eval_deepsearchqa.sh
```

**缺失标记**:replay 时若某题没录过 → 自动回退真实搜索 + 录入 + 在该题 report 打
`search_cache_miss=true`(并在 `<dir>/cache_misses.jsonl` 记一行)。这题含网络延迟,
分析时按 `search_cache_miss` 剔除即可。

## 7. 结果

- **第一阶段(效率,10 题 × 2 轮)已完成**,详见
  [phase1_blend_cost_analysis.md](phase1_blend_cost_analysis.md):
  blend 成本 ≈ `455ms + 0.0146ms/token`(107 事件回归),盈亏交叉点约
  4-6k 复用 token;小上下文 B 胜(64 vs 91s/题),大上下文 A 胜
  (94 vs 106s/题,复用请求 prefill 快 2-4 倍);decode 占 93% 封顶端到端收益。
- 第二阶段(50 题 + qwen-flash 判分)待跑。

<!-- RESULTS_PLACEHOLDER -->
