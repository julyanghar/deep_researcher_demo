# exp-test-supervisor 运行指引:Supervisor 复用 + Writer full prefill(SW 臂)

**目标**:隔离"是不是 supervisor 的 KV 复用拖累了 report"。做法:让 **Writer 走 full prefill(不复用)、Supervisor 仍 KV 复用**(记 **SW 臂**),跑 40 题,和已有 C(全复用)/B(全真算带分隔符)/A_orig(真算无分隔符)对照。
- SW ≈ C → 清理 writer 没用,**害来自 supervisor 复用**。
- SW ≈ B/A → supervisor 复用无害(之前 C 的差是 writer 或噪声)。

> 已有数据(别动)在 `/home/yilin/tmp/ExpB_fullseg_4arm/`:`exp_fullseg`(A_orig)、`exp_fullseg_A2`(A_orig2)、`exp_fullseg_van`(B)、`exp_fullseg_pure`(C)。
> 跑这俩 env:server 用 `lmcache`(`/home/yilin/anaconda3/envs/lmcache/bin`),driver/分析用 `gpt-deep`(`/home/yilin/anaconda3/envs/gpt-deep/bin`)。
> **日志统一放 `/home/yilin/tmp/logs/`(先 `mkdir -p /home/yilin/tmp/logs`),不要放进 `ExpB_fullseg_4arm/`(那只放数据,别混)。** SW 的数据(traj/报告)仍进 `ExpB_fullseg_4arm/exp_fullseg_SW/`。

---

## Step 0 — 代码已改好(我改的;以下说明改了什么 + 为什么)

**为什么改**:原来 `config.kv_reuse_separator` **一个值**同时喂给 Supervisor / Researcher / FinalWriter(`cli.py` 构造处),没法让"writer 不复用、supervisor 复用"。改成**按角色独立的 separator + env 覆盖**,默认继承全局 → **不影响以前的实验**;只有显式设 `FINAL_KV_REUSE_SEPARATOR=`(空)时,writer 才单独走 prefill。

**改了两个文件(已落地):**

**① `deep_researcher_demo/config.py`** —— 加 3 个 per-role 字段 + `from_env` 里按角色读 env(默认 = 全局 `sep`):
```python
# dataclass 里(kv_reuse_separator 下面):
    supervisor_kv_reuse_separator: str = ""
    researcher_kv_reuse_separator: str = ""
    final_kv_reuse_separator: str = ""
# from_env 里:
        sep = os.getenv("KV_REUSE_SEPARATOR", "")          # 全局
        ... kv_reuse_separator=sep,
            supervisor_kv_reuse_separator=os.getenv("SUPERVISOR_KV_REUSE_SEPARATOR", sep),
            researcher_kv_reuse_separator=os.getenv("RESEARCHER_KV_REUSE_SEPARATOR", sep),
            final_kv_reuse_separator=os.getenv("FINAL_KV_REUSE_SEPARATOR", sep),
```

**② `deep_researcher_demo/cli.py`** —— 三个 agent 各传**对应角色**的 separator(原来都传 `config.kv_reuse_separator`):
```python
        supervisor=Supervisor(llm, supervisor_model, config.supervisor_kv_reuse_separator),
        researcher=Researcher(llm, researcher_model, summary_model,
                              kv_reuse_separator=config.researcher_kv_reuse_separator),
        final_writer=FinalWriter(llm, final_model, config.final_kv_reuse_separator),
```

**原理**:`FINAL_KV_REUSE_SEPARATOR=`(空串)覆盖全局 → writer 的 findings 用 `\n\n` join、无 `<|fim_pad|>`、不触发 blend = **full prefill**;supervisor/researcher 仍用全局 `<|fim_pad|>` = **复用**。warmup 各角色自己判 separator(writer 空 → 自动跳过)。

**已验证(你也可复跑确认)**:
```bash
# 默认(不设 per-role)→ 三角色都 = 全局,旧行为不变:
cd /home/yilin/deep_researcher_demo
KV_REUSE_SEPARATOR='<|fim_pad|>' /home/yilin/anaconda3/envs/gpt-deep/bin/python -c \
"from deep_researcher_demo.config import AppConfig as C;c=C.from_env();print(c.supervisor_kv_reuse_separator, c.researcher_kv_reuse_separator, repr(c.final_kv_reuse_separator))"
#  → <|fim_pad|> <|fim_pad|> '<|fim_pad|>'
# SW 配置(FINAL 设空)→ writer 走 prefill:
KV_REUSE_SEPARATOR='<|fim_pad|>' FINAL_KV_REUSE_SEPARATOR= /home/yilin/anaconda3/envs/gpt-deep/bin/python -c \
"from deep_researcher_demo.config import AppConfig as C;c=C.from_env();print(c.supervisor_kv_reuse_separator, c.researcher_kv_reuse_separator, repr(c.final_kv_reuse_separator))"
#  → <|fim_pad|> <|fim_pad|> ''   ← writer 空 = prefill
```

---

## Step 1 — 启 CacheBlend server(和 C 同款,TP4)
先 `nvidia-smi` 找 4 张空卡,填进 `CUDA_VISIBLE_DEVICES`(下面以 4,5,6,7 为例):
```bash
cd /home/yilin
export CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONHASHSEED=0
export LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=true LMCACHE_MAX_LOCAL_CPU_SIZE=16
export LMCACHE_ENABLE_BLENDING=true LMCACHE_BLEND_SPECIAL_STR='<|fim_pad|>'
export LMCACHE_USE_LAYERWISE=true LMCACHE_SAVE_UNFULL_CHUNK=true LMCACHE_SAVE_DECODE_CACHE=false
export LMCACHE_BLEND_CHECK_LAYERS=1 LMCACHE_BLEND_RECOMPUTE_RATIOS=0
# 不设 LMCACHE_BLEND_CONTROL / CONTROL_ENABLED → 走原生 CacheBlend(纯复用)
nohup /home/yilin/anaconda3/envs/lmcache/bin/vllm serve \
  --config /home/yilin/LMCache/server/vllm/Qwen3-32B/config.yaml \
  --max-logprobs 200 --return-tokens-as-token-ids \
  > /home/yilin/tmp/logs/server_SW.log 2>&1 &
# 等就绪(~3-4min):
until curl -s -m3 http://localhost:30000/v1/models | grep -q Qwen3-32B; do sleep 10; done; echo "server ready"
```

## Step 2 — 跑 SW driver(reuse-only + Writer=prefill)
关键:`FINAL_KV_REUSE_SEPARATOR=`(空,让 writer 走 prefill);其余同 C 臂(reuse-only、replay、共享缓存)。
```bash
cd /home/yilin/deep_researcher_demo
EXP_FS_OUT=/home/yilin/tmp/ExpB_fullseg_4arm/exp_fullseg_SW \
EXP_FS_QUESTIONS=/home/yilin/tmp/ExpB_fullseg_4arm/exp_a_questions.json \
EXP_FS_N=40 EXP_FS_Q_TIMEOUT=2400 EXP_FS_SKIP_PREFILL=1 \
FINAL_KV_REUSE_SEPARATOR= \
LLM_CALL_LOG=/home/yilin/tmp/logs/llm_calls_SW.jsonl \
EXP_SERVER=http://localhost:30000 OPENAI_BASE_URL=http://localhost:30000/v1 \
MODEL=Qwen3-32B OPENAI_API_KEY=EMPTY \
SEARCH_CACHE=replay SEARCH_CACHE_DIR=/home/yilin/deep_researcher_demo/eval/results/search_cache \
SEARCH_PROVIDER=duckduckgo EXP_DEMO_CWD=/home/yilin/deep_researcher_demo \
nohup /home/yilin/anaconda3/envs/gpt-deep/bin/python \
  /home/yilin/LMCache/server/vllm/Exp_fullseg/exp_fullseg_run.py \
  > /home/yilin/tmp/logs/exp_fullseg_SW_resume.log 2>&1 &
```
> 注意 `FINAL_KV_REUSE_SEPARATOR=` 后面紧跟空格(bash 把它设成空串)。~1.7h 跑完 40 题。
>
> **`LLM_CALL_LOG=...`**(可选,新加):开 `llm.py` 的每次 LLM 调用计时日志(JSONL,记 tag / prompt+completion tokens / `prefill_s` / `decode_s` / `ttft_s` 等)。和 harvest 是两套(harvest 记 messages 原文,这个记**时延**),互不影响。
> - 40 题**追加到同一个文件**(无 per-题分隔;记录里有 `tag`/`req_id` 但没 sample_id)→ 适合看**聚合时延分布**(如 supervisor 复用是否真省了 TTFT/prefill)。
> - 看一眼:`tail -2 /home/yilin/tmp/logs/llm_calls_SW.jsonl`;按 tag 聚合 prefill/ttft 用 jq/python 自己拉。
> - 不想要就删掉这行 env(默认 `LLM_CALL_LOG` 不设 → `_log_call` 直接 return、不写)。

## Step 3 — 第 1 题就验证(别等跑完)
```bash
# (a) supervisor 在复用:server 日志该有 CacheBlend、reuse~99%、0 CONTROL
grep -ac 'BLEND_PATH=CacheBlend' /home/yilin/tmp/logs/server_SW.log
grep -ac 'BLEND_PATH=CONTROL'    /home/yilin/tmp/logs/server_SW.log   # 应=0
grep -a  'BLEND_PATH=CacheBlend' /home/yilin/tmp/logs/server_SW.log | head -1   # 看 reuse 百分比

# (b) writer 走 prefill:抽 q0 的报告调用,findings 该用 \n\n、无 <|fim_pad|>
python3 -c "
import json
for l in open('/home/yilin/tmp/ExpB_fullseg_4arm/exp_fullseg_SW/q0/harvest_reuse.jsonl'):
    r=json.loads(l)
    if r.get('tag')=='FINAL_REPORT_MARKDOWN' or 'final' in (r.get('tag') or '').lower():
        u=[m['content'] for m in r['messages'] if m['role']=='user'][0]
        print('writer prompt 含 <|fim_pad|> :', '<|fim_pad|>' in u, '(应 False)')
        break
"
```
> 预期:SW 的 `BLEND_PATH=CacheBlend` 条数 ≈ C 少 ~1 条/题(少了 writer 那次 blend)。若 writer prompt 含 `<|fim_pad|>` → 说明 per-role 没生效,检查 Step 0。

---

## Step 4 — 分析(merge/metrics/sources 我已替你改好,直接跑)
> **我已改这三个脚本**(路径指向 `ExpB_fullseg_4arm/` + 加了 SW,且 SW 数据没跑时自动跳过):
> - `exp_fullseg_merge.py`:常量 `BASE` 指向 ExpB_fullseg_4arm/、加 `SW` + SW 三个合成。
> - `exp_fullseg_metrics.py`:`ARMS` 路径修正 + 加 `SW`、目录不存在自动跳过。
> - `exp_fullseg_sources.py`:`ARMS` 加 `SW`。
> **无需手改、无需符号链接。** 直接按下面跑(都在 SW 数据跑完后)。

### 4a. 合成对比目录
```bash
cd /home/yilin/LMCache/server/vllm
/home/yilin/anaconda3/envs/gpt-deep/bin/python Exp_fullseg/exp_fullseg_merge.py
```
→ 生成 `ExpB_fullseg_4arm/exp_fullseg_{SWvC, BvSW, A2vSW}`:**SWvC=writer 复用效应**(pf−ru=SW−C)、**BvSW=supervisor 复用效应**(pf−ru=B−SW)、A2vSW=SW 总。

### 4b. env4 配对 F1 + bootstrap CI(qwen-flash)
```bash
cd /home/yilin/LMCache/server/vllm
for tag in SWvC BvSW A2vSW; do
  echo "==== $tag ===="
  EXP_FS_OUT=/home/yilin/tmp/ExpB_fullseg_4arm/exp_fullseg_$tag \
  /home/yilin/anaconda3/envs/gpt-deep/bin/python Exp_fullseg/exp_fullseg_score.py 2>&1 | tail -8
done
```
env2/3(决策/内容偏离)同理:把 `exp_fullseg_score.py` 换成 `exp_fullseg_analyze.py`。

### 4c. SW 整体 metrics.json(会重跑全部臂,~5×40 次评委调用)
```bash
EXP_FS_REGRESS_OUT=/home/yilin/LMCache/exp-docx/ExpB_fullseg \
/home/yilin/anaconda3/envs/gpt-deep/bin/python /home/yilin/LMCache/server/vllm/Exp_fullseg/exp_fullseg_metrics.py
```
→ 出 `exp-docx/ExpB_fullseg/metrics_SW.json`,和已有 `metrics_C/B/A_orig.json` 比 F1/全对率。

### 4d.(可选)数据源 + sub-query on-topic
```bash
/home/yilin/anaconda3/envs/gpt-deep/bin/python /home/yilin/LMCache/server/vllm/Exp_fullseg/exp_fullseg_sources.py
```
→ 看 SW 的 URL/子查询 on-topic 是否和 C 一致(已含 SW 臂)。

---

## Step 5 — 判读
| 比较 | 看什么 | 结论 |
|---|---|---|
| **SW vs C**(writer 效应)| F1 差≈0?| ≈0 → writer 复用无关,**害在 supervisor**;C 明显差于 SW → 是 writer 复用 |
| **SW vs B**(supervisor 效应)| SW 比 B 差?| SW 显著差于 B(超噪声底)→ **supervisor 复用真害**(研究轨迹被带偏);≈B → supervisor 复用无害 |
| 对照 | metrics_SW F1 落在 C(26%)还是 B(34%)附近 | 靠 C → supervisor 主导;靠 B → 不是 supervisor |

判据:任何效应都要**超过噪声底**(A_orig vs A_orig2:配对 Set F1 0.14)才算真。

---

## 跑完清理(腾 GPU,按 worker PID 杀,别用 pkill -f 防误杀)
```bash
# 找 server 占卡的 worker PID,直接 kill(先确认是自己的、别杀别人的)
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
kill -9 <你的 server worker PID...>
```

## 涉及文件一览
- 改:`deep_researcher_demo/{config.py,cli.py}`(per-role sep)、`server/vllm/{exp_fullseg_merge.py,exp_fullseg_metrics.py}`(加 SW)。
- 用(不改):`server/vllm/{exp_fullseg_run.py,exp_fullseg_runner.py,exp_fullseg_analyze.py,exp_fullseg_score.py}`、`Qwen3-32B/config.yaml`。
- 数据:`/home/yilin/tmp/ExpB_fullseg_4arm/`(已有 A/A2/B/C + 新产 exp_fullseg_SW)。
- 背景(在 `../ExpB_fullseg/`):[mainline_40q_report.md](../ExpB_fullseg/mainline_40q_report.md)(四臂结论)、[experiment_workflow.md](../ExpB_fullseg/experiment_workflow.md)(四臂跑法)、[blend_path_logging.md](../ExpB_fullseg/blend_path_logging.md)(BLEND_PATH 含义)、[callchain_cdriver_and_control.md](../ExpB_fullseg/callchain_cdriver_and_control.md)。
