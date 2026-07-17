# Writer 隔离实验 · 运行指引(用哪些命令、跑哪些代码)

> 代码已就绪并测过(见 [code_changes_report.md](code_changes_report.md));本文是**端到端跑实验的操作手册**。设计/判读见 [part2_run_experiment.md](part2_run_experiment.md)。
> 三步:**起 server(control+dump)→ 跑 driver(主轨+18 writer 重放)→ kimi 打分**。

## 0. 环境与前置(先看一眼)
| 角色 | 用哪个 python / env | 为什么 |
|---|---|---|
| **server**(vllm serve) | `/home/yilin/anaconda3/envs/lmcache/bin/vllm` | LMCache blend 那套 |
| **driver**([exp_writer_run.py](../../server/vllm/Exp_writer_isolation/exp_writer_run.py))+ **打分**([exp_writer_score.py](../../server/vllm/Exp_writer_isolation/exp_writer_score.py)) | `/home/yilin/anaconda3/envs/gpt-deep/bin/python` | 唯一同时齐 torch/transformers/**bs4**/datasets/deep_researcher_demo/eval 的 env(driver 要起 demo 子进程,demo 依赖 bs4) |

- **GPU**:server 要 4 张空卡(TP4,每张 ~44GB);算 M 会**自动另挑空闲卡**(`_pick_m_device`,躲开模型卡)。
- **磁盘**:`/home/yilin/tmp` 要留 ~30GB——每题 teacher-force dump 可达十几 GB,driver **读完即删**(只是瞬时占用)。
- **关键对齐**:driver 写 control / 读 dump 的路径 = server 启动时的 `LMCACHE_BLEND_CONTROL` / `LMCACHE_BLEND_DUMP_KV`(driver 直接读这俩 env,保证同一文件)。

### 📂 日志 / 产物一览(调试看这张表;`$EXP_W_DIR=/home/yilin/tmp/exp_writer`)
| 路径 | 内容 | 调试用途 |
|---|---|---|
| `$EXP_W_DIR/server.log` | vllm + LMCache,含 `BLEND_PATH=CONTROL mode=.. reuse%` | **核每次 blend 走没走 control、复用比例对不对**(本实验首要看这个) |
| `$EXP_W_DIR/driver.log` | driver 进度(每题 ctx、dump_event、每分支报告长度) | 跑到哪、哪题失败、分段对不对 |
| `$EXP_W_DIR/llm_calls.jsonl` | 主轨逐 LLM 调用计时(tag/req/timing) | 配 server.log 的 BLEND_PATH 按时间戳查**单请求**复用(见 memory `kv-reuse-per-request-diagnosis`) |
| `$EXP_W_DIR/score.log` | kimi 打分进度 + compare 判读 | 看打分过程 / 最终结论 |
| `$EXP_W_DIR/q<N>/harvest.jsonl` | 该题主轨每次 LLM 调用 messages 原文 | 看 writer prompt、summary 段切得对不对 |
| `$EXP_W_DIR/q<N>/meta.json` | segments / summary_pos / prompt_len / m_event | 核分段、M dump 是哪个 event |
| `$EXP_W_DIR/q<N>/controls/branch_*.json` + `diag.json` | 各分支 control(选了哪些 token/层)+ ΔK/ΔV 相关性 | 看每分支选择是否合理 |
| `$EXP_W_DIR/q<N>/reports.json` | 18 分支生成的报告 | 人眼对比报告质量 |
| `$EXP_W_SCORE_OUT/ratings_branch_*.jsonl` `metrics_*.json` | 逐题打分 + 汇总(默认 `exp-docx/Exp_writer_isolation/scores/`) | 复查分数 |

---

## Step 0 — 确保有 ≥4 张空闲 GPU(server TP4 要 4 张,每张 ≥42GB 空)
> 不挑固定卡号(Step 1 自动选当前空闲的);只需保证至少 4 张空。若旧 server / 别的任务占着且不再需要,按下面 kill。
```bash
nvidia-smi --query-gpu=index,memory.free --format=csv          # 看哪几张空
# 若 0-3 仍被旧 server 占(workers 是 yilin 的 vllm 进程)且不再需要 → 直接 kill worker PID
#   注意:pgid/TERM 杀不到 worker(自成进程组);务必按 user=yilin 过滤,别误杀别人(如 armaan)的进程
nvidia-smi --query-compute-apps=pid,used_memory --format=csv | sort -t, -k2 -rh   # 找占卡大户 PID
ps -o pid,user,cmd -p <PID>                                   # 确认是 yilin 的 vllm worker 再杀
kill -9 <worker_pids...>
```

## Step 1 — 起 control 模式 server(TP4 + dump,固定 control/dump 路径)
在一个**独立终端**跑(这个终端的 `CUDA_VISIBLE_DEVICES` 给 server 用):
```bash
export EXP_W_DIR=/home/yilin/tmp/exp_writer
mkdir -p $EXP_W_DIR/dump
echo '{"reuse_token_ranges": []}' > $EXP_W_DIR/blend_control.json   # 初始 truth control(server 启动即可读)
# 动态挑 4 张当前空闲卡(≥42GB,留余量给 0.85×47≈40GB)。别硬编码 0-3——空卡随时被别人抢(实测踩过:
# 腾空 0 号后立刻被抢 21GB → server "Free memory on cuda:0 < desired" 崩)。
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | awk -F', ' '$2>42000{print $1}' | head -4 | paste -sd,)
echo "server 用卡: $CUDA_VISIBLE_DEVICES"   # 必须正好 4 张;不足 4 张就等/腾卡再起
export PYTHONHASHSEED=0
# 原生 CacheBlend(和四臂 C 同款)
export LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=true LMCACHE_MAX_LOCAL_CPU_SIZE=16
export LMCACHE_ENABLE_BLENDING=true LMCACHE_BLEND_SPECIAL_STR='<|fim_pad|>'
export LMCACHE_USE_LAYERWISE=true LMCACHE_SAVE_UNFULL_CHUNK=true LMCACHE_SAVE_DECODE_CACHE=false
export LMCACHE_BLEND_CHECK_LAYERS=1 LMCACHE_BLEND_RECOMPUTE_RATIOS=0
# control 双开关 + dump(本实验的命根子;两个开关必须同时设才走 control)
export LMCACHE_BLEND_CONTROL_ENABLED=1
export LMCACHE_BLEND_CONTROL=$EXP_W_DIR/blend_control.json
export LMCACHE_BLEND_DUMP_KV=$EXP_W_DIR/dump
cd /home/yilin/LMCache/server/vllm
# --return-tokens-as-token-ids 必加!否则 logprobs.tokens 是解码字符串,driver 的 _parse_tid 解析不出
#   gen_ids → teacher-force 的 report 部分为空 → M dump 全 0 → M×ΔV 分支废掉(实测确认)。
# --max-logprobs 给 free_gen_report 的 logprobs 请求留余量。tee 到 server.log 供 grep BLEND_PATH。
/home/yilin/anaconda3/envs/lmcache/bin/vllm serve --config Qwen3-32B/config.yaml \
    --max-logprobs 200 --return-tokens-as-token-ids \
    2>&1 | tee $EXP_W_DIR/server.log          # 端口 30000
```
就绪检查(另开终端):`curl -s http://localhost:30000/v1/models | grep Qwen3-32B && echo READY`
控制是否生效:`grep -m1 "BLEND_PATH=CONTROL" $EXP_W_DIR/server.log`(出现 = 走 control,不是原生 CacheBlend)

## Step 2 — 跑 driver(主轨 + 18 writer 重放,40 题)
**新开一个终端**(别带 Step 1 的 `CUDA_VISIBLE_DEVICES=0,1,2,3`,否则 driver 看不到空闲卡算 M):
```bash
cd /home/yilin/LMCache/server/vllm
export EXP_W_DIR=/home/yilin/tmp/exp_writer
# 关键:control/dump 路径与 server 完全一致(driver 写 control 到这、读 dump 从这)
export LMCACHE_BLEND_CONTROL=$EXP_W_DIR/blend_control.json
export LMCACHE_BLEND_DUMP_KV=$EXP_W_DIR/dump
export EXP_W_SERVER=http://localhost:30000        # server 在 30000(不是默认 8000!)
export EXP_W_OUT=$EXP_W_DIR                        # 每题输出 $EXP_W_DIR/qN/reports.json
export EXP_W_N=40                                  # 跑前 40 题(deepsearchqa eval)
export EXP_W_M_DEVICE=cuda:6                       # 算 M 用哪张空闲卡(留空=自动挑最空闲的)
export KV_REUSE_SEPARATOR='<|fim_pad|>'            # 全角色复用分隔符(writer 才会 blend summary)
export SEARCH_CACHE=replay                         # 复现检索(没现成缓存就改 record 先录)
export LLM_CALL_LOG=$EXP_W_DIR/llm_calls.jsonl     # 主轨逐调用计时(配 server.log BLEND_PATH 查单请求复用)
nohup /home/yilin/anaconda3/envs/gpt-deep/bin/python Exp_writer_isolation/exp_writer_run.py \
      > $EXP_W_DIR/driver.log 2>&1 &
tail -f $EXP_W_DIR/driver.log
```
- 单题调试:`... python Exp_writer_isolation/exp_writer_run.py "你的问题" q0`。
- 断点续:`reports.json` 在的题自动跳过,直接重跑同命令即可。
- **边跑边核**:`grep BLEND_PATH=CONTROL $EXP_W_DIR/server.log` —— 每次 blend 应见 `mode=...`;25% 档那次 `recompute≈25%`、full reuse 那次 `reuse≈100%`、branch1 truth 那次 `reuse 0%`、layer 档那次 `recompute K/L layers`。哪条对应哪分支:按时间戳和 driver.log 的 `[branch N]` 对齐。

## Step 3 — kimi 打分(18 分支逐题 → 汇总 → 判读)
```bash
cd /home/yilin/LMCache/server/vllm
export EXP_W_DIR=/home/yilin/tmp/exp_writer EXP_W_BASE=/home/yilin/tmp/exp_writer EXP_W_N=40
# .env 里要有 JUDGE_BASE_URL / JUDGE_API_KEY / JUDGE_MODEL=kimi-k2.5(DashScope)
/home/yilin/anaconda3/envs/gpt-deep/bin/python Exp_writer_isolation/exp_writer_score.py all 2>&1 | tee $EXP_W_DIR/score.log
#   ① score  逐题打分存 ratings_branch_<bb>.jsonl(resumable)
#   ② agg    每分支 metrics_branch_<bb>.json
#   ③ compare 四方对比 + 噪声底 + 总判读
```
分项也可单跑:`... exp_writer_score.py score` / `agg` / `compare`。输出在 `$EXP_W_SCORE_OUT`(默认 `exp-docx/Exp_writer_isolation/scores/`)。

## Step 4 — 看结论
`compare` 末尾那句 **"25% 档总判读"**:M×ΔV 是否**同时打过 random 且超 full_reuse(差 > 噪声底 0.14)**——
是 → 选择性刷新有用、M×Δ 选得准;否 → 选不准/无用。详细判据见 [part2_run_experiment.md](part2_run_experiment.md) 的"怎么读结果"。

---

## 验证清单(跑之前/跑之中核对)
1. **server**:`curl .../v1/models` 通;日志出现 `BLEND_PATH=CONTROL mode=...`(不是 `CacheBlend`)= control 生效。
2. **路径对齐**:driver 的 `LMCACHE_BLEND_CONTROL`/`LMCACHE_BLEND_DUMP_KV` 与 server 完全相同(否则 control 不生效 / dump 找不到)。
3. **算 M 不碰模型卡**:driver 日志/`nvidia-smi` 确认 M 跑在空闲卡(或 `EXP_W_M_DEVICE` 指定);模型卡显存不被 driver 吃。
4. **pinned summary**:同题 18 分支的 writer prompt 一致(driver 全程用同一份 `prompt_ids`)。
5. **打分**:每分支 `ratings_branch_*.jsonl` 满 40 题、`per_q_f1` 非全 None;重跑跳过已打题。

## 常见坑(已知)
- **端口**:server 在 **30000**,driver 必设 `EXP_W_SERVER=http://localhost:30000`(默认 8000 会连不上)。
- **CUDA_VISIBLE_DEVICES 串台**:driver 终端别继承 server 的 `0,1,2,3`,否则算 M 找不到空闲卡 → 退 CPU 变慢(或显式 `EXP_W_M_DEVICE`)。
- **磁盘**:dump 瞬时十几 GB/题,确认 `/home/yilin/tmp` 有 ~30GB 余量(driver 每题读完即删 M dump)。
- **kill server**:worker 自成进程组,`kill -TERM -$pgid` 杀不到 → 按 PID 直接 kill,且先按 user 过滤。
- **search 缓存**:首次没现成缓存时 `SEARCH_CACHE=replay` 会冷 miss 退 live;想完全复现可先 `record` 跑一遍或指向四臂已录的 `SEARCH_CACHE_DIR`。

*相关:[code_changes_report.md](code_changes_report.md)、[part1_code_and_test.md](part1_code_and_test.md)、[part2_run_experiment.md](part2_run_experiment.md);memory `blend-three-modes-env`、`kv-reuse-per-request-diagnosis`。*
