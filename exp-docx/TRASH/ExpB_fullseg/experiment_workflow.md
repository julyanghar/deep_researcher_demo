# 全段实验(纯复用效应)· 完整流程

> 这次实验:在 deep-research agent 的**结果**上,干净测出**纯 KV 复用效应**。
> 只产两条 reuse 轨 —— **C(纯复用·带分隔符)**、**B(vanilla 真算·带分隔符)**,核心比 **C − B**;
> 辅以 **B vs A_orig**(分隔符效应,带 confound)。全程只用 4 张卡、顺序跑、replay 共享搜索缓存。

---

## 一、为什么这么设计(一句话)

- 原"复用 vs prefill"作废:① 旧"复用"轨被残留 `LMCACHE_BLEND_CONTROL` 强制全重算(没真复用);② agent 轨迹非确定性极大。
- 已证 control-truth ≡ 干净 prefill(KV cos 中位 0.0001)→ 旧轨差异主要来自 **`<|fim_pad|>` 分隔符**,不是 KV 复用。
- **分解**:总(C−A)= 分隔符效应(B−A) + **纯复用效应(C−B)**。C−B 两臂 prompt 都带分隔符,唯一差别 = KV 到底复用没复用 → 最干净。

## 二、三臂 + 数据来源

| 臂 | 含义 | prompt join | KV | 数据目录 |
|---|---|---|---|---|
| **A_orig** | 真算·无分隔符(基线) | `\n\n` | 无复用 | `/home/yilin/tmp/exp_fullseg`(早先跑,prefill 臂有效) |
| **A_orig2** | 真算·无分隔符(第二条,**噪声底**)| `\n\n` | 无复用 | `/home/yilin/tmp/exp_fullseg_A2`(本轮 vanilla 重跑) |
| **B** | 真算·带分隔符(vanilla 全重算) | `<\|fim_pad\|>` | 无复用 | `/home/yilin/tmp/exp_fullseg_van`(reuse 臂) |
| **C** | 复用·带分隔符(CacheBlend) | `<\|fim_pad\|>` | 复用~99% | `/home/yilin/tmp/exp_fullseg_pure`(reuse 臂) |

> 注:本轮用两个开关切单臂(省时间)——
> - **B/C** 用 `EXP_FS_SKIP_PREFILL=1` → 每题**只跑 reuse 臂**(带分隔符)。
> - **A_orig2** 用 `EXP_FS_SKIP_REUSE=1` → 每题**只跑 prefill 臂**(无分隔符)= 第二条 full prefill。
>
> **噪声底 = A_orig vs A_orig2**(两条 prefill·无分隔符 互比 = 纯 vllm 非确定性 + 搜索漂移);判据:B−A / C−B 的效应要**超过这个噪声底**才算真。⚠ 之前缺这条第二 prefill(只有 q1 spot-check `trace_det_run1/2`,37 vs 15 个 LLM call,已显非确定性极大),本轮补全。

---

## 三、用到的代码文件

| 文件 | 作用 | 关键行 |
|---|---|---|
| [blender.py](../../lmcache/v1/compute/blend/blender.py) | LMCache blend 三模式 + `BLEND_PATH=` 日志 + patch C(q-dump) | 见下 §3.1 |
| [exp_fullseg_runner.py](../../server/vllm/Exp_fullseg/exp_fullseg_runner.py) | 单题 runner:`run_trajectory`(跑一条轨)、`measure_question`(prefill+reuse) | `run_trajectory` [:28](../../server/vllm/Exp_fullseg/exp_fullseg_runner.py#L28);分隔符开关 [:39](../../server/vllm/Exp_fullseg/exp_fullseg_runner.py#L39);起 demo 子进程 [:51](../../server/vllm/Exp_fullseg/exp_fullseg_runner.py#L51);`measure_question` [:88](../../server/vllm/Exp_fullseg/exp_fullseg_runner.py#L88);**`EXP_FS_SKIP_PREFILL`(只 reuse)/ `EXP_FS_SKIP_REUSE`(只 prefill=A_orig2)** [:95-99](../../server/vllm/Exp_fullseg/exp_fullseg_runner.py#L95-L99) |
| [exp_fullseg_run.py](../../server/vllm/Exp_fullseg/exp_fullseg_run.py) | 40 题驱动(可断点续跑:跳过已有 `traj.json`) | 读 `EXP_FS_OUT/QUESTIONS/N` |
| [exp_fullseg_merge.py](../../server/vllm/Exp_fullseg/exp_fullseg_merge.py) | **本次新写**:把 C、B 的 reuse 臂合成 `{prefill:B, reuse:C}`,让现有分析脚本原样跑跨臂对比 | `merge()` 构造合成 traj |
| [exp_fullseg_analyze.py](../../server/vllm/Exp_fullseg/exp_fullseg_analyze.py) | 环2(决策偏离)+ 环3(内容分叉:followup/报告 URL Jaccard) | `first_divergence`、`report_url_jac` |
| [exp_fullseg_score.py](../../server/vllm/Exp_fullseg/exp_fullseg_score.py) | 环4:autorater(qwen-flash via DashScope)给报告打 F1 + bootstrap CI + on-topic 分层 | `score_report` [:74](../../server/vllm/Exp_fullseg/exp_fullseg_score.py#L74) |
| [search.py](../../../deep_researcher_demo/deep_researcher_demo/search.py) | demo 搜索 + 缓存;**本次加 `SEARCH_PATH=` 日志** | logger [:22](../../../deep_researcher_demo/deep_researcher_demo/search.py#L22);`SEARCH_PATH=LIVE(record)` [:345](../../../deep_researcher_demo/deep_researcher_demo/search.py#L345);`SEARCH_PATH=CACHE(replay)` [:373](../../../deep_researcher_demo/deep_researcher_demo/search.py#L373);first-write-wins(record 不覆盖旧缓存)[:454](../../../deep_researcher_demo/deep_researcher_demo/search.py#L454),[:460](../../../deep_researcher_demo/deep_researcher_demo/search.py#L460) |
| [config.py](../../../deep_researcher_demo/deep_researcher_demo/config.py) | demo 读 `SEARCH_CACHE`(模式)/ `SEARCH_CACHE_DIR`(目录) | [:86](../../../deep_researcher_demo/deep_researcher_demo/config.py#L86)、[:88](../../../deep_researcher_demo/deep_researcher_demo/config.py#L88) |
| [cli.py](../../../deep_researcher_demo/deep_researcher_demo/cli.py) | demo 入口调 `wrap_with_cache(mode, cache_dir)` | [:85-88](../../../deep_researcher_demo/deep_researcher_demo/cli.py#L85-L88) |
| [Qwen3-32B/config.yaml](../../server/vllm/Qwen3-32B/config.yaml) | LMCache server 配置(带 kv-transfer-config)| — |
| `/home/yilin/tmp/config_vanilla_p1.yaml` | vanilla server 配置(**去掉 kv-transfer-config** = 无 LMCache,port 30001)| — |

### 3.1 blender.py 关键行(三模式 + 日志 + patch C)
- **control 双开关**(根除"路径残留→静默全重算"坑):`LMCACHE_BLEND_CONTROL` 路径 [:100](../../lmcache/v1/compute/blend/blender.py#L100) + `LMCACHE_BLEND_CONTROL_ENABLED` 开关 [:106](../../lmcache/v1/compute/blend/blender.py#L106),两者**同时**设才走 control。
- **`BLEND_PATH=` 日志**(grep 可核实实际走哪条):CONTROL 分支 [:178](../../lmcache/v1/compute/blend/blender.py#L178);CacheBlend 分支 [:206](../../lmcache/v1/compute/blend/blender.py#L206)(`topk_num=max(int(N*ratio),1)` [:204](../../lmcache/v1/compute/blend/blender.py#L204),ratio=0 → 复用~100%、重算 1 token)。
- **patch C(探针轨算 M 用,本次写,纯复用实验里恒不触发)**:`_dump_q_layers` 初始化 [:84](../../lmcache/v1/compute/blend/blender.py#L84);三重 gate `dump_active` [:148](../../lmcache/v1/compute/blend/blender.py#L148) 内多抓 query [:163-164](../../lmcache/v1/compute/blend/blender.py#L163-L164);落盘 `_flush_dump` [:299](../../lmcache/v1/compute/blend/blender.py#L299) 加 `"q"` 键 [:333-335](../../lmcache/v1/compute/blend/blender.py#L333-L335)。

---

## 四、服务器配置 + 环境变量(blend 三模式)

> 直接用 `/home/yilin/anaconda3/envs/lmcache/bin/vllm serve` + `export`(**别走 conda run**,可能不传环境变量)。

### C(pure-reuse,CacheBlend 纯复用)
```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONHASHSEED=0
export LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=true LMCACHE_MAX_LOCAL_CPU_SIZE=16
export LMCACHE_ENABLE_BLENDING=true LMCACHE_BLEND_SPECIAL_STR='<|fim_pad|>'
export LMCACHE_USE_LAYERWISE=true LMCACHE_SAVE_UNFULL_CHUNK=true LMCACHE_SAVE_DECODE_CACHE=false
export LMCACHE_BLEND_CHECK_LAYERS=1 LMCACHE_BLEND_RECOMPUTE_RATIOS=0   # 0=纯复用
# 关键:不设 LMCACHE_BLEND_CONTROL / CONTROL_ENABLED → 走原生 CacheBlend
/home/yilin/anaconda3/envs/lmcache/bin/vllm serve \
  --config /home/yilin/LMCache/server/vllm/Qwen3-32B/config.yaml \
  --max-logprobs 200 --return-tokens-as-token-ids > /home/yilin/tmp/server_C_purereuse.log 2>&1 &
```
→ 服务在 **:30000**。

### B(vanilla,纯 prefill 无 LMCache)
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONHASHSEED=0
# 不设任何 LMCACHE_*;config_vanilla_p1.yaml 已删 kv-transfer-config
/home/yilin/anaconda3/envs/lmcache/bin/vllm serve \
  --config /home/yilin/tmp/config_vanilla_p1.yaml \
  --max-logprobs 200 --return-tokens-as-token-ids > /home/yilin/tmp/server_B_vanilla.log 2>&1 &
```
→ 服务在 **:30001**。它收到 reuse 臂的 `kv_transfer_params` 会打 `Got kv_transfer_params, but no KVConnector found. Disabling KVTransfer` → **全重算**(= B 臂)。

---

## 五、运行顺序(只用 4 卡 → C、B 顺序跑)

> 约束:一次只能开一个 TP4 server。先 C 后 B。

1. **启 C server**(§4),轮询 `:30000` 直到返回 `Qwen3-32B`。
2. **跑 C driver**(reuse-only,replay,共享缓存):
   ```bash
   cd /home/yilin/deep_researcher_demo
   EXP_FS_OUT=/home/yilin/tmp/exp_fullseg_pure EXP_FS_QUESTIONS=/home/yilin/tmp/exp_a_questions.json \
   EXP_FS_N=40 EXP_FS_Q_TIMEOUT=2400 EXP_FS_SKIP_PREFILL=1 \
   EXP_SERVER=http://localhost:30000 OPENAI_BASE_URL=http://localhost:30000/v1 MODEL=Qwen3-32B OPENAI_API_KEY=EMPTY \
   SEARCH_CACHE=replay SEARCH_CACHE_DIR=/home/yilin/deep_researcher_demo/eval/results/search_cache \
   SEARCH_PROVIDER=duckduckgo EXP_DEMO_CWD=/home/yilin/deep_researcher_demo \
   /home/yilin/anaconda3/envs/gpt-deep/bin/python /home/yilin/LMCache/server/vllm/Exp_fullseg/exp_fullseg_run.py \
     > /home/yilin/tmp/exp_fullseg_pure.log 2>&1 &
   ```
3. **第 1 题验证 C**(见 §6):`BLEND_PATH=CacheBlend reuse~100%/0 CONTROL` + `SEARCH_PATH=CACHE` 命中率。
4. **C 跑完 → 交接 B**:按 **worker PID** 杀 C server(见 §7 教训)→ 验 GPU 释放 → 启 B server(§4)。
5. **跑 B driver**:同步骤 2,但 `EXP_FS_OUT=.../exp_fullseg_van`、`:30001`。
6. **第 1 题验证 B**:`BLEND_PATH` **0 条**(纯 prefill)+ `SEARCH_PATH=CACHE` 命中率。
7. **A_orig2(噪声底,A 模式)** —— B 跑完后**复用同一个 vanilla server(0-3:30001,无 LMCache)**,只跑 prefill 臂、**无分隔符**:
   ```bash
   cd /home/yilin/deep_researcher_demo
   EXP_FS_OUT=/home/yilin/tmp/exp_fullseg_A2 EXP_FS_QUESTIONS=/home/yilin/tmp/exp_a_questions.json \
   EXP_FS_N=40 EXP_FS_Q_TIMEOUT=2400 EXP_FS_SKIP_REUSE=1 \    # ← 只跑 prefill 臂(KV_REUSE_SEPARATOR="")
   EXP_SERVER=http://localhost:30001 OPENAI_BASE_URL=http://localhost:30001/v1 MODEL=Qwen3-32B OPENAI_API_KEY=EMPTY \
   SEARCH_CACHE=replay SEARCH_CACHE_DIR=/home/yilin/deep_researcher_demo/eval/results/search_cache \
   SEARCH_PROVIDER=duckduckgo EXP_DEMO_CWD=/home/yilin/deep_researcher_demo \
   /home/yilin/anaconda3/envs/gpt-deep/bin/python /home/yilin/LMCache/server/vllm/Exp_fullseg/exp_fullseg_run.py \
     > /home/yilin/tmp/exp_fullseg_A2.log 2>&1 &
   ```
   验证:`sep=off`(无分隔符)、`BLEND_PATH` **0 条**(无 LMCache、纯 prefill)、prefill 报告非空。
   → 它和 A_orig 互比 = **40 题噪声底**(纯非确定性)。

> driver 可断点续跑:已有 `traj.json` 的题自动跳过(`[N/40] qX 已完成,跳过`)。

---

## 六、验证方法(边跑边查,别等跑完)

```bash
# 模式(server 日志):C 该全是 CacheBlend、0 CONTROL;B 该 0 条
grep -ac 'BLEND_PATH=CacheBlend' /home/yilin/tmp/server_C_purereuse.log
grep -ac 'BLEND_PATH=CONTROL'    /home/yilin/tmp/server_C_purereuse.log   # =0
grep -ac 'BLEND_PATH'            /home/yilin/tmp/server_B_vanilla.log     # =0

# 搜索缓存命中率(driver 日志):确认 replay 真用缓存、不是全 live
awk -F'命中=|cold_live=' '/SEARCH_PATH=CACHE/{h+=$2;c+=$3}
  END{printf "命中率 %.0f%% (命中%d/冷%d)\n",(h+c>0)?100*h/(h+c):0,h+0,c+0}' /home/yilin/tmp/exp_fullseg_pure.log
```
本轮实测:C 命中率 60%、B 82%(B 后跑,replay 了 C 记下的冷 miss);两臂模式均正确。

---

## 七、分析流程(B 跑完后)

```bash
cd /home/yilin/LMCache/server/vllm
# 1) 合成跨臂 traj:CvB(prefill槽=B, reuse槽=C)、BvA(prefill槽=A_orig, reuse槽=B)
/home/yilin/anaconda3/envs/gpt-deep/bin/python Exp_fullseg/exp_fullseg_merge.py
# 2) 核心:纯复用效应 C−B
EXP_FS_OUT=/home/yilin/tmp/exp_fullseg_CvB python Exp_fullseg/exp_fullseg_analyze.py   # 环2/3
EXP_FS_OUT=/home/yilin/tmp/exp_fullseg_CvB python Exp_fullseg/exp_fullseg_score.py     # 环4(autorater)
# 3) 辅:分隔符效应 B−A_orig(带 confound)
EXP_FS_OUT=/home/yilin/tmp/exp_fullseg_BvA python Exp_fullseg/exp_fullseg_analyze.py
EXP_FS_OUT=/home/yilin/tmp/exp_fullseg_BvA python Exp_fullseg/exp_fullseg_score.py
```
- 合成脚本把"两个独立 run 的 reuse 臂"塞进一个 traj 的 prefill/reuse 槽 → 现有"prefill vs reuse"脚本原样可用,输出的 `pf−ru` 即 `B−C`(基线−处理)。
- 环4 主判据 = autorater F1 差(Set/Single 分开)+ bootstrap 95% CI;on-topic 分层(judge 多判);旁证 = %答对差、本地 exact。
- **局限**:本轮无噪声底(去了 A_new/A_new2),C−B 含单轮非确定性 → 靠 bootstrap CI 判系统性差异;且搜索仅 ~60-80% held(query 计划漂移导致冷 miss)。

---

## 八、数据 / 日志位置

| 路径 | 内容 |
|---|---|
| `/home/yilin/tmp/exp_fullseg_pure/qN/traj.json` | C:`{prefill:null, reuse:{decisions,report}}` |
| `/home/yilin/tmp/exp_fullseg_van/qN/traj.json` | B:同上 |
| `/home/yilin/tmp/exp_fullseg/qN/traj.json` | A_orig(prefill 臂有效) |
| `/home/yilin/tmp/exp_fullseg_{CvB,BvA}/` | merge 合成的跨臂 traj |
| `/home/yilin/deep_researcher_demo/eval/results/search_cache/` | 共享搜索缓存(replay 读、冷 miss 写;first-write-wins 不覆盖) |
| `/home/yilin/tmp/server_{C_purereuse,B_vanilla}.log` | server 日志(grep `BLEND_PATH`) |
| `/home/yilin/tmp/exp_fullseg_{pure,van}.log` | driver 日志(grep `SEARCH_PATH`) |
| `exp-docx/ExpB_fullseg/mainline_{env23,env4}.json` | 分析脚本输出 |

---

## 九、这次踩的坑 + 教训

1. **搜索缓存模式**:`SEARCH_CACHE=record` 是**每个 query 都联网 live**(只记录、不读缓存加速,[search.py:345](../../../deep_researcher_demo/deep_researcher_demo/search.py#L345));要 held 住搜索必须用 **`replay`**([:373](../../../deep_researcher_demo/deep_researcher_demo/search.py#L373))。判定靠 `SEARCH_PATH=` 日志,别靠猜。
2. **慢的瓶颈是 LLM 不是搜索**:每题 ~70 次 LLM 调用(3×3×3 扇出),32B 单调用 ~4-5s;replay 省的是搜索那小头。
3. **kill vllm server**:`kill -TERM -$pgid` **杀不到 worker**(worker 自成进程组)→ 残留占卡 → 下一个 server OOM 崩。**正解:`nvidia-smi --query-compute-apps=pid` 列 worker PID 直接 kill**;且**先按用户过滤**(别误杀别人的,如同机 armaan 的进程)。
4. **并行 vs 顺序**:两个 driver 共享同一搜索缓存目录会**写竞争**(`_write_json` 固定 `.tmp` 名、`_CACHE_LOCK` 仅进程内)→ 并行须各用一份缓存副本;顺序跑则共享缓存最干净(先跑的冷 miss 被后跑 replay,held 更齐)。
5. **跳过 prefill 臂安全**:reuse 臂的 KV 复用 = **既有的 CacheBlend + decode→prefill KV reuse(`blend_store_generated`)+ warmup 请求**(之前已实现/验证的功能)——它本就在 reuse 臂自己的轨迹内"decode 生成 summary→存 KV→prefill 决策时复用",**不依赖同题 prefill 臂**。所以 `EXP_FS_SKIP_PREFILL=1` 跳过 prefill 臂安全;搜索侧用 replay 读预存缓存。已 q0 实测(报告非空 + `BLEND_PATH=CacheBlend`)。

---
*相关:[mainline_40q_report.md](mainline_40q_report.md)(结果报告)、[全段实验.md](../全段实验.md)(总计划,含探针轨/解法验证那条独立线)、[claude-docx/14](../../claude-docx/14-blend-control-truth-pitfall.md)(control 坑);memory `blend-three-modes-env`。*
