# Exp A 完整版 · 代码实现说明(最终设计,改了哪、为什么)

> 对应计划 [EXP-A_KV漂移与决策归因_完整实验计划.md](../EXP-A_KV漂移与决策归因_完整实验计划.md)。
> 模型 Qwen3-32B(64 层 / 8 KV 头 / TP4),测量栈 = vLLM+LMCache。本文档随设计迭代已更新到**最终版**。

---

## 0. 一句话:要证明什么

把"KV 复用漂移"做成一条因果链:**锁住轨迹 → 只换某个 summary 段 s_i 的 KV(真值 KV\* ↔ 复用值 KV^r)→ 看 supervisor 的决策怎么变 → 同时量 s_i 的几何漂移 → 回归:哪个几何量最能预测决策变化 → 它就是 gate 该用的 Δ。**

为什么:几何"偏多少" ≠ 决策"错多少"。gate 要拦的是"会改变决策"的复用,不是几何上偏得多的复用。所以拿**决策变化**当靶子,反过来挑几何量。

---

## 1. 决策怎么度量 —— 按字段拆,不是整段一把梭(最终定稿)

supervisor 输出是 `SupervisorDecision`,**3 个字段**:
```json
{"status": "continue"|"complete",  "followup_questions": [...],  "reason": "..."}
```
按字段各用对的尺子:

| 字段 | 度量 | 为什么 |
|---|---|---|
| `status`(动作) | **continue(token 9534)/ complete(token 14737) 两点的 2-way 精确 KL** + 翻转标志 | 离散二元枢纽;两个都是单 token、互不同 → 取这俩 logprob 归一化算 KL 就是**精确**(非 top-k 近似) |
| `followup_questions`(下一轮搜什么) | 把 truth/swap 的 followups 各拿去 **search → URL 集合 Jaccard 差**;(continue 时才有) | followup 是自由文本,token-KL 被措辞噪声淹没;测**下游检索变没变** = 直接量"轨迹分叉",措辞换法不影响 |
| `reason` | **丢弃** | 事后 justification,自由文本,最噪、非决策 |

**外加:存 truth/swap 两条自由生成的决策全文** → 供之后用 LLM 对比 full-prefill 轨迹 vs swap-reuse 轨迹。

**实现要点(一次自由生成拿全部)**:truth 和每个 swap 各 `temp=0` 自由生成一遍决策 → 在第一个 continue/complete token 那位读两点分布(action KL + 翻转)、解析 followups(Jaccard)、整段文本存档;truth 那遍顺带触发 KV dump(几何曲线)。**不用 teacher-force 整段、不啃上万 token 的 prompt_logprobs**。

---

## 2. 逐处改动(文件 / 为什么)

### 2.1 LMCache · blender 受控 swap + dump opt-in
[lmcache/v1/compute/blend/blender.py](../../lmcache/v1/compute/blend/blender.py)
- `LMCACHE_BLEND_CONTROL=<json>`:每次 blend 读控制文件。`reuse_token_ranges` 为空=truth(全重算=KV*),非空=那些 token 留复用 KV^r、其余重算。
- 受控时在 **layer 0** 用控制覆盖"重算谁"(`imp_indices`),贯穿所有层 → s_i 每层都复用(真实复用语义)。覆盖 imp_indices **不碰 blend 载入/拼接、全段都缓存、无 contiguous 限制**。
- **dump opt-in**:控制文件带 `"dump": true` 且 truth 时才落 KV dump。这样每题只 truth 那一遍落一份(demo/swap 都不落,省 I/O + 不污染)。
- 为什么用控制文件而非请求参数:每个 s_i 单独控制,串行写文件最简单、改动全在 blender 一个文件。

### 2.2 demo · 接 search 缓存 + 强制多轮 knob
[cli.py](../../../deep_researcher_demo/deep_researcher_demo/cli.py) 接两级 per-question 缓存(冻结检索可复现);
[workflow.py](../../../deep_researcher_demo/deep_researcher_demo/workflow.py)+[config.py](../../../deep_researcher_demo/deep_researcher_demo/config.py) 加 `MIN_ROUNDS`(全量用默认 3×3,不强制)。

### 2.3 LMCache · teacher-forcing harness(测量本体)
[server/vllm/Exp_A/exp_a_tf_runner.py](../../server/vllm/Exp_A/exp_a_tf_runner.py)
- demo 产轨迹 → 取最后一轮上下文 → 按分隔符切段 → **用 harvest 记录精确匹配**每段是哪条 summary、哪一轮(不靠"像不像 JSON"的猜)。
- truth(`{reuse_token_ranges:[],dump:true}`)+ 每 summary swap(`{reuse_token_ranges:[[s,e]]}`)各自由生成 → action 2-way KL / 翻转 / followup Jaccard / 轨迹文本。
- 落 `events.jsonl`(度量)+ `trajectories.json`(决策全文)+ `meta.json`。

### 2.4 LMCache · 几何 + 回归
[server/vllm/Exp_A/exp_a_regress.py](../../server/vllm/Exp_A/exp_a_regress.py)
- `geom <qdir>`:从 truth dump 算每段 6 个几何量(K/V × {方向 1-cos、纯幅度 `|‖KV^r‖-‖KV*‖|/‖KV*‖`、欧氏 ‖Δ‖},整体 mean/max + 浅/中/深分段)→ `geometry.json`;并做 **RoPE 前缀核对**(纯 P0 漂移逐层应平坦≈bf16 地板,不平坦=位置错位)。
- 无参:汇总所有题,几何特征 vs **action_kl / followup_jaccard** 求 Pearson/Spearman → 排名、Δ 候选、欧氏 vs 方向 vs 幅度家族对照。

### 2.5 LMCache · 全量 driver
[server/vllm/Exp_A/exp_a_run.py](../../server/vllm/Exp_A/exp_a_run.py):循环 N 题,每题 harness → 算几何 → **删 2GB dump**(60 题不然 ~120GB);可断点续;末尾汇总。

---

## 3. ③.5 单题验证时**实跑暴露并修掉的 bug**(真问题)
1. **prompt_logprobs 把 engine 干崩**:它在 prompt 位置 0 放 `-1` 哨兵,vLLM 反解码 `decode([-1])` 负数转无符号 → OverflowError → 整个 engine 挂。**改用生成 logprobs**(只看生成 token,不碰位置0)+ 服务端 `--return-tokens-as-token-ids`(候选给 id、跳过反解码)。
2. **段分类**:原假设严格 `[out,r,out,r]` 交替;实际每轮多 researcher → 多 summary 只 1 个 r。改用 **harvest 记录精确匹配** + 标轮次。
3. **chat 模板没对齐**:重建 prompt 漏 `enable_thinking=false` → token 对不上、blend 命中不了。补上。
4. **API 模型名**:拿路径当 model 字段,server 认 served name `Qwen3-32B`。拆开。
5. **dump 闸太松**:无控制时也落 → demo 自跑落一堆无效 dump。收紧成 **truth + 显式 opt-in** 才落。
6. **RoPE 核对阈值太严**:1e-3 对 bf16 太苛刻(精度地板≈0.0127、逐层平坦),放到 0.03 + 看逐层是否平坦。

## 4. 怎么跑(全量 60 题)
```bash
# server(GPU 0,2,6,7;dump/control 用固定共享路径)
LMCACHE_BLEND_DUMP_KV=/home/yilin/tmp/exp_a_shared/dump \
LMCACHE_BLEND_CONTROL=/home/yilin/tmp/exp_a_shared/control.json \
  <32B+LMCache+CacheBlend 启动> --max-logprobs 200 --return-tokens-as-token-ids
# driver(60 题,默认 3×3 多轮,record 检索)
EXP_A_SERVER=http://localhost:30000 EXP_A_SERVED_MODEL=Qwen3-32B MODEL=Qwen3-32B \
EXP_A_DUMP_DIR=/home/yilin/tmp/exp_a_shared/dump EXP_A_CONTROL=/home/yilin/tmp/exp_a_shared/control.json \
SEARCH_CACHE=record  python server/vllm/Exp_A/exp_a_run.py
```

## 5. 跑前必过的核对 + 产出
- RoPE 前缀≈bf16 地板(平坦)、决策 token 锁对(9534/14737)、GQA H=8、swap 只动一段。
- 原始 KV dump ~2GB/题 → `/home/yilin/tmp/exp_a_shared/dump`,**算完即删**;结果 → `exp-docx/ExpA_kv_drift/run_full/`。
- Δ 定义(哪个 K/V 量、哪些层)→ 交 Phase 2 gate;truth/swap 轨迹 → 供 LLM 对比分析。

## 6. 二期(未做,文档标明)
- 全词表 logits dump(patch B);swap-one-layer 单层归因;曲线 7(注意力输出)+ 注意力 M(都需 attention hook)。
- followup Jaccard 现走 record 检索(swap 的新 followup 可能联网搜);可加嵌入余弦作零搜索次选。
