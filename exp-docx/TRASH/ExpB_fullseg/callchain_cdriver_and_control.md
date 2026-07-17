# C driver 调用链 + Control 功能详解

> 两部分:**(一)** 跑 C(纯复用)driver 从命令到服务端 CacheBlend KV 复用的完整调用链;
> **(二)** Control 功能是什么、怎么实现、调用链。所有引用带文件 + 行号。

---

# 一、C driver 调用链(命令 → agent → 服务端 KV 复用)

一句话:**driver 循环起子进程跑 demo;demo 里 agent 每生成一段 summary 就把它的 KV 存进 LMCache(`blend_store_generated`),拼决策 prompt 时用分隔符切段,服务端 CacheBlend 直接复用这些存好的 summary KV。**

## 1. 驱动层 — [exp_fullseg_run.py](../../server/vllm/Exp_fullseg/exp_fullseg_run.py)
```
main() 循环 N 题 [:20]
  └─ 有 traj.json 的题跳过(断点续跑)[:25]
  └─ subprocess.run([python, exp_fullseg_runner.py, q, sid], env=os.environ) [:30]
```
透传所有 env(含 `EXP_FS_SKIP_PREFILL`、`SEARCH_CACHE`、`EXP_SERVER` 等)给 runner。

## 2. Runner 层 — [exp_fullseg_runner.py](../../server/vllm/Exp_fullseg/exp_fullseg_runner.py)
```
measure_question(q, sid) [:88]
  └─ skip_prefill = EXP_FS_SKIP_PREFILL [:95]  → 本轮只跑 reuse 臂
  └─ run_trajectory(q, sid, qdir, "reuse") [:96]
        └─ env["KV_REUSE_SEPARATOR"] = SEP="<|fim_pad|>"  (reuse 模式)[:39]
        └─ subprocess.run([python, -m deep_researcher_demo, --output report, q], env, cwd=DEMO_CWD) [:51]
  └─ json.dump({prefill:None, reuse:{decisions,report}}, traj.json)
```
**关键:`KV_REUSE_SEPARATOR=<|fim_pad|>` 是开启复用的总开关** —— 它传进 demo,决定 summary 之间用原子特殊 token 切段(给 blend 切成可复用 segment)。

## 3. demo / agent 层 — [deep_researcher_demo](../../../deep_researcher_demo/deep_researcher_demo/)
入口 [cli.py](../../../deep_researcher_demo/deep_researcher_demo/cli.py) → 读 `KV_REUSE_SEPARATOR`([config.py:85](../../../deep_researcher_demo/deep_researcher_demo/config.py#L85))→ `workflow.run(question)`。

### 3a. warmup(复用的前置,必需)— [workflow.py:62-69](../../../deep_researcher_demo/deep_researcher_demo/workflow.py#L62-L69)
```
if kv_reuse_separator:
    await supervisor.warmup_kv_prefix(max_followups)   # agents.py:194
    await final_writer.warmup_kv_prefix()              # agents.py:516
```
warmup 发一个带分隔符的小请求([agents.py:204-214](../../../deep_researcher_demo/deep_researcher_demo/agents.py#L204-L214)),`store_generated_kv=True` → **把"决策/报告角色的公共前缀 KV"先存进 LMCache**,这样后面真请求里 summary 段第一次出现就能被复用。

### 3b. 研究循环(每轮产 summary,存 KV)— [workflow.py:92-110](../../../deep_researcher_demo/deep_researcher_demo/workflow.py#L92-L110)
```
for iteration in 1..max_iterations [:92]
  round_results = _run_researchers(current_questions) [:104]
     └─ researcher.summarize_results(...) [agents.py:316]
          └─ LLM 调用,store_generated_kv=bool(sep)=True, tag="RESEARCH_SUMMARY_TEXT" [agents.py:358-359]
                └─ llm.py: payload["kv_transfer_params"]={"lmcache.blend_store_generated":True} [llm.py:159]
                   → 服务端把这段 summary 的 decode KV 存下来(decode→prefill 复用的来源)
  summaries.extend(round_summaries) [:107]
  decision = supervisor.decide(summaries=summaries, ...) [:110 / agents.py:219]
```

### 3c. 拼决策 prompt(用分隔符切段)— [agents.py:39-53](../../../deep_researcher_demo/deep_researcher_demo/agents.py#L39-L53)
```
join_reusable_segments(parts, separator):
   sep 为空  → "\n\n---\n\n".join(parts)        # prefill 基线
   sep=fim_pad → sep + sep.join(parts) + sep    # 每段两侧都加分隔符 → blend 各自切段复用
```
`supervisor.decide` 把各 summary 用 `join_reusable_segments` 拼进 findings([agents.py:234](../../../deep_researcher_demo/deep_researcher_demo/agents.py#L234))→ 发决策 LLM 请求([agents.py:248-253](../../../deep_researcher_demo/deep_researcher_demo/agents.py#L248))。**这次 prefill 就是复用前面存好的 summary KV 的地方。**

## 4. 服务端层(vLLM + LMCache)
决策 prompt(带分隔符)到服务端,LMCache 发现这些 summary 段命中了存好的 KV → 触发 blend:
```
vLLM forward → LMCache connector
  └─ [vllm_v1_adapter.py:902] self.blender.blend(tokens[:hit], mask[:hit], kvcaches, slot_mapping, ...)
        └─ [blender.py:370] blend()
              ├─ _load_control() [:383]   → C 没设 control → _control=None
              └─ blend_layer(tokens, mask) [调用:384 / 定义:239] 逐层:
                   ├─ compute_layer(tokens)   现场重算(layerwise)
                   ├─ retrieve_layer(tokens, mask)  载入存好的 KV^r
                   └─ process_qkv(q,k,v,...) [:110]  ← 融合点
```
`process_qkv` 里 `_control is None` → **走 Normal CacheBlend 分支**([blender.py:199+](../../lmcache/v1/compute/blend/blender.py#L199)):
- 在 `check_layers` 算各 token 偏差,选 top-k 重算:`topk_num = max(int(total_len * ratio), 1)`([:204](../../lmcache/v1/compute/blend/blender.py#L204));**ratio=0 → 只重算 1 个 token,其余全复用存好的 KV**。
- 打日志 `BLEND_PATH=CacheBlend total=N | recompute(top-k)=1 (0.x%) | reuse=N-1 (99.x%)`([:206](../../lmcache/v1/compute/blend/blender.py#L206))。

**这就是 C 臂的"纯复用":summary KV 复用 ~99%、重算 ~1 token。** grep `BLEND_PATH=CacheBlend` 即可核实。

---

# 二、Control 功能详解

## 1. 是什么
Control 是给 **Exp A 单段诊断 / 探针轨 |Δz|** 加的**人工接管开关**:用一个 JSON 文件**强行指定哪些 token 复用(KV^r)、哪些重算(KV\*)**,**覆盖** CacheBlend 自动 top-k。两个用途:
- `{"reuse_token_ranges": []}`(空)→ **全部重算 = KV\* = truth = 等价 full prefill**(reuse 0%)。
- `{"reuse_token_ranges": [[s,e],...]}` → **只这些区间复用**、其余重算 → 测"复用段 s 单独造成多少决策偏移"。

### control 文件长什么样 + 怎么控制 reuse 范围(举例)
就一个 JSON:
```json
{"reuse_token_ranges": [[120, 200]], "dump": false, "dump_q": false}
```
- **`reuse_token_ranges`**:`[[s,e],...]`,**命中 token 序列里的下标区间(左闭右开 `[s,e)`),列出"要保留复用"的段** = 复用白名单。核心字段。
- `dump`/`dump_q`:可选,诊断时要不要 dump KV / query(默认不写)。

**机制**(control 分支 [blender.py:172-194](../../lmcache/v1/compute/blend/blender.py#L172),`_controlled_reuse_indices` 把 `[s,e]` 展成 `range(s,e)` 下标 [:362](../../lmcache/v1/compute/blend/blender.py#L362)):
1. 全部 token 先重算一遍 → 干净 KV\*(`old_k[:]=k`);
2. 把 `reuse_token_ranges` 列的 token 改回载入的复用值 KV^r(`old_k[reuse_idx]=reuse_k`);
3. → **白名单里的段复用旧 KV、没列的全重算**。

**举例**(设某次 blend 命中 300 token,3 段 summary:段1=[50,120)、段2=[120,200)、段3=[200,290)):

| control 文件 | 效果 | 复用 |
|---|---|---|
| `{"reuse_token_ranges": []}` | 白名单空 → **全重算 = KV\* = 等价 full prefill** | 0%(truth) |
| `{"reuse_token_ranges": [[120,200]]}` | **只段2 复用**,其余 220 token 重算 | 段2 |
| `{"reuse_token_ranges": [[50,120],[200,290]]}` | 段1+段3 复用,段2 等重算 | 段1+段3 |
| `{"reuse_token_ranges": [[0,300]]}` | 全部复用、无重算 | 100% |

**Exp A / 探针轨用法**(编排器每个 pass 前写一次 control 文件,blend 即读到):
- **truth pass**:`{"reuse_token_ranges": [], "dump": true}` → 全重算 → 基准决策 + dump KV\*。
- **swap pass(测段 i)**:`{"reuse_token_ranges": [[s_i,e_i]]}` → 只复用段 i、其余真算 → `|Δz_i|=|swap决策 − truth决策|` = 单独复用段 i 把决策推了多少。
- 段下标 `[s_i,e_i]` 由 harness 从 prompt + 分隔符算出(`find_segments`/`classify_segments`)再写进 control。

## 2. 怎么实现([blender.py](../../lmcache/v1/compute/blend/blender.py#L20))
### 2a. 双开关(默认关,根除残留坑)
```
self._control_path    = os.getenv("LMCACHE_BLEND_CONTROL", "")          [blender.py:100]
self._control_enabled = os.getenv("LMCACHE_BLEND_CONTROL_ENABLED","0")  [blender.py:106]
```
**两者同时设才生效**;任一缺 → `_control=None` = 正常 CacheBlend。(为什么加 ENABLED 开关:见 §4 坑。)

### 2b. per-blend 重读 control 文件 — [_load_control() blender.py:348](../../lmcache/v1/compute/blend/blender.py#L348)
```
def _load_control():
    if not self._control_enabled or not self._control_path:  # 任一缺
        self._control = None; return
    self._control = json.load(open(self._control_path))      # 每次 blend 都重读
```
在 `blend()` 里每次调用([blender.py:383](../../lmcache/v1/compute/blend/blender.py#L383))→ **编排器每个请求前写一次 control 文件,blend 就读到对应设置**(truth pass / swap pass 切换)。

### 2c. reuse_token_ranges → token 下标 — [_controlled_reuse_indices() blender.py:362](../../lmcache/v1/compute/blend/blender.py#L362)
把 `[[s,e],...]` 展平成要"保留复用"的 token 索引;**空 → None(全重算 = truth)**。

### 2d. 控制分支(clone-revert,关键)— [process_qkv blender.py:172-196](../../lmcache/v1/compute/blend/blender.py#L172)
```
if self._control is not None:
    reuse_idx = _controlled_reuse_indices(...)              # 要保留复用的 token
    [layer 0 打日志 BLEND_PATH=CONTROL reuse=ns/recompute=...]   # :178
    if reuse_idx 非空:
        reuse_k = old_k[reuse_idx].clone()  # 先存下这些段的复用值 KV^r
        old_k[:] = k                        # 全部覆盖成现算的 KV*(全重算)
        old_k[reuse_idx] = reuse_k          # 再把指定段改回复用值
    else:                                   # 复用空 = truth
        old_k[:] = k                        # 全部 = 现算 KV* = full prefill
    return (q, old_k, old_v, ...)           # 不做 subset
```
**为什么这么绕(先全算、再改回)**:CacheBlend 正常的"只重算 top-k 子集"会把 blend 之后的自由生成搞坏(decode 出空/乱码)。诊断要 blend 后自由生成拿决策分布,所以**先全重算保住一份完整干净 KV、再把指定 复用段强制改回复用值**;`reuse_token_ranges=[]` 退化成纯全重算 = truth。

## 3. 调用链(和 CacheBlend 共用入口,只在 process_qkv 分叉)
```
vllm_v1_adapter.blend() [adapter:902]
  └─ blender.blend() [blender:370]
        ├─ _load_control()  [:383]  ──读 control 文件→ _control = {"reuse_token_ranges":...} 或 None
        └─ blend_layer() 逐层 → process_qkv() [:110]
              ├─ _control 非空 → CONTROL 分支(clone-revert)[:172]  → BLEND_PATH=CONTROL
              └─ _control is None → Normal CacheBlend 分支 [:199]   → BLEND_PATH=CacheBlend
```

## 4. 坑 + 修复(为什么有 CONTROL_ENABLED)
- **坑**:`_load_control` 是 **per-blend 重读**。早先只有路径变量 `LMCACHE_BLEND_CONTROL`,没独立开关 → Exp A 跑完没清,下一个"复用实验"的 server 继承了这个残留路径(指向 `{"reuse_token_ranges":[]}`)→ **每个 blend 被静默强制全重算 = truth = 根本没复用**,还被误判成"复用有害"。详见 [claude-docx/14](../../claude-docx/14-blend-control-truth-pitfall.md)。
- **修复**:加独立开关 `LMCACHE_BLEND_CONTROL_ENABLED`(默认关,[blender.py:106](../../lmcache/v1/compute/blend/blender.py#L106))→ 即便路径残留,开关不开 control 也不触发。**用 control 必须同时设两者。**

## 5. 已验证的等价性
`control + reuse_token_ranges=[]`(全重算)的 KV **==** 干净 full prefill 的 KV:32B/TP4 实测 cos-dist 中位 0.0001(单个离群 token 来自 massive-activation 的存取精度)。→ control 路径**没污染**,可信地用作"指定范围 prefill/复用"的诊断工具。

## 6. 逐轮改重算范围:不用重启 server(per-round / 探针轨)

**常见误解**:"`LMCACHE_BLEND_CONTROL` 是环境变量,server 一起来范围就定死了,改不了重算范围 —— 那 supervisor round1 和 round2 重算范围不同怎么办?"

**真相:环境变量只定『文件路径』,不定『范围』。** 范围写在那个 JSON 文件的**内容**(`reuse_token_ranges`)里,而 `blend()` **每次 blend 都调一次 `_load_control()` 重读文件**([blender.py:383](../../lmcache/v1/compute/blend/blender.py#L383),不是启动时读一次)。所以:

> **改重算范围 = 重写 control 文件(原子写)**,server 不重启、env 不动。下一个 blend 请求就读到新范围。

### 怎么做到逐轮不同范围
**串行**地「写文件 → 发请求 → 再写 → 再发」:
```
写 control = {"reuse_token_ranges": [[s1,e1]]}  → 发 round1 决策请求 → blend 读到 [[s1,e1]]
写 control = {"reuse_token_ranges": [[s2,e2]]}  → 发 round2 决策请求 → blend 读到 [[s2,e2]]
```
只要**同一时刻只有一个 blend 在跑**(请求串行),共享的 control 文件就不会被竞争,每个 blend 拿到的就是它前面刚写的范围。这正是 Exp A harness 现在对 final 决策做的([write_control](../../server/vllm/Exp_A/exp_a_tf_runner.py#L121) 原子写 + 串行 `free_gen_decision`):truth pass 写 `{"reuse_token_ranges":[]}`、每个 swap pass 写 `{"reuse_token_ranges":[[s,e]]}`。

### 为什么不能在 live 轨迹里逐轮控制(要离线 replay)
live demo 是**一个子进程把所有轮自动背靠背跑完**的,round1 决策和 round2 决策之间 harness **插不进去**写文件。所以探针轨的正确姿势(Exp A 既有思路):
1. **harvest 一次**:先跑一遍干净全 prefill(**不带 control**,measure_question 跑前先删 control 文件),把每一轮 supervisor 决策的上下文(`SUPERVISOR_DECISION_JSON`)和各 summary 都记录下来;同时各轮 summary 的 KV 也被存进 LMCache。
2. **离线逐轮 replay**:对**每一轮 r** 的决策上下文,harness 串行地:`find_segments` 算出**该轮**的段 token 区间 → 写 control(该轮该段的 `reuse_token_ranges`)→ 把该轮决策 prompt 重新发一次(命中 step 1 存的 summary KV → 触发 blend → 读当前 control)→ 拿决策分布 / `|Δz|`。

### 举例(round1 vs round2 范围天然不同)
设 round1 决策 prompt 累积了 2 段 summary、round2 累积了 4 段:
- **round1**:`find_segments(prompt_round1)` → 段在 `[[60,130],[130,200]]`;测段1 就写 `{"reuse_token_ranges":[[60,130]]}` 发 round1 请求。
- **round2**:`find_segments(prompt_round2)` → 段在 `[[60,130],[130,200],[200,280],[280,360]]`(prompt 更长、段更多、区间不同);测某段就写对应区间发 round2 请求。

同一个 control 文件,两轮被写入不同 `reuse_token_ranges`;blend 各自重读 → **逐轮范围不同的问题自然解决,全程 server 不重启、control 机制不改**。详见探针轨实现计划。

---
*相关:[experiment_workflow.md](experiment_workflow.md)(本实验完整流程)、[claude-docx/14](../../claude-docx/14-blend-control-truth-pitfall.md)(control 坑 + 修复)、[全段实验.md](../全段实验.md)(探针轨 |Δz| 用 control 那条线);memory `blend-three-modes-env`。*
