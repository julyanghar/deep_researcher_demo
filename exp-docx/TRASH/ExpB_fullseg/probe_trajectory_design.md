# 探针轨设计:主轨 = full prefill·带分隔符,reuse 是分叉

> 探针轨要回答:**沿着一条干净的 agent 轨迹,每个决策点上,"复用第 k 段 summary 的 KV"相对"全重算它"把决策推了多少(|Δz_k|)** —— 即哪些段在驱动决策。
> **骨架:主轨全程 full recompute(干净参考),只在测量点临时分叉一次 reuse;主轨绝不能让 reuse 驱动。**
> ⚠ 当前 [exp_a_tf_runner.py](../../server/vllm/Exp_A/exp_a_tf_runner.py) 的 `run_demo` 是用**普通 reuse** 跑主轨的(取巧,只为取一个 prompt 切段),**不符合本设计**;按本文改。

---

## 一、骨架图

```
主轨(全程 control-truth = full recompute,但带分隔符 + 存段 KV):
  round1 supervisor 决策(干净) ──fork──► swap 段k → |Δz_1k|   (分叉测完即弃,不回灌主轨)
  round2 supervisor 决策(干净) ──fork──► swap 段k → |Δz_2k|
   ⋮
  每个决策点都从【干净全重算的决策】出发,逐段分叉一次 reuse
```
- **主轨**:决定"实际走哪条剧情"(每轮哪些 summary、哪个决策),**必须干净**。
- **分叉(swap)**:在某个决策点的**固定 prompt** 上,只把段 k 改成复用 KV^r、其余仍全重算,看决策分布偏移。分叉**不影响主轨往下走**。

## 二、为什么主轨必须 full prefill·带分隔符(不能 reuse 驱动)

1. **参考要干净**:|Δz_k| 是"复用段 k **相对干净全重算**"的偏移。若主轨本身 reuse 驱动,reuse 误差已一路累积进每条 summary、每个决策 → 主轨自己是脏的 → |Δz| 测的是"脏上加脏",归因不到单段。
2. **单点隔离**:干净主轨上每个分叉只引入**一次**reuse(单段)→ |Δz_k| 干净归因。reuse 驱动主轨做不到(误差到处都是)。
3. **沿轨前进**:多决策点探针轨要跟着主轨一轮轮往前,每轮从干净决策分叉 → 主轨必须是那条确定、干净的 full prefill 轨。

## 三、"full prefill·带分隔符"怎么配(3 要素,缺一不可)

| 要素 | 怎么设 | 为什么 |
|---|---|---|
| **带分隔符** | demo 端 `KV_REUSE_SEPARATOR=<\|fim_pad\|>` | 段才切得出([find_segments](../../server/vllm/Exp_A/exp_a_tf_runner.py#L82))、KV 才存得下 |
| **存段 KV** | summary decode `store_generated_kv=True`([agents.py:358](../../../deep_researcher_demo/deep_researcher_demo/agents.py#L358)) | 分叉要复用的 KV^r 来源 |
| **决策全重算** | server control-truth:`reuse_token_ranges=[]` | 主轨决策 = 全重算 = 等价 full prefill·带分隔符那条 |

> 即:**主轨 = 在 control-truth 下跑 demo**(分隔符开、存 KV,但每个 supervisor 决策 blend 全重算)。分隔符开但要全重算,只能靠 control-truth(`[]`)强制——普通 CacheBlend 会自动复用,就脏了。

## 四、调用链

### 0. 起 control server(双开关都设)
```bash
export LMCACHE_ENABLE_BLENDING=true LMCACHE_BLEND_SPECIAL_STR='<|fim_pad|>'
export LMCACHE_USE_LAYERWISE=true LMCACHE_BLEND_CHECK_LAYERS=1
export LMCACHE_BLEND_CONTROL_ENABLED=1                                    # 开 control
export LMCACHE_BLEND_CONTROL=/home/yilin/tmp/exp_a/<sid>/blend_control.json
export LMCACHE_BLEND_DUMP_KV=/home/yilin/tmp/exp_a/<sid>/dump             # 要测几何 Δ / 算 M 才设
vllm serve --config Qwen3-32B/config.yaml ...
```

### 1. 跑主轨(control-truth 驱动,产出干净轨迹 + 所有决策 prompt + 存好的段 KV)
```
write_control(control, [])                 # 主轨期间 control 文件 = {"reuse_token_ranges":[]} = 全重算
run_demo(question)  (KV_REUSE_SEPARATOR=SEP, store_generated_kv=True)
  → 每个 supervisor 决策 blend 都 per-blend 重读 control=[] → 全重算 → 干净决策
  → harvest 记下【每一轮】的 SUPERVISOR_DECISION_JSON(决策 prompt)
  → 各 summary 的 KV 已 store 进 LMCache(给分叉复用)
```
**和当前代码的差:** 现在 `run_demo` 删掉 control 文件 → 走普通 CacheBlend reuse(脏);**应改成主轨期间 control 写 `[]`(control-truth)**。

### 2. 逐决策点 + 逐段分叉(offline 重放每轮的决策 prompt)
```
for 每一轮决策 call in harvest(不再只取最后一轮 pick_decision_call):   ← patch C 要扩这里
    prompt_ids = context_token_ids(call.messages)         # 那轮的决策 prompt
    segs = find_segments(prompt_ids, sep_id)              # 切段
    classify_segments(...)                                # 标段(第几条 summary)
    # truth(重放,应复现主轨的干净决策)
    write_control(control, [], dump=True)
    truth = free_gen_decision(prompt_ids)                 # 全重算 → 基准决策分布(+dump KV*)
    # swap:每段单独复用
    for 段 k=[s,e]:
        write_control(control, [[s,e]])
        swap_k = free_gen_decision(prompt_ids)            # 只复用段 k → 决策分布
        |Δz_k| = action_kl(truth.dist, swap_k.dist)       # 段 k 复用 vs 重算的决策偏移
```

### 3. 每个 pass 的服务端链(control 怎么生效)
```
free_gen_decision 发 prompt → vLLM → vllm_v1_adapter.blend() [adapter:902]
  └─ blender.blend() [blender:370]
       ├─ _load_control()  per-blend 重读 control.json [blender:383]   # [] or [[s,e]]
       └─ process_qkv → CONTROL 分支(clone-revert)[blender:172]
            old_k[:] = k                  # 全重算 = KV*
            old_k[reuse_idx] = KV^r        # 只把段 k 改回复用(swap);truth 时 复用空=不改
            → BLEND_PATH=CONTROL [blender:178]
```

## 五、|Δz_k| 与 M(patch C)

- **|Δz_k|** = status 位 {continue,complete} 两 token 上 truth‖swap 的 **2-way KL**([action_kl:171](../../server/vllm/Exp_A/exp_a_tf_runner.py#L171));`flip` = 决策翻转。大 → 段 k 的复用强烈影响决策。
- **M_k(决策对段 k 的注意力)= patch C**:control 多带 `"dump_q":true`,**teacher-force** `[prompt + 决策前缀到 status 前]` 跑一个 truth-dump pass → dump 里 `q` 末位 = 预测 status 的 query;离线 `M_k = softmax(q · K*_段k)`。配合 `Δ_k`(段 k 的 KV^r vs KV* 差)对照 |Δz_k|,验证"注意力×漂移"能否解释决策偏移。(blender 已写好 q-dump,见 [blend q-dump](../../lmcache/v1/compute/blend/blender.py#L163)。)

## 六、和当前代码的差距(实现 patch C 时要改)

| 项 | 现状 | 应改成 |
|---|---|---|
| 主轨模式 | `run_demo` 普通 reuse(脏)[exp_a_tf_runner.py:52](../../server/vllm/Exp_A/exp_a_tf_runner.py#L52) | 主轨期间 control 写 `[]`(control-truth = 干净 full prefill·带分隔符) |
| 决策点 | 只取最后一轮 [pick_decision_call:66](../../server/vllm/Exp_A/exp_a_tf_runner.py#L66) | 遍历**每一轮**决策 call,各自切段 + 分叉 |
| M | 未算 | teacher-force + `dump_q`(blender q-dump 已就位,harness/离线 M 待写) |

## 七、产出
每个决策点 × 每段:`{round, seg_k, |Δz_k|, flip, M_k, Δ_k, followup_jac}` → events.jsonl;
聚合看"哪些段(哪一轮、什么内容)驱动决策" + "M×Δ 能否预测 |Δz|"。

---
*相关:[callchain_cdriver_and_control.md](callchain_cdriver_and_control.md)(blend/control 调用链)、[harvest_format.md](harvest_format.md)(决策 prompt 从哪来)、[blend_path_logging.md](blend_path_logging.md)(BLEND_PATH 核实)、[claude-docx/14](../../claude-docx/14-blend-control-truth-pitfall.md);代码 [exp_a_tf_runner.py](../../server/vllm/Exp_A/exp_a_tf_runner.py)、[blender.py](../../lmcache/v1/compute/blend/blender.py)。*
