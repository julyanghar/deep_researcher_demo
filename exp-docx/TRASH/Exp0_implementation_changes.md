# Exp0(= Exp A,KV 漂移)实现改动与理由

> 本文逐处记录为跑通 Exp A 所做的全部代码改动 + 为什么这么改。
> 配套实验方案见同目录 `EfficientDeepResearchAgent_实验方案.md`。
> **状态(写作时)**:deep_researcher_demo 的改动**均未提交**,在 commit `1b28c0d` 之上;
> LMCache 只新增一个 harness 脚本(**未改 LMCache 源码**)。测试 64 passed,合成打通 PASS。

---

## 0. 概览:改了哪几块

| 块 | 仓库 | 性质 | 目的 |
|---|---|---|---|
| A1 r_t 递归推理 | deep_researcher_demo | 改逻辑 | supervisor 输出作可复用 KV 段、构造递归 trace(Exp C 前置) |
| A2 DRIFT_HARVEST 采集 | deep_researcher_demo | 加埋点 | 落盘真实 s_i/r_t + worker/supervisor 上下文,喂离线 harness |
| A3 两级 search 缓存 | deep_researcher_demo | 重写 | query→url→content 可复现(替换旧 per-question 共享池) |
| A4 测试 | deep_researcher_demo | 加/改测试 | 覆盖上面改动,保"关闭即回归" |
| B1 离线 harness | LMCache(新文件) | 新增脚本 | 逐层逐 token dump KV^w/KV^r/KV* 并算漂移 |

**贯穿性原则**:所有行为改动默认**关**(env 开关),关时与改前**逐字一致** → 既能对照 baseline,也保证回归安全。

---

## A. deep_researcher_demo(未提交,在 `1b28c0d` 之上)

### A1. supervisor 逐轮推理 r_t —— `SUPERVISOR_REASONING`

**目标**:把 supervisor 每轮的决策 JSON 作为可复用 KV 段 `r_t`,并交错进后续 supervisor 上下文
`[sys] + out_1 + r_1 + out_2 + r_2 + …`(out 在前、r 在后,已与你确认)。这样复用误差经
reasoning trace 这个递归状态逐轮放大 —— 把"value 级累积"从已死的 summary 通道搬到真实递归状态
(实验方案 §0.2 / Exp C)。

**为什么 r_t = 直接复用 decide 的 JSON 段**(你确认的方案,非新增独立推理调用):改动最小。代价是
trace 薄(真正推理只有 `reason` 一句、含易变 followups)。**靠两点绕开坑**:
1. 下游嵌入的是 `chat()` 返回的**原始 content**(不是 `model_dump_json()` 重序列化 → 否则 key 序/空格变 → content-hash miss);
2. parse 失败触发 repair 时,被存的是**第一次 decode** 的 KV,故 `r_t` 取第一次 content、`decision` 取 repair 后的值,两者分离才稳。

**改的文件 / 函数**(点击跳转):
- `agents.py`
  - 新增 env `SUPERVISOR_REASONING`(默认关):[agents.py:36](../../deep_researcher_demo/deep_researcher_demo/agents.py#L36)
  - 新增 `interleave_segments(summaries, reasonings)` → `[out_1, r_1, out_2, …]`(reasonings 短一个时末尾停在 out_t):[agents.py:56](../../deep_researcher_demo/deep_researcher_demo/agents.py#L56)
  - `Agent.call_json`:加 `store_generated_kv` 入参、返回改 **`(parsed, raw_content)`**(raw=第一次 decode,repair 不覆盖):[agents.py:80](../../deep_researcher_demo/deep_researcher_demo/agents.py#L80)。调用方改解包:[initialize_question agents.py:156](../../deep_researcher_demo/deep_researcher_demo/agents.py#L156)、[plan_queries agents.py:300](../../deep_researcher_demo/deep_researcher_demo/agents.py#L300)
  - `Supervisor.decide`:签名加 `reasonings`/`store_generated_kv`、findings 交错(`interleave_segments`+`join_reusable_segments`)、返回 **`(decision, raw_content)`**:[agents.py:219](../../deep_researcher_demo/deep_researcher_demo/agents.py#L219)(交错点 [agents.py:233](../../deep_researcher_demo/deep_researcher_demo/agents.py#L233),返回 [agents.py:267](../../deep_researcher_demo/deep_researcher_demo/agents.py#L267))
- `workflow.py`:import 开关 [workflow.py:6](../../deep_researcher_demo/deep_researcher_demo/workflow.py#L6);维护 `reasonings` [workflow.py:86](../../deep_researcher_demo/deep_researcher_demo/workflow.py#L86);decide 解包 [workflow.py:106](../../deep_researcher_demo/deep_researcher_demo/workflow.py#L106) + gate/append [workflow.py:113](../../deep_researcher_demo/deep_researcher_demo/workflow.py#L113);带入结果 [workflow.py:169](../../deep_researcher_demo/deep_researcher_demo/workflow.py#L169)
- `schemas.py`:`WorkflowResult.supervisor_reasonings` 字段 [schemas.py:47](../../deep_researcher_demo/deep_researcher_demo/schemas.py#L47)
- `tests/test_agents.py`:`decide()` 返回 tuple,测试解包 + 断言 raw 含 `"status"` [test_agents.py:30](../../deep_researcher_demo/tests/test_agents.py#L30)

**未改**:FinalWriter(仍只用 summaries,已确认);warmup 前缀逻辑(decide 前缀不变)。

---

### A2. Exp A stage-1 采集 —— `DRIFT_HARVEST`

**目标 / 为什么需要它**:Exp A 要量"摘要的 KV 复用偏了多少",得在离线 harness 里**按原样重建**两个上下文、
各算一遍 KV 再比。但 gpt-deep 走 HTTP,只看得到文本进/出、拿不到张量 —— 所以唯一办法是**把发给 server 的
精确 prompt 录下来**,让 harness 照着复现。`DRIFT_HARVEST` 就是这台"录音机":每条相关调用落
`{messages, content, token_ids}`。靠**段文本匹配**关联 s_i(摘要文本同时出现在 worker 记录的 content
与 supervisor 记录的 messages 里),不必在并发里穿 query_id。

**只录两类调用(刚收窄)**:一次研究会发很多种调用(分解问题 / 规划 query / 摘要 / 决策 / 写报告 / 修 JSON),
Exp A 只用两类 —— `RESEARCH_SUMMARY_TEXT`(worker 上下文 + s_i → KV^w)、`SUPERVISOR_DECISION_JSON`
(supervisor 上下文 + r_t → KV*/KV^r)。其余是死重(尤其 ≤1 万 token 的 final report),默认不录;
`DRIFT_HARVEST_TAGS` 可覆盖(`*`=全部)。

**为什么落在 llm.py**:它是唯一的真实 HTTP 客户端出口,所有角色调用都经过它,最小侵入、零业务逻辑改动。

**改的文件**(点击跳转):
- `llm.py`
  - `_HARVEST_PATH` [llm.py:32](../../deep_researcher_demo/deep_researcher_demo/llm.py#L32) + tag 白名单 `_HARVEST_TAGS`(默认 2 类,`*`=全部)[llm.py:35](../../deep_researcher_demo/deep_researcher_demo/llm.py#L35) + `_harvest_call(record)` [llm.py:93](../../deep_researcher_demo/deep_researcher_demo/llm.py#L93)。
  - `chat()`:仅当 `harvest_this`(path 开 且 tag 命中)才强制 `return_token_ids` [llm.py:163](../../deep_researcher_demo/deep_researcher_demo/llm.py#L163) 并落盘 [llm.py:233](../../deep_researcher_demo/deep_researcher_demo/llm.py#L233);抽 `finish_reason`/`token_ids` 复用原截断修正 [llm.py:190](../../deep_researcher_demo/deep_researcher_demo/llm.py#L190)。
  - **注意**:dump 的 `content` 是**截断修正后**的文本(= 字节级可复用段),`token_ids` 是完整生成串(被存段 = `[:-1]`)。默认关、零开销。

---

### A3. 两级 query→url→content search 缓存(替换旧 per-question 池)

**问题**:多轮研究里"同一 query 搜到的 url 不同" → 不可复现;且旧 `CachingSearchProvider` 的 replay 是
**per-question 共享 doc 池、无视 query 字符串**(为 A/B 计时公平设计),会把 Exp B 的轨迹分叉**抹平**。

**你的决定**:删掉旧 replay,换成两级结构。

**改的文件**(点击跳转):
- `search.py`:**整体重写** [CachingSearchProvider search.py:284](../../deep_researcher_demo/deep_researcher_demo/search.py#L284)(沿用 `record`/`replay` 模式名,故 config / `run_deepsearchqa.py` 无需改;`wrap_with_cache` 不变 [search.py:497](../../deep_researcher_demo/deep_researcher_demo/search.py#L497))。**每个 question 一个目录** `<dir>/q<sample_id>/`,各存自己的两级缓存(per-question、目录内去重、first-write-wins):
  - 目录:`q<sample_id>/search_cache.json` = `{归一化query: [url,…]}`;`pages_index.json` = `{url:{file,title}}` + `pages/<hash>.txt` = 内容(question 内去重)。目录设置 [search.py:326](../../deep_researcher_demo/deep_researcher_demo/search.py#L326)。
  - `record`:每 query 一次 live(per-query url 忠实)→ 写 query→urls + url→content,见 [_record_query search.py:425](../../deep_researcher_demo/deep_researcher_demo/search.py#L425)。
  - `replay`:query→urls→content,零网络;cold query 回退 live + flag miss,见 [_search_replay search.py:354](../../deep_researcher_demo/deep_researcher_demo/search.py#L354)。`fix_n`:每 query 取前 N 个 url。
  - 新增 [_norm_query search.py:275](../../deep_researcher_demo/deep_researcher_demo/search.py#L275)、[_url_filename search.py:280](../../deep_researcher_demo/deep_researcher_demo/search.py#L280);删除旧 `_strip_front_matter`/`_load_pool`/`_select`/manifest 系列。
- `config.py`:缓存注释更新为两级语义(字段名不变)[config.py:43](../../deep_researcher_demo/deep_researcher_demo/config.py#L43)。

**注意**:CLI 单 query 路径(`cli.py`)**没接** `wrap_with_cache` → 永远 live。Exp A 不需要复现(只要 s_i 真实),
故无碍;**Exp B 要复现需给 cli 加一行 wrap_with_cache**(或走 `run_deepsearchqa`)。

**影响**:旧 per-question pool replay 已删 → **重跑之前那批 AB 实验需重新 record**;`eval/EVAL_GUIDE.md`/`AB_EXPERIMENT.md`
仍描述旧行为(历史 writeup,未动)。

---

### A4. 测试

- `tests/test_agents.py`:`decide()` tuple 解包(见 A1)[test_agents.py:30](../../deep_researcher_demo/tests/test_agents.py#L30)。
- `tests/test_search.py`:新增 4 个用例 —— [roundtrip test_search.py:245](../../deep_researcher_demo/tests/test_search.py#L245)(query 忠实 + 零网络)、[cold 回退+flag test_search.py:264](../../deep_researcher_demo/tests/test_search.py#L264)、[fix_n 截断 test_search.py:274](../../deep_researcher_demo/tests/test_search.py#L274)、[url→content first-write-wins test_search.py:282](../../deep_researcher_demo/tests/test_search.py#L282)。
- **结果:64 passed**(原 60 + 新 4)。注意:测试全走 `StubChatClient`/fake base,**未触达** llm.py 真实 HTTP 路径(A2 的 harvest 落盘是在真实跑 harvest 时才执行/验证的)。

---

## B. LMCache(只新增脚本,未改源码)

### B1. 离线 harness [exp_a_drift_measure.py](../server/vllm/Exp_A/exp_a_drift_measure.py)(新文件)

**目标**:对同一段 token `s`,在两上下文各 prefill 一次,取逐层逐 token KV,得三份:
`KV^w`(s 在 worker 上下文)、`KV*`(s 在 supervisor 上下文完整 prefill,真值)、
`KV^r`(把 KV^w 的 K 用 FusedRope 从旧位旋到新位;V 无 RoPE)。

**怎么取 KV —— 不改 LMCache 源码**(点击跳转):
- in-process `vllm.LLM(enforce_eager=True)` + LMCache,**单进程** `VLLM_ENABLE_V1_MULTIPROCESSING=0`(否则引擎在子进程、monkeypatch 与 in-process 取模型都失效):[exp_a_drift_measure.py:33](../server/vllm/Exp_A/exp_a_drift_measure.py#L33);env 设置 [setup_env exp_a_drift_measure.py:108](../server/vllm/Exp_A/exp_a_drift_measure.py#L108)。
- **monkeypatch** `VLLMPagedMemGPUConnectorV2.from_gpu`:store 时把 paged KV 搬成连续 `[2,L,T,hidden]`(KV_2LTD)写进 `memory_obj.tensor`,在那一刻 clone 到 CPU f32;chunk 调大让整段一次落 → [install_dump_hook exp_a_drift_measure.py:59](../server/vllm/Exp_A/exp_a_drift_measure.py#L59)、收集拼段 [prefill_and_collect exp_a_drift_measure.py:77](../server/vllm/Exp_A/exp_a_drift_measure.py#L77)。
- **blend 关掉**:测量不需要 server 的复用,`KV^r` 自己用现成 `FusedRope` 算 → [rotate_k exp_a_drift_measure.py:154](../server/vllm/Exp_A/exp_a_drift_measure.py#L154)。
- **FusedRope 来源**:不调 `get_fused_rope`(会重建 vLLM RotaryEmbedding custom op、需引擎 config 上下文),而是**直接包住模型已构造好的 `rotary_emb`**(经 `VLLMModelTracker.get_model`)→ [build_fused_rope_from_model exp_a_drift_measure.py:141](../server/vllm/Exp_A/exp_a_drift_measure.py#L141)。
- 合成验证主流程 [main exp_a_drift_measure.py:199](../server/vllm/Exp_A/exp_a_drift_measure.py#L199)。

**合成打通验证结果(Qwen2.5-0.5B,PASS)**:
- 形状 `(2, 24, 113, 128)` 正确;
- `dist(K^r,K*)=0.0116 << dist(K^w,K*)=0.1602`(RoPE 对齐去掉位置错配,只剩内容漂移)→ RoPE 接线正确;
- `dist(V^w,V*)=0.0423`(V 纯内容漂移);逐层已显 **U 形**(浅层稳、中层最狠、末层回落)。

**待办**(本文档之后):扩 harness 读 `DRIFT_HARVEST` 的 JSONL、定位 s_i、算实验方案 §1 Exp A 的六条曲线。

---

## C. 环境 / 基础设施(非业务代码)

- 把 `1b28c0d` 之前的 WIP(generated-kv-reuse + 搜索缓存 + AB eval)按你指示提交成一个 commit。
- 给 `gpt-deep` conda env 装了 `pytest`(pyproject 声明的 dev 依赖,跑测试用)。
- 记忆:加了"中文优先"偏好。
- 跑通验证用 GPU:0.5B 在单卡;Qwen3-32B(将用于真实采集+测量)需 2 卡 TP2。

---

## D. 当前未改动 / 保真点

- `SUPERVISOR_REASONING` 关 → workflow / supervisor 行为与改前**逐字一致**(回归安全 + baseline 对照)。
- `DRIFT_HARVEST` 不设 → llm.py 零额外开销。
- search 缓存 `off` → 原 live 行为不变。
- FinalWriter、Researcher 业务逻辑未变。
- **LMCache 源码未改**(只加 `server/vllm/Exp_A/exp_a_drift_measure.py` 一个脚本)。

---

## E. 提交计划(待你 review 后)

deep_researcher_demo 改动建议分 3 个 commit,边界清晰好回滚:
1. `feat: supervisor r_t recurrent trace (SUPERVISOR_REASONING, default off)` —— agents/workflow/schemas + test_agents
2. `feat: DRIFT_HARVEST full-payload dump for Exp A` —— llm.py
3. `refactor: query-keyed search cache (replace per-question pool)` —— search/config + test_search

LMCache 的 harness 脚本单独提交或留在工作区(你定)。
