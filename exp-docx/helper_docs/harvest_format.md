# harvest 是什么、长什么样、怎么生成

> harvest = demo 跑的时候,把**选定的 LLM 调用的完整原始数据**(prompt + 生成文本 + token id)逐条落到一个 JSONL 文件。
> 它是"agent 实际发了什么、生成了什么"的**地面真值记录**,后续分析(切段、重建决策序列、teacher-force)全靠它。

---

## 一、是什么 / 为什么有

Exp A / 全段实验要在 token 级别复盘 agent:
- 把某轮**决策 prompt** 按分隔符切成段(每段对应一条 summary)→ 需要那条决策调用的**完整 messages**;
- 重建每轮的**决策序列**(continue/complete)→ 需要每条决策调用的**生成文本**;
- teacher-force / 比对 → 需要**精确 token id**。

普通日志只有时延/条数,不够。所以加了 **harvest:把白名单内的 LLM 调用整条 payload 存下来**。

## 二、怎么生成

### 2a. 两个环境变量 — [llm.py:32-37](../../../deep_researcher_demo/deep_researcher_demo/llm.py#L32-L37)
```python
_HARVEST_PATH = os.getenv("DRIFT_HARVEST", "")        # 设了路径才开;落到这个 jsonl
_HARVEST_TAGS = os.getenv("DRIFT_HARVEST_TAGS",
                          "RESEARCH_SUMMARY_TEXT,SUPERVISOR_DECISION_JSON")  # 白名单;"*"=全记
```
- **`DRIFT_HARVEST=<path>`**:不设 → 不 harvest;设了 → 选定调用逐条 append 到这个文件。
- **`DRIFT_HARVEST_TAGS`**:只记这些 tag 的调用(默认就 summary + decision 两类;`*` = 所有 LLM 调用)。

本实验由 runner 设置 — [exp_fullseg_runner.py:41](../../server/vllm/Exp_fullseg/exp_fullseg_runner.py#L41):`"DRIFT_HARVEST": harvest`,其中 `harvest = <qdir>/harvest_{mode}.jsonl`([:32](../../server/vllm/Exp_fullseg/exp_fullseg_runner.py#L32))→ **每题每臂一个 harvest 文件**。

### 2b. 每个 LLM 调用后判断要不要记 — [llm.py:163-169](../../../deep_researcher_demo/deep_researcher_demo/llm.py#L163)
```python
harvest_this = bool(_HARVEST_PATH) and (
    _HARVEST_TAGS is None or (tag or _infer_call_tag(messages)) in _HARVEST_TAGS)
if harvest_this:
    payload["return_token_ids"] = True   # 顺便让 server 回传 token id(给精确切段/teacher-force)
```
tag 来自调用方显式传(如 summarize 传 `RESEARCH_SUMMARY_TEXT`、decide 传 `SUPERVISOR_DECISION_JSON`);没传则从 system prompt 首行推断([_infer_call_tag llm.py:76](../../../deep_researcher_demo/deep_researcher_demo/llm.py#L76))。

### 2c. 落盘:整条 payload append — [llm.py:233-246](../../../deep_researcher_demo/deep_researcher_demo/llm.py#L233) + [_harvest_call:93-99](../../../deep_researcher_demo/deep_researcher_demo/llm.py#L93)
```python
if harvest_this:
    _harvest_call({...整条记录...})     # → open(_HARVEST_PATH,"a").write(json.dumps(record)+"\n")
```
**JSONL:一行一个 JSON = 一次 LLM 调用,按调用先后顺序 append。**

## 三、长什么样(9 个字段)

每行结构 — [llm.py:235-245](../../../deep_researcher_demo/deep_researcher_demo/llm.py#L235):

| 字段 | 含义 |
|---|---|
| `tag` | 调用类型:`RESEARCH_SUMMARY_TEXT` / `SUPERVISOR_DECISION_JSON` |
| `req_id` | vLLM 请求 id |
| `model` | 模型名 |
| `store_generated_kv` | 这次有没有让 server 存 decode KV(复用的来源,reuse 臂为 True) |
| `finish_reason` | `stop`(正常)/ `length`(撞 max_tokens 被截断) |
| `max_tokens` | 本次生成上限 |
| **`messages`** | **完整 prompt**(`[{role,content},...]`,含拼好分隔符的 summary)← 切段就用它 |
| **`content`** | **生成文本**(已做截断修复 = 字节级可复用 segment 原文)← 重建决策/比对就用它 |
| **`token_ids`** | **完整生成 token id 列表**(存下来的 segment = `[:-1]`)← 精确切段/teacher-force |

### 真实例子(`exp_fullseg_pure/q0/harvest_reuse.jsonl`,31 行 = 27 summary + 4 decision)
```jsonc
// 一条 SUPERVISOR_DECISION_JSON(这条其实是 warmup 调用,见 §五):
{
  "tag": "SUPERVISOR_DECISION_JSON",
  "req_id": "chatcmpl-...",
  "model": "Qwen3-32B",
  "store_generated_kv": true,
  "finish_reason": "length",          // max_tokens=8 撞上限被截
  "max_tokens": 8,
  "messages": [
    {"role": "system", "content": "...supervisor 系统提示..."},
    {"role": "user",   "content": "<research_summaries>\n<|fim_pad|>warmup<|fim_pad|>\n</research_summaries>\n\n<origin..."}
  ],
  "content": "```json\n{\"status\": \"complete",   // 截断的决策 JSON 开头
  "token_ids": [/* 8 个 id */]
}
```
真正的决策行 `messages` 里 `<research_summaries>` 段会是**真 summary 用 `<|fim_pad|>` 拼起来**的长文本,`content` 是完整 `{"status":"continue"/"complete", "followup_questions":[...]}`。

## 四、谁消费它

| 消费方 | 用 harvest 做什么 |
|---|---|
| [exp_fullseg_runner.py `parse_decisions`](../../server/vllm/Exp_fullseg/exp_fullseg_runner.py#L58) | 按出现顺序读 `SUPERVISOR_DECISION_JSON` 的 `content` → 重建每轮决策序列(continue/complete)→ 存进 traj.json |
| [exp_a_tf_runner.py `pick_decision_call`](../../server/vllm/Exp_A/exp_a_tf_runner.py#L66) | 挑那条决策调用(content 最长=summary 最多的一轮)→ 取 `messages` |
| [`context_token_ids` + `find_segments`](../../server/vllm/Exp_A/exp_a_tf_runner.py#L75) | 把决策 `messages` tokenize → 扫 `<\|fim_pad\|>` 切段 |
| [`classify_segments`](../../server/vllm/Exp_A/exp_a_tf_runner.py#L92) | 把每段解码,和 harvest 里 `RESEARCH_SUMMARY_TEXT` 的 `content` 比对 → 标"第几轮第几条 summary" |

## 五、坑:warmup 调用也会被记

reuse 模式开头有个 `warmup_kv_prefix`(预热决策/报告角色的 KV),它发的也是 `SUPERVISOR_DECISION_JSON` 标签的请求,但:
- summaries 段是占位 `<|fim_pad|>warmup<|fim_pad|>`、`max_tokens=8`、`finish_reason=length`(被截断)。
- 所以 harvest 里第一条 decision 往往是 warmup,不是真决策。**消费方要跳过它**(`parse_decisions` 按 content/被截断过滤,见 [runner:60](../../server/vllm/Exp_fullseg/exp_fullseg_runner.py#L60) 注释)。

## 六、和"call log"的区别

llm.py 还有个更轻的 `_log_call`([:209-228](../../../deep_researcher_demo/deep_researcher_demo/llm.py#L209))——**所有**调用都记,但只存**元数据 + 时延**(tag/req_id/prompt_tokens/prefill_s/decode_s...),**不含 messages/content/token_ids**。
- **call log**:看时延、调用数(轻)。
- **harvest**:看完整 prompt + 生成 + token(重,只白名单 tag)。本实验/Exp A 切段、重建决策靠的是 harvest。

---
*相关:[callchain_cdriver_and_control.md](callchain_cdriver_and_control.md)(harvest 里的 tag 怎么来 / 段怎么切)、[experiment_workflow.md](experiment_workflow.md);代码 [llm.py](../../../deep_researcher_demo/deep_researcher_demo/llm.py)、[exp_a_tf_runner.py](../../server/vllm/Exp_A/exp_a_tf_runner.py)。*
