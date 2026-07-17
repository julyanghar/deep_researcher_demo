# helper · `meta.json` 格式(Exp_M / writer 隔离实验每题落的元信息)

> 路径:`$EXP_W_OUT/q<N>/meta.json`。
> **谁产**:`exp_writer_run.py` 的 `measure_question`(每题跑完 M 抓取 + writer dump 后落)。
> **谁消费**:`exp_writer_select.py` 的 `select_and_write`(读它去算 ΔK/ΔV、M,写各分支 control)。
> 代码:[../../server/vllm/Exp_writer_isolation/exp_writer_run.py](../../server/vllm/Exp_writer_isolation/exp_writer_run.py)、[../../server/vllm/Exp_writer_isolation/exp_writer_select.py](../../server/vllm/Exp_writer_isolation/exp_writer_select.py)。

---

## 一、它干嘛用的
一题里有**两份 dump**(writer 的 ΔK/ΔV、supervisor 的 M),和**两套坐标系**(同一批 summary 在 writer prompt 里、在 supervisor decide 上下文里位置不同)。`meta.json` 就是把这些**对应关系**记下来,好让离线的 `select_and_write` 知道:
- 哪份 dump 是哪个(`writer_event` / `m_event`);
- summary 在两边各落在哪些位(`summary_pos` / `super_summary_pos`、`segments` / `super_segments`);
- 算 M 时决策位从哪开始(`super_prompt_len`)。

---

## 二、字段逐个解析
| 字段 | 类型 | 是什么 | 给谁用 |
|---|---|---|---|
| `dump_dir` | str | dump 文件目录(`blend_<event>_tp*.pt` 在这)| `load_blend_dump` |
| `control_dir` | str | 各分支 control(`branch_*.json`)+ `diag.json` 写到这 | `select_and_write` 落 control |
| `question` | str | 题面原文 | 记录/打分 |
| `sample_id` | str | 题号,如 `q0` | 记录 |
| **`writer_event`** | int | **writer 的 ΔK/ΔV dump** 事件号(全重算 pass)| `load_blend_dump(dump_dir, writer_event)` → `delta_kv` 出 ΔK/ΔV |
| **`summary_pos`** | list[int] | **writer 帧**里所有 summary token 的**绝对位**(各 segment 展开)| `delta_kv` 取这些位算 ΔK/ΔV;`transfer_M` 的目标坐标 |
| **`segments`** | list[[s,e)] | **writer 帧**的 summary **段**(每段一个 `[起, 止)` 区间)| 截到缓存命中区、`transfer_M` 按段对齐 |
| **`m_event`** | int | **supervisor 的 M dump** 事件号(复用 pass、含决策位 q)| `load_blend_dump(dump_dir, m_event)` → `compute_M` |
| **`super_summary_pos`** | list[int] | **supervisor 帧**里所有 summary token 的绝对位 | `compute_M` 取这些位当 K(复用的 K^r)|
| **`super_segments`** | list[[s,e)] | **supervisor 帧**的 summary 段 | `transfer_M` 的源坐标(按内容映射到 writer 段)|
| **`super_prompt_len`** | int | **决策位的起点**(`compute_M` 里 `decision_pos = arange(super_prompt_len, T)`)| `compute_M` 决定哪些位是"决策 q" |

样例(q0):
```jsonc
{
  "dump_dir": ".../dump", "control_dir": ".../q0/controls",
  "question": "Consider the OECD countries ...", "sample_id": "q0",
  "writer_event": 8,
  "summary_pos": [41,42,43,44, ...],          // list[4499]
  "segments": [[41,153],[154,232],[233,303], ...],   // 25 段
  "m_event": 7,
  "super_summary_pos": [79,80,81,82, ...],    // list[4500]
  "super_segments": [[79,191],[192,270], ...],       // 25 段
  "super_prompt_len": 4604
}
```

---

## 二·5、"事件号"是什么(`writer_event` / `m_event`)
**"事件" = blender 的一次 dump**(某遍 blend 时 control 带 `dump:true`,就把那遍的 KV+q 写盘)。
**"事件号" = blender 里计数器 `_dump_counter` 的值**:从 0 起、**每 dump 一次 +1**,就是文件名里的号——`blend_<事件号>_tp<rank>.pt`。一个事件 = **4 个文件**(TP0–3 各 1/4 head,`load_blend_dump` cat 回)。
- **全局递增、不按题、不重置**:`m_event=7`/`writer_event=8` 是"server 自启动以来第 8、第 9 次 dump",不是"第 7 题"。
- **一题两个事件**:`m_event`(supervisor 复用 pass 抓 M)、`writer_event`(writer 全重算 pass 抓 ΔK/ΔV)。
- driver 靠"dump 前 snapshot 目录、dump 后取新增号"(`_last_dump_event`)认出这次是哪个事件。
- **dump 读完即删**(单题十几 GB),所以跑完目录是空的——`meta.json` 里只留事件号当"指针",真要复看得在删之前抓。

---

## 三、两个坐标系(writer 帧 vs supervisor 帧)——最该搞懂的点
**同一批 summary,出现在两个 prompt 里、位置不同**:
- **writer 帧**:writer 的报告 prompt = `<findings>` + summary 段。→ `segments` / `summary_pos`。
- **supervisor 帧**:supervisor 最后一轮 decide 上下文 = `<research_summaries>` + summary 段 + 问题。→ `super_segments` / `super_summary_pos`。

数据流:
```
m_event   dump → compute_M(super_summary_pos, super_prompt_len) → M[L, Ns_super]   (supervisor 帧)
                                                       │ transfer_M(按段内容对齐 super→writer)
                                                       ▼
writer_event dump → delta_kv(summary_pos) → ΔK/ΔV[L, Ns_writer]   (writer 帧)
                                  mxdv = M_writer × ΔV → 选位 → 写 control
```

---

## 四、两个容易踩的点
1. **`super_prompt_len` 不是 super_prompt 的长度,是它 +1**。
   抓 M 时把决策包成 `[super_prompt + SEP + 决策]`,SEP 占一位;`super_prompt_len` 设成 `len(super_prompt)+1` 是为了**跳过那个 SEP**,让 `decision_pos` 正好从决策第一个 token 起。别当成"上下文长度"。
2. **`segments` 的最后一段往往不是真 summary,而是"问题块/闭合标签"**(find_segments 把最后一个分隔符之后的尾巴也当成一段)。
   - 例:writer 末段 `[4444,4564]`、super 末段 `[4482,4603]` 是 `</research_summaries> 问题` / `</findings>` 那截。
   - 所以 `summary_pos` 严格说混进了少量非 summary 位。**无害**:`transfer_M` 按**内容**对齐两帧,两边的尾巴块内容不同 → 匹配不上 → 不参与 M 转移(只对得上真正的 summary 段)。这也是 `[transfer_M] writer 25 段 super 25 段 对齐 24 段` 里"少一段"的来源。

---

## 五、自洽校验(复核时可用)
- `summary_pos` == 把 `segments` 每段 `[s,e)` 展开 `range(s,e)` 拼起来(实测一致)。
- `writer_event` / `m_event` 在 `dump_dir` 里都有对应 `blend_<event>_tp*.pt`。
- `super_prompt_len` ≈ `super_segments` 末段的 `e` + 1(决策接在 super_prompt 之后)。

> 相关:dump 文件本身的格式见 `exp_writer_select.py` 顶部 docstring;复核全流程见 [../Exp_M/verification_method.md](../Exp_M/verification_method.md)。
