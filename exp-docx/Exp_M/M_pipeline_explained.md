# Exp_M · M 是怎么算出来的(大白话全流程)

> 一句话:**M = supervisor 做最后决策时,对每段 summary 有多上心(注意力)**。
> 要算它得凑齐两样:**决策的"提问向量" q** 和 **被复用的 summary 的"被查向量" K^r**;凑齐了做 `softmax(q·K^r)`,再把结果对齐到 writer 的 summary 段,乘上 writer 的漂移 ΔV,就是主方案 mxdv 的选位分。
> 涉及代码:[../../lmcache/v1/compute/blend/blender.py](../../lmcache/v1/compute/blend/blender.py)、[../../lmcache/v1/token_database.py](../../lmcache/v1/token_database.py)、[../../server/vllm/Exp_writer_isolation/exp_writer_run.py](../../server/vllm/Exp_writer_isolation/exp_writer_run.py)、[../../server/vllm/Exp_writer_isolation/exp_writer_select.py](../../server/vllm/Exp_writer_isolation/exp_writer_select.py)。

---

## 全景图(先看一眼整条流水线)
```
supervisor 最后一轮决策(summary 全复用条件下)
        │
   ┌────┴─────────────────────────────────────────────┐
   │ 这一遍要同时抓两样东西,都落进同一份 dump          │
   │   ① summary 的 K^r(复用版被查向量)               │
   │   ② 决策位的 q(提问向量)                          │
   └────┬─────────────────────────────────────────────┘
        ▼
   compute_M:  M[层, summary位] = Σ_决策位 Σ_head softmax_j(q·K^r)|取 summary 列
        │   (supervisor 坐标系:summary 在 decide 上下文里的位置)
        ▼
   transfer_M: 按"段内容"把 M 搬到 writer 坐标系(summary 在 <findings> 里的位置)
        │
        ▼
   mxdv = M_writer × ΔV(writer)  → 选位重算  → 各分支 control
```

---

## 一、怎么"精确"存下 summary 段的 KV(K^r)

**先搞清 K^r 是什么**:supervisor 决策时,summary 的 KV **不是现算的**,是从缓存**复用**载入的(载入后还做了位置编码校正,把它搬到当前 prompt 该在的位置)。这份复用来的 K 叫 **K^r**(r = reused)。我们要的就是它——因为要还原"supervisor 真·复用条件下"的注意力。

**怎么存下来**:这一遍我们让 supervisor 走"复用 pass"(control 里 summary 标成复用)。blender 做 blend 时,本来手里就同时有两份东西:
- `old_k/old_v` = 从 staging buffer 拿到的**复用版 K^r/V^r**;
- 当层重算出来的**新鲜版 K*/V***。

我们在 control 里打开 `dump`,blender 就在每层把这两份都 `detach().cpu()` 存进字典 `_dump_layers`,最后 `_flush_dump` 落盘([blender.py:338](../../lmcache/v1/compute/blend/blender.py#L338)),文件 `blend_<event>_tp<rank>.pt` 里:
- `kv_r` = 复用版(算 M 用这个);
- `kv_star` = 新鲜版(算 writer ΔK/ΔV 用,另一份 dump);
- `tokens` = 这遍的完整 token 序列(离线按分隔符切段用)。

**"精确"体现在三点**:
1. **按 TP rank 分文件**:每个 worker 只持有 1/4 的注意力头,4 个文件离线按 head 维 cat 回完整(`load_blend_dump` 干这事),否则会互相覆盖。
2. **存的是 post-RoPE 的 K**:位置编码之后的,和 q 同一"帧",这样 q·K 才是真实注意力。
3. **只覆盖命中区**:T = 缓存覆盖长度(summary 段),不多存。

---

## 二、怎么"精确"抓下决策段的 query(q)—— 全程最难的一环

**难在哪**:决策是 supervisor **一个 token 一个 token 解码**出来的,它不在 prompt 里。而 blender **只能 dump"复用区"**(= 命中缓存、被分隔符切出来的段)。决策天生不在复用区 → blender 看不到它的 q。(早期 M 一直是 0,就卡在这。)

**配方(三步,缺一不可)**——见 [exp_writer_run.py](../../server/vllm/Exp_writer_isolation/exp_writer_run.py) `measure_question` 第 4 步:
1. **存决策段 KV**:summary 全复用下解码出决策,这一发请求带 `kv_transfer_params={"lmcache.blend_store_generated": true}`。LMCache 会用 `gen_start` 在"prompt 和生成内容交界处"切一刀,把决策段存成一个**纯内容哈希的 chunk**([token_database.py:534](../../lmcache/v1/token_database.py#L534))——这样它以后能靠"内容"被命中复用。
2. **丢决策末位 `gen_ids[:-1]`**(⚠ 头号坑):解码是"用第 i 个 token 算第 i+1 个",**最后采样出来那个 token 从没喂回模型、压根没有 KV**,所以存的是 `[:-1]`。**再喂时也必须丢末位**,否则你喂的决策比存的多一个 token、末段内容哈希对不上 → 命不中 → 决策又进不了复用区 → M 又是 0,而且静默。
3. **再喂一遍 + 抓 q**:把 `[super_prompt + SEP + 决策[:-1]]` 再喂一发(`SEP` 把决策包成一个 blend 段),control 开 `dump_q` + `dump_in_reuse`。这遍决策段**命中缓存、进了 blend 复用区**,blender 在 process_qkv 里 rotary 之后把决策位的 post-RoPE q `detach().cpu()` 存进 `_dump_q_layers`([blender.py:182](../../lmcache/v1/compute/blend/blender.py#L182)),`_flush_dump` 落成 `q` 键([blender.py:372](../../lmcache/v1/compute/blend/blender.py#L372))。

**结果**:这份 dump 里 summary 的 K^r 和决策位的 q **同帧**(都 post-RoPE),万事俱备。

> 决策位从哪开始?`super_prompt_len = len(super_prompt)+1`(那个 +1 是跳过包决策的 SEP),决策位 = `[super_prompt_len, T)`。

---

## 三、怎么算 M —— [exp_writer_select.py `compute_M`](../../server/vllm/Exp_writer_isolation/exp_writer_select.py)

打个比方:决策 token 是**考官**,summary 是**考卷**。M = 考官批卷时,**眼睛在每张卷子上停了多久**。

逐层、逐决策位算:
1. 取这一层的 q(只取决策位那些行)和 K^r(`key_kind="kv_r"`,GQA 展开到和 q 同头数)。
2. 对每个决策位 g、每个 head:`logits = q_g · K_j / √hd`,对**全因果上下文** j(j ≤ g,屏蔽未来)做 softmax → 决策对每个位置的注意力分布。
3. **只取 summary 那些列**(`super_summary_pos`),对所有决策位 + 所有 head 求和 → `M[层, 每个 summary token]`。

```
M[l] += softmax_j(q_decision · K^r_j)[:, summary列].sum(决策位, head)
```
含义:**所有决策 token,落在每个 summary token 上的注意力质量总和**。softmax 是在全上下文上归一的,所以拿到的是"真实落在 summary 上"的那部分(不是 summary 内部的相对比例)。

显存两招:q/K 留 CPU、**逐层**搬卡;决策位**分块**(g_chunk);OOM 就缩块、再不行退 CPU。

---

## 四、怎么把 M 对齐到 writer 的 summary 段 —— `transfer_M`(最容易想错)

**为什么要"搬"**:M 是在 **supervisor 坐标系**算的(summary 在 decide 上下文里的位置,比如从第 79 位起);而 writer 的漂移 ΔV 在 **writer 坐标系**(summary 在 `<findings>` 里的位置,从第 41 位起)。**两边位置不一样、段数还可能不等**(writer 末段常被缓存边界截掉),所以**绝对不能按下标硬对**。

**怎么对**:按**段内容(token-id 序列)**对——
1. supervisor 每段的 token-id 元组 → 记下它在 M_super 里占哪几列;
2. writer 每段,拿自己的 token-id 元组去 supervisor 那查:**查到** → 把那几列 M **逐 token** 搬过来;**查不到** → 这段 M=0。

**为什么"按内容"能精确对上**(这点我专门验证过):
- 每段 summary 前面都顶着一个 `<|fim_pad|>` 分隔符,它是个**干净的 token 边界**;
- 所以**同一段 summary 文本,在两帧里 tokenize 出完全相同的 token-id**(实测:writer 的 24/25 段文本逐字出现在 supervisor 上下文里,且都以分隔符开头);
- 于是"按 token-id 元组精确匹配"成立,不会错配。

**对齐后为什么能直接乘 ΔV**:`transfer_M` 返回的 `M_writer` 列序 = **writer 段序**,而 ΔV 也是按 writer 段序排的 → **同列序** → `mxdv = M_writer × ΔV` 就是"同一段 summary 的 注意力 × 漂移"。

**没对上的那 1 段**:是 `</findings>`/问题块那种**非 summary 的尾巴**,M=0 正确(它本来就不该被注意力加权)。这也是日志 `[transfer_M] writer N 段, supervisor N+1 段, 对齐 N 段` 里"少一段"的来源。

---

## 五、三个坑 + 怎么自己核
**坑**
1. **末位无 KV**:store_generated 存 `[:-1]`,再喂也要丢末位,否则决策段命不中、M=0(静默)。
2. **两个坐标系**:M 在 super 帧、ΔV 在 writer 帧,只能按内容对,别按下标。
3. **离线复算别瞎重建**:别用 `context_token_ids` 重新 tokenize 一遍、再拿 meta 的位置去切——重建的 token 序列未必和真实 dump 逐 token 一致,位置一错全错(我就这么踩过一次,得出假的"0 段对齐")。要核就用 **dump 自带的 `tokens`**,或干脆按**文本子串**对。

**怎么核 M 对齐对不对**
- 看 `driver.log` 的 `[transfer_M] … 对齐 N 段`(N ≈ summary 段数,差 1 是尾块);
- 文本验证:writer 每段 summary 文本能否逐字在 supervisor 上下文里找到(能 → 同内容,对齐有效);
- `diag.json` 的 `M_rowsum_mean > 0` 且非均匀。

---

## 相关
- store_generated / gen_start / 末位坑:[../../claude-docx/16-decode-generated-kv-reuse.md](../../claude-docx/16-decode-generated-kv-reuse.md)。
- meta.json 两帧字段:[../helper_docs/meta_json_format.md](../helper_docs/meta_json_format.md)。
- 当时抓 M 的调试过程:[m_capture_debug.md](m_capture_debug.md);整体实验设计:[experiment_design.md](experiment_design.md)。
