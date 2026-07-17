# Exp_M · 代码改造 plan(M = supervisor 决策注意力 → 指导 writer 复用)

> 配套:[execution_plan.md](execution_plan.md)。先审核,暂不动手。
> **本版纠正了 M 的来源**(见下),作废上一版的 report-q / dump_q_report / store_generated。

## 背景与目标
14 分支实验已完成(87 题,见 [../Exp_writer_isolation/results_14branch_final.md](../Exp_writer_isolation/results_14branch_final.md)),但 **M×ΔV 主方案没跑成**。

**M 的正确定义(用户纠正,关键)**:
- M = **上游 supervisor 的决策 token 对复用 summary 段的注意力分数**,**不是** writer 的 report 对 summary 的注意力。
- 结构:`[summary1][summary2][summary3][supervisor decision]`,M = `decision` 对各 summary token 的 attn。
- **粒度**:原始是**每层 × 每个决策 token × 每个 head** 各一个 attn score(对 summary 位的注意力)。
- **时序**:**必须等 supervisor 把决策解码完**才能算(要拿到完整决策 token 序列,才能 teacher-force 它们抓 q)。
- **条件(关键,用户定)**:M 必须在 **supervisor 全复用 summary KV(0 重算)的真实条件下**算——决策 q 在**复用 pass** 解码、注意力打到**复用的 `K^r`**(不是新鲜 `K*`)。所以要**从头解题**(重跑 pipeline、supervisor 走 blender 复用),不能用旧的 clean/全重算 harvest。
- **用途**:把 M(聚合成每个 summary token 一个权重)**搬到 writer**,M×ΔV 指导 writer 选哪些 summary token 重算。

**为什么 14 分支那轮 M 全 0**:源头找错了——`exp_writer_run` 在 **writer 自己的前向**上算 M(writer 没有"决策位"、report 没进 dump)。M 根本不该从 writer 取,要从 supervisor 的 decide 取。

**硬约束(用户定)**:① 不要 random 对照;② **不存 writer report 的生成 KV**——但 supervisor 决策段为抓 M 需短暂 `store_generated`(让决策段进 blend 区,见 A1.5),两者不冲突。

---

## A. 解锁 M —— 在 supervisor 全复用 summary KV 的真实条件下抓决策注意力(核心)
patch-C 的 `dump_q` 现成可借,但 **exp_a 的口径不能直接照搬**:exp_a 在"全重算 pass(recompute-all)"里抓决策 q、注意力打到新鲜 `K*`;本实验要**复用条件**——决策 q 在 supervisor 复用 summary KV(`K^r`)那一遍解码、注意力打到 `K^r`。这逼出一处 **LMCache 核心改动:blender 的 dump 门控**(现在 dump 只在 recompute-all 触发)。

### A0. 从头解题 + supervisor 全复用 summary KV
- **重跑 pipeline**(不能用旧的 clean/全重算 harvest);supervisor.decide 走 **blender 全复用**(summary KV 0 重算 = 原生 CacheBlend / control 全复用 summary 段)。
- 现 `run_demo` 用 `control={reuse_token_ranges:[]}`(全重算,见 [../../server/vllm/Exp_writer_isolation/exp_writer_run.py](../../server/vllm/Exp_writer_isolation/exp_writer_run.py) 第 50 行)→ 改成让 supervisor 那几发**全复用 summary**(decide 上下文 summary 段 0 重算)。

### A1. 在复用 pass 里抓 M(决策 q + 复用的 `K^r`)
**关键认知(已查证)**:blender 的 dump/`dump_q` **只覆盖命中缓存的复用前缀(summary)**,**生成段(决策 token)不进 blender**(由 vLLM 正常算,见 blender.py:120-126 文档)。所以决策 Q 抓不到——**除非先把决策段变成"可命中缓存区"**。用 `store_generated_kv` 的逻辑做到这点(它从分页缓存按 `slot_mapping` 读生成段 KV,见 [../../lmcache/integration/vllm/vllm_v1_adapter.py](../../lmcache/integration/vllm/vllm_v1_adapter.py) `_process_final_saves` 第 1179 行)。

每题:
1. 取最后一轮 `supervisor.decide` 上下文 `prompt_ids`(布局 = **纯 summary + 问题**,`SUPERVISOR_REASONING=0`):
   `[P0(chat模板+system+<research_summaries>头), summary1 SEP summary2 SEP summary3, </research_summaries>, <original_question>问题</original_question>]`(依据 [deep_researcher_demo/agents.py](/home/yilin/deep_researcher_demo/deep_researcher_demo/agents.py) 第 56-68、179-188、243 行)。`find_segments` 切 summary 段。
2. supervisor 在 **summary 全复用**下解码完整决策 → `gen_ids`,**且对这次调用开 `blend_store_generated`**(`kv_transfer_params`)→ 决策段 KV 存进 LMCache、变成可命中。
   - **决策位(选项 A)= 全部生成 token** `[prompt_len, prompt_len+len(gen_ids))`(complete → followups 空,≈ `status`+`reason`,reason 也算)。
3. **复用 pass 抓 M**:再喂 `[prompt_ids + gen_ids]`,此时**决策段也命中缓存 → 进入 blend 区**;control = **summary+决策段全复用 + dump + dump_q**(需 A1.5 放开复用 pass 的 dump)→ blender 现在能抓:① **决策位的 post-RoPE q**(每层,注意力打到 `K^r`);② summary 复用的 `K^r`(`old_k`/`kv_r`)。
4. `compute_M` 改用 **`K^r`(`kv_r`)**:`M[l] = Σ_{g∈决策位} Σ_head softmax_j( q_{g,l} · K^r_summary_{j,l} )` → `[L, Ns]`(per-head softmax 后求和、per-token 求和;保 head 维可选)。

### A1.5. 两处 LMCache 改动(让决策 q 抓得到)
1. **`store_generated` 存决策段**:supervisor decode 那发开 `blend_store_generated`(参考 `_process_final_saves` 从分页缓存读生成段 KV 的逻辑),让决策段可命中缓存——**这是把决策 token 弄进 blend 区的前提**(blender 本身够不着生成段)。
2. **dump 门控放开复用 pass**:现 `dump_active` 要求 `not reuse_token_ranges`(只在 recompute-all dump,blender.py:155-160)。改成**复用 pass 也能 dump**(如 control 带 `dump_in_reuse`):① `dump_q` 抓决策 q;② 抓 summary 复用的 `K^r`(此 pass summary 不重算 → `kv_star` 无意义,M 只用 `kv_r`)。
- **旧的 recompute-all dump 路径不动**(writer 的 ΔK/ΔV 还要用,见 A3 第 4 步)。
- 注:这里存的是 **supervisor 决策段**(M 的来源,该存),与"不存 **writer** report KV"不冲突。

### A2. 把 M 从 supervisor 帧搬到 writer 帧(新工程量)
- supervisor decide 上下文 = `[P0, summary..., 问题]`(纯 summary);writer prompt = `<findings> + 纯 summary`。两边 **summary 同文本、同分隔符切段** → 对齐干净。
- 对齐:两边都用 `find_segments` 切 summary 段 → 按 **summary 序号 + 段内 token 偏移**逐 token 映射 → 得 **writer summary 位**坐标下的 M(只对齐 SEP 之间的 summary 内容,包裹标签 `<research_summaries>` vs `<findings>` 不参与)。

### A3. driver 流程改造([../../server/vllm/Exp_writer_isolation/exp_writer_run.py](../../server/vllm/Exp_writer_isolation/exp_writer_run.py))
每题:
1. **A0**:重跑 demo,supervisor 全复用 summary;最后一轮 decide **开 `store_generated`** 存决策段。
2. **A1**:再喂 `[上下文+决策]`(决策已命中)→ 复用 pass 抓 M(`K^r` 条件,决策位 = 全决策 token)。
3. **A2**:M 搬到 writer summary 坐标。
4. **writer 的 ΔK/ΔV dump**:仍是 **recompute-all/truth pass**(在 writer prompt 上,给 `ΔK=||K^r−K*||`、`ΔV`)——这步用旧 dump 路径,不受 A1.5 影响。
5. 跑 writer 的 full / 50% / **M×ΔV** 分支(M 来自 A2、ΔV 来自第 4 步)。
- **删掉**上一版错的:report-q 捕获、`dump_q_report`、writer 端 teacher-force 算 M。(`store_generated` **保留**,但只用在 supervisor 决策段、不用在 writer report。)
- ⚠ **两份 dump 别混**:M dump(supervisor,复用 pass,`K^r`)vs ΔK/ΔV dump(writer,全重算 pass,`K^r`+`K*`)。

---

## B. 新分支集(12 支,无 random)
全部在 **summary 段**做选择性重算(非 summary 恒重算);复用现成 `select_token`/`select_layer`/`_topx_idx`。
| 组 | 分支 | 数量 |
|---|---|---|
| baseline | full prefill(上界)、full reuse(下界) | 2 |
| 非 M(仅 50%) | {ΔK, ΔV} × {token, layer} @ **50%** | 4 |
| **M×ΔV(主)** | **{token, layer} × {75%, 50%, 25%}**(M = 转移过来的 supervisor 注意力) | **6** |
| | **合计** | **12** |

- 改 [../../server/vllm/Exp_writer_isolation/exp_writer_select.py](../../server/vllm/Exp_writer_isolation/exp_writer_select.py):`BUDGETS`/`branch_id` 去 random、加 M×ΔV 三档。M 现非 0 → `mats["mxdv"]=M*dV` 生效。
- 同步 [../../server/vllm/Exp_writer_isolation/exp_writer_score.py](../../server/vllm/Exp_writer_isolation/exp_writer_score.py) 的 `BRANCHES`/`branch_label`/`compare`:主判据 = **M×ΔV vs ΔV(同 50% 同粒度)是否超噪声底 0.14**(注意力加权选位是否优于纯 ΔV);75/25 档只看相对 full reuse 的恢复。

---

## C. CPU 内存(每题清库存,防 OOM)
M 的 dump = supervisor 决策 q(决策位 × 64 层 f32)+ summary K*,单题可数 GB;LMCache 里 main-track 存的 summary KV 逐题累积。**每题清两样**:
1. **dump 文件**:算完 M 立刻删(沿用 [../../server/vllm/Exp_writer_isolation/exp_writer_run.py](../../server/vllm/Exp_writer_isolation/exp_writer_run.py) 第 157-159 行的删 dump,提前到 M 算完)。
2. **LMCache summary 库存**:本题所有分支跑完后 `DELETE /cache/clear?locations=LocalCPUBackend`(见 [../../lmcache/v1/internal_api_server/vllm/cache_api.py](../../lmcache/v1/internal_api_server/vllm/cache_api.py) 第 316 行)。
   - ⚠ 需确认 internal API server 端口开着;没开就 server 启动加上,或调小 `LMCACHE_MAX_LOCAL_CPU_SIZE` 靠 LRU。
   - ⚠ 时序:writer 分支重放还要用本题 summary KV → 清理放**本题最后**。

---

## 文件清单
| 文件 | 改动 |
|---|---|
| [../../lmcache/integration/vllm/vllm_v1_adapter.py](../../lmcache/integration/vllm/vllm_v1_adapter.py) | **核心**:supervisor 决策段用 `blend_store_generated`(现成,`_process_final_saves` 第 1179 行)存生成段 KV → 让决策段可命中、能进 blend 区 |
| [../../lmcache/v1/compute/blend/blender.py](../../lmcache/v1/compute/blend/blender.py) | **核心**:dump 门控放开"复用 pass 也 dump"(A1.5),抓复用 `K^r` + 决策 q;recompute-all 旧路径不动 |
| [../../server/vllm/Exp_writer_isolation/exp_writer_run.py](../../server/vllm/Exp_writer_isolation/exp_writer_run.py) | A0 重跑 demo(supervisor 全复用 + 决策段 store_generated)→ A1 再喂抓 M → A2 转移 → A3 writer ΔK/ΔV 全重算 dump + 跑分支;每题清库存 |
| [../../server/vllm/Exp_A/exp_a_tf_runner.py](../../server/vllm/Exp_A/exp_a_tf_runner.py) | 复用 `pick_decision_call`/`find_segments`/`free_gen_decision`;**M 用完整决策 + 复用 pass(非 status_pos、非全重算)** |
| [../../server/vllm/Exp_writer_isolation/exp_writer_select.py](../../server/vllm/Exp_writer_isolation/exp_writer_select.py) | `compute_M` 改用 `kv_r`(`K^r`)+ 全决策位;新增 M 的 supervisor→writer 对齐转移;分支集去 random、加 M×ΔV 三档 |
| [../../server/vllm/Exp_writer_isolation/exp_writer_score.py](../../server/vllm/Exp_writer_isolation/exp_writer_score.py) | BRANCHES/label/compare 同步(去 random、主判据 M×ΔV vs ΔV@50%) |

> ⚠ 与上一版不同:**本版要改 LMCache 核心(store_generated 用法 + blender dump 门控)**,因为决策段不进 blender、且 M 要在复用 pass 抓。

## 开放/风险
- **头号风险(单题先验)**:`store_generated` 存决策段后,再喂 `[上下文+决策]` 决策段是否**真命中、真进 blend 区**,从而 `dump_q` 抓到决策 q。门禁验 `M_rowsum_mean>0` 且非均匀。
- **复用 pass 的 q 是否"打到 K^r"**:确认该 pass summary 段确实走复用(`old_k` 不被覆盖)、决策位 attention 的 K 列 = 复用键。blender 日志核 `BLEND_PATH=CONTROL ... reuse 100%`。
- **M 对齐转移**:supervisor 帧 ↔ writer 帧 summary 逐 token 映射,需两边 summary 分词一致(门禁抽查)。
- **从头解题成本**:每题要重跑完整 pipeline(supervisor 复用)+ writer 12 分支,比旧版重;`SEARCH_CACHE=replay` 仍能免联网。
- 是否保 head 维 M(现默认对 head 求和聚合)——待定。
