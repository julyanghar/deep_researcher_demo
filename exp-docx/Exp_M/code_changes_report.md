# Exp_M · 代码改动报告(供审核)

> 配套设计:[code_change_plan.md](code_change_plan.md)、[execution_plan.md](execution_plan.md)。
> 状态:**只做了代码改动 + 语法编译通过(`py_compile`),没跑过任何一题**。机制能否成立要靠 execution_plan 的**阶段0 单题门禁**验证(见文末)。

## 概览
目标:把主方案 **M×ΔV** 跑通——M = **supervisor 决策在 summary 全复用条件下,对复用 summary 段(K^r)的注意力**,搬到 writer 指导其选择性重算。共改 4 个文件,2 个是 LMCache 核心、2 个是实验脚本。

| 文件 | 性质 | 一句话 |
|---|---|---|
| [../../lmcache/v1/compute/blend/blender.py](../../lmcache/v1/compute/blend/blender.py) | LMCache 核心 | dump 门控放开"复用 pass 也 dump" |
| [../../server/vllm/Exp_writer_isolation/exp_writer_run.py](../../server/vllm/Exp_writer_isolation/exp_writer_run.py) | 实验编排 | 新增 supervisor 复用 pass 抓 M 的链路 + 每题清库存 |
| [../../server/vllm/Exp_writer_isolation/exp_writer_select.py](../../server/vllm/Exp_writer_isolation/exp_writer_select.py) | 信号/选择 | `compute_M` 改用 `kv_r`;M 跨帧转移;12 分支(去 random) |
| [../../server/vllm/Exp_writer_isolation/exp_writer_score.py](../../server/vllm/Exp_writer_isolation/exp_writer_score.py) | 打分 | 分支表 + 主判据(M×ΔV vs ΔV@50%) |

---

## 1. blender.py —— dump 门控放开复用 pass
**改了什么**:`process_qkv` 里的 `dump_active` 条件,新增 `dump_in_reuse` 出口。
```python
# 原:and not self._control.get("reuse_token_ranges")
# 新:and (not self._control.get("reuse_token_ranges") or self._control.get("dump_in_reuse"))
```
**为什么**:原 dump 只在"全重算 truth pass"触发。M 要在 **summary 复用**条件下抓决策 q(决策 q 必须打到复用的 `K^r`),所以要让复用 pass 也能 dump。
**注**:实测我们的复用 pass 用的是 `recompute_token_idx` 格式(不是 `reuse_token_ranges`),原条件 `not reuse_token_ranges` 已为真、dump 本就会触发;`dump_in_reuse` 是给 `reuse_token_ranges` 格式留的兜底,**两条路都通**。dump 时机在 clone-revert 之前,`old_k`=载入的 `K^r`、`k`=新鲜 `K*`,两者都抓,M 只用 `kv_r`。

## 2. exp_writer_select.py —— M 用 K^r、跨帧转移、12 分支
**(a) `compute_M` / `_compute_M_on`**:加 `key_kind="kv_r"` 参数,注意力的 K 改用**复用的 `K^r`**(原来用 `kv_star`=K*)。`report_pos` 改名 `decision_pos`(语义=决策位 `[prompt_len, T)`)。
- 为什么:supervisor 真复用 → 决策注意力打到 `K^r`,不是新鲜 K*(用户确认的"复用条件下")。

**(b) 新增 `transfer_M(M_super, super_segs, writer_segs)`**:把 M 从 supervisor 坐标系搬到 writer 坐标系。
- 两边 summary 同文本、同分词、同分隔符切段 → 列一一对应;函数校验**段数一致、对应段长一致**,不一致即抛错(阶段0 门禁会接住)。

**(c) 分支表 `EXP_M_BRANCHES`(替换旧 `branch_id` + random 分支)**:
```
3=full_reuse;40-43 = {dv,dk}×{token,layer}@50%;50-55 = mxdv×{token,layer}×{75,50,25}
```
共 11 个(+driver 直接生成的 branch1=full prefill = 12)。**去掉 random**。

**(d) `select_and_write` 重写**:吃**两份 dump**——
- ΔK/ΔV 来自 `meta["writer_event"]`(writer 全重算 dump,writer 帧 summary 位);
- M 来自 `meta["m_event"]`(supervisor 复用 pass dump),`compute_M(key_kind="kv_r")` → `transfer_M` 到 writer 帧;
- `mats={dv,dk,mxdv=M·dV}` → 按 `EXP_M_BRANCHES` 写 control。

## 3. exp_writer_run.py —— 抓 M 的编排(核心)
**新增辅助**:`free_gen_report(..., store_generated=False)`(带 `kv_transfer_params={"lmcache.blend_store_generated": true}`)、`_non_summary_idx`(非 summary 位=要重算的位)、`_clear_cpu_cache`(`DELETE /cache/clear?locations=LocalCPUBackend`)。

**`run_demo` 修正(重要)**:旧版跑 demo 前写 `control={reuse_token_ranges:[]}` = **强制全重算 → 整条轨迹 full prefill、supervisor 决策不在复用下做**(错)。改成**删 control 文件** → blender 回退**原生 CacheBlend 复用**(server `RECOMPUTE_RATIOS=0`,≈全复用;同 exp_a 做法)→ demo 里 supervisor/writer **真复用 summary KV**,轨迹是真实条件。summary 仍照存。

**`measure_question` 重写为**(每题):
1. 跑 demo(**原生 CacheBlend 复用**)→ harvest(产 summary、存 K^r 进 LMCache)。
2. **writer 帧**:`pick_writer_call` → `writer_prompt` + `find_segments` → `writer_summary_pos`。
3. **supervisor 帧**:`pick_decision_call`(复用 exp_a)→ `super_prompt` + `find_segments` → `super_summary_pos`。
4. **抓 M(复用条件)**:
   - 4a:control=`{recompute_token_idx: 非summary}`(summary 复用)下 `free_gen_report(super_prompt, store_generated=True)` → 解码决策 + 决策段 KV 进缓存(可命中)。
   - 4b:control=`{recompute_token_idx: 非summary, dump, dump_q, dump_in_reuse}`,再喂 `[super_prompt+决策]` → 决策段命中、进 blend 区 → dump 抓决策 q + summary 的 `K^r` → `m_event`。
5. **branch1 full prefill report** + **writer ΔK/ΔV dump**(`{reuse_token_ranges:[], dump}` 全重算 teacher-force writer_prompt+report)→ `writer_event`。
6. `select_and_write(meta)`(meta 带两份 event + 两帧 summary 信息)→ 各分支 control;**两份 dump 读完即删**。
7. 各分支:逐个 control → `free_gen_report(writer_prompt)` 出报告。
8. 落 `reports.json` + 复位 control + `_clear_cpu_cache()`。

**删/换**:旧的"writer 端 teacher-force `[writer+report]` 算 M"换成上面 4(M 改从 supervisor 取);`store_generated` **保留但只用于 supervisor 决策段**。

## 4. exp_writer_score.py —— 分支表 + 判据
- `_LABEL` / `BRANCHES` 改成 Exp_M 的 12 分支(去 random),`branch_label` 查表。
- `do_compare` 主判据改为:① 各分支 − full_reuse(是否救回);② **M×ΔV − ΔV(同 50% 同粒度)**(注意力加权是否优于纯 ΔV);总判读 = 50% token 的 M×ΔV 是否既超 full_reuse 又优于 ΔV。`do_score`/`do_agg` 自动跟随新 `BRANCHES`。

---

## M 抓取链路(串起来看)
```
demo(存 summary KV) → 取 supervisor decide 上下文 [P0, summary..., 问题]
  → 4a: summary 复用下解码决策 + store_generated(决策段 KV 进缓存=可命中)
  → 4b: 再喂 [上下文+决策](决策命中→进 blend 区)+ summary 复用 + dump
        → blender dump: 决策位 q(每层)+ summary 的 K^r
  → compute_M(kv_r): M[L,Ns_super] = Σ_决策位 Σ_head softmax(q·K^r_summary)
  → transfer_M → M[L,Ns_writer]
writer 帧: 全重算 dump → ΔK/ΔV;  mxdv = M·ΔV → 选位 → 12 分支报告 → 打分
```

## 没改 / 复用的
- blender 的 patch-C `dump_q`、clone-revert、`store_generated`(`_process_final_saves`)都是现成的。
- `delta_kv`、`select_token`/`select_layer`/`_topx_idx`、`load_blend_dump`、exp_a 的 `pick_decision_call`/`context_token_ids`/`find_segments` 复用未改。

## ⚠ 未验证 / 风险(阶段0 必须先验,别直接全量跑)
1. **`kv_transfer_params` 能否经 `/v1/completions` 透传**(store_generated 是否真生效)。验:server.log 出现 `Queued final save ... generated N`。
2. **再喂时决策段是否真整条命中、进 blend 区**(否则决策 q 不在 dump,M=0)。验:`diag.json` `M_rowsum_mean>0` 且非均匀。
3. **复用 pass 的决策 q 是否打到 `K^r`**(summary 真复用):核 `BLEND_PATH=CONTROL ... reuse`。
4. **transfer_M 的两帧 summary 段长一致**(同文本同分词):不一致会抛错——阶段0 抽查。
5. **writer ΔK/ΔV dump 的 summary 是否仍命中**(store_generated 之后缓存没被挤掉)。
6. 内存:两份 dump + store_generated 的决策段 KV,靠每题 `_clear_cpu_cache` + 删 dump 控住;阶段1 连跑 5 题看 `free -g` 是否回落。

## 怎么验(对应 execution_plan 阶段0)
起 control 模式 server(`/cache/clear` 可用)→ `python Exp_writer_isolation/exp_writer_run.py "<q0>" q0` → 看 `q0/controls/diag.json` 的 `M_rowsum_mean>0` 且非均匀、`reports.json` 含 12 分支 key、server.log 有 store_generated + 复用日志。**门禁过了再铺开。**
