# Exp_M · 执行 plan(改完代码后怎么跑,分阶段带门禁)

> 配套:[code_change_plan.md](code_change_plan.md)。先审核,暂不执行。
> **本版 M 来自 supervisor 的 decide 决策注意力**(非 writer report);M 必须等 supervisor 解码完决策才能算。

原则:**先单题验通 supervisor 端的 M(最关键),再铺开**。M 不通就别全量跑。

---

## 阶段 0 · 单题验 M(最关键,先过这关)
1. 起 control 模式 server(同 [../Exp_writer_isolation/run_instruction.md](../Exp_writer_isolation/run_instruction.md) Step 1:TP4 + 双开关 control + DUMP_KV + 分隔符 + `--return-tokens-as-token-ids`)。
2. 确认清库存接口:`curl -X DELETE "http://localhost:30000/cache/clear?locations=LocalCPUBackend"`(不通见 code_change_plan C2 兜底)。
3. 跑单题:`python Exp_writer_isolation/exp_writer_run.py "<q0 problem>" q0`。流程内部应:**从头跑 demo(supervisor 全复用 summary)** → 取最后一轮 decide 上下文 → 全复用下解码完整决策(**该发开 `store_generated` 存决策段**)→ **再喂 `[上下文+决策]`(决策已命中、进 blend 区)** 的复用 pass 抓 M(决策 q + `K^r`)→ 转移到 writer → writer 全重算 dump 出 ΔK/ΔV → 跑 12 分支。
4. **🚦 门禁(必须过)**:
   - **决策段命中**:server.log 见决策段 `store_generated` 存成功(`Queued final save ... generated N`)、再喂时整条命中缓存 → 决策段进 blend 区(否则 `dump_q` 抓不到决策 q)。
   - **复用真生效**:supervisor decide / 再喂那发 `BLEND_PATH=CONTROL ... reuse≈100%`(summary 0 重算);M dump 来自**复用 pass**、不是全重算 pass。
   - **M 真非 0**:`q0/controls/diag.json` 的 `M_rowsum_mean > 0` 且 M 非均匀(决策位 q 进了 dump、注意力打到 `K^r`)。
   - **M 用了全决策位**:prefix = 完整决策 token(不是只到 status_pos);dump `T` 覆盖决策位。
   - **两份 dump 没混**:M dump 用 `kv_r`(复用 pass);writer ΔK/ΔV dump 用 `kv_r`+`kv_star`(全重算 pass)。
   - **对齐正确**:抽查 supervisor 帧某 summary token 的 M,确认映射到 writer 帧同一 summary token(同文本同分词、逐 token 对上)。
   - **不过则停**:查 ① 决策段有没有 store+命中、② dump 门控有没有在复用 pass 触发、③ 决策 q 有没有进 dump、④ 两边 summary 分词是否一致。**别进阶段1**。

## 阶段 1 · 小批验内存 + 分支齐(5 题)
5. `EXP_W_N=5` 跑前 5 题。
6. **🚦 门禁**:
   - 每题 `reports.json` 含**全部 12 分支 key**(M×ΔV 的 6 支都在:token/layer × 75/50/25);
   - 连跑 5 题 `free -g` 的 CPU 占用**每题回落、不单调涨**(清库存生效);dump 目录不堆积。

## 阶段 2 · 全量 + 打分 + 出报告
7. 全量跑(`EXP_W_N` 设目标题数;断点续沿用现机制)。
8. 打分:`python Exp_writer_isolation/exp_writer_score.py all`(kimi-k2.5,deepsearchqa 官方口径)。
9. `compare` 出结论:
   - **M×ΔV vs ΔV(同 50% 同粒度)** 配对显著性 —— 超噪声底 0.14 → supervisor 注意力加权选位**优于纯 ΔV**(即"用上游决策的注意力指导下游 writer 复用"有效);
   - M×ΔV 各档(75/50/25)相对 full reuse 的恢复、相对 full prefill 上界的差距。
10. 出报告(单独 md 放本目录,沿用 results_14branch 的多指标 + 配对显著性风格)。

---

## 验证总览(每阶段一条硬指标)
| 阶段 | 门禁 |
|---|---|
| 0 单题 | supervisor 端 `M_rowsum_mean>0` 且非均匀 + 全决策位 + 对齐抽查对 |
| 1 小批 | 12 分支全在 + CPU 占用每题回落 |
| 2 全量 | `compare` 给出 M×ΔV vs ΔV@50% 是否显著(>0.14) |

## 关键环境(沿用现实验,见 run_instruction)
- server:`/home/yilin/anaconda3/envs/lmcache/bin/vllm`,端口 30000,TP4。
- driver / 打分:`/home/yilin/anaconda3/envs/gpt-deep/bin/python`。
- 路径对齐:driver 的 `LMCACHE_BLEND_CONTROL` / `LMCACHE_BLEND_DUMP_KV` 必须与 server 启动时一致。
- 搜索:`SEARCH_CACHE=replay` + `SEARCH_CACHE_DIR=/home/yilin/deep_researcher_demo/eval/results/search_cache`。

## 失败回退
- **M 全 0 / 门禁0 不过** → ① blender dump 门控没在复用 pass 触发(查 A1.5 改对没);② 决策位 q 没进 dump(查 dump `T` 覆盖 + prefix=完整 gen_ids);③ 复用没生效(summary 被重算了 → 注意力打到 `K*` 而非 `K^r`,查 `BLEND_PATH reuse%`)。
- **对齐错位** → supervisor 与 writer 的 summary 分词不一致:改用 harvest 原文按 summary 序号匹配,段内按文本对齐而非纯位置。
- **CPU OOM / 门禁1 不过** → 确认 `/cache/clear` 生效;不行调小 `LMCACHE_MAX_LOCAL_CPU_SIZE` 靠 LRU。
