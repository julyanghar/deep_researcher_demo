# Writer 隔离实验 · Part 1:代码实现 + 测试

> 两段式:**本文 = Part 1**(把机制建好、逐件单测验真,不烧大实验)→ 全绿才进 [part2_run_experiment.md](part2_run_experiment.md)(正式跑 18 分支 × N 题)。
> **当前状态:只写 plan,先不改代码**(用户要求)。下面每件标了「现状 / 怎么改 / 为什么 / 测试」。

## Context(这套代码服务的实验)
四臂结论:纯 KV 复用对答案无统计显著危害,但 micro F1 低 ~6-8 分、源于 ~3 道崩题;复用发生在 **supervisor + writer** 两处。SW 已隔离 supervisor。**本实验隔离 writer**:除 writer 外全 full prefill、researcher summary 走 prefill 但 **store generated KV**,只在 writer 分支化,喂同一份逐 token 钉死的 summary,只切"writer 读 summary KV 怎么处理"。
更进一步:测 **M×ΔV 选择性刷新**——只重算"注意力加权 value 漂移"高的 top-X% token/layer,看能否用少量重算救回质量、且选得比 random 准。
骨架沿用 [../ExpB_fullseg/probe_trajectory_design.md](../ExpB_fullseg/probe_trajectory_design.md) 的**主轨(control-truth 干净)/ 分叉(只换 writer KV 处理)**范式;18 分支细节见 [part2_run_experiment.md](part2_run_experiment.md)。

## 需求 → 代码映射(能复用的别重写)
| 件 | 可直接复用 | 要新写 |
|---|---|---|
| teacher-forcing harness / 搜索 replay / 逐 pass 编排 | [../../server/vllm/Exp_A/exp_a_tf_runner.py](../../server/vllm/Exp_A/exp_a_tf_runner.py)(`write_control` / `free_gen_decision` / `find_segments` / `make_searcher` / `measure_question`) | **D** 套它改 |
| KV 漂移 ΔK/ΔV(逐层逐 token 范数) | [../../server/vllm/Exp_A/exp_a_drift_measure.py](../../server/vllm/Exp_A/exp_a_drift_measure.py)、[../../server/vllm/blend_debug/blend_offline.py](../../server/vllm/blend_debug/blend_offline.py) | **C** 调它 |
| control(clone-revert + reuse_token_ranges)+ KV dump + dump_q | [../../lmcache/v1/compute/blend/blender.py](../../lmcache/v1/compute/blend/blender.py)(:172-195 控制分支 / :373 reuse 下标 / :163 q-dump) | **A:加 per-layer / token-set** |
| writer 单独配 prefill/reuse(per-role separator) | [../../../deep_researcher_demo/deep_researcher_demo/config.py](../../../deep_researcher_demo/deep_researcher_demo/config.py)、[cli.py](../../../deep_researcher_demo/deep_researcher_demo/cli.py)(已改、已验) | — |
| 评分官方链 | [../../eval/scoring.py](../../eval/scoring.py)、[../../eval/judge.py](../../eval/judge.py)、[../../eval/run_deepsearchqa.py](../../eval/run_deepsearchqa.py) | **E** 包脚本 |

---

## P1.A — blender 加 per-layer + token-set 选择性重算(唯一服务端改动)

**现状**:控制分支用 `_controlled_reuse_indices(total_len, device)`([blender.py:373](../../lmcache/v1/compute/blend/blender.py#L373)),只从 `reuse_token_ranges` 出"保持复用(KV^r)"的 token 下标,且**层间统一**(每层同一套复用);clone-revert 主体在 [blender.py:184-193](../../lmcache/v1/compute/blend/blender.py#L184)(`old_k[:]=k` 全算 → `old_k[reuse_idx]=KV^r` 把选中的改回复用)。→ **做不到"选 layer"**(整层重算 / 整层复用),也不方便表达散点 top-X% token。

**怎么改**(向后兼容,默认仍走老 `reuse_token_ranges`,supervisor 探针轨**不受影响**):
1. `_controlled_reuse_indices` 签名加 `layer_id`;按 control 字段分三模式:
   - 有 `"recompute_layers": [ℓ,...]` → `layer_id∈列表`:返回 `None`(整层重算 KV*);否则返回全下标 `0..total_len`(整层复用 KV^r)。**layer 分支**用。
   - 有 `"recompute_token_idx": [i,...]` → 返回**补集**(全部 token 减这些;选中的全层重算、其余全层复用)。**token 分支**用。
   - 都没有 → 老 `reuse_token_ranges` 逻辑(原样)。
2. clone-revert 主体(:184-193)不动,只换 `reuse_idx` 来源;`BLEND_PATH=CONTROL` 日志已报 `reuse=.. recompute=..`([blender.py:177](../../lmcache/v1/compute/blend/blender.py#L177))。

**为什么这么改**:① clone-revert 是"全算再按复用改回复用",所以无论 token 还是 layer 维度,都只是**换一组 reuse_idx**,主体零改动、最不容易碰坏已验证的复用通路;② per-layer 靠 `layer_id` 参数(process_qkv 本就有)天然能分;③ control 文件 per-blend 重读([blender.py:394](../../lmcache/v1/compute/blend/blender.py#L394)),分支间换文件即可,server 不重启。

**测试 A**(小 prompt 直发 blend,看日志 + 抽 KV):
- **等价性**:`recompute_token_idx`=全部 ≡ `reuse_token_ranges=[]`(truth,recompute 100%);`recompute_token_idx=[]` ≡ 全复用(reuse 100%)。两端必须和现有 truth/全复用**逐 token 一致**。
- **token 分支比例**:`recompute_token_idx`=N/4 → 日志 `recompute≈25%`。
- **layer 分支**:`recompute_layers`=前 75% 层 → 抽一层在列表内(`recompute=100%`)、一层不在(`reuse=100%`),逐层日志对得上。

## P1.B — writer M 提取(oracle,离线,最重的一块)

`M_{i,ℓ}=Σ_{g∈G}Σ_h softmax_j(q_g·k_j/√d)|_{j=i}` —— 报告生成 token g 对 summary token i 的注意力,来自**分支1 全真值** writer forward。

**现状缺口**:blender 的 dump_q([blender.py:163](../../lmcache/v1/compute/blend/blender.py#L163))抓的是 **summary 段 prefill 时的 query**,**不是生成态报告 token 的 query** → 直接拿来算 M 是错的。

**怎么做**:teacher-force `[writer_prompt + 钉死的报告]` 一次 forward,抓**各层报告位的 query `Q_g`** + **summary 段的 key `K*`**(K* 已在分支1 的 KV dump 里),离线 `softmax(Q_g·K*/√d)` 对 g、h 求和 → `M`(N×L)。实现复用 [exp_a_drift_measure.py](../../server/vllm/Exp_A/exp_a_drift_measure.py) 的**单进程 + monkeypatch 抓 hook** 范式(`install_dump_hook` :59),和服务端 TP4 blend **解耦**——M 是纯离线 oracle,单 GPU 加载一次模型抓注意力即可。

**为什么单列**:M 来自实际 forward 注意力 → **已含"K 漂移如何改变注意力"**的效应,所以 `M×Δ` 只乘 ΔV、ΔK 另作独立信号。

**测试 B**:M 形状 N×L、非负;每 (g, head) 行 softmax 和=1 → 总行和 ≈ H×|G|;抽查 M 高的 summary token = 报告实际复述/引用的内容(人眼 sanity)。

## P1.C — 离线"算分→选 top-X%→写 control"(新脚本 `server/vllm/Exp_writer_isolation/exp_writer_select.py`)

读分支1 dump:`ΔK=‖K^r−K*‖₂`、`ΔV=‖V^r−V*‖₂`(逐层逐 token,范数算法复用 `exp_a_drift_measure` 的 `summarize`/`cos_dist` 那套);`(M×Δ)=M·ΔV` **逐元素乘(先乘后求和,铁律)**。聚合:per-token 分数=`Σ_layer`、per-layer 分数=`Σ_token`。取 top-{75,25}%。`random`=固定多 seed 均匀抽。每分支输出一个 control JSON(`recompute_token_idx` 或 `recompute_layers`)。

**测试 C**:top-25% token 数≈0.25N、top-75% layer 数≈0.75L;四信号(M×ΔV/ΔV/ΔK/random)control 文件各自合法且互不相同。附**两个零成本检查**(Part 2 判读要用):② ΔK/ΔV 逐 token 相关性(强相关→两信号近似;弱相关→各有信息);③ layer 求和是否被 softmax 归一化偏向"弥散层"污染。

## P1.D — 18 分支编排器(新脚本 `server/vllm/Exp_writer_isolation/exp_writer_run.py`,套 `exp_a_tf_runner`/`exp_fullseg_runner`)

> 编号沿用原设计、**去掉原分支2(第二条 full prefill 噪声底)→ 共 18 分支(1、3-19,无 2)**;噪声底改用四臂现成本底(A_orig vs A_orig2 ≈ Set F1 0.14),不再单跑第二条 full prefill。

**核心:不是每题跑 18 条完整轨迹。** writer 是整条链最后一步、不反馈轨迹 → supervisor 多轮 + researcher 多 summary + 搜索(最贵)**每题只跑 1 次**;18 个分支 = 在那条主轨**钉死的 writer prompt** 上**只重放 writer 这一步**(每分支 = 1 次 writer 调用,prompt byte 一致,唯一变量 = 当次的 control 文件)。机制 = `exp_a_tf_runner` 的"固定 prompt 反复 replay + blender per-blend 重读 control"([blender.py:394](../../lmcache/v1/compute/blend/blender.py#L394)):串行发 18 个请求、每个请求前把对应 control 写进去,server 不重启。成本 ≈ **1 条轨迹 + 18 次报告生成**(≈31 次调用),而非 18×轨迹(≈234 次)。

每题:
1. **主轨(只 1 次,最贵)**:control-truth `run_demo`(分隔符开、summary `store_generated_kv=True`、每个 supervisor 决策 blend control=`[]` 全重算)→ harvest **writer prompt** + **pin summary**(内容+token 固定存盘)+ store 好各 summary 的 KV^r。
2. **分支1(writer 重放)**:writer full prefill(control=`[]`,dump=true)→ KV* dump + 报告_1;再 teacher-force 报告_1 跑 **P1.B** 取 M(多一次 teacher-force pass,仍是 writer 单步)。
3. **分支3(writer 重放)**:**writer-only full reuse**——只在 writer 这步对 summary KV 全复用(reuse=全 → reuse 100%);supervisor/researcher 仍 full prefill(与整套"除 writer 外全 full prefill"一致)。
4. **分支4-19(writer 重放,各 1 次)**:写 P1.C 产的 control → 同一 writer prompt → **free-gen** 报告(F1 要真生成,不是 teacher-force)。
各报告落盘待打分;`SEARCH_CACHE=replay` 共享检索(主轨录一次、各分支不再 live)。

**为什么 pin summary**:18 分支 writer prompt 必须 **byte 级一致** → 唯一变量 = summary token 的 KV 怎么算。否则混入"summary 内容也变了"的 confound,隔离失败。

**测试 D(smoke:1-2 题,只跑锚点 1/3(full prefill + writer-only full reuse)+ 1 个 25% M×ΔV 分支)**:
- pinned writer prompt 在这几个分支 **byte 一致**;
- 各分支报告**非空**(control clone-revert 保证 free-gen 不崩);
- `BLEND_PATH=CONTROL` 日志证明 control 被读到、`reuse%` 和分支设定对得上;
- 报告正确落盘到 `EXP_*_OUT/<sid>/<branch>/`。

## P1.E — 评分链 kimi-k2.5(新脚本 `server/vllm/Exp_writer_isolation/exp_writer_score.py`)

判据模型 **kimi-k2.5**(`.env` 现成 DashScope:`JUDGE_BASE_URL` / `JUDGE_API_KEY` / `JUDGE_MODEL=kimi-k2.5`,同端点同 key;跑前先 sanity 一份)。**不自己写打分逻辑**——照 [run_deepsearchqa.py](../../eval/run_deepsearchqa.py) 的 `rate_report_record`:`score_answer → build_item_rating_from_report → rate_report(kimi) → reduce_autorater_response → ItemRating`,汇总 `aggregate_ratings → metrics`(building blocks 在 [eval/scoring.py](../../eval/scoring.py) + [eval/judge.py](../../eval/judge.py),直接 import)。**逐题 append `ratings_<branch>.jsonl`、已打 sample_id 跳过**(resumable、每题只打一次、不重复花钱)。逐题 F1:`tp=Σgrader_ratings`、`fp=len(wrong)`、`fn=len(expected)−tp` → `calculate_metric`。

> 同一套 `exp_writer_score` **顺手也给已跑完的 SW(B/SW/C 三轨)打分**(原"统一打分"plan 的目标)——同链、同尺子,不丢。

**测试 E**:kimi sanity 打 1 份报告→返回评分 JSON→parse 成 ItemRating;resume 重跑跳过已打题。

---

## ✅ Part 1 完成判据(全绿才进 Part 2)
A 等价性 + token/layer 比例对 · B M 形状/归一/人眼 sanity 过 · C 选择比例对 + 两零成本检查出数 · D smoke 报告非空且 prompt 钉死、control 生效 · E kimi 通 + resume 生效。

## ⚠ 执行顺序
**现在只定稿本 plan,不动代码**。批准后实现顺序:A(服务端,先验等价性别碰坏复用)→ B(M)→ C(选择)→ D(编排,smoke)→ E(评分)。每件做完先过它的测试,再做下一件;五件全绿 → 转 [part2_run_experiment.md](part2_run_experiment.md)。

*相关:[../ExpB_fullseg/probe_trajectory_design.md](../ExpB_fullseg/probe_trajectory_design.md)、[../ExpB_fullseg/callchain_cdriver_and_control.md](../ExpB_fullseg/callchain_cdriver_and_control.md)、[../ExpB_fullseg/kv_dump_implementation.md](../ExpB_fullseg/kv_dump_implementation.md);memory `blend-three-modes-env`、`kvcomm-attn-weighted-reuse-error`。*
