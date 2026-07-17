# Exp_writer_isolation · Part 1 代码改动报告(大白话版)

> ## 🚩 当前状态(2026-06-25,实跑 q0 后更新)
> - ✅ **14 个非 M 分支已端到端跑通并在日志逐分支核对正确**(branch 1/3 + ΔV/ΔK/random × {token,layer};reuse 率与 control 文件精确一致,见 [run_q0_verification.md](run_q0_verification.md))。
> - ❌ **M×ΔV 的 4 个分支(4/5/12/13)暂未实现、已在代码里跳过**。
>   **原因(实跑才发现)**:算 M 需要"报告 token 的 query",但 branch1 的 teacher-force **blend dump 只覆盖 LMCache 缓存命中的可复用前缀区**(summary 那段,`tokens[:lmcache_cached_tokens]`,见 [vllm_v1_adapter.py:903](../../lmcache/integration/vllm/vllm_v1_adapter.py#L903)),**根本不含报告位的 query**。所以"用 dump 离线算 M、把 B 并进 C"的简化**行不通**。
>   **待办**:M 要单独做一次**抓注意力的前向**(HF `output_attentions` / 模型 hook),在报告位抓对 summary 的注意力。这是原计划 B 那块、省不掉。

> 对应 [part1_code_and_test.md](part1_code_and_test.md) 的 A→E。**只动了一处已验证代码**([blender.py](../../lmcache/v1/compute/blend/blender.py)),**新写四个脚本**;原计划单列的 B(算 M)**没能并进 C(见上方状态)**。
> 全部**离线单测通过**(含 18 分支全覆盖);D/E 真机端到端:**非 M 分支已验,M 分支待补**。测试:[test_exp_writer.py](../../server/vllm/Exp_writer_isolation/test_exp_writer.py)。

---

## 这套代码要干嘛(先说人话)
这个实验只想搞清一件事:**写报告的那一步(writer)在复用 summary 的 KV 时,到底是"全复用"伤报告,还是"挑几个关键的 token 重算一下"就能救回来**。
为此每道题:前面的研究(supervisor + researcher)照常老老实实算一遍(干净),只把**最后"写报告"这一步**反复换不同的"KV 怎么处理"跑 18 次,看哪种写出来的报告质量好。

四块新东西各管一段:
- **A** 让服务端能接受"挑哪些 token / 哪些层重算"的指令(原来只会按连续区间);
- **C** 离线算"哪些 summary token 最该重算",生成上面那些指令;
- **D** 把整件事串起来:跑一遍研究 + 重放 18 次 writer;
- **E** 给 18 份报告用 kimi 打分、判输赢。

---

## A — [blender.py](../../lmcache/v1/compute/blend/blender.py):给"复用开关"加两种新挑法

**原来**:服务端用 [`_controlled_reuse_indices()`](../../lmcache/v1/compute/blend/blender.py#L383) 决定"哪些 token 保持复用、其余重算",但只认 `reuse_token_ranges`(一段连续区间),而且**所有层用同一套**。这做不到本实验要的两种新粒度。

**改成**:同一个函数 [blender.py:383](../../lmcache/v1/compute/blend/blender.py#L383) 加了个 `layer_id` 参数,按 control 文件里的字段分三种挑法:
1. `recompute_layers`(**按层挑**)—— 这一层在名单里就整层重算、不在就整层复用;
2. `recompute_token_idx`(**按 token 挑**,可以是散开的)—— 名单里的 token 重算、其余复用,每层一样;
3. `reuse_token_ranges`(**老的**)—— 前两个字段都没有时走它,所以 Exp A 那条老探针轨**一点不受影响**。

调用点也跟着把 `layer_id` 传进去([blender.py:173](../../lmcache/v1/compute/blend/blender.py#L173)),日志加了 `mode=` 标识([blender.py:188](../../lmcache/v1/compute/blend/blender.py#L188)),按层挑时改打"重算 K/L 层"的整体口径([blender.py:177-184](../../lmcache/v1/compute/blend/blender.py#L177-L184))。

**为什么敢这么改**:真正干活的那段"先全算、再把要复用的位置塞回旧值"(clone-revert,[blender.py:197](../../lmcache/v1/compute/blend/blender.py#L197) 和 [:202](../../lmcache/v1/compute/blend/blender.py#L202))**一个字没动**——三种挑法只是换了"哪些位置算复用"这份名单。最不容易碰坏已经验证过的复用通路。
**测过的等价性**:`recompute_token_idx=全部` 等于 `reuse_token_ranges=[]`(全重算=真值)、`=[]` 等于全复用;没有新字段就走老逻辑。旧 control 文件行为不变。

---

## C — [exp_writer_select.py](../../server/vllm/Exp_writer_isolation/exp_writer_select.py):给每个 summary token 打分、挑出该重算的

**先解释"算三种逐(层,summary token)信号"是什么意思**(原报告这句太拗口):
就是算**三个打分矩阵,每个矩阵的形状是 [层数 L × summary token 数 Ns]**——也就是**给每一个(层, summary token)的组合各算一个分**。三个分分别是:
- **ΔV**:这个 summary token 在这一层,"复用的 V"和"重算的 V"差多大([delta_kv()](../../server/vllm/Exp_writer_isolation/exp_writer_select.py#L72) 里逐 head 求 L2 再对 head 求和)。差大 = 复用误差大。
- **ΔK**:同上,但看 K(独立信号,也用于检查 K/V 漂移相不相关)。
- **M**:写报告时,**报告里的字对这个 summary token"看"得多重**(注意力权重,对所有报告 token 和所有 head 求和;[compute_M()](../../server/vllm/Exp_writer_isolation/exp_writer_select.py#L85))。看得越重 = 这个 token 越关键。

然后 **M×Δ := M·ΔV**(逐元素相乘)——既漂得多、又被报告看得重的 token,最该重算。把矩阵压成一维选:选 token 时对层求和([select_token()](../../server/vllm/Exp_writer_isolation/exp_writer_select.py#L123)),选层时对 token 求和([select_layer()](../../server/vllm/Exp_writer_isolation/exp_writer_select.py#L136))。每档(75%/25%)取分最高的 top-X% 生成 control,[select_and_write()](../../server/vllm/Exp_writer_isolation/exp_writer_select.py#L152) 一次性写出 branch 3-19 的 control 文件;[branch_id()](../../server/vllm/Exp_writer_isolation/exp_writer_select.py#L145) 负责把(档位, 信号, 粒度)映射成文档里的分支号 4-19。

**关键巧妙处(也是 B 为什么能并进来)**:算 M 需要"报告 token 的 query"和"summary token 的 key"。这两样在 branch1 那个 teacher-force dump 里**同时都有**(dump 覆盖 `[writer_prompt + 报告]`,summary 位给 K、报告位给 query)。所以 **M 纯离线一行 `softmax(q·K*)` 就能算**,不用像原计划那样单独起进程 hook 模型——原 plan 里最重的一块直接省了。代码里 GQA(query head 比 KV head 多)、因果遮罩、TP 分头存盘要拼回([load_blend_dump()](../../server/vllm/Exp_writer_isolation/exp_writer_select.py#L44) 按 rank 拼)都处理了。

**算 M 的显存安全(实测过)**:M 用 einsum 向量化(去掉原来逐报告 token 的 Python 循环),并有两道防线防 OOM——① [`_pick_m_device()`](../../server/vllm/Exp_writer_isolation/exp_writer_select.py#L98) **自动挑最空闲的 GPU(≥4GB)、躲开那 4 张被模型占着只剩 ~1GB 的 TP 卡**,都不够就回 CPU(可 `EXP_W_M_DEVICE` 覆盖);② q/K* 留 CPU、**逐层搬上卡**(峰值 ≈ 一层 q+k,而非整个 `[64, T, 8192]` 上卡)。真机实测:最大尺寸(T=13000、Ns=5000)逐层峰值仅 **4.27GB**(整 64 层 q 一次上卡要 27GB),自动选到空闲的 cuda:5;旧法整 q 往模型卡 cuda:0(1GB 空)如期 OOM。

> token 挑法的 control 是"非 summary 位 + 选中的 summary 位都重算",也就是**只复用没被选中的 summary token**;P0 和分隔符在所有分支都重算、保持一致 → 真正做到"唯一变量 = summary token 的 KV 怎么算"。

---

## B — 算 writer 的 M:**已并进 C**
原 plan 担心要单独 monkeypatch 模型抓注意力。实测不用:见上面 C 的"关键巧妙处",M 直接从 branch1 的 dump 离线算,代码就是 [compute_M()](../../server/vllm/Exp_writer_isolation/exp_writer_select.py#L85)。所以 B 没有独立文件。

---

## D — [exp_writer_run.py](../../server/vllm/Exp_writer_isolation/exp_writer_run.py):跑一遍研究,然后只把"写报告"重放 18 次

**不是每题跑 18 条完整轨迹**。每题就一条主轨 + 18 次"只重放 writer 那一步":
1. **主轨**:control 设成 `{reuse_token_ranges:[]}` = 全程干净全重算([run_demo() blender 主轨写法 exp_writer_run.py:50](../../server/vllm/Exp_writer_isolation/exp_writer_run.py#L50)),跑完整 demo,researcher 把 summary 的 KV 存下来;从 harvest 里拿出 writer 那次调用(tag = `FINAL_REPORT_MARKDOWN`,[pick_writer_call()](../../server/vllm/Exp_writer_isolation/exp_writer_run.py#L69))当作"钉死的 writer prompt"。
2. **branch1**:先正常写一份 report_1;再 teacher-force `[prompt+report_1]` 触发一次带 dump 的前向([exp_writer_run.py:133](../../server/vllm/Exp_writer_isolation/exp_writer_run.py#L133)),给 C 算 ΔK/ΔV/M。
3. 调 [select_and_write()](../../server/vllm/Exp_writer_isolation/exp_writer_run.py#L146) 生成 branch 3-19 的 control。
4. **branch 3-19**:逐个写 control → 同一个 writer prompt 重新生成报告([free_gen_report()](../../server/vllm/Exp_writer_isolation/exp_writer_run.py#L81))。
5. 全部存成 `reports.json {分支号: 报告}` 给 E。

**顺手修的一个坑**:每题的输出目录从默认 `EXP_W_OUT/{sid}` 改成 `EXP_W_OUT(当基目录)/{sid}`([exp_writer_run.py:103](../../server/vllm/Exp_writer_isolation/exp_writer_run.py#L103))——否则跑多题时设了 `EXP_W_OUT` 会互相覆盖。`reports.json` 在就跳过(可断点续)。

---

## E — [exp_writer_score.py](../../server/vllm/Exp_writer_isolation/exp_writer_score.py):给 18 份报告用 kimi 打分、判输赢

照 [exp_sw_score.py](../../server/vllm/Exp_sw/exp_sw_score.py) **同一条官方打分链**,只是改成**按分支**:每题读 `reports.json`,对 18 个分支逐题用 kimi 打分,逐题追加到 `ratings_branch_<bb>.jsonl`(可断点续、每题只打一次,[score_branch()](../../server/vllm/Exp_writer_isolation/exp_writer_score.py#L84))。[branch_label()](../../server/vllm/Exp_writer_isolation/exp_writer_score.py#L38) 把分支号翻译回(信号, 档位, 粒度)。[do_compare()](../../server/vllm/Exp_writer_isolation/exp_writer_score.py#L165) 出定性判据:每档四方各自比 full reuse(分支3)、M×ΔV 比 random,差要**超过噪声底 0.14**([NOISE_FLOOR](../../server/vllm/Exp_writer_isolation/exp_writer_score.py#L34),来自四臂 A_orig vs A_orig2)才算数,最后打一句总判读。

---

## 测了什么、过没过([test_exp_writer.py](../../server/vllm/Exp_writer_isolation/test_exp_writer.py),本机 torch 2.10 CPU)

| 测什么 | 结果 |
|---|---|
| A 的 [`_controlled_reuse_indices`](../../lmcache/v1/compute/blend/blender.py#L383) 三模式 + 向后兼容(7 项) | ✓ |
| C 的 [ΔK/ΔV](../../server/vllm/Exp_writer_isolation/exp_writer_select.py#L72)、[M](../../server/vllm/Exp_writer_isolation/exp_writer_select.py#L85)、M×ΔV:形状对、非负、`复用==真值则Δ=0`、`M≤报告数×head 数` | ✓ |
| **全 18 分支端到端**:1(真值)、3(全复用)、4-19(75%/25% × {M×ΔV,ΔV,ΔK,random} × {token,layer}),每个 control 喂回 A 逻辑、断言复用/重算模式与计数 | ✓ |
| 预算↔复用%(25% 档复用更多)、覆盖恰好 {1,3..19}、产出 17 个 control | ✓ |
| 四个文件语法编译;D 导入;E 的 [eval](../../../deep_researcher_demo/eval/scoring.py) 函数名 grep 确认存在 | ✓ |

**本机跑不了、要你真机验**:
- D 端到端 smoke(主轨→dump→选择→18 份报告):验 18 分支 writer prompt 字节一致、`BLEND_PATH=CONTROL` 的 reuse% 与档位对得上、报告非空。
- E 的 kimi 打分:本机缺 `bs4`(deep_researcher_demo 的网页依赖)无法直接 import E,这是**环境问题不是代码 bug**,打分机 gpt-deep 上有,import 那一组与 [exp_sw_score.py](../../server/vllm/Exp_sw/exp_sw_score.py) 完全一致。
- M 的数值合理性(检查1):要真 dump 才能看 M 高的 summary token 是不是报告真去复述/引用的内容。

## 没动什么 / 边界
- per-role separator(`config.py`/`cli.py`/`agents.py`)上一轮已改已验已提交,本轮没碰。
- **没跑任何真实实验**(按你"先测试再写报告");以上是离线能验的范围,实跑按 [part2_run_experiment.md](part2_run_experiment.md)。
