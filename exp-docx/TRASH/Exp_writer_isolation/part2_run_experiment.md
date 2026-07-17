# Writer 隔离实验 · Part 2:正式跑 18 分支(定性判断 M×ΔV 选择性刷新)

> 两段式:[part1_code_and_test.md](part1_code_and_test.md)(代码+测试)**全绿后**才跑本文。
> **本版只做定性**:25%/75% 两端点,不画精细预算曲线;先判"选择性刷新有没有用、M×Δ 选得准不准"。
> **结论是性能上界**:M 从分支1(全真值)拿 = oracle M,部署拿不到(鸡生蛋)→ 表述为"上界"。

## 实验定位(一句话)
除 writer 外全程 full prefill;researcher summarize 走 prefill 但 **store generated KV**;只在 writer 分支化,喂**同一份逐 token 钉死的 summary**,只切"writer 读 summary 时 KV 怎么处理",跑题统计报告质量。

## 骨架(主轨 / 分叉)
- **主轨 = control-truth 跑 demo**(supervisor 决策全重算=干净、researcher summary prefill+store KV、分隔符开)→ 固定 writer prompt(pinned summary)+ 各 summary 的 KV^r。
- **分叉**:18 分支共用同一 writer prompt,**唯一变量 = summary token 的 KV 怎么算**。

## 选择信号矩阵(18 分支骨架)
每档(仅 **75% / 25%** 两档)下,**4 信号 × 2 粒度 = 8 分支**,两档共 16,加 2 锚点 = **18**(去掉原分支2 第二条 full prefill,编号沿用、无 2)。
| 信号 | 选什么重算 | 对照意图 |
|---|---|---|
| **M×ΔV** | M×ΔV 高的 | 主方案(注意力加权 value 漂移)|
| **ΔV** | value 漂移高的 | vs M×ΔV → 加不加注意力 M 的增量 |
| **ΔK** | key 漂移高的 | vs ΔV → key vs value 谁更适合当信号 |
| **random** | 随机(固定多 seed)| vs 上三者 → 选得准不准(非刷新量的功劳)|

**聚合(求和)**:per-token 分数 `=Σ_layer 各层值` → 选 token(重算其全层 KV);per-layer 分数 `=Σ_token 各 token 值` → 选 layer(重算该层全 token KV)。⚠ layer 求和受 softmax 归一化偏向"弥散层",见检查3。

## 18 个分支(编号沿用原设计、去掉原分支2 → 无 2,共 18)
**对照锚点(2)**
```
1.  full prefill      : baseline,记录 oracle M(供选 token/layer)+ 记录 ΔK、ΔV
3.  full kv reuse     : writer-only full reuse——只在 writer 这步对 summary KV 全复用
                        (supervisor/researcher 仍 full prefill);0% 重算 = 下界锚点
```
> 噪声底不再单跑(原分支2),改借四臂现成本底 A_orig vs A_orig2(两条 full prefill,paired Set F1 ≈ 0.14)。
**75% 重算档(8)**:`4-5 M×ΔV(token/layer)`、`6-7 ΔV`、`8-9 ΔK`、`10-11 random`
**25% 重算档(8,预算最紧、最见"选得准"=主战场)**:`12-13 M×ΔV`、`14-15 ΔV`、`16-17 ΔK`、`18-19 random`
> 每档 8 分支严格对齐(同档同粒度,唯一变量=选择信号);随机基线固定/多 seed,可复现。

## 符号定义(算分用)
writer 上下文 `[system][summary_1]...[summary_M]`;被复用 summary token 集 `S`(|S|=N);L 层、每层 H 个 KV head、head 维 d;writer 生成报告 token 集 `G`。
- `ΔV_{i,ℓ}=‖V^r_{i,ℓ}−V*_{i,ℓ}‖₂`(进 M×Δ + 独立 ΔV 分支)
- `ΔK_{i,ℓ}=‖K^r_{i,ℓ}−K*_{i,ℓ}‖₂`(独立 ΔK 分支 + ΔK/ΔV 相关性)
- `M_{i,ℓ}=Σ_{g∈G}Σ_h softmax_j(q_g·k_j/√d)|_{j=i}`(下游依赖度,oracle,**已含 K 漂移的注意力效应**)
- `(M×Δ)_{i,ℓ}=M_{i,ℓ}·ΔV_{i,ℓ}`(**先逐元素乘、再求和**)
`*`=writer 完整上下文重算真值;`^r`=researcher store + RoPE 重定位复用值。各为 N×L 非负矩阵。

## 起 server(control 模式,双开关 + dump + 分隔符,TP4 找 4 张空卡)
```bash
export LMCACHE_ENABLE_BLENDING=true LMCACHE_BLEND_SPECIAL_STR='<|fim_pad|>'
export LMCACHE_USE_LAYERWISE=true LMCACHE_BLEND_CHECK_LAYERS=1
export LMCACHE_BLEND_CONTROL_ENABLED=1                              # 开 control(双开关之一)
export LMCACHE_BLEND_CONTROL=/home/yilin/tmp/exp_writer/<sid>/blend_control.json
export LMCACHE_BLEND_DUMP_KV=/home/yilin/tmp/exp_writer/<sid>/dump  # 分支1 抓 KV*/ΔK/ΔV
vllm serve --config /home/yilin/LMCache/server/vllm/Qwen3-32B/config.yaml ...
```

## 跑什么 / 规模
- **N**:定性先用现有 **40 题**(规模 = 18 分支 × 40 = 720 次 writer 生成 + 720 次 kimi 打分;resumable、成本可控)。80 题留给通过后的定量。
- 每分支落 `/home/yilin/tmp/exp_writer/<sid>/<branch>/report.md`;日志放 `/home/yilin/tmp/logs/`。

## 流程(每题)
主轨(control-truth,pin summary)→ 分支1(dump KV* + 抓 M)→ 离线 `exp_writer_select` 由 M/ΔK/ΔV 生成 16 个 control → 分支3(writer-only full reuse)+ 4-19 free-gen 报告 → `exp_writer_score` kimi 逐题打分(存中间)。

## 怎么读结果(定性判据)
**主判据(token 维为主):每档四方对比 M×ΔV vs ΔV vs ΔK vs random(+ full reuse)**
- vs full reuse(分支3):回升了吗 → 刷新**有用**。
- vs random(同档):高于随机吗 → 选得**准**。
- 任一信号要立住:须"比 full reuse 回升 **且** 比 random 高"。

**两端点逻辑(砍了 50% 中间档后怎么判)**
- **25% 档(主战场)**:`M×ΔV(12) > random(18)` 且 `> full reuse(3)` → **选择性刷新有用、M×Δ 选得准**(定性成立);若 `M×ΔV ≈ random` → 选不准、方案存疑。
- **75% 档(上界参照)**:刷得多、各信号大概率都贴近 baseline 且彼此贴近(区分度低);只确认"刷够多质量能救回",不指望看出信号差异。

**三个对照**
- M×ΔV **>** ΔV → 注意力加权值得(保留 M);**≈** → 纯 ΔV 够(prefill 可得,解鸡生蛋)。
- ΔV **>** ΔK → value 漂移更关键,印证 M×Δ 用 ΔV;**ΔK>ΔV** 意外要回审;**≈** → K/V 相关(检查2)。
- 三者均 **≈ random** → 选择无效,危害不可用单点 KV 漂移定位 → 方案需重想。

**噪声底门槛**:任何分支 vs baseline 的 F1 差须 **超四臂现成本底**(A_orig vs A_orig2 = 两条 full prefill,paired Set F1 ≈ 0.14;本版不再单跑第二条 full prefill,直接借此本底)才算数;带 bootstrap 95% CI。

**指标**:① 报告答案 F1(主,DeepResearch QA autorater = kimi-k2.5,配噪声底);② writer 输出 token 级偏移(teacher-force,更敏感);③ 错误定位。

## 必做检查(零成本 + 防自欺)
1. **oracle M 假设验证**:记分支3 的 M,离线比 全真值 M vs 全复用 M。差小→上界有部署意义;差大→oracle 不现实(结论写明)。
2. **ΔK/ΔV 相关性**:每 token ΔK 与 ΔV 相关性。强相关→ΔK/ΔV 分支选出近似相同 token(可互证);弱相关→两者区分度真实(最有信息)。
3. **layer 求和弥散度污染**:查各层信号求和是否与注意力弥散度强相关;**若 layer 维各信号均 ≈ random layer → layer 选择无效**(类比 Exp A head 无效)→ 可砍 layer 分支。

## 关键设置(writer 隔离命根子)
- **同一份 summary 钉死**:分支1 的 summary(内容+token)固定存盘,3-19 全用同一份喂 writer → 唯一变量 = KV 怎么处理。
- **teacher forcing 锁生成序列**(指标②):相同 summary 上下文、相同已生成序列,只比 KV 来源对下一 token 的影响。(指标① F1 仍各分支 free-gen 真报告。)
- **researcher summarize 走 prefill 但 store generated KV**:供复用/选择性重算取用。

## 产出 + 下一步
每分支 control + 报告 + `ratings_<branch>.jsonl` / `metrics_<branch>.json`;**"M×ΔV 选择性刷新是否定性有用、选得准"的直接答案**,挂回 [../ExpB_fullseg/mainline_40q_report.md](../ExpB_fullseg/mainline_40q_report.md) 新增小节。
**定性通过**(25% 档 M×ΔV 同时打过 random + full reuse)→ 补回 50% 档(画预算-质量曲线)+ 上 80 题 + bootstrap CI 定量坐实;按检查3 砍无效维度(多半 layer)、按对照砍冗余信号(若 ΔK≈ΔV 或 M×ΔV≈ΔV)。**打不过** → 回头重想信号。

## 验证清单
1. per-layer/token `reuse%`(`BLEND_PATH=CONTROL`)与选择比例吻合(25% 档 recompute≈25%)。
2. pinned summary:18 分支 writer prompt **byte 一致**。
3. oracle M 检查(分支1 全真 M vs 全复用 M)。
4. 噪声底:各档效应超四臂本底(A_orig−A_orig2,paired Set F1 ≈ 0.14)。
5. kimi:每分支 `ratings_*` 满 N 题、f1 非全 None;resume 不重打。

*相关:[part1_code_and_test.md](part1_code_and_test.md)(代码+测试)、[../ExpB_fullseg/probe_trajectory_design.md](../ExpB_fullseg/probe_trajectory_design.md)、[../ExpB_fullseg/callchain_cdriver_and_control.md](../ExpB_fullseg/callchain_cdriver_and_control.md);memory `blend-three-modes-env`、`kvcomm-attn-weighted-reuse-error`、`kv-reuse-per-request-diagnosis`。*
