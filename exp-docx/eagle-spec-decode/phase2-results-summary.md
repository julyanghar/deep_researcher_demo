# Phase2 二期域训 EAGLE-3 实验结果汇总(自包含)

把 phase2(二期域训 EAGLE-3 draft head)的训练 + 评测 + 机制分析汇总到一处,一个文档看全。
散在 [paper-submission 主索引](../paper-submission/experiment-data-summary.md) + exp-result/ 6 子文件 +
[phase2-12k-3ep/README](../../phase2-12k-3ep/README.md) + [裁剪工作](phase2-trimming-work.md) +
[一期结果](phase1-eagle-vs-suffix-results.md) 的数据,这里整合。每个数字带来源。

## TL;DR

- **核心发现:投机解码的优劣强烈依赖 workload 形态。** 长检索 prompt(DRGym 英文 ~40K 字符)上
  **EAGLE 域训碾压 suffix**——EAGLE 二期干净 held-out **1.303×**,suffix **0.976×(净减速)**;
  短 prompt(DRBench 中文 ~11K 字符)上反过来 **suffix 赢**(1.16× vs EAGLE 1.11×)。
- **二期比一期强**:数据 ×2.13(3607→7681)、max_len 8192→12288、epoch 2→3,DRGym 接受率
  34.2%→41.2%、加速比 1.230×→1.303×。
- **使能技术**:位置维双裁剪省 7.05GiB/卡,解锁 TP4@12288(单机 4×48GB 可行)。
- **反直觉点**:suffix 在 summary 上接受率有 24-28%却净减速——因为它的起草成本随 prompt 长度暴涨
  (每步扫全 prompt 的 suffix tree),长负载下草稿税压过接受收益(见 [§5](#5-为什么-suffix-接受率-24-28-还净减速机制))。

---

## 1. 训练配置

| 项 | 值 |
|---|---|
| Target | Qwen3-32B(TP4, bf16) |
| Draft | EAGLE-3 |
| Warm-start | Tencent AngelSlim/Qwen3-32B_eagle3(第三方开源 head,非本产出) |
| 框架 | SpecForge 本地 fork(online, sglang target)+ 位置维裁剪;四补丁 ropebuf/chunk-acc/nocompile/ckpt-norm |
| TP / max-length | TP4 @ 12288 |
| 数据量 | 7,681 条(指纹 3a1078d2),有效监督 3.98M token |
| 语言分布 | zh 3935 / en 3725 / mix 21 |
| epoch / lr | 3 / 2e-5(warmup 0.05, max-grad-norm 0.5, accum 16) |
| ttt-length / attn | 3 / flex_attention |
| 数据形态 | prompt:assistant ≈ 13-15:1;prompt 占每条 93.8%,assistant 中位仅 507 token |
| 部署/评测 | vLLM 0.18, num_speculative_tokens=3, --enforce-eager, 温度 0 |

**双裁剪**(是什么/省多少):A 级 `--trim-loss-positions`(teacher target_p + draft logits/loss 只在监督位置)
省 3.37GiB/卡;B-i 级 `--trim-prompt-rows`(TTT 步 2..k 只跑监督行,prompt 行仅作步 1 KV 上下文)再省 3.68GiB/卡;
合计 **7.05GiB/卡 @8192**,解锁 TP4@12288。来源:[01-setup.md:9-26](../paper-submission/exp-result/01-setup.md#L9)、
[05-memory-efficiency.md:11-13](../paper-submission/exp-result/05-memory-efficiency.md#L11)、[README:30-38](../../phase2-12k-3ep/README.md#L30)。

## 2. 训练指标

- 二期逐 epoch 训练 acc:**0.44 → 0.47 → 0.48**;最终 run 全程 **0 OOM**([README:50](../../phase2-12k-3ep/README.md#L50))。
- 裁剪调试早期曾 epoch2 贴顶 OOM(实测余量仅 1-2GB,[phase2-trimming-work.md:74](phase2-trimming-work.md#L74));最终成功 run 无 OOM。
- 有效监督覆盖:8192 时 72.7% → 12288 时 **99.4%**——12288 才把长 summary 尾巴的监督基本吃全([phase2-trimming-work.md:22](phase2-trimming-work.md#L22))。

## 3. 评测结果:两数据集反转(核心)

指标口径:接受率 = accepted/drafted(vLLM /metrics 全局差分);加速比 = vs vanilla(无投机)逐题墙钟中位;
本期评测**全部是 summary 段 replay**(heldout summary prompt)。

### DRBench(中文短 prompt,~11K 字符,46 条)

| 臂 | 接受率 | 加速比 vs vanilla |
|---|--:|--:|
| vanilla | — | 1.00× |
| AngelSlim(未域训) | 11.1% | 0.94×(净减速) |
| 一期 8k-2ep | 17.6% | 1.10× |
| 二期 12k-2ep | 18.3% | 1.11× |
| 二期 12k-3ep | 18.4% | 1.11× |
| **suffix** | **33.7%** | **1.16×**(赢) |

结论:域训救活 EAGLE(0.94→1.11×)但**仍输 suffix**;域训边际枯竭(一期→二期只 +0.8pp,epoch 2→3 推理接受率停)。
来源:[02-drbench-short-prompt.md](../paper-submission/exp-result/02-drbench-short-prompt.md)。

### DRGym(英文长 prompt,~40K 字符≈10K token,672 条)——反转

分层:干净 held-out 126 条(公平数)/ 训练集内 546 条(偏高)。

| 臂 | 干净(126) | 全体 | 接受率 |
|---|--:|--:|--:|
| AngelSlim(未域训) | 1.071× | 1.060× | 22.8% |
| 一期 8k-2ep | 1.230× | 1.228× | 34.2% |
| **二期 12k-3ep** | **1.303×** | 1.329× | **41.2%** |
| **suffix** | **0.976×(净减速)** | 0.980× | 24.5% |

结论:EAGLE 二期 **1.303× 碾压 suffix 0.976×**,干净样本差 **+0.33×**。suffix 配置已逐字核实为 --enforce-eager
(与 DRBench 同 config),反转是**纯 workload 效应**非配置差异。来源:[03-drgym-long-prompt.md](../paper-submission/exp-result/03-drgym-long-prompt.md)。

## 4. 逐位置接受率(a1/a2/a3)

DRBench 逐位置(第 1/2/3 草稿 token 接受率,[02:16](../paper-submission/exp-result/02-drbench-short-prompt.md#L16)):

| 臂 | a1 | a2 | a3 |
|---|--:|--:|--:|
| AngelSlim | 0.228 | 0.042 | 0.019 |
| 一期 8k | 0.281 | 0.081 | 0.044 |
| 二期 12k-3ep | 0.303 | 0.095 | 0.054 |

域训主要抬 a1(第一个草稿 token),深层 a2/a3 仍快速衰减——k=4 时 P(前3全中)≈0.16%,所以 num_spec=3 够用。

## 5. 为什么 suffix 接受率 24-28% 还净减速(机制)

phase2 最反直觉的点。**加速 = 接受收益 − 起草成本**,接受率不是唯一因素。

- **suffix 起草成本随 prompt 长度暴涨**:suffix 每个 decode step 要在整个 prompt 建的 suffix tree 里
  扫描找匹配。DRGym 的 ~40K 字符 prompt 让每步扫描成本 4× 于 DRBench(~11K)。这个"草稿税"是固定开销,与接受无关。
- **实测**([03:21](../paper-submission/exp-result/03-drgym-long-prompt.md#L21)):suffix 在 DRGym 起草
  **419,938 token 只中 102,905(24.5%)**——75% 白起草,每个白起草 token 付了「扫 40K tree + target verify 被拒位置」双重税。
- **两因素同时恶化**:prompt 长 4× → 起草税 4×;接受率还从 DRBench 33.7% 掉到 24.5%(检索内容越长改写越难字面命中)。
  加上 spec decode 必须 --enforce-eager(关 cudagraph)的固定 decode 税——接受率够高(33.7%)补得回、净赚,不够高(24.5%)补不回、净亏。
- **EAGLE 不吃这亏**:EAGLE 的 draft 是固定小 head 前向(1 层,GPU),成本**不随 prompt 长度变**。所以长 prompt 上
  EAGLE 草稿税几乎不涨,域训后接受率 41.2%,净加速 1.303×。**suffix 的软肋(起草 ∝ prompt 长度)恰是 EAGLE 的强项。**

一句话:**suffix 慢不是接受率低,而是它扫全 prompt 的起草方式在长检索负载下太贵,24.5% 接受率补不回这个税。**

## 6. 增益拆解(架构 / workload / 域训)

DRGym 干净 held-out 三级拆解([04-gain-decomposition.md:9-15](../paper-submission/exp-result/04-gain-decomposition.md#L9)):

| 台阶 | 加速比 | 增量 | 归因 |
|---|--:|--:|---|
| vanilla | 1.00× | — | 基准 |
| AngelSlim(未域训) | 1.071× | +0.07× | 架构红利(勉强超 suffix) |
| 一期 8k-2ep | 1.230× | +0.16× | 域训(窄) |
| 二期 12k-3ep | 1.303× | +0.07× | 域训(强:监督×2+12288+3ep) |

- **workload 效应**(同一未域训 head 跨数据集):AngelSlim 从 DRBench 0.94× → DRGym 1.071×,+0.13×;接受率 11.1%→22.8%(×2.05)。
- **域训效应**:DRGym 上 1.071×→1.303×,累计 +0.23×(决定性)。
- 三句话:架构必要条件 +0.07×、workload 放大器(接受率翻倍)+0.13×、域训决定性 +0.23×(把"勉强赢"变"碾压")。

## 7. 显存 / 训练效率(位置维裁剪,使能技术)

| 项 | 值 |
|---|---|
| A 级 `--trim-loss-positions` | 省 3.37GiB/卡 @8192 |
| B-i 级 `--trim-prompt-rows` | 再省 3.68GiB/卡 |
| 合计 | 7.05GiB/卡 → 解锁 TP4@12288 |
| 等价性 | 在位双算 72/72,最大误差 1.9e-3(bf16 噪声级) |
| 边际成本 | 每 prompt token 从 960KB 降到 ~25KB |
| 新颖性 | A 级 TorchSpec 已有;B 级(backbone/TTT 内部裁剪)全网 9 框架无先例 |
| 原理 | 94% 显存花在「算出来就被 loss_mask 乘零」的位置;裁剪使算量随监督 token 数而非序列长度伸缩 |

来源:[05-memory-efficiency.md](../paper-submission/exp-result/05-memory-efficiency.md)、[phase2-trimming-work.md](phase2-trimming-work.md)。
(12288 first-principles 算 ~33GB 但实测峰值 46-47GB,差 ~13GB 为框架实现开销,真实余量仅 1-2GB。)

## 8. 一期 vs 二期对比

| 项 | 一期 8k-2ep | 二期 12k-3ep | 提升 |
|---|--:|--:|--:|
| max_len | 8192 | 12288 | +50% |
| 数据条数 | 3,607 | 7,681(zh/en 均衡) | ×2.13 |
| 有效监督 | 1.91M token | 3.98M token | ×2.08 |
| epoch | 2 | 3 | +1 |
| DRBench 接受率 | 17.6% | 18.4%(+0.8pp,枯竭) | — |
| DRBench 加速比 | 1.10× | 1.11× | — |
| **DRGym 接受率** | 34.2% | **41.2%(+7pp)** | — |
| **DRGym 加速比(干净)** | 1.230× | **1.303×(+0.07×)** | — |

一期域训 vs 未域训(DRBench 46 题):接受率 11.13%→17.55%(×1.58)、平均接受长度 AL 1.334→1.527、
e2e 46 题 790s→703s(−11%)、输出无漂移。来源:[phase1-eagle-vs-suffix-results.md](phase1-eagle-vs-suffix-results.md)。

## 9. 诚实边界 / caveats

1. **两变量同变(最重要)**:DRBench(中文 11K)与 DRGym(英文 40K)语言和长度都不同,不能断言单一变量致反转;
   机制分析支持长度是主因但未做控制变量隔离语言效应。
2. **接受率 41.2% 为混合值**(含训练集内 546 条偏高);加速比已分层,干净 126 条 1.303× 为公平数;
   过拟合 gap 极小(二期 +0.036×、一期 +0.002×,两代泛化良好)。
3. **DRGym 干净样本仅来自 7 题**(126 条 summary),题级多样性有限,泛化待更多干净题验证。
4. **加速比可能被超长 prefill 稀释**:40K prefill 占墙钟大头,spec decode 只加速 decode 段,报告的绝对加速比是
   保守下界;但对 EAGLE-vs-suffix 相对结论无影响(同题两 server 同 prefill)。
5. **本期全为 summary 段**;report 段独立接受率/加速比 phase2 未单独测(一期口径提及 40q 全流程 suffix 1.89-2.41×,与 summary 段口径别混)。

## 对论文的意义

**主命题**:投机解码选型必须匹配 workload——agent/RAG 长检索 prompt 场景,学习式 draft(EAGLE 域训)显著优于
输入匹配式(suffix),后者甚至净减速。这为"何时用哪种投机解码"提供 characterization 证据,而非笼统的"某方法更快"。
配套位置维裁剪是使能技术。三期(开源 PR + 显存进一步优化)见 [phase3 计划](phase3-opensource-and-memory-plan.md)。
