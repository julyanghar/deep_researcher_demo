# 实验数据汇总(主索引):EAGLE-3 域训 vs suffix decode

论文投稿用实验数据的主索引。各部分实验独立成子文件(见 [目录](#目录)),本文只留概览、核心发现与导航。

> 状态:672 条 DRGym 主结果已定;一期 8k 臂 672 条评测收尾中(部分数据 1.231× 已稳定,完整后更新 [03](exp-result/03-drgym-long-prompt.md))。数据物证 `~/modify-code-runs/eagle3-trim/evalext_*`、`eval5_*`。

## 一句话核心发现

**投机解码方法的优劣强烈依赖 workload 形态**——短 prompt 场景 suffix 胜,长检索 prompt 场景 EAGLE-3 域训碾压 suffix(后者甚至净减速)。

## 两个数据集的反转(全文骨架)

| workload | prompt 长度 | 二期 EAGLE | suffix | 谁赢 |
|---|--:|--:|--:|---|
| DRBench(中文) | ~11K 字符 | 1.11× | **1.16-1.22×** | **suffix** |
| DRGym(英文) | ~40K 字符 | **1.303×**(干净) | 0.976×(净减速) | **EAGLE** |

同样的 Ceager suffix 配置(逐字核实),两数据集结果反转——纯 workload 效应。增益拆解:workload 效应 +0.13×(主导)+ 域训效应 +0.07×(次要)。

## 目录

| # | 子文件 | 内容 |
|---|---|---|
| 1 | [01-setup.md](exp-result/01-setup.md) | 实验设置:模型/框架/指标定义/两代训练配置/两个 workload |
| 2 | [02-drbench-short-prompt.md](exp-result/02-drbench-short-prompt.md) | 主结果 A:DRBench 中文短 prompt(EAGLE 输 suffix) |
| 3 | [03-drgym-long-prompt.md](exp-result/03-drgym-long-prompt.md) | 主结果 B:DRGym 英文长 prompt(反转,EAGLE 碾压 suffix)+ suffix 配置核实 |
| 4 | [04-gain-decomposition.md](exp-result/04-gain-decomposition.md) | 增益拆解(workload vs 域训)+ 过拟合诊断 |
| 5 | [05-memory-efficiency.md](exp-result/05-memory-efficiency.md) | 显存/训练效率(位置维裁剪,使能技术) |
| 6 | [06-caveats.md](exp-result/06-caveats.md) | 诚实边界(两变量同变/接受率混合/样本多样性/prefill 稀释) |

## 对论文的意义

**主命题**:投机解码选型必须匹配 workload——**agent/RAG 长检索 prompt 场景,学习式 draft(EAGLE 域训)显著优于输入匹配式(suffix),后者甚至净减速**。这为"何时用哪种投机解码"提供 characterization 证据,而非笼统的"某方法更快"。配套的位置维裁剪([05](exp-result/05-memory-efficiency.md))使长 prompt 域训在单机 4×48GB 可行,是使能技术。

相关规划:工程/开源见 exp-docx/eagle-spec-decode/[phase2](../eagle-spec-decode/phase2-trimming-work.md)、[phase3](../eagle-spec-decode/phase3-opensource-and-memory-plan.md)。
