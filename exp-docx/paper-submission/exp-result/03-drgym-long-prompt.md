# 主结果 B:DRGym(英文长 prompt,672 条)—— 反转

← 返回 [experiment-data-summary.md](../experiment-data-summary.md) · 设置见 [01-setup.md](01-setup.md)

逐题加速比 vs vanilla(中位),分层(干净 held-out / 训练集内);干净 126 条为公平数,训练集内 546 条含偏高(见 [06-caveats.md](06-caveats.md))。

| 臂 | 干净(126) | 训练集内(546) | 全体 | 接受率(混合) |
|---|--:|--:|--:|--:|
| **AngelSlim(未域训)** | **1.071×** | 1.058× | 1.060× | 22.8% |
| 一期 8k-2ep | 1.230× | 1.226× | 1.228× | 34.2% |
| **二期 12k-3ep** | **1.303×** | 1.339× | 1.329× | 41.2% |
| **suffix** | **0.976×(净减速)** | 0.981× | 0.980× | 24.5% |

**四级台阶(关键)**:未域训 AngelSlim **1.071× 就已翻正、小胜 suffix**(0.976×)——长 prompt 利好 EAGLE 有一部分是**架构本身**(学习式 draft 不吃输入长度的亏)。但架构红利薄(+0.07×),**域训才是决定性因素**:1.071×→一期 1.230×→二期 1.303×,域训贡献 **+0.23×**,远大于架构红利。

## 结论 B(核心发现)

长英文检索 prompt 上,结论完全反转:

1. **EAGLE 碾压 suffix**:二期 1.303× vs suffix **0.976×(净减速!)**,干净样本差 +0.33×;
2. **suffix 为何净减速**:suffix 靠输入字面 n-gram 重复起草。超长检索 prompt(40K 字符)+ 压缩改写的 summary,输出与输入字面重复少,suffix 起草 42 万 token 只中 10 万(24.5%),却每次扫 40K 字符找匹配,**草稿税压过收益**;EAGLE 学习式 draft 不依赖输入重复、不受 prompt 长度影响。

## suffix 配置核实(排除测量陷阱)

用户质疑 suffix 是否 Ceager 配置,已逐项核实:

- **启动命令**:`--enforce-eager` 在命令行(config_suffix.yaml + `method: suffix`);
- **server 日志实证**:`Enforce eager set, disabling torch.compile and CUDAGraphs` + `Cudagraph is disabled under eager mode`——cudagraph 确实禁用;
- **suffix 真在起草**:SUFFIX_TRAJ 19MB 逐请求 trace,全局 draft 419,938 / accepted 102,905 = 24.5%;
- **两数据集配置逐字一致**:DRBench 与 DRGym 的 suffix 均 `enforce_eager=True`+`method='suffix'`+同一 config。

**→ 反转纯 workload 效应,非配置差异。** 同样 Ceager suffix,短中文 1.16× 赢、长英文 0.976× 净减速。

增益拆解(workload 效应 vs 域训效应)见 [04-gain-decomposition.md](04-gain-decomposition.md)。
