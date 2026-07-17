# 显存 / 训练效率(位置维裁剪)

← 返回 [experiment-data-summary.md](../experiment-data-summary.md)

二期引入两级 loss-mask 感知位置裁剪,是长 prompt 域训在单机 4×48GB 可行的使能技术。

## 裁剪收益(实测)

| 级别 | 改什么 | 实测省 @8192 |
|---|---|--:|
| **A** `--trim-loss-positions` | teacher target_p + draft logits/loss 只在监督位置计算 | 3.37 GiB/卡 |
| **B-i** `--trim-prompt-rows` | TTT 步 2..k 只跑监督行,prompt 行仅作步 1 KV 上下文 | 再省 3.68 GiB/卡 |
| **合计(A+B-i)** | | **7.05 GiB/卡** |
| B-ii(未做,留三期) | 步 1 prompt 行也短路到 k/v | 理论 ~2.6 GiB |

## 关键指标

| 指标 | 值 |
|---|--:|
| 解锁 | TP4@12288(原 first-principles 模型 + 实测均 OOM 证伪) |
| 等价性 | 在位双算 72/72,最大误差 1.9e-3(bf16 噪声级) |
| 边际成本 | 每 prompt token 从 960KB(320KB×k)降到 ~25KB |
| 新颖性 | A 级 TorchSpec 已有;**B 级(backbone/TTT 内部裁剪)全网 9 框架无先例** |

## 原理一句话

数据形态 prompt:assistant ≈ 15:1,94% 的显存花在"算出来就被 loss_mask 乘零"的位置。裁剪把 loss 侧词表宽张量 + TTT 后续步激活裁到只算监督位置,使训练算量与激活随**监督 token 数**(而非序列长度)伸缩。

## 显存实测校准(诚实记录)

12288 k=3 的 first-principles 模型算 ~33GB,**实测峰值 46-47GB**,差 ~13GB 为"激活的框架实现开销"(flex kernel 工作区 + 分配器碎片 + cache/RoPE/autograd 冗余)。**显存以实测为准,first-principles 只配当下界**。此为三期显存进一步优化的课题。

工程与开源规划见 exp-docx/eagle-spec-decode/[phase2-trimming-work.md](../../eagle-spec-decode/phase2-trimming-work.md) 与 [phase3-opensource-and-memory-plan.md](../../eagle-spec-decode/phase3-opensource-and-memory-plan.md)。
