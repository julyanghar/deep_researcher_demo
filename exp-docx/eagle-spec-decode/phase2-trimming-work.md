# 二期工作:两级 loss-mask 感知裁剪(A + B-i)——已完成

状态:**已实现并七闸全过**(2026-07-13,`/modify-code` 收敛循环)。目标:把"为整条序列物化、却只有 assistant 位置有用"的张量裁到只算监督位置,省 GPU 显存,解锁 TP4@12288。物证 + 审查报告在 `~/modify-code-runs/eagle3-trim/`(review-report.md 已用户确认)。

> 三期(开源 PR + 显存进一步优化 + 论文)已独立成文:[phase3-opensource-and-memory-plan.md](phase3-opensource-and-memory-plan.md)。本文只记二期。

## 〇、成果速览

| 项 | 结果 |
|---|---|
| 实现 | `--trim-loss-positions`(A)+ `--trim-prompt-rows`(B-i,步 2..k 只跑监督行) |
| **实测省显存** | **7.05 GiB/卡 @8192**(A 省 3.37 + B-i 再省 3.68) |
| **解锁** | **TP4@12288 用 drgym 真样本(12.2K token)零 OOM** |
| 等价性 | 在位双算协议,72/72 最差 1.9e-3(bf16 噪声级) |
| 边界 | 仅 batch=1 / lk_loss=None / 非 VL / online sglang;其余自动回退全长 |
| 未做 | B-ii(step1 prompt 行短路到 k/v,再省 ~2.6GB)→ 移交三期 |

## 一、为什么值得做(动机,大白话)

我们的 summary 训练数据形态是 **prompt:assistant ≈ 13-15:1**(数据第六版 7,681 条实测:prompt 占每条 93.8%,assistant 中位仅 **507 token**,全集有效监督 3.98M token)。loss 只在 assistant 位置计入(prompt 位置的 loss 项被 loss_mask 乘 0),但原实现把"词表宽"张量按**全长 L** 物化——等于 94% 的显存花在"算出来就为了被乘零"的位置上。

省出的显存直接兑换成 max_length:**TP4 从 8192 → 12288**(有效监督覆盖 72.7% → 99.4%)。一期训练(TP4@8192)丢掉的 27% 有效监督,靠它赎回;DRGym 232 题英文增广(~9-11K token)也只有裁剪后才能在 TP4 入训。

本轮 OOM 史反证了这条路的正确性:14336/10240 的失败全是"∝L 的基座顶死容量"(见 [eagle3-domain-training-plan.md](eagle3-domain-training-plan.md) §四c #9-#12),贴线微调(0.5GB 级)五次全输——只有结构性削减能翻盘。

## 二、原理:哪些能裁、哪些不能(判据一句话)

**判据:凡是"从 loss 走回可训参数"的反向路径需要的张量,不能裁;只为 loss 数值服务、且 prompt 位置贡献恒为零的,能裁。**(推导全文见主文档 §八b 反向传播梯子)

- **能裁**:logits 及其下游(softmax/CE/target_p/acceptance 指标)——prompt 位置的这些值算出来就被乘零,且 logits 不进 TTT 下一步(下一步吃的是 backbone hidden);
- **不能裁的只有一小段**:prompt 行的"X→k_p/v_p"路径(**~25KB/位**)+ teacher 的 3 层 aux hidden。依据=TTT mask 结构(`generate_eagle3_mask`,已核源码):跨位置注意力只读**第 1 步**的 K/V,第 2/3 步的 K/V 是**对角线私有**——prompt 行的 q/attn/MLP、以及步 2+ 整行,下游全部终结在被 mask 的 loss,梯度死路,**既不用存也不用算**。

## 三、两级裁剪(实测数据)

| 级别 | 改什么 | 实测省 @8192 |
|---|---|--:|
| **A** `--trim-loss-positions` | teacher target_p + draft logits/loss 只在监督位置(`index_select(1, sup)`);重标定 mean 分母 | **3.37 GiB** |
| **B-i** `--trim-prompt-rows` | 步 2..k 只跑监督行(compact mask + RoPE 绝对位),prompt 行仅作步 1 KV 上下文 | **再省 3.68 GiB** |
| B-ii(未做) | 步 1 prompt 行也短路到 k/v(不跑 attn 输出/MLP) | 理论 ~2.6GB → 三期 |

⚠️ **实测 7.05 vs 早期文档"~10GiB"的差额说明**:早期 §动机引用的"A+B 累计 ~10GiB / draft 激活 7.5→0.64"是**完整 B 级(B-i + B-ii)**的理论值。二期只做 B-i,B-ii(砍 step1 全长 backbone 的 ∝L 项 ~2.6GB)留三期,故实测 A+B-i = **7.05GB = 10 − 2.6(B-ii 缺口) − ~0.4(A 级 n_sup 分布/框架开销折损)**。B2 验收阈值当初就按此预设成 ≥7(非 10),实测卡线过,非未达标。三期补 B-ii 兑现剩余 ~2.6GB,届时 16384 可开。

## 四、改动点(源码钉子,已落地)

均在 `~/modify-code-runs/eagle3-trim/`(current.diff,4 文件 ~570 净行,两 flag 默认关、off 路径零扰动):

1. **teacher 侧** [eagle3.py](../../../SpecForge/specforge/core/eagle3.py):`_build_trim_pack`(监督位∪k 步滑窗上算 fp32,pad 行逐值对齐 F.pad)、`_compute_target_p_eager`(不套 compile 防变长重编译;后按 256 行分块压瞬时);
2. **draft 侧**:TTT 循环内 `compute_logits(hidden[sup])`;`_acc_and_loss` 重标定 `loss_scale=n_sup/L`、`full_positions=L`;
3. **B-i 骨架**:[flex_attention.py](../../../SpecForge/specforge/modeling/draft/flex_attention.py) 新增 `generate_eagle3_mask_compact`(q 行 r↔绝对位 sup[r]:步 1 块绝对 causal、后续块对角线私有)、[llama3_eagle.py](../../../SpecForge/specforge/modeling/draft/llama3_eagle.py) `trim_ctx` 全链穿参 + RoPE 直传绝对位置;
4. **flag** [train_eagle3.py](../../../SpecForge/scripts/train_eagle3.py):`--trim-loss-positions` / `--trim-prompt-rows`;
5. **不动**:backbone 数学、FSDP、原 flex mask——裁剪只在"hidden→logits→loss"支路和"步 2..k 行子集"入口。

## 五、验收结论(七闸全过)

| 闸 | 判据 | 结果 |
|---|---|---|
| A1/B1 等价性 | 同 run 在位双算,fp64 相对误差 <5e-3 | ✅ 72/72 最差 1.9e-3 |
| A2 显存 | 最长样本冒烟省 ≥3GiB | ✅ 3.37 GiB |
| A3 训练不漂 | smoke24 1 epoch 优化器全开在位双算 | ✅ 最差 2.2e-3 |
| A4 产物兼容 | Gate D + vLLM 加载 + spec decode | ✅ t2d/d2t==AngelSlim,drafted 48/accepted 9 |
| B2 显存+解锁 | 合计省 ≥7GiB + 12288 零 OOM | ✅ 7.05 GiB,drgym 真样本 exit 0 |
| B3 12288 训练信号 | loss 有限、acc>0 | ✅ loss 0.33/acc 0.07 |

**关键教训**:等价性协议曾误判 FAIL——跨 run 对拍无法区分"裁剪 bug"和"sglang run 间浮点噪声"(后者达 58%),AMEND 为**同 run 在位双算**(阈 5e-3)才对。这是"验收协议本身也要能被证伪"的一课。

## 六、与前后期的衔接

- **上承一期**(已完结):TP4@8192 域训 P0 达标(接受率 11.1%→17.6%,净减速翻正),裁剪解锁的 12288 用来赎回丢掉的 27% 监督 + drgym 增广;
- **本期产出**:数据第六版 7,681 条(zh/en 均衡,监督 3.98M ×一期 2.08)已就绪;12288+双裁剪 3-epoch 重训进行中(epoch1 acc 0.42-0.58,远超一期同期);
- **下接三期**:见 [phase3-opensource-and-memory-plan.md](phase3-opensource-and-memory-plan.md)——B-ii 补完 + 显存 13GB 拆解优化 + 开源 PR + 论文。

## 七、实测校准原则(2026-07-13 追记)

12288 k=3 的 first-principles 显存模型算出 ~33GB,**实测峰值 46-47GB**,差 ~13GB 全在"激活的框架实现开销"(flex kernel 工作区 + 分配器碎片 + cache/RoPE/autograd 冗余)。**显存必须以实测为准,first-principles 只配当下界参考**——之前几条用模型值说"余量充足"是错的,真实余量仅 1-2GB,这也是 epoch2 OOM(贴顶 + 250MB 瞬时尖峰 + 碎片)的根因。那 13GB 的拆解与进一步优化,是三期的正式课题。
