# SpecForge 训练显存优化 —— 贡献定位(诚实重估版)

> 本目录 = 针对 SpecForge EAGLE3 训练的显存优化工作的独立文档区。
> 本文回答一个问题:**相对已有 PR / 方法,我们的贡献到底在哪、能立住的是什么、别吹的是什么。**
> 定位在 2026-07-16 的显存实测拆解 + B-ii 实测之后**重写**——旧的 novelty 判决(phase3 §4.1,2026-07-13)是拆解之前写的,有几条现在站不住,见 §四。

- 上游代码/物证:fork [julyanghar/specforge-yilin](https://github.com/julyanghar/specforge-yilin)(main,含 A+B-i+B-ii+MCMEM,commit 78a6432)
- 二期(A+B-i)记录:[phase2-trimming-work.md](../eagle-spec-decode/phase2-trimming-work.md)
- 三期计划:[phase3-opensource-and-memory-plan.md](../eagle-spec-decode/phase3-opensource-and-memory-plan.md)
- 显存拆解物证:`/home/yilin/modify-code-runs/eagle3-mem-breakdown/review-report.md`
- B-ii 物证:`/home/yilin/modify-code-runs/eagle3-trim-bii/review-report.md`

---

## 一、一句话:我们攻的是一条**别人没攻的轴**

EAGLE3 训练显存有几条可攻的轴,现有 PR / 方法各占一条,**唯独没人动"draft backbone 内部的位置维"**:

| 方法 | 轴 | 机制 | 降 OOM 峰值? | 状态 |
|---|---|---|---|---|
| SpecForge #669 | **重算** | TTT 反向重算激活 | ✅ 大 | Open 未合并 |
| SpecForge #673 | **分块** | 位置分块削峰,**总算量不变** | ✅(削峰非省算) | Open 未合并 |
| SpecForge #671 | **搬运** | 优化器/embed CPU offload | 省 base | Open 未合并 |
| Liger FLCE / Apple CCE | **词表** | fused CE 融 vocab 维 | ✅(logits) | 已有 |
| TorchSpec(EAGLE-3.1 官方) | 位置(**仅损失末端**) | 裁 loss,backbone 仍全长 | 部分 | 已有 |
| **我们 A** | 位置(损失末端) | 同 TorchSpec | ✅ ~3.4G | 我们实现 |
| **我们 B-i / B-ii** | **位置(backbone/TTT 内部)** | **消除**被 mask 行的 backbone 算量 | B-i ✅ / B-ii ❌ | **全网无先例** |

**唯一属于我们的机制** = 把 loss-mask 感知的裁剪**推进到 draft backbone 与 TTT 展开内部**,让训练**算量 + 前向激活随监督 token 数(而非序列长)伸缩**。它是"**消除**"——不同于重算(#669)、分块(#673)、搬运(#671)、词表融合(Liger),和它们全部**正交、可叠加**。

---

## 二、实证基础(这个会话新产出,是重估的依据)

### 2.1 首个 EAGLE3 训练显存的 allocator 级拆解(独立 systems 贡献)

用 env 门控探针把 12288/k=3 的 46-47GB 逐桶拆开(物证 eagle3-mem-breakdown),**推翻了两个流行假设**:
- **没有"13GB 框架开销"**:非-torch(CUDA ctx+NCCL)恒 **0.7GB**,不存在神秘的 flex 工作区大头;
- **46-47GB = 驻留 base ~20-24GB(主体 sglang teacher ~17-19GB 同进程,不可攻)+ backward 激活/梯度瞬时尖峰 ~21.5GB(∝L)+ reserved 缓存**;峰值由**反向那一下**顶起,不是常驻;
- **draft 优化器 base 是 FSDP 分片、每卡小**(非满量),8-bit Adam / 修分片不是大杠杆。

社区没人做过这个拆解,它本身可作论文的 characterization 章。

### 2.2 B-ii 实测:位置轴单独降不了反向峰值(正交性的硬证据)

B-ii(step1 prompt 行短路)等价性铁证通过(SC4 最差 2.6e-3、off-path 逐位一致),但显存上:
- **前向驻留激活省 2.69G@12288**(∝L,正好对上计划估的 2.6G);
- **backward 峰值 max_alloc 完全不变**(off/on 相同)——**位置/前向轴单独降不了 OOM 临界峰值**。

机制:峰值在反向,B-ii 省在前向侧、被反向 transient(梯度 + 全长 K/V 注意力反向工作区,这段 B-ii 照留)填回。**这条恰好证明位置轴(我们)与重算轴(#669)正交**:要降反向峰值得靠 #669,要省前向算量/激活靠我们,两者叠加。

---

## 三、我们真正能立住的三块贡献(排序)

1. **🎯 backbone 内部的位置维消除(B 级)= 唯一原创机制**。
   - 最强 pitch **不是"省显存"**(会和 vocab/重算方法打架),而是:
     - **算量/吞吐**:prompt-heavy 数据(agent 轨迹/RAG,我们 15:1,94% 位置被 mask)下,backbone 少算 94% 无用行 → 训练更快(#669 反而更慢);
     - **动机独特**:唯一从"agent/RAG 训练数据极端 prompt 占比"出发的显存工作。
2. **🎯 首个 EAGLE3 训练显存的 allocator 级 characterization**(§2.1)——独立于代码的 systems 贡献。
3. **🎯 位置轴 × 重算轴的实测正交性**(§2.2)——把我们定位成 **#669 的互补而非对手**,反而更好合并。

---

## 四、诚实重新校准(旧叙事里站不住、必须改的三条)

不改这几条,PR/论文会被 maintainer/reviewer 一眼看穿:

1. **A 级不是我们的原创**——TorchSpec 已做损失末端裁剪,Liger/CCE 覆盖 vocab 维。A 级 PR 要老实标"移植 TorchSpec 到 SpecForge",**A 级的 ~3.4G 不算独家**。
2. **B-ii 不降峰值**——"省显存解锁大配置"这个 headline **只属于 A+B-i 的峰值降(实测 5.5G@8192)**,且其中一半是 A 级(logits/loss)贡献、与 vocab 方法重叠。**B-ii 的价值 = 前向驻留省 2.7G + 算量省 + 把 B 级做完整 + 论文新颖点**,不是峰值。
3. **真正降峰值的最大杠杆是 #669(重算),不是我们**——所以对 maintainer 的话术是"和 #669 正交可叠加",不是"我们更省显存"。

---

## 五、PR 策略(按诚实度 / 接受度排)

1. **ropebuf bugfix 先发**:真 bug(meta-load 不回填 non-persistent RoPE buffer → warm-start loss=nan)、零新颖性争议、攒信誉;
2. **A 级**:标"port TorchSpec 的损失末端裁剪到 SpecForge",不吹原创;补一个 ShareGPT 常规形态的 benchmark 证普适;
3. **B 级(先 RFC)**:主打**算量/吞吐 + agent 数据动机 + 与 #669 正交可叠加**的 characterization 数据,**别主打"省峰值显存"**;
4. 给 maintainer 的定位话术:"#669 攻重算轴、#673 攻分块;我们攻**位置轴消除**,正交、这是实测叠加数据"。

> PR 就绪度的逐条评估(抽取干净度 / 上游 rebase / 拦路石)见 [pr-readiness.md](pr-readiness.md)。
