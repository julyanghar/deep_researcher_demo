# 三期工作:开源贡献 + 显存进一步优化 + 论文

状态:**已立项设计**(2026-07-13),等二期 12288 重训 + 评测收官后启动。三条线共用同一份代码,互为因果:显存优化产出更强的裁剪代码 → 成为开源 PR → 支撑论文的 systems 证据。

- 上承二期(A + B-i 已落地,七闸全过):[phase2-trimming-work.md](phase2-trimming-work.md)
- 主线文档:[eagle3-domain-training-plan.md](eagle3-domain-training-plan.md)

---

## 一、总览:三条线一份代码

```
        ┌─────────────────────────────────────────┐
        │  同一份裁剪代码(A + B-i + B-ii + 配套)   │
        └──────┬──────────────┬─────────────┬──────┘
               ↓              ↓             ↓
         ①显存优化       ②开源 PR       ③论文
         拆 13GB +      SpecForge/     位置维裁剪
         补 B-ii +      TorchSpec      characterization
         正交叠加       双投           + 正交性证据
```

核心资产:B 级(prompt 行退化为纯 KV 上下文 + 步 2..k 只跑监督行)在全网 9 个 EAGLE 训练框架中**无先例**(新颖性判决见 §四),既是论文原创点,也是 PR 卖点。

---

## 二、显存进一步优化(把那 13GB 拆开)

**现状**:二期落地后实测 12288 k=3 峰值 **46-47GB**(贴 48.5 顶),first-principles 模型只算到 33GB,差 **~13GB 是"激活的框架实现开销"**(flex kernel 工作区 + 分配器碎片 + cache/RoPE/autograd 冗余),**未实测分解**。三期第一步就是拆开它、按可优化性分层攻。

### 2.1 杠杆全表(按分工)

| 手段 | 省 | 代价 | 定位 |
|---|--:|---|---|
| **B-ii**(step1 prompt 行短路到 k/v) | -2.6GB | 中(attention 行子集前向) | 🎯 论文主线+PR:把 B 级做完整 |
| **memory_summary 实测分解** | 0(测量) | 无 | 🎯 论文 characterization:13GB 拆成"可优化 vs 框架固有" |
| sglang memfrac 0.44→0.42 | -1GB | 验 KV 池够 | 自用调参 |
| draft 冻结 embed CPU offload | -1.5GB | 前向多一次 H2D | PR 配套(呼应 #671) |
| TTT unroll 梯度检查点 | -5~8GB | **慢 ~30%** | 🎯 正交叠加对比(对标 #669) |
| 8-bit AdamW(m/v fp32→int8) | -1.2GB | 收敛风险 | 备选,不主推 |
| flex 工作区 + 碎片 | 13GB 主体 | 框架固有,难动 | 已靠 GC 阈值缓解 |

### 2.2 显存实验设计(论文的 systems 证据)

1. **拆解**:跑 `torch.cuda.memory_summary()`,把 46GB 拆成 {固定基座 / ∝L / ∝n_sup×k / 框架固有},坐实 13GB 构成;
2. **消融四臂**:baseline(原版全长)/ A / A+B-i(二期)/ A+B-i+B-ii,各测峰值 + 训练吞吐,画"显存随监督占比伸缩"曲线——这是论文核心图;
3. **正交叠加**:A+B 全套 vs 梯度检查点(#669 移植)vs 两者叠加——证 B 级砍"算量"、检查点砍"驻留",两条轴独立;
4. **解锁验证**:全套优化下能否 **16384 单机**(数据 p90=14328,覆盖 98.8%)+ k=4/k=7 显存可行性(账:k=7 需 B-ii 才可行)。

### 2.3 ⚠️ 上游 PR 成熟度(2026-07-13 核实,必读)

引用的两个"省显存 PR" **均 Open 未合并**,不是"上游现成":

| PR | 内容 | 状态 |
|---|---|---|
| #669 | `--draft-gradient-checkpointing`(TTT unroll 梯度检查点,声称省 ~85% 激活) | **Open 未合并** |
| #671 | single-GPU memory options(优化器/embed CPU offload + 分块 acceptance) | **Open 未合并** |

含义:要用得从 PR 分支移植 + 自测。好消息:①#671 的分块 acceptance 我们已独立实现(chunk-acc 补丁);②#669 未合并 = TTT 省显存仍是开放地带,B 级(位置维)与它(重算维)正交,叠加是论文的关键对比。**教训:引用上游 PR 必标合并状态,不能凭 PR 号说"现成"。**

---

## 三、开源 PR(SpecForge + TorchSpec 双投)

自用与上游是同一份代码,按接受度排序:

1. **ropebuf bugfix PR(即刻可提,攒信誉)**:meta-load 不回填 non-persistent buffer → warm-start loss=nan,真 bug、影响所有 `--ckpt-dir` 用户(上游连 vocab 覆盖的 PR#534 都未合,此路径少有人踩通);
2. **A 级裁剪 PR**:flag 门控默认关 + loss 逐元素等价证明 + 显存 benchmark(补一个 ShareGPT 常规形态的数字,证普适非特例);实现前先读 TorchSpec 的 `maybe_mark_dynamic`(治 dynamic shape × torch.compile,HF PR#40065 因此搁置的前车之鉴);
3. **B 级(含 B-ii)先 RFC/issue**:讲清 mask 对角线结构→prompt 行梯度死路的数学 + "16K 在 4×48GB 不再需要 USP/offline"的对照表(直接消解上游 issue#380/USP 求助),maintainer 表态再动 TTT 循环;
4. **B 级同样贡献给 TorchSpec**:TorchSpec 只裁到损失末端,TTT 每步 backbone 仍全长(已核 eagle3.py L211-257)——B 级对它同样适用,且**接受概率高于 SpecForge**(maintainer 已自证认可位置裁剪方向,只需延伸;生产采用方 CoreWeave/fal/DigitalOcean 放大影响)。顺序:我们 fork(SpecForge)先落地拿验证数字 → 同一数学、两份适配 PR 分投两家;
5. **配套 memory options**:memfrac 微调 + embed CPU offload 作为 PR 附带项(呼应未合并的 #671);
6. 前置核查:上游 main 现状(USP/adapter 重构中,需 rebase)、实验室开源署名政策、benchmark 场景选型、TorchSpec 的 TTT mask 是否同样"对角线私有"(**EAGLE-3.1 是新架构需单独验证**)。

---

## 四、论文

### 4.1 新颖性判决(2026-07-13,9 框架 + issue/PR 区 + 论文先例全查)

- **A 级:新颖性不成立**——TorchSpec(EAGLE-3.1 官方合作训练库)已双侧实现;但 SpecForge 及其余 8 家(EAGLE 官方/speculators/Model-Optimizer/NeMo/AngelSlim/BaldEagle/HF 邻域)均无 → A 级仍可作 SpecForge 首个移植 PR;
- **B 级:新颖性成立**——全部 9 家的 backbone 在每个 TTT 步都对全长位置前向,"prompt 行退化为纯 KV 提供者 + 步 2..k 只跑监督行"在任何框架/issue/论文中未出现;
- 最接近先例:TorchSpec(裁损失末端,backbone 全长)、HF `logits_to_keep`(非 spec、无 TTT)、SpecForge PR#673/#671(位置维**分块**压峰值,算量不减)、Liger FLCE/Apple CCE(vocab 维)。

### 4.2 positioning 一句话

> **首次把 loss-mask 感知的裁剪从损失末端推进到 draft backbone 与 TTT 展开内部——prompt 行退化为纯 KV 提供者、步 2..k 只前向监督行,使训练算量与激活随监督 token 数(而非序列长度)伸缩;对 prompt 占比高的 agent 轨迹数据(我们 15:1)收益最大,16K 域训练从"需 USP/offline"变为 4×48GB 单机可跑。**

### 4.3 论文骨架(systems note / MLSys 一节 + artifact)

1. 动机:agent/RAG 训练数据的极端 prompt 占比(实测 93.8%),显存全花在乘零位置;
2. 方法:两级位置维裁剪(A 损失末端 + B backbone/TTT 内部)+ 梯度死路的 mask 结构证明;
3. characterization:13GB 实测拆解 + 消融四臂显存伸缩曲线 + 与重算维/vocab 维的正交性;
4. 评测:被两大训练栈 PR 采纳的 artifact 叙事;域训接受率同台(五臂)作应用侧佐证;
5. 诚实边界:A 级引用 TorchSpec 为先例;分块类(#673/Liger)、重算类(#669)列为正交手段。

---

## 五、执行顺序

1. 二期重训 + 五臂评测收官(拿到 12k-3ep 逐位接受率,决定 k 是否上调);
2. 补 B-ii + memory_summary 拆解 → 消融四臂 + 正交叠加实验;
3. ropebuf bugfix PR(先发,攒信誉)→ A 级 PR → B 级 RFC 双投;
4. 论文成文(systems note 或主论文一节 + 开源 artifact)。

## 六、界外(明确不做)

- **盲目整层梯度检查点**:TTT 的 cache_hidden 在 forward 中被原位修改,盲套 checkpoint 重算会读到污染 cache → 梯度错;三期只移植 #669 已处理好 cache 的 TTT-aware 版本做正交对比,不自己乱套;
- USP/offline:磁盘不够,已证死;
- kernel 级 fused CE(Liger 风格):收益与 A 级重叠,复杂度更高,不做。
