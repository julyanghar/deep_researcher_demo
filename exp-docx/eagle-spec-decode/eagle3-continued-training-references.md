# 在已训好的 EAGLE-3 上继续训练：先例配方与可行性判决

> **缘起**：用户要在现成 AngelSlim/Qwen3-32B_eagle3 head 上用自有域数据（deep research summary 轨迹 1,744 条）继续训练，问"现有工作怎么训的、是否可行"。
> **调研方式**（2026-07-11）：4 路 agent 并行深读一手来源（Aurora 全文 / SpecPV+域适配论文全文 / SpecForge 主干源码+issue / 在线持续更新一族），第 5 路反面证据对抗核查因 API 限额未跑完（待补，见 §四）。所有引文均从论文 HTML 全文或源码 HEAD 357a97e 逐字核出。
> **一句话判决**：**可行、有直接先例、配方可抄**——但照现计划直接跑会踩一个源码级大坑（t2d/d2t 覆盖），且 lr 应比计划再降一档。

---

## 〇、TL;DR：对现有计划（[eagle3-domain-training-plan.md](eagle3-domain-training-plan.md)）的三个必改动作

1. **🔴 必须先处理 t2d/d2t 覆盖坑**：SpecForge 主干 `train_eagle3.py` L1036 在 `--ckpt-dir` 加载**之后**会无条件用**你的新训练数据**重算 top-32000 频次词表映射并覆盖 checkpoint 自带的 t2d/d2t buffer → **预训练 lm_head 整体错位，且 train/acc 指标看不出来，只有部署后接受率崩**。修复 PR #534（2026-04-15 提出）至今未合并。三选一：打 #534 补丁 / 把 AngelSlim bin 里的 d2t、t2d 导出预存到 vocab_mapping 缓存路径抢占 / 注释 L1036。
2. **lr 从计划的 3e-5 降到 1e-5~2e-5**：三处独立证据同向——Aurora warm-start 用恒定 1e-5、SpecPV 续训用 2e-5、域适配论文 offline 蒸馏 2e-5；SpecForge 默认 1e-4 是**从头训**的口径且 issue #577 实测它在 Qwen3 上早期梯度爆点频出。
3. **计划 §一.4 的诚实注记已过时**："没有找到现成 head→续训的干净公开案例"——**SpecPV（arXiv:2512.02337）就是**：在公开发布的 EAGLE-3 权重上用 SpecForge 续训，超参全表公开（Appendix A Table 5）。

---

## 一、直接先例（按对我们的可搬程度排序）

### 1. SpecPV（arXiv:2512.02337）——"现成 head → SpecForge 续训"的最干净存在证明

原文自述："The publicly released EAGLE-3 models are trained with a 2K context window. **Building on these released weights**, we further apply YARN-based long-context adaptation…"；"We **fine-tune the EAGLE-3 module based on SpecForge**, and our training hyperparameters are listed in Table 5."（target 含 Qwen3 4B/8B/14B——大概率就是 AngelSlim/SpecForge 那批公开 head，论文未点名）

| 项 | 值 |
|---|---|
| 初始化 | 公开发布的已训好 EAGLE-3 权重 |
| 数据 | PG-19 抽 **6,400 条**、32K 窗口 |
| lr / warmup | **2e-5** / ratio 0.05 |
| epoch | **1** |
| batch | global 16（micro 1 × DP4 × accum 4） |
| loss | SpecForge 默认 TTT 损失 L=Σαⁱ·Lᵢ，TTT 长度 4 |
| 结果 | EAGLE3-YARN 在 10K–60K 上下文 τ≈3.38–3.57、加速 2.49–3.11×（无前后对比消融） |

⚠️ 外推限制：它续训的目的是**修位置编码**（论文原话"只修 positional embedding、不注入知识，所以小数据够"）；我们的域适配要注入分布知识，数据/epoch 需求可能更高。

### 2. Aurora（arXiv:2602.06932，Together AI）——唯一有 "trained-from-static" 对照臂的工作

系统形态是"边服务边训"（SGLang 推理 + 1 张独立 GPU 异步训练，热切换回服务），但它的 **Aurora (trained) 臂就是字面意义的"在已训好的 EAGLE-3 head 上继续训练"**（现成 head：Tengyunw/qwen3_8b_eagle3，>200k 数据离线训的）：

| 项 | 值 |
|---|---|
| 初始化两臂 | trained=载入现成 head、**lr 恒定 1e-5**；scratch=随机初始化、lr 1e-4 |
| 优化器 | AdamW（wd=0、clip 0.5、warmup 400 步、**恒定 lr 不 decay**——刻意为之，防遗忘保可塑性） |
| loss | accepted token 上 KL(p_target‖p_draft)（最终取 Reverse KL）+ λ=1.0 的 rejected-token Discard KL（top-10 过滤）；**无 hard-label CE** |
| 数据 | 不用离线语料——serving 侧把 EAGLE-3 需要的三层 hidden states + target logits（裁 32K 小词表+top-256，每 token 256KB→2KB）+ accept/reject 轨迹经 RPC 流给训练 GPU |
| batch / seq | global 8 / max seq 2048 / TTT 5 |
| 节奏 | 梯度更新每 100 requests；权重热切换每 ~80 requests 是 Pareto 点 |

**结果**（Qwen3-8B，五域混合 40k prompts）：static 冻结 2.63 → **trained 2.99**（接受长度）；对已充分训练的 static head，在线继续训额外带来 **1.25× 吞吐**。scratch 长期 3.08 反超 trained，但要 ~10k requests 才收敛、初期吞吐受损；**warm-start 的价值=起点高、收敛快**（换域瞬间先小掉后超）。

### 3. arXiv:2604.26779 §3.3——"继续训练 = 防分布错配的保险"（最重要的期望管理）

RL rollout 中在线训 EAGLE-3 head（复用同一次 Megatron 前向的 hidden states，`.detach()` 梯度隔离只训 head）。关键消融（Table 5，k=3）：

- 初始化数据**已对齐**目标分布（policy 自生成 responses）→ 在线更新**几乎零增益**：1.77x vs 1.78x；
- 初始化数据**错配**（UltraChat 通用）→ 在线更新有实增益：**1.51x→1.63x**、1.19x→1.26x。

论文结论原话："insurance against distribution mismatch rather than a general improvement strategy"。**对我们**：AngelSlim head 在本域接受率只有 ~11%（vs 它通用域宣称 AL 1.95-2.6）——错配已量化坐实，我们恰好处在"继续训练有钱赚"的那一格。

### 4. arXiv:2503.07807（域适配最佳实践）——注意它训的是 standalone 1B draft，非 EAGLE head，只可类比

- warm-start 自现成 Llama-3.2-1B-Instruct 续训；**offline 蒸馏 lr 2e-5 / 3 epoch / forward KL 最优**；online 必须 1e-6（用 2e-5 反掉 8%）；
- 换域崩数字（Table 1，换 target 口径）：Biology 60.7→37.5、Chinese 49.6→35.7、Coding −7.1%、Math −3.7%；
- **scaling**：Function Calling 这类**结构化输出域 2,000 条即达最优**（82.6%，+37.0%）；Biology/Chinese 知识域 2k→19k 持续提升（终点 +17.2%/+14.0%）；
- **遗忘：全文没测**（域适配后没回通用域评过）——公开文献在这点上是空白。

### 5. 在线持续一族（见效速度的数量级参考）

- **OSD**（ICML'24，独立 draft）：只收集 draft 被拒位置的 (位置, target logits) 做 forward-KL 蒸馏，**100–200 个请求**接受率即显著上升，~2K 条到平台（Spider α 0.28→0.76）；
- **Baseten**：生产在线续训（不落盘 hidden states），**中位接受率 +20%**，受限流量模式 +100%+（实现细节专有未公开）；
- **Together ATLAS**：静态+轻量自适应双 speculator，RL 场景 1.4k 步内接受率 <10%→>80%（架构未披露）。

---

## 二、SpecForge warm-start 的工程事实（源码级，HEAD 357a97e）

- `--ckpt-dir` 做两件事：用 ckpt 自己的 config.json **顶替** draft config（你传的 `--draft-model-config` 会被覆盖）；`from_pretrained` 加载 draft 权重（midlayer/fc/norm/lm_head + t2d/d2t buffer 若在文件里）。**optimizer/step 不加载**，scheduler 带 warmup 从零（train_eagle3.py L506-530、L1115-1135）。`--resume` 才是断点续跑（另加载 training_state.pt），两者语义不同。
- **embedding 一律从 target 复制并冻结**（唯一冻结项；其余 draft 参数全量可训）。AngelSlim 的 pytorch_model.bin 只有 1.46GB、按参数量估算**不含 embed_tokens**——恰好匹配这个流程（缺 embedding 的警告被框架刻意静音）。
- **AngelSlim 兼容性**：architectures=LlamaForCausalLMEagle3 在注册表内 ✓；键名与 SpecForge 逐一对齐（Tencent/AngelSlim 训练源码 vs SpecForge llama3_eagle.py 核过）✓；唯一 config 差异 **head_dim 80 vs SpecForge 官方 config 的 128**——因走 ckpt 自己的 config 且 attention 优先读 config.head_dim，形状自洽能吃，但**部署时也必须沿用 ckpt 的 config**（与官方 config 从零训的头不同构）。
- **训前实证一次**：`torch.load` 打印 bin 的 keys，确认 d2t/t2d 在文件里（persistent buffer 默认入 state_dict，且该头在 vLLM 上以 32K draft vocab 正常工作理应携带；但眼见为实）。
- **缓存坑**：processed_dataset 与 vocab_mapping 按 md5(数据路径+max_length+template+target路径) 缓存——**同路径换数据内容会静默吃旧缓存**。
- **失败先例 issue #300**（拿 lmsys 现成头续训）：acc 从 0.00 起步、部署 accept length 退化到 1.2；维护者确认"微调数据分布与预训练头差异大时 acc 低开是预期"；楼主最终把 `--attention-backend` 从 flex_attention 换 **sdpa** 才恢复——续训不顺时这是排查点之一。
- max-length 默认 2048（例子 4096）——我们 prompt 中位 6K，**必须调大**（计划的 10240 ✓），#300 评论区有"max_length 小于 prompt 导致 acc 0"的先例。
- 官方对 finetune 的全部指引 = 博客一句 "can be efficiently fine-tuned for domain-specific tasks"，零超参。

---

## 三、可行性判决（针对我们 1,744 条的设定）

**方向可行，证据链完整**：

1. **错配是价值来源，而我们的错配已量化**：2604.26779 证明"初始化对齐→续训白费、初始化错配→续训有钱赚"；AngelSlim 在本域 ~11% vs 通用域 AL 1.95-2.6 = 错配坐实。
2. **数据量级不离谱但偏薄**：可见的最小有效量——OSD 100-200 请求见效、FC 结构化域 2k 到顶、SpecPV 6.4k（但那是修位置编码）、知识域要 2k→19k 爬坡。**1,744 条略低于 2k 下限**；summary 半结构化（引用+固定小结骨架），预期落在 FC 与知识域之间。欠拟合/不达标就走计划已有的扩产路径（其它非并行 run +4,500 条、题库扩产 500 题≈+9K）。
3. **on-policy 天然满足**：harvest content 是 Qwen3-32B temp=0 自产——正是 Aurora/OSD/2604.26779 全都强调的"用 target 自己的分布训"。
4. **遗忘风险对我们天然低**：公开文献没人测过域适配后通用域掉多少（空白）；但我们的部署是**两 server tag 路由，summary 专用 head 不服务通用流量**——通用能力掉了也不影响生产。heldout 45 条守域内，不必为遗忘加实验。

**落地 checklist（照抄可用）**：

```
① torch.load AngelSlim bin → 确认 d2t/t2d keys 在
② 处理 L1036：打 PR#534 补丁 / 预置 vocab_mapping 缓存 / 注释（三选一）
③ lr 2e-5（保守可 1e-5），恒定不 decay；epoch 1-3（SpecPV 1 / 域适配 3）
④ --max-length 10240+；--chat-template qwen3-instruct（与 serving enable_thinking=false 对齐）
⑤ 训中看 train/acc 低开不慌（#300 预期行为），持续 0 再查 attention-backend（换 sdpa）
⑥ 验收走预注册判据（P0 15% / P1 20% / P2 35%，heldout 45 条），部署沿用 ckpt config（head_dim=80）
```

**loss 口径备忘**（回应"prompt 不是 LLM 生成的"之问）：SpecForge 的 loss_mask 只在 assistant 回复段置 1（parse.py L257-291），**prompt 位置只当上下文输入、不当预测目标**——训练分布与推理分布在这点上天然一致（推理时 prompt 同样是人给的、同样只做上下文）。担心的事框架已经处理了；真正要守的 on-policy 约束在 response 标签上，harvest 数据已满足。

---

## 三b、追问回填（2026-07-11）

**Q：t2d/d2t 覆盖坑到底导致什么后果？**（通俗版）draft 用 32K 小词表省 lm_head 参数；lm_head 第 i 行的权重是**专为"坐在 i 号座位的那个 token"练的**，d2t 就是座位表（行号→真实 token id，偏移形式 target_id = i + d2t[i]）、t2d 是反查表。L1036 用你的 1,744 条新数据重算词频 top-32K 生成**新座位表**并覆盖旧表——**换了座位表、没换打分器**：第 i 行权重还是给旧座客练的，现在却给新座客打分，整个 lm_head 张冠李戴。静默的原因：训练/评估都按新表自洽计算，指标只表现为"acc 从 ~0 慢慢爬"（issue #300 的症状），等于**把 warm-start 最值钱的资产（练好的 lm_head + 与其对齐的特征）无声清零，退化成用 1,744 条偏斜词频从头重练一个乱序头**——你以为在微调，实际在小数据从头训，部署接受率远低于"现成 head 起点"，甚至不如不训。

**Q：prompt 位置不当预测目标，为什么还要它们的 3 层 hidden state？只要 KV 不够吗？**
直觉对一半：draft 起草时消费的确实是前缀的 KV——但是 **draft 自己的 KV，不是 target 的**（draft 是独立 1 层小网络，有自己的 q/k/v 投影，target 64 层的 KV 空间它用不了）。draft 在位置 j 的 K/V 是从 `concat(embed(token_j), FC([h_low;h_mid;h_high]_j))` 算出来的（q_proj 输入 2h 可证，SpecForge llama3_eagle.py）——**3 层 hidden state 是制造 draft KV 的原料**。训练时不能"只存 KV"：造 KV 的机器（fc/k_proj/v_proj）正是被训练的对象，每步权重都变，同样的原料造出的 KV 不同，必须保留原料每个 step 现造。**易混提醒**：①保存的那个东西叫"三层 hidden state"（三个层的残差流输出向量），**不是任何人的 KV**——target 的 KV 全程不参与 draft 的计算；②draft 的 KV 训练时**从不保存**，它只是每个训练 step 前向里的临时中间量，用完即弃；③推理时恰好相反——权重冻结了，draft KV 才可以缓存（这就是 vLLM 里的 draft KV cache）。"训练不能存、推理可以存"的分界线就一条：造 KV 的权重变不变。推理时同构：vLLM eagle3 在 target prefill 时 extract_hidden_states 抽所有 prompt 位置的 3 层，draft 跑一遍自己的"prefill"建 draft KV cache——训练消费方式与推理一致。prompt 位置的角色="在场但不考试"：有 hidden state、有 KV、参与注意力，唯独没有 loss。成本不亏：这些 hidden 是 target 同一次 prefill 顺手产出。

## 四、反面证据覆盖情况（诚实声明）

对抗核查 agent（专找"续训不如从头/灾难遗忘/小样本过拟合"反例）因 API session 限额**未跑完**。目前已覆盖的反面证据：issue #300 的实锤失败案例（见 §二）、2604.26779 的"对齐初始化→零增益"情形（见 §一.3）、Aurora 的"scratch 长期反超 warm-start"（§一.2，但代价是 10k requests + 初期受损）。**尚未系统排查**：小样本（1-2K）微调 2B head 的过拟合报告、AngelSlim 语料 license 问题。限额恢复后可 resume 补跑（run wf_626aab11-8fc）。

## 相关链接

- 训练总计划（本文的三个修正点落到它身上）：[eagle3-domain-training-plan.md](eagle3-domain-training-plan.md)
- 现成 head 证伪实录：[eagle-idea.md](../suffix-spec-decode/docs/eagle-idea/eagle-idea.md)
- 新颖性收窄定案（Aurora/Qwen MTP/2604.26779 三篇 + 可认领交集）：[Once Generated.md](../suffix-spec-decode/docs/eagle-idea/Once%20Generated.md) 缺口末条
- 源码/issue：SpecForge PR #534（vocab mapping 覆盖修复，未合并）、issue #300（现成头续训失败实录）、issue #577（lr 1e-4 梯度爆点）
