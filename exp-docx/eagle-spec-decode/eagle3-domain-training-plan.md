# EAGLE-3 域内训练方案:给"总结分析"场景训专用 head(后端 vLLM)

> **缘起**:现成 eagle3 head(RedHat/AngelSlim,都是通用 chat 数据训的)在本 workload 净减速、分析臂被证伪([eagle-idea.md](../suffix-spec-decode/docs/eagle-idea.md))。本方案回答:**在已训好的 eagle3 基础上,怎么继续训一个专打"summary 分析"场景的 head**,产物直接给 vLLM 用。
> **调研口径**:2026-07 当前生态,5 路并行调研(SpecForge / 官方 EAGLE / RedHat speculators / 域适配先例 / 本地数据盘点),来源见文末。

## 〇、TL;DR 路线图

```
数据:现有 harvest(target 自产 on-policy,1.2万条 summary)去重 → 不够就用题库扩产(DRGym 2110 题+DRQA 900 题)
   ↓ 转 SpecForge jsonl 格式
训练:SpecForge + configs/qwen3-32b-eagle3.json + --ckpt-dir <AngelSlim head> warm-start
      online 模式(target 用内置 SGLang TP4 跑前向),max-length 8-12K,lr 2e-5~5e-5,2-4 epoch
   ↓ 产物 = HF 格式 LlamaForCausalLMEagle3
部署:vLLM 直接加载(零转换):--speculative-config '{"method":"eagle3","model":"<ckpt>","num_speculative_tokens":3}'
   ↓
验收:复用 measure_speedup.py 回放(held-out 题)→ 接受率对着预注册判据(P0 15% / P1 20% / P2 35%)
```

## 一、为什么这事有戏(证据)

1. **败因确诊是域错配,不是机制**:RedHat head 在英文 chat 基准上 AL 2.15-2.49(官方 model card),到我们中文研究域掉到 1.40-1.61;域适配论文(arXiv 2503.07807)显示 target/域一换接受率崩(Biology 60.7%→37.5%)——和我们看到的一模一样。
2. **域内数据见效快**:同论文显示 **2K-19K 条域内样本**即显著移动接受率且符合 scaling law;Draft-OPD(2605.29343)在数学/代码域把 EAGLE-3 接受长度再提 +23%;OSD(ICML'24)持续域适配接受率 +0.1~0.65 绝对值。
3. **我们的数据天生是对的形态**:EAGLE-3 生态强共识是"用 target 模型自产的回复训练"(on-policy)——**harvest 里的 content 就是 Qwen3-32B temp=0 自己生成的**,零成本满足。
4. **诚实注记**:没有找到一篇"拿现成 head→域内续训→接受率 a→b"的干净公开案例(全表超参);上述是最接近的旁证。我们的预注册判据(§六)就是来补这个空白的。

## 二、框架选型:SpecForge 主路线,speculators 备选

> **2026-07-13 追记(选型盲区,用户追问暴露)**:选型时未评估 **TorchSpec**(lightseekorg,EAGLE-3.1 官方合作训练库,2026-05 发布)。回头核查:当时三理由(vLLM 零转换已实证 / AngelSlim warm-start 格式兼容 / sglang 共驻 TP4 匹配 4×48GB)仍成立——TorchSpec 是训推卡分离架构(Ray+vLLM worker extension,4卡=2训+2推,target TP2 每卡 32GB 权重更紧)、产出需 FSDP→HF 转换、无 Qwen3-32B 配方、当时仅 2 个月大,SpecForge 之选可辩护。**但盲区有代价**:TorchSpec 已实现 A 级位置裁剪(valid_idx gather),若当时知晓,七次发车中的 OOM 拉锯可省一半。教训:框架选型要做全面扫描,不能只看最响的两家。TorchSpec 的正确入场时机=EAGLE-3.1 从头训(3.1 不能续训旧 head)。

| | **SpecForge(SGLang 团队)⭐主路线** | speculators(vLLM 官方)备选 | AngelSlim | 官方 EAGLE repo |
|---|---|---|---|---|
| Qwen3-32B 支持 | ✅ 自带 config+QwQ-32B 4卡TP4 示例 | ✅(RedHat 的 head 就是它训的) | ✅ 训练代码公开 | ❌ TODO 未勾,只有 Llama |
| warm-start 已有 head | ✅ `--ckpt-dir`(打印"Finetuning from base model") | ✅ `--from-pretrained` 官方教程明写 | ❌ 只能 resume 自己的 trainer ckpt,热启动要改代码 | — |
| 长上下文 | ✅ USP(Ulysses+Ring)序列并行,实测到 64K | 🔶 教程示例 seq 8192,更长未见实证 | ❌ 示例 max_len 2048 | ❌ 2048 |
| 产物 vLLM 可加载 | ✅ **零转换**(LlamaForCausalLMEagle3,vLLM registry 直接映射;issue #42508 实跑证据) | ✅ 原生(speculators_config) | ✅(我们已跑过 AngelSlim head) | 需 speculators 转换 |
| 4×48GB 可行性 | 🔶 8×24GB(同总量)有人跑通 32B online(len 512);长 seq 需 USP/调参 | 🔶 未见 32B+长 seq 实证 | — | — |
| 官方态度 | EAGLE 官方 README **两处推荐用 SpecForge** | Red Hat 生产化路线 | — | — |

**warm-start 用哪个 head**:**AngelSlim/Qwen3-32B_eagle3**——它就是标准 `LlamaForCausalLMEagle3` 格式(和 SpecForge 的 config 同架构、同 draft_vocab 32000),`--ckpt-dir` 大概率直接吃(⚠️ t2d/d2t 词表映射 buffer 兼容性要冒烟验证,§八风险1);RedHat head 是 speculators 格式,只适合走 speculators 路线续训。
(也可考虑从头训:Baseten 经验值"专域 ~100K 样本";我们只有 1-2 万,**所以 warm-start 是刚需**——通用能力白拿,只做域迁移。)

## 三、数据管线(我们最大的本钱)

**存量盘点**(39 个 run 的 harvest.jsonl):
- RESEARCH_SUMMARY_TEXT **12,373 条** / FINAL_REPORT_MARKDOWN 1,104 条(⚠️ 跨 run 大量同题重复——同一批题不同配置反复跑);
- **去重后的独立量级**:online100_v2 单 run 覆盖 100 题 = 1,812 summary + 100 report;全库按题面去重估计 **~3-5K 独立 summary**;
- 每条已含 `messages`(system+user)+ `content`(target 自产输出)+ `token_ids`——**近乎无损转 SFT 格式**;
- token 形态:summary prompt≈7.9K/out≈0.5K;report prompt≈12.2K/out≈5.1K。

**扩产**(如需冲 1-2 万独立样本):题库有货——DRGym researchy_queries **2,110 题** + deepresearchqa **900 题** + DRBench 100 题。用 Ceager server 跑 agent(harvest 模式)每题产 ~18 条 summary → **500 题 ≈ +9K 独立 summary**(~20h,4 卡)。

**转换与切分**:
1. harvest → SpecForge jsonl:`{"id":..., "language":..., "conversations":[{"role":"system"},{"role":"user"},{"role":"assistant","content":harvest 的 content}]}`,`--chat-template qwen3-instruct`(与 serving 的 enable_thinking=false 对齐);
2. **按 user 内容去重**、**按 `<research_question>` 题面隔离 held-out**(禁止进训练);
3. **只训 summary 臂**:report 继续由 suffix 承担,max-length 只需覆盖 ~8K,避开 report 17K+ 长序列。

**★ 已执行(2026-07-11,产物在 `deep_researcher_demo/train/Eagle3/data/`;当天两次修订:①排除分节 summary ②统一只用 online100_v2 单 run)**:

| 文件 | 条数 | 说明 |
|---|--:|---|
| `summary_train_main.jsonl` | **1,744** | ≤16384 token(zh 890/en 847 ≈ 中英对半 = DRBench 真实分布) |
| `summary_train_longtail.jsonl` | 21 | >16384,忽略 |
| `summary_heldout.jsonl` | 45 | 验收池,15 题面完全隔离(泄漏=0 已复核) |
| `summary_all_raw.jsonl` | 1,810 | 无损备份 |

漏斗:1,812 原始(online100_v2)→ 去重+过滤 1,810 → 主集 1,744。MAXLEN=16384(该 run 中位 6192/p90 14328,覆盖 98.8%;16K 训练需 USP 或 offline)。**已排除**分节 section 调用(含 `section_to_write`,只留部署实际走的非并行整段 summary)。数据薄(1,744)靠 warm-start 补;欠拟合再扩其它非并行 run(c1/c2/census 系还有 ~4,500 条)。详见 [train/Eagle3/README.md](../../train/Eagle3/README.md)。

**⚠️ 训练前 2 个必验项(装 SpecForge 后)**:①**确切字段名/schema**——本产物用调研得到的 `{id,conversations:[{role,content}]}`(role=system/user/assistant),SpecForge 若要求 `{human,gpt}` 或预格式化 `text` 则字段重命名即可(无损备份在手);②**template 对齐**——用 SpecForge 的 tokenizer 预览渲染后的序列,确认与 vLLM serving(enable_thinking=false)拼出的字节一致,否则训练分布≠推理分布。

## 四、训练配方

```bash
# SpecForge,online 模式(target=Qwen3-32B 由内置 SGLang TP4 承载)
torchrun --nproc_per_node=4 scripts/train_eagle3.py \
  --target-model-path /data/yilin/huggingface/Qwen3-32B \
  --draft-model-config configs/qwen3-32b-eagle3.json \
  --ckpt-dir /data/yilin/huggingface/Qwen3-32B-eagle3-angelslim \   # warm-start
  --train-data-path <our_summary_sft.jsonl> \
  --chat-template qwen3-instruct \
  --target-model-backend sglang --tp-size 4 --sglang-mem-fraction-static 0.35 \
  --max-length 10240 \
  --learning-rate 3e-5 --num-epochs 3 \
  --output-dir <out>
```

- **lr 2e-5~5e-5**(warm-start 比从头训的 1e-4 低一档;域适配论文 2e-5/3epoch,NVIDIA recipe 2e-4 是从头);**2-4 epoch**(Red Hat:收益大头在前 2-3 个);
- **max-length 10240** 覆盖绝大多数 summary(均值 8.4K);OOM 就 8192 + USP(`--attention-backend usp_fa --sp-ulysses-size 2 --sp-ring-size 2`);
- 显存账(4×48GB):target TP4 权重 ~16.3GB/卡 + SGLang runtime(mem-fraction 0.35≈17GB/卡)+ draft FSDP(head ~2B 参数,Adam 状态摊 4 卡 ~6GB/卡)+ 10K 序列激活——**紧但有先例**(8×24GB 同总量跑通过 32B online);
- 冻结:target 全冻 + draft embedding 冻(标准配方),只训 draft transformer 层 + lm_head。

## 四b、兼容冒烟已过(2026-07-11)

训练前最该先做的风险验证——**SpecForge 格式产物能否被我们 vLLM 0.18 加载**——已实测:下载 `Zhihu-ai/Zhi-Create-Qwen3-32B-Eagle3`(SpecForge 训、`LlamaForCausalLMEagle3`),`--speculative-config method=eagle3` 在 vLLM 0.18 **零转换加载成功 + spec decode 运行**(draft 9558/accept 904)。→ **训练产物必能服务,部署端零风险**。剩下只有训练输入端 schema/template 对齐(装 SpecForge 现场核)。物证 `~/modify-code-runs/eagle3-test/server_specforge_smoke.log`;训练数据已备(`deep_researcher_demo/train/Eagle3/`,8560 主集)。

## 四c、训练执行实录(2026-07-12,Gate A-C 踩坑账本)

**已打补丁(都有 .bak 备份,物证 `~/modify-code-runs/eagle3-train/`)**:

| # | 补丁/配置 | 为什么 |
|---|---|---|
| 1 | `train_eagle3.py` L1036:`--ckpt-dir` 时**跳过 load_vocab_mapping 覆盖** | ckpt 的 t2d/d2t 与其 lm_head 行语义绑定,新数据重算映射会静默错位(上游 PR#534 未合并)。Gate B 张量级断言:from_pretrained 后 t2d/d2t/lm_head 与 AngelSlim bin 完全相等 ✅ |
| 2 | `eagle3_target_model.py`:ServerArgs 加 `chunked_prefill_size=-1` | SpecForge `_extend` 一次性提交 target_batch(=tp_size×batch)×max_length token,而 chunked_prefill>0 时 sglang eager buffer 只按 ~4096 分配 → foreach_copy 尺寸崩溃(a=4096 vs b=16384) |
| 3 | `--sglang-mem-fraction-static ≥0.354`(用 0.42) | 0.35 连 TP4 权重都放不下(sglang 报最低 0.354) |
| 4 | `--shard-target-output` | teacher 全词表 logits fp32 一次性物化 = L×151936×4B(12288 时 9.27GB)直接 OOM;shard 后 ÷4 |
| 5 | `--ttt-length 3`(默认 7) | 对齐部署 k=3 + TTT 链激活 ∝ 步数(SpecPV 先例用 4) |
| 6 | ~~max_length 定案 8192~~ → **2026-07-12 改判:TP8 全 8 卡 + max_length 14336**(数据重转 MAXLEN=14336,train_main **4,764 条**,zh 3935/en 808,longtail 清零,100% 覆盖——实测最长样本 13483) | TP4 下 12288/10240 均 draft 侧 OOM;用户要求覆盖完整样本 → 唯一物理可行解=**权重切更薄**:TP8 每卡权重 16.8→8.4GB,省出的 ~8.4GB 背 14336 的 ∝L 激活增量。**6 卡恰好无解**:64 q 头/8 kv 头对 6 不整除,TP2×DP3 权重 32GB/卡装不下(SpecForge 支持 DP=world/tp,train_eagle3.py:450)。∝L 大头(TTT 激活)每卡复制不随卡数摊薄——多卡本身买不来长 L,只有薄权重能 |
| 7 | **数据文件名带内容指纹**(如 `smoke24_13b25802.jsonl`)+ 清 `~/.cache/huggingface/datasets/generator` | 🔴 **最深的坑**:HF `Dataset.from_generator` 按**文件路径**缓存、不看内容——同路径覆盖新数据后训练静默读旧 Arrow 缓存。症状伪装成"换数据后仍 loss=0 / mask=0 / 同样 OOM",连环误导了 parser/模板/显存三层排查。SpecForge 自己的 cache_key 同样只含路径+参数 |
| 8 | `train_eagle3.py` PATCH(ropebuf):from_pretrained 后对每个 attention 重跑 `_init_rope()` | 🔴 **loss=nan 的真凶**:transformers **meta-device 加载不回填 non-persistent buffer**——RoPE 的 `inv_freq/cos_cached/sin_cached` 不在 ckpt 里(persistent=False),`to_empty()` 后是**未初始化内存**(CPU 恰好零页,GPU 上随机垃圾含 nan,每 rank 分布都不同)。只有 `--ckpt-dir`/续训(from_pretrained)路径中招,from_config 从头训正常走 `__init__` 不触发——所以上游没人报 |
| 9 | `lk_loss.py` PATCH(chunk-acc):acceptance_rate 指标的 fp32 softmax 按 2048 位置分块 + memfrac 0.30→0.27 | TP8@14336 首炮 OOM 差 1.61GiB,炸点=监控指标(非 loss 本体)的整条 fp32 softmax:`softmax(logits.to(fp32))` 瞬时 2×L×32000×4B(fp32 副本+输出),L=13483(全集最长样本,rank7 中签)时 ~3.2GiB。分块后逐元素相同(CPU 自检 maxdiff=0.0)。冒烟 B:全集最长 16 条连轰 2/2 步过,峰值 48.0/48.5GB 贴线成立 |
| 10 | `llama3_eagle.py` PATCH(nocompile):摘除 RMSNorm.forward / apply_rotary_pos_emb / RotaryEmbedding.forward 三处 `@torch.compile(dynamic=True)` | 全量 TP8 第 2 步崩:`InternalTorchDynamoError: attempting to assign a gradient of size [s0] to a tensor of size [s19]`——变长 batch 触发 dynamo 重编译,fake-tensor 化带累积梯度的 FSDP 参数时内部报错(compile+FSDP+梯度累积组合 bug)。这三个是小算子,eager 损失极小;吃显存的 flex attention 编译不受影响。另:此前同配置首启曾静默挂死(42 进程全睡、util 0,启动竞态,干净重启即好)——紧跟前一 run 退出就发车有风险,发车前确认旧进程清零 |
| 11 | 全量脚本 memfrac 统一为 **0.26**(冒烟验证 0.27,全量脚本曾漏改留在 0.30) | 🔴 **自摆乌龙**:改 memfrac 时只 sed 了冒烟脚本,全量脚本仍 0.30 → sglang 多占 1.42GB,~40 分钟后长批 OOM(只差 250MiB)。教训:**同一参数改动必须 grep 所有引用它的脚本**,"冒烟验证的配置"和"全量实际配置"要 diff 核对。0.26 的下界依据:KV 池 3.92GB > 理论最坏批(8×14336 token×32KB/token=3.76GB),不会出现"单批永远调度不上"的静默死等;比验证过的 0.27 再多 0.47GB 余量,覆盖 nocompile 后 eager RMSNorm 的 fp32 瞬时(L×5120×4B×2≈0.55GB@13.5K,与差额吻合) |
| 12 | **max_length 终判 12288**(数据第三版:main 4,744 条 99.6% 覆盖,longtail 20 条) | 修完 #11 后 0.26 档**仍在同位置 OOM**(GPU4 差 250MiB,46.2GB 已分配)→ 改判:**14336 包络被两次实测证伪**——最长样本 ~13.5K 时峰值 ≈48.0-48.2GB vs 48.5GB 容量,0.26/0.27 两档都是 break-even,分配器碎片一波动就死;0.5GB 级微调救不了。12288 = 结构性削减:砍最长 20 条(0.4%)→ 最坏 rank 峰值 -1.2GB(1195 位置×~1MB),余量 +1.7GB。**100% 覆盖的正道是二期 assistant-only 裁剪**(省 ~5GB 后 14336 宽裕),不是硬挤 |
| 13 | `llama3_eagle.py` PATCH(ckpt-norm):5 处 norm 调用套 `torch.utils.checkpoint`(use_reentrant=False) | 🔴 **12288 仍 OOM(第 5 崩)后真凶归位**:那笔反复出现的 250MiB=25600×5120×2B=**FSDP 的 MLP 权重形梯度缓冲**(只是压垮者);抬高基座的是 **#10 nocompile 补丁的显存回归**——eager RMSNorm 把 fp32 中间量(2×L×5120×4B/次)存进反向账本,3 norm×3 TTT 步+final ≈ **+4GB@12K**,恰好吞掉 #11/#12 挤出的 1.7GB。checkpoint 反向重算 norm(便宜、无副作用、躲开 dynamo):GPU 实测输出/输入梯度/权重梯度**逐元素相等**,单次省 360MiB、全前向省 ~4.3GB。教训:**摘 torch.compile 不是免费的——compiled kernel 融合掉的 fp32 中间量会回到 autograd 账本**,治 A 病引 B 病要对着显存表复查 |

**Gate C 关键教训 ①:`max_length < prompt 长度`时 loss 静默为 0**——截断砍的是序列尾部,恰是 assistant(loss 区);SpecForge 不报警,只有 vocab counting 的 "token frequency zero" 一条侧面线索。预检必须做"逐样本 loss_mask.sum>0"。

**Gate C 关键教训 ②:loss=nan 的定位链(2026-07-12,教科书级反例:第一嫌疑人是错的)**。症状:flex 后端 6/6 步跑完但 loss=nan(step 1 起)。排查按"缩小包围圈"走了四层:
1. **A/B 隔离**:flex@4096 短样本(无 padding、batch=1)**仍 nan** → 排除"长序列/padding 行全屏蔽"假设;sdpa@4096 在 backward 物化 L² 分数时 OOM(3.25GiB)→ sdpa 死于显存、无法作证;
2. **独立复现反证**:用 SpecForge 原函数(编译后 triton kernel + eagle3 mask + GQA + head_dim 80,探针同形状)喂随机数据 → **全干净** → flex kernel 本身无罪,nan 依赖真实权重/数据;
3. **探针下沉**(MCDBG_NAN 分层打点):入口 target hidden/target_p/fc 投影全干净 → **第一个 nan 在 q/k 过 RoPE 之后**,v(不过 RoPE)干净,q:k 的 nan 数=8:1=**恰好 GQA 头数比** → 锁定 cos/sin 表;
4. **离线一枪毙命**:按训练脚本同路径 from_pretrained 后直接读 `cos_cached` → **全 0**(应为 [-1,1] 且 cos(0)=1),`inv_freq` max=3.33e-12(应为 1.0)→ 未初始化 buffer 实锤;`_init_rope()` 重建后 cos(0)=1.0 恢复。
教训:**"换 backend 治 nan"是错误的第一反应**——flex/sdpa 都被冤枉;若当时硬啃 flash-attn 编译(30-60 分钟)会白干,nan 照旧。真正省时间的是第 2 步"随机数据反证":它把嫌疑从 kernel(与数据无关)推向加载路径(与权重有关),直接指路。

## 四d、4 卡 vs 8 卡参数配置对照(2026-07-12,应对卡被他人占用)

物理约束回顾:Qwen3-32B=64 q 头/8 kv 头 → TP 只能 1/2/4/8;**6 卡无解**;权重每卡=64GB/TP。∝L 的 draft 侧大头(TTT 激活等)每卡复制、不随卡数摊薄,**卡数买不来长 L,只有"权重切薄"能**。

| 参数 | 4 卡(TP4) | 8 卡(TP8) | 为什么不同 |
|---|---|---|---|
| CUDA_VISIBLE_DEVICES | 0,1,2,3 | 0-7 | 4 卡选连续段,避开常被占的 5 |
| --nproc_per_node / --tp-size | 4 | 8 | SpecForge online:world=tp(dp 需 world>tp) |
| 权重/卡 | **16.8GB** | **8.4GB** | 64GB bf16 ÷ TP——两配置差异的根源 |
| --max-length | **10240**(覆盖 90.7%) | **12288**(覆盖 99.6%;14336 已证伪) | TP4 权重多占 8.4GB → L 上限低两档 |
| 训练数据 | main=≤10240 桶(~4.3K 条) | main=≤12288 桶(4,744 条) | convert MAXLEN 与 --max-length 配对,保证零截断 |
| --sglang-mem-fraction-static | **0.42**(池 3.1GB>最坏批 4×10240×65.5KB=2.7GB) | **0.26**(池 3.9GB>最坏批 8×12288×32KB=3.1GB) | 池下界=最坏批 KV;KV/token 随 TP 减半(kv 头数分片) |
| --draft-accumulation-steps | **16** | **8** | 有效 batch 统一 64(= ranks×accum),lr 2e-5 口径不变 |
| 步数/epoch | ~1080(每步 4 样本) | ~593(每步 8 样本) | 样本数/ranks |
| 预计墙钟(2 epoch) | ~7-8h | ~6-7h | TP8 每步吞吐高但 L 更长,总时相近 |
| 不变项 | flex_attention、TTT k=3、lr 2e-5、warmup 0.05、clip 0.5、--shard-target-output、ckpt-dir warm-start、chunk-acc/ckpt-norm/ropebuf/nocompile 四补丁 | 同左 | 与卡数无关 |

修订(2026-07-12 深夜):TP4 的 max-length 定格 **8192**——10240 实测证伪(基座 46.3GB 顶死;错账复盘:chunk-acc 省的是**瞬时**峰值不动基座,ckpt-norm 相对 compiled 基线只省 ~1GB,之前把两者都记成基座节省)。**各 max-length 的有效监督量**(assistant 段 token,Qwen3 tokenizer 实测;样本超长即整条不训,prompt 不产生 loss):

| MAXLEN | 入选样本 | 样本覆盖 | 有效监督 token | 有效覆盖 | 均值/条 |
|--:|--:|--:|--:|--:|--:|
| 8192 | 3,645 | 75.8% | 1.91M | **72.7%** | 523 |
| 9216 | 4,103 | 85.3% | 2.19M | 83.4% | 533 |
| 10240 | 4,365 | 90.7% | 2.33M | 89.1% | 535 |
| 12288 | 4,790 | 99.6% | 2.61M | 99.4% | 544 |
| 14336 | 4,810 | 100% | 2.62M | 100% | 545 |

两个要点:①全集有效监督仅 **2.62M token**(prompt 占每条 ~93%,assistant 均值 545)——TP4@8192 拿到其中 72.7%;②这个"prompt:assistant ≈ 13:1"的形态让**二期 assistant-only 裁剪的收益极大**:loss 侧词表宽张量只需 ~545 位置而非 8192,理论 ~15×缩减,做完后 TP4 也能开 12288+。**二期已完成**(A+B-i 七闸全过,实测省 7.05GiB/卡、TP4@12288 解锁):[phase2-trimming-work.md](phase2-trimming-work.md);**三期**(B-ii+显存13GB拆解+开源PR+论文):[phase3-opensource-and-memory-plan.md](phase3-opensource-and-memory-plan.md)。

## 五、部署回 vLLM(和现在完全一样)

产物是 HF 目录(`LlamaForCausalLMEagle3`),**vLLM 0.18+ 直接加载零转换**(我们跑 AngelSlim head 的命令原样换路径):

```bash
vllm serve /data/yilin/huggingface/Qwen3-32B --tensor-parallel-size 4 \
  --speculative-config '{"method":"eagle3","model":"<训练输出目录>","num_speculative_tokens":3}'
```

**生产架构 = D2 两 server tag 路由**(既有设计):summary 调用 → eagle3-domain server;report 调用 → suffix(Ceager)server。**不需要 per-request 切投机**(已判死)。

## 六、预注册判据(用现成 harness,别刷数)

验收 = held-out 20 题的 summary prompt 回放([measure_speedup.py](../../../modify-code-runs/eagle3-test/measure_speedup.py) 原样可用),对照已钉死的盈亏公式($r_{\text{need}}=\frac{C_{\text{fix}}+ck}{kT_0}$,[eagle3-cost-anatomy.md](../suffix-spec-decode/docs/examine-spec-tax/eagle3-cost-anatomy.md)):

| 档 | summary 接受率(k=3) | 含义 |
|---|--:|---|
| 起点(现成 AngelSlim/RedHat) | 10.7-13.6% | 净减速 0.92-0.97× |
| **P0 保本** | **≥15.1%** | 胜过 vanilla |
| **P1 胜过 suffix 的 summary 段贡献** | **≥~20%**(AL≥1.6) | suffix 在 e2e summary 段只有 ~1.07×,这条线不高——**过了混合架构就成立** |
| **P2 论文级** | **≥35-40%**(AL≥2) | summary decode ~1.4×,"分析臂"作为机制贡献立得住 |
| 参考:追平 suffix replay | ~58% | 大概率不现实,不设为目标 |

失败判据也预注册:3 epoch 后 held-out 接受率 <15% → 判"域数据量不足或长上下文结构性瓶颈",按 §八回退,不无限调参。

## 六b、一期结果(2026-07-12 深夜,heldout 46 replay,vLLM TP4 + k=3,温度 0)

训练:TP4@8192,3,607 条,2 epoch(1,802 步),4h54m,末段训练侧 acc 0.32-0.38。Gate D 终验过(t2d/d2t==AngelSlim,lm_head 有更新)。评测两臂(SpecForge 只存最终 ckpt,无 epoch1 臂),指标为 vLLM `/metrics` 全程计数差分:

| 指标 | AngelSlim 基线 | **域训 2 epoch** | 变化 |
|---|--:|--:|--:|
| 接受率(accepted/drafted) | 11.13%(7,210/64,761) | **17.55%**(9,990/56,919) | **×1.58** |
| 平均接受长度 AL | 1.334 | **1.527** | +0.19 |
| 逐位置接受率 | 0.253/0.058/0.023 | **0.333/0.125/0.070** | 位 2 翻倍、位 3 三倍 |
| e2e 总时长(46 题) | 790s | **703s** | **−11%** |
| 逐题总时长中位 | 17.2s | **14.5s** | **−15.7%** |
| 输出长度中位(字符) | 1,025 | 1,028 | 无漂移 |

**四方同台**(2026-07-12 深夜补齐,同一 heldout 46、同一 replay、同温度;用户指正:对比必须同 workload,且要有 vanilla 绝对基准与 suffix 臂):

| 臂 | 接受率 | 逐题总时中位 | vs vanilla(中位/总和) |
|---|--:|--:|--:|
| vanilla(无投机) | — | 16.0s | 1.00× |
| AngelSlim(未域训 eagle3) | 11.1% | 17.2s | **0.95× / 0.97×(净减速)** |
| **域训 2 epoch** | **17.6%** | **14.5s** | **1.06× / 1.09×** |
| suffix | **33.7%**(13,080/38,860) | **13.1s** | **1.16× / 1.22×** |

(注:suffix 每轮草稿长度自适应,accepted/drafted 口径与 eagle3 固定 k=3 不完全同义,跨方法比较以 vs vanilla 墙钟为准。suffix 旧数字 1.89-2.41× 来自 40q 全流程(含 report 段,重复结构多、suffix 最吃香),summary 段单独看只有 1.16-1.22×——两个数字口径不同,别混用。)

**判决 vs 预注册判据**:
- **P0(打赢未域训基线)✅ 明确通过**:接受率 ×1.58(11.1%→17.6%),三位置齐涨,且把 eagle3 从**净减速(0.95×)救回净加速(1.06-1.09×)**——域训有效的主命题成立;
- **P1(接受率 ≥20% / 本意=summary 段 ≥1.07×)⚠ 贴线**:17.6% 差 2.5pp;vs vanilla 1.06×(中位)/1.09×(总和)正好骑在 1.07× 线上——记"贴线未决",不宣胜负;
- **P2(35-40% paper 级)✖ 远未及**;
- **suffix 仍是该 workload 赢家**:同台 33.7% 接受率、1.16-1.22×,域训 eagle3(1.06-1.09×)未追平。结论:**域训能救活 eagle3,但在块级检索 summary 这种"输入长、输出短、无历史可背"的形态下,天花板仍低于 suffix**。
- **k=4 终判(回答遗留问题)**:域训后 P(前 3 位全中)=0.333×0.125×0.070≈**0.3%**——第 4 猜每轮多付 1/3 draft 税换 0.3% 触发率,**否决**。
- 备注:两臂输出 hash 仅 2/46 一致——vLLM 投机解码的浮点平票分歧(不同 draft → batch 数值路径不同 → 近平票 argmax 偶发翻转,一处分叉全后缀变),两臂发出 token 总量(28.8K vs 29.0K)与长度分布一致,非质量信号。
- 后续杠杆:①epoch 数/数据量(本次只 2 epoch、1.91M 有效 token,Baseten 配方是 3-4 epoch);②二期 assistant-only 裁剪 → 12288(有效监督 +27pp);③若追 P2,需混入通用语料防域过拟合并加大数据。

## 七、时间/资源预算

| 步骤 | 卡 | 时长 |
|---|---|---|
| 数据去重+转换(存量) | CPU | ~半天 |
| (可选)扩产 500 题 | 4×48 跑 Ceager | ~20h |
| 训练 3 epoch(~1-2 万样本×~9K tok) | 4×48 | **~1-2 天**(online,target 前向是大头) |

**训练时间成分拆解(追问回填)**:每 token 计算 ≈ ①老师前向 64 GFLOP(**~65%**,产 aux hidden states+监督轨迹,online 下每 epoch 重跑)+ ②drafter TTT 展开 ~7-10(~12%,每位置串行 ~7 步)+ ③drafter 反向 ~15-20(~18%)+ ④杂项 5-10%。**harvest 省掉的是标准管线的"数据生成"步**(让 target decode 生成回复,别人最烦的一步,我们 temp=0 自产白送);**省不掉 ①**——harvest 没存 hidden states(~30KB/token≈4TB 存不起,serving 也不吐),drafter 没有它连输入都没有。①占 65% 且逐 epoch 重复 ⇒ offline 模式(跑一遍存盘)后每 epoch 只剩 ②③ ≈ 3-5h,多轮调参时明显更优。

**hidden states 三问(追问回填)**:①prefill 确实快(4 卡 ~2-3K tok/s,比 decode 快 60 倍),贵在**量**——135M token/epoch,online 下逐 epoch 重跑;②存的是**所有位置 × 固定 3 层**(EAGLE-3 招牌:低/中/高三层 aux hidden states 融合,64 层取 ~2/32/61;EAGLE-1/2 才只用最后一层)——每个位置都是一条训练样本,只存最后 token 就没有训练数据了;监督信号(下个 token id)harvest 已有,EAGLE-3 无 hidden 回归损失,不必存;③4TB = 3×5120×bf16(30KB/token)× 135M token;压缩:fp8 存(SpecForge offline 重构自带)→ ~2TB,数据砍 8K 条×8K 长 → ~1.9TB,叠加 → **~1TB**。

**为什么冻结了还要 1-2 天(追问回填)**:冻结省的是反向/优化器,**省不掉前向**——EAGLE-3 每个训练 token 都必须过 target 前向(①drafter 的输入=target 的 3 层 aux hidden states;②监督信号=target 的下一步轨迹),且 online 模式每个 epoch 重跑。账:135M token/epoch × 3 ÷ (4 卡 prefill ~2-3K tok/s) ≈ 40-55h;另有 TTT(drafter 每位置串行 ~7 次自身前向)也不小。**压时间四旋钮**:①offline 模式(target 前向只跑一遍存盘,~4TB 磁盘,后续 epoch 只训 head,迭代最快);②epoch 3→2;③数据挑 8K 条精华;④max-length 10K→8K。全用上 online ~12-18h,offline 首跑后每轮几小时。
| 每 epoch 验收回放 | 复用训练间隙换 server / 或训完统一测 | ~15min/次 |

## 八、风险与回退

1. **AngelSlim head 喂 `--ckpt-dir` 的格式兼容**(t2d/d2t buffer):先做 5 分钟冒烟(加载+打印参数名对齐);不行 → 从头训(SpecForge 同 config 随机初始化,代价是需要更多数据/epoch)或换 speculators 路线(`--from-pretrained` RedHat head)。
2. **4×48GB 长序列 OOM**:唯一可靠杠杆=**降 max-length**(已定案 8192)。⚠️ 2026-07-12 修订:**USP 退路实测已死**——SpecForge `sp_sanity_check` 硬断言 USP 只支持 offline 模式(`assert args.train_hidden_states_path is not None`),而 offline 需 ~1TB hidden states 磁盘、/data 仅剩 169GB;也没有梯度检查点 flag。
3. **长上下文结构性天花板**:EAGLE-3 在 8K+ 有已知 attention-drift 退化(LongSpec/SpecExtend);EAGLE-3.1(2026-05)有训练侧修复但 SpecForge 支持未确认——若 P0 都过不了且诊断指向长上下文,这是首要嫌疑,可试 SpecExtend(training-free)或等 3.1 落地。
4. **数据量不够**(1-2 万 vs Baseten 经验 10 万):靠 warm-start 补;若 epoch 曲线显示欠拟合域分布 → 扩产(题库还有 2,900 题没用过)。

## 八b、追问区(2026-07-11 回填,全部源码钉死)

**Q:online 和 local 的 summary prompt 为什么长度差很多?**
喂法不同:local(replay embed)只贴**命中的块**——全池 embedding 检索全局 top-10 块(`relevance.py _TOP_K=10`),约 2-4K token/查询;online 是**整页正文粘贴**,每条结果截到 `max_content_chars=12000` 字符(`search.py:46,186`)、每查询 6 条 → 可到 20K+ 字符。英文页常顶满 12K 上限 → 统一数据里长样本多为英文题(p90 14328 那批)。

**Q:EAGLE 推理输入——第 2 个草稿 token 用谁的 hidden?**
第 1 个草稿:输入=target 的 hidden(本步新接受 token 的,3 层融合;历史在 drafter 自己的 KV 里,每个位置的 target hidden 只进一次);**第 2..k 个:输入=eagle 自己上一步的 hidden**(vLLM eagle.py 草稿循环:`input_ids=draft_token_ids_list[-1]` + `self.hidden_states=hidden_states`←上一步自产输出)。验证后 target 的真 hidden 替换错误的自产 hidden。**训练要"所有位置"是因为每个位置都是一条"第一步"监督样本;而"第 2..k 步吃自产 hidden"正是 TTT 存在的理由**(否则推理第 2 步就是没见过的输入分布=exposure bias)。

**Q:TTT(training-time test)到底是什么?(2026-07-12 回填)**
名字有迷惑性——不是"测试",而是**训练时提前排练推理时的接力**。背景(见上条 Q):EAGLE 推理时第 1 个草稿 token 吃 target 的 hidden,第 2..k 个吃**自己上一步自产的 hidden**。若训练只教"吃 target hidden→出下一个词",推理第 2 步的输入(自产 hidden,带着自己的误差)就是训练没见过的分布,错会越滚越大(exposure bias)。TTT 的做法:每个 batch 把 draft **串行展开 k 步**(我们 k=3,SpecForge 默认 7):第 1 步吃 target hidden 算 loss;第 2 步吃第 1 步自产 hidden 再算 loss;第 3 步同理;k 步 loss 相加一起反向。draft 在训练时就练过"拿自己带误差的 hidden 继续猜",推理时第 2、3 个草稿 token 的命中率才不塌。代价:k 步的激活都得留到反向(反向要从第 k 步穿回第 1 步),显存 ×k。源码钉子:specforge/core/eagle3.py online forward 的 `for idx in range(self.length)` 循环,`self.length` 即 `--ttt-length`。

**Q:为什么 TTT ×7?**
SpecForge 默认 `--ttt-length 7`(train_eagle3.py:183)。训练时每样本做 7 次**串行** drafter 前向(step j 输入=step j-1 自产 hidden),7 步各算 loss、**反向穿过整条链** → 7 步激活全部保留到反向 ≈ 单步激活 ×7;且每个 TTT 步是全序列并行(16K 位置同时模拟),单步激活本身 ∝ L。

**Q:激活不是应该"对应要训练的参数"吗?参数少激活就该少?(2026-07-12 回填,常见误解)**
不对应——这是最容易搞混的一点。**参数决定"机器有多少台",激活决定"流水线上有多少零件"**:反向传播算某个权重 W 的梯度,公式是 `grad_W = 输入激活ᵀ × 输出梯度`,需要的是**前向时流过 W 的那批数据**。W 自己不管多大,它的输入激活是 `L × 输入宽度` 个数——**每个 token 位置都得存一份**。所以:
- 参数量 → 决定权重 3.2GB + 优化器 ~2.5GB(这两项确实小,固定不随 L);
- 激活量 → ∝ **L(数据量)× 每位置中间张量宽度 × TTT 步数**,和参数个数无关。8192 个位置 × 3 步,一层小模型的激活照样 ~7.5GB。
同一误解的镜像问题:"target 64 层参数多,激活岂不爆炸?"——不会,**冻结层不存激活**:不用对它的权重求梯度,梯度也不需要流过它(draft 拿到的 target hidden 是 detach 的,反向到此为止)。所以出现反直觉现象:64 层大模型零激活,1 层小模型激活是显存大头——**存不存激活看"要不要训",激活多少看"喂了多少数据"**,都不看参数多少。

**Q:单步激活具体由哪些张量组成?(2026-07-12 回填)**
先钉定义:"激活"= autograd 为反向保存的**前向中间结果**(例:算 `W_down` 梯度需要它当时的输入=SiLU(gate)×up 的值,前向不存反向就没法算)。draft 单层每个位置要留的中间量(hidden 宽 5120,数字=元素个数):

| 部件 | 每位置元素数 | 说明 |
|---|--:|---|
| 输入拼接(embed 5120 + target hidden 5120) | ~10K | 两个 layernorm 的输入输出 |
| QKV 投影输出 | ~10K | q:64头×80=5120,kv:8头×80×2=1280,+RoPE 中间 |
| attention 输出 + o_proj | ~13K | flex 的 out/lse + o_proj 输入输出 |
| **MLP gate/up 中间(25600 宽 ×2)** | **~51K** | **单项最大**——中间宽度是 hidden 的 5 倍 |
| norm/残差 | ~15K | 各 norm 的输入 |
| **logits(draft 词表 32000,fp32 折算)** | **~64K** | **第二大**——loss 要 fp32 softmax |
| 合计 | **~163K ≈ 320KB**(bf16 2B 计) | |

乘法:320KB/位置 × L × TTT 步数(**k 步激活同时驻卡**,反向要从第 k 步穿回第 1 步,谁都不能提前释放)。8192×320KB×3 ≈ **7.5GB**,即上表 #7。k=3 是**我们的训练配置**(`--ttt-length 3`,SpecForge 默认 7)——从 7 降 3 正是为了把这项从 ~17GB 压到 7.5GB。

**Q:上表每行数字具体怎么算出来的?(2026-07-12 回填,逐行推导)**
一条记账规则管全部:**对矩阵乘 `y=Wx`,反向算 `∂W` 需要当时的输入 `x`,所以每个矩阵乘的"输入"必存**;激活函数(SiLU)存自己的输入;RMSNorm 存输入;残差加法什么都不存(梯度直通)。把 draft 层里每个算子的"必存输入"排出来加总,就是一层的激活。权重形状是铁证(直接读训练产出 model.safetensors):

| 行 | 推导(元素个数,每位置) |
|---|---|
| 输入拼接 ~10K | q/k/v 三个矩阵乘**共享的输入** = concat(embed 5120, target hidden 5120) = **10240**,存一份。铁证:`q_proj.weight (5120, 10240)`——第二维 10240 就是输入宽度 |
| QKV 投影 ~10-12K | 输出:q 64头×80=5120 + k 8头×80=640 + v 640 = **6400**(q/k 是 RoPE 的输入要存,v 是 attention 的输入要存);RoPE 旋转又产出 q'/k' 新张量 ≈ +5760 |
| attention 输出 ~13K | flex 输出 5120(它是 o_proj 的输入,必存)+ lse 64 + reshape/contiguous 副本 5120 + o_proj 输出 5120 |
| MLP gate/up ~51K | gate 输出 **25600**(SiLU 的输入)+ up 输出 **25600**(逐元素乘的操作数)= **51200**;若算子未融合,SiLU(gate)×up 乘积(down_proj 的输入)再 +25600。铁证:`gate_proj.weight (25600, 5120)`——中间宽度 25600 = hidden 的 5 倍,这就是它是最大件的原因 |
| norm/残差 ~15K | hidden_norm 输入 5120 + input_layernorm 输入 5120 + post_attention_layernorm 输入 5120 |
| logits ~64K(fp32 折算) | lm_head 输出 32000(bf16)+ loss 的 fp32 log_softmax 保存 32000×4B——折成 2B 元素记 64K。铁证:`lm_head.weight (32000, 5120)` |

合计 ≈163K 元素 ≈ **326KB/位置**(bf16 2B/元素折算)——上文 320KB 的来历。

**Q:TTT 激活/draft logits/老师交付物为什么都 ∝ 输入长度 L?是不是所有位置都算了 loss?(2026-07-12 回填)**
你的理解基本正确,钉准三个细节:
1. **每个 assistant 位置 = 一条独立监督样本,且是并行算的**(2026-07-12 追问修正:prompt 位置只当上下文——供 K/V、不被预测、不产生监督信号;"监督样本"只有 assistant 位置)。这些样本 L 个位置**一次前向同时算**(和普通 transformer 训练一样),不是逐位置循环。"loss 在所有位置上算"的准确拆法:向量化实现先对**全长**算出 per-position loss 项(全长 logits→softmax→CE,显存照付),再乘 loss_mask(prompt=0)——**计入最终 loss、驱动更新的只有 assistant 位置**。max_length=4096 事故是反面证据:assistant 被截 → mask 全 0 → loss 恒 0。TTT 的 k 步是在这之上**串行重复**:第 1 步 = 全部 L 个位置一起做"第 1 猜";第 2 步 = 全部 L 个位置一起吃各自第 1 步自产的 hidden 做"第 2 猜";第 3 步同理。所以是**位置维并行 × TTT 步维串行**——推理时那种"单点滚动 k 步"的画面搬到训练里,变成"L 条轨迹同时滚 k 步"。attention 的花式 mask(causal+suffix 对角线)就是为了让第 j 步的位置 i 只看见它该看的历史。每步都产出全长 logits、每步都算 loss,k 步 loss 相加一起反向 → 激活 L×宽×k、draft logits L×32000×k,全 ∝L×k。
2. **loss 只在 assistant 位置计入,但显存按全 L 付**。loss_mask 把 prompt 位置的 loss 乘 0——可是 prompt 位置照样要**前向**(它们的 K/V 是 assistant 位置 attention 的历史),而且 assistant 的梯度会**穿过 attention 流回 prompt 位置的 K/V 投影**(算权重梯度),所以 prompt 位置的激活一个都不能少存。这就是"显存看输入总长 L,不看 assistant 长度"的根本原因。
3. **老师交付物 ∝L 同理**:target 对整条序列做一次 prefill,每个位置都要交 3 层 aux hidden(学生每个位置的输入)+ 该位置的全词表分布(每条监督样本的"标准答案")。另注:全长 logits/softmax 里 prompt 位置那部分对 loss 是零贡献,理论上可以只算 assistant 位置省显存——SpecForge 没做这个裁剪(position_mask 只做乘 0,张量仍全长),属于它显存账单里的可优化项,不是算法必然。

**Q:既然 prompt 位置的 loss 被乘零,显存是否有优化空间?(2026-07-12 回填,二期第一杠杆)**
有,且我们任务形态(prompt 长/输出短,assistant 占比粗估 ~30%)放大收益。分两类:
- **省得掉——loss 侧"词表宽"张量**(prompt 位置算出来就为被乘零):draft logits ×3 步(~2.75GB@14336)、draft fp32 softmax 瞬时(~1.8GB)、teacher target_p L×32000 fp32(~1.8GB)、teacher 全词表 logits 分片(~1.1GB/卡)。做法=lm_head/target_p 构造前先 `index_select` assistant 位置,loss/指标在紧凑张量上算。合计可省 **~5GB/卡** ≈ 够把 TTT 升 k=4 或推 16K+。SpecForge 上游没做(position_mask 只乘零不裁剪),中等侵入度(eagle3.py 内百行)。
- ~~省不掉——backbone 激活(~320KB/位全长存)~~ **2026-07-13 修正(用户指正,《full-ft-vs-eagle3-memory》§11 同判)**:prompt 行必须保的只有"X→k/v"一小段(**~25KB/位**)。依据=TTT mask 结构(generate_eagle3_mask,本会话核过源码):跨位置注意力只读**第 1 步**的 K/V(causal 块),第 2/3 步的 K/V 是**对角线私有**(只有本行自己读)→ prompt 行的 q/attn/**MLP**、以及步 2+ 的整行,所有下游都终结在被 mask 的 loss,**梯度死路,无需保存**(甚至无需计算)。修正后上限:draft 激活 7.5→**0.64 GiB**@8192,合计可省 **~10 GiB/卡**(原计划书的 ~5GB 低估一半);且裁剪后每加长 1 个 prompt token 只花 ~25KB(原 960KB),**16384 也开得起,"14336 证伪"在裁剪后不成立**。教训:把"实现现状"(全长算+乘零)说成了"数学必然"——判死路要看 mask 的真实连接结构,不能只看"K/V 有人读"一层。
  细化(2026-07-12 追问):prompt 位置具体要留两笔——①**权重梯度的现场**:`grad_W_k = x_pᵀ×∂L/∂k_p`,W_k/W_v 在 prompt 位置也"干了活",算它们的梯度要用当时的输入 x_p(norm 后拼接向量);②**attention 反向自身**要保存的 Q/K/V/out/lse,prompt 的 k_p/v_p 在册。且链条不止于投影:梯度到 x_p 后继续上游——hidden 分支来自 fc(可训)或上一 TTT 步整层输出 → prompt 位置的 norm/上一步 attention/MLP 全在回程路上。唯一理论可裁:最后一个 TTT 步 prompt 位置的"输出侧"(只通向乘零的 logits),收益小不值做。直觉:prompt 不是训练目标,但是可训权重的**工作现场**,现场必须封存到反向结束。
当前 run(TP8@14336)显存最坏情况已实测成立,不中途动刀;此优化立项条件=k=4 需求或 16K+ 需求出现。

**Q:"需要存的激活"=需要训练的层的输出?(2026-07-12 回填)**
不完全对,准确规则是:**梯度要路过的每个算子,其反向所需的中间张量都要存**——多数算子存的是自己的**输入**(不是输出;只不过 A 的输入=B 的输出,口语说"中间结果"都算对)。"能不存"要同时满足两个条件:①梯度不进该算子自己的参数(冻结);②梯度不需要**穿过**它去更上游的可训参数。target 64 层两条全占(冻结 + draft 拿到的 hidden 是 detach 的,梯度到此断流)→ 零激活。反例帮你钉住第②条:假如只训 embedding、冻结全部中间层,中间层激活**照样全存**——梯度要穿过它们流回 embedding。所以判据不是"这层训不训",是"梯度的回程路经不经过这里"。

**Q:"反向传播每经过一个算子要乘它的局部导数"是什么意思?(2026-07-12 回填,从零版)**
训练要回答"每个数动一点,loss 动多少"(=梯度=放大率),网络是函数套函数,总放大率=沿路各节放大率**相乘**(链式法则)。手算例:y=x²(算子A)、L=3y(算子B),x=2 → y=4,L=12。①y 动 1,L 动 3(∂L/∂y=3,上游梯度);②x 动 1,y 动 2x=**4**(∂y/∂x,A 的**局部导数**——"局部"=只看这一节自己);③串乘:∂L/∂x=3×4=12。验算:x=2.01 → L=12.1203,动了 0.1203≈12×0.01 ✓。**"反向"**=计算从 loss 端往回接力,每个算子只报自己那节放大率;**"存激活"**=局部导数公式里常含当时的输入(2x 在 x=2 处是 4、x=3 处是 6),不存 x 反向就报不出数——y=x² 无任何可训参数照样要存,这就是无参算子要存的最小例子。对照:y=x+5 放大率恒为 1,与数据无关 → 免费(残差加法)。

**Q:只算 loss 对权重的导数不就行了,为什么还要算 ∂L/∂x(对中间数据的导数)?(2026-07-12 回填)**
∂L/∂x 不是目的,是**通往深层权重的接力棒**。两层例:h=W₁·a,y=W₂·h,L=f(y)。W₂ 紧挨 loss,∂L/∂W₂=∂L/∂y×hᵀ,确实不需要数据导数;但 W₁ 影响 L 的唯一途径是"先改 h→再改 y→再改 L",所以 ∂L/∂W₁=**(∂L/∂h)**×aᵀ,而 ∂L/∂h=W₂ᵀ×∂L/∂y——**必须先算出 loss 对中间激活的导数,深层权重的梯度才有原料**。数字(接上条):a=1,W=2,x=Wa=2,y=x²,L=3y → ∂L/∂W=3×4×1=12,中间的 3×4=∂L/∂x 是必经之站(验算 W=2.01→L=12.1203 ✓)。什么时候可以不算 ∂L/∂x:x 的上游再没有任何可训参数时——autograd 靠 requires_grad 传播剪枝;target hidden 的 detach 就是宣告"这条路上游没人要信号",整棵子树跳过 → 冻结 target 零开销的机制本体。网络越深,对中间数据求导的次数越多——"深"度学习的账单记在这。

**Q:无参算子(SiLU/softmax/norm)凭什么也要存?(2026-07-12 回填)**
反向传播每经过一个算子要乘它的**局部导数**(∂L/∂输入 = ∂L/∂输出 × ∂输出/∂输入),**存不存取决于局部导数是不是常数**:
- 加法(残差):导数恒为 1 → 免费,啥也不存;
- 线性 y=Wx:对输入的导数 = W,**权重本身就在显存里** → 梯度免费穿过(冻结矩阵乘零成本);只有 W 可训时才另存输入 x(算 ∂W 用);
- SiLU:y=x·σ(x),导数 = σ(x)+xσ(x)(1−σ(x)),**是 x 的函数**。手算:前向 x=2 → 输出 1.762;反向收到 ∂L/∂y=0.5,∂L/∂x=0.5×1.091=0.545——那个 1.091 不知道"当时 x=2"就算不出 → x 必存;
- 逐元素乘 a×b:∂/∂a = b,**要对方的值** → 两边都存;softmax 雅可比依赖输出 → 存输出;norm 依赖输入+统计量 → 存输入。
一句话:**局部导数=常数 → 免费;局部导数=权重 → 免费穿行;局部导数依赖数据 → 数据必须留下**。这也解释了"冻结层也存激活"的真凶是层里的无参算子,不是冻结的矩阵乘。

追问(2026-07-12):**没有 SiLU 处的 ∂L/∂x,具体哪个权重的梯度算不出?——W_gate,点名到人**。权重梯度公式要两个原料:∂L/∂W =(自己的输入)ᵀ×(**自己输出处收到的梯度**)。MLP 链 g=W_gate·h → s=SiLU(g) → out=W_down·(s×u) 里,W_gate 的第二个原料 ∂L/∂g **只能由 SiLU 反向算出**(∂L/∂g=∂L/∂s×SiLU′(g),要存 g)。手算:h=1,W_gate=2,W_down=3,L=W_down·s → ∂L/∂g=3×1.091=3.273,∂L/∂W_gate=3.273×1=3.273;验算 W_gate=2.01 → L 从 5.285 到 5.3175,Δ=0.0327≈3.273×0.01 ✓。且断的不止 W_gate:接力棒在 SiLU 站断掉,这条支路更上游的 norm γ/W_qkv/o_proj/fc/上一 TTT 步整层全部收不到信号。收束:**每一站的 ∂L/∂x 都是为它上游第一个带参数算子准备的"第二个原料"**。

**终版一句话**(2026-07-12 追问收敛。注意"只留可训权重的输入"还不完备):**凡是"从 loss 走回任一可训参数"的反向路径所需要的张量,都要留**——拆两类:①路径终点=可训矩阵乘,存"它的输入"(最大宗);②路径中途的**无参算子**(SiLU/softmax/norm),存"它们反向要用的输入/统计量"。第②类容易漏,而显存表最大件恰好属于它:MLP 51K 里 gate 输出是 SiLU(无参)的输入、up 输出是逐元素乘(无参)的操作数——都不是任何可训权重的输入,但不存它们,梯度就穿不过去、到不了 gate_proj/up_proj。自检三例:target 冻结+detach → 不在任何回程路上 → 零激活;prompt 位置 → 在回程路上(K/V 现场)→ 全存;推理 → 没有回程 → 即算即扔。

**Q:假设 target 32B 不冻结(全参微调),激活要多少显存?(2026-07-12 回填)**
按同一套记账规则算 Qwen3-32B(hidden 5120、64 层、q 64头×128、kv 8头×128、MLP 中间 25600),每位置每层:norm 4×5120≈20K + QKV 输出 10.2K + RoPE 副本 9.2K + attention/o_proj ≈13K + MLP(gate+up+乘积)76.8K ≈ **130K 元素 ≈260KB**。往上乘:
- ×64 层 = **16.6MB/位置**(draft 只有 1 层 ×3 TTT 步 ≈ 1MB/位置,差 16 倍)
- ×8192 位置 = **~136GB**——单条样本、batch=1、无 TTT!
- + 词表 logits 8192×151936×(2B+4B) ≈ 7.5GB → **纯激活 ~140GB**
还没完,全参训练态:权重 bf16 64GB + 梯度 64GB + AdamW fp32 m/v 256GB + fp32 主权重 128GB ≈ **512GB**。合计 ~650GB ≈ 14 张 48GB 卡**只装状态**。这就是为什么全参微调 32B 必须 ZeRO-3 分片 + 梯度检查点(检查点=只存 64 个层边界 hidden ≈5.4GB,层内反向时重算,拿 ~30% 算力换 25 倍激活显存)。对照我们的方案:只训 1 层 draft,激活 7.5GB + 训练态 ~6GB,4 张卡背得动——**"只训小 head"的全部显存红利就在这一对比里**。

**Q:eagle 参数少,激活应该也少?/"激活"就是 hidden states 吗?**
不是——hidden(5120 宽)只是其中最细的一根。训练态的"激活"= autograd 为反向保存的**全部中间张量**(算 down_proj 权重梯度需要它当时的输入=SiLU 输出,所以前向必须存):每位置每步 ≈ 输入拼接 10K + QKV 10K + attn/o 13K + **MLP gate/up 51K(最大件)** + norm/残差 15K + **logits 64K(fp32 折算,第二大件)** ≈ **16 万元素 ≈ 320KB/位置**。公式:**激活显存 ≈ L × W_saved(每位置保存的中间元素总和) × TTT 步数** ≈ 16384×320KB×7 ≈ **35GB**——与参数量无关(参数只决定权重的 3.2GB)。推理时这些算完即扔所以无感;反直觉对照:**冻结的 64 层 target 不留激活,要训的 1 层反而是显存大头**。缓解:梯度检查点/logits 分块/USP/offline。

**Q:每张卡 48GB 训练时分别消耗在哪?(2026-07-12 实测锚定,online 模式,TP4×4 卡,L=max_length)**

实测锚点(L=10240 时 OOM 现场):PyTorch 已分配 **45.25 GiB**、进程共 **46.22 GiB**、仅剩 1.14 GiB,压死骆驼的是 draft 的 fp32 softmax(L×32000×4B:12288→1.43GiB、10240→1.22GiB,**与报错字节完全吻合**)。逐项账(单卡):

| # | 项 | 大小(GiB) | 随 L 涨? | 说明 |
|---|---|--:|:--:|---|
| 1 | target 权重(TP4 分片) | **16.8** | ✗ | 64GB bf16 ÷4;sglang 实测下限 mem-frac=0.354 |
| 2 | target KV 池(sglang 静态) | **~3.1** | ✗(池固定) | mem-frac 0.42×47.37=19.9 减权重;池内按 target_batch×L 消耗 |
| 3 | draft 模型参数 fp16 | ~3.2 | ✗ | 1 层+fc+lm_head 1.65 + 冻结 embed 1.55 |
| 4 | draft 优化器(AdamW fp32 m/v,FSDP÷4) | ~2.5 | ✗ | 可训 ~0.83B×12B÷4 |
| 5 | draft 梯度(分片) | ~0.5 | ✗ | |
| 6 | teacher 交付物:logits(--shard-target-output ÷4)+ 3 层 aux hidden | ~1.5-2.5 | **✓ ∝L** | 不 shard 时 12288 一项就 9.27GiB(实测 OOM) |
| 7 | **draft TTT 激活(3 步链全保留)** | **~6-8 @10K** | **✓ ∝L×TTT** | 每步每位置 ~200KB(MLP 中间 51K 元素为大件)×L×3 步,反向要用不能丢 |
| 8 | draft 各 TTT 步 logits(bf16)+CE 中间量 | ~2-3 | **✓ ∝L** | L×32000×2B×3 步 |
| 9 | **draft fp32 softmax 瞬时(炸点)** | 1.2-1.6 | **✓ ∝L** | L×32000×4B,单笔最大瞬时分配 |
| 10 | CUDA ctx/NCCL/flex_attention 工作区/碎片 | ~1.5-2.5 | ~ | 报错里"非 PyTorch 内存"~1GB |
| | **合计 @L=10240** | **~44-46** | | 47.37 可用 → **骑线,任何 ∝L 项再涨即 OOM** |

**Q:为什么长序列训练会 OOM?**
表里 #6-#9 全部 **∝L**(#7 还要 ×TTT 步数):L 从 8192→12288(+50%)时这四项合计从 ~9GB 涨到 ~14GB,而 #1-#5 的 ~26GB 固定成本不动——48GB 卡上 target+draft 共驻,预算没有伸缩空间。且三条泄压阀全关:USP 序列并行=offline-only(磁盘不够)、无梯度检查点 flag、offline 模式磁盘不够 → **唯一杠杆是降 L(定案 8192)**。对照推理为什么没这问题:推理激活即算即扔、无 TTT 链、无优化器,同一张卡 serving 40960 上下文都没事——**贵的是"训练态"不是"长序列"本身**。

**Q:sdpa / flex_attention / fa 是什么?为什么换个 backend 能治 nan 又能引来 OOM?(2026-07-12 回填)**
它们是**同一个数学(注意力)的三种 kernel 实现**,SpecForge `--attention-backend` 三选一。数学等价,但显存和数值行为完全不同:
- **sdpa** = scaled dot-product attention,PyTorch 官方 `F.scaled_dot_product_attention`。在需要自定义 mask 时走的实现会**物化 L×L 注意力分数矩阵**(每头 L²×4B fp32)。实测:8192 时 backward 单笔要 12.82GiB(炸),4096 也要 3.25GiB 而卡上只剩 3GB(炸)。管线把样本 pad/按 max_length 算,sdpa 永远付满 L² 的账。数值上最保守可靠。
- **flex_attention** = PyTorch 2.5+ 的"可编程注意力":自定义 mask(EAGLE TTT 那种"causal+suffix 对角线"花式 mask,见 specforge/modeling/draft/flex_attention.py `generate_eagle3_mask`)被 torch.compile 编成 triton kernel,**不物化 L² 矩阵**,显存 O(L),8192 稳跑。但数值路径新、坑多(SpecForge issue#300 也报过异常)。我们一度实测它 loss=nan——**后被平反**:nan 真凶是 RoPE buffer 未初始化(§四c 教训②),与 backend 无关,flex 是最终选用的后端。
- **fa** = FlashAttention 库,O(L) 显存 + 数值久经考验,理想解;但 SpecForge 代码 import 的是 `flash_attn` 的 varlen 接口 + `bert_padding`,环境里装的 flash-attn-4(beta)是**空壳命名空间**(`dir()` 为空,三个符号全缺)→ 启动即警告 "flash_attn is not found" 回退 flex。要用得真编译 flash-attn v2(cu13+torch2.11 兼容风险,30-60 分钟)。
一句话:**数学一样,账单和数值不一样**——sdpa 稳但 O(L²) 付不起;flex O(L) 付得起但当前 nan;fa 两全但装不上。这是 Gate C 卡住的三角。

**Q:训练的 `--max-length` 是什么?**
一条训练样本(system+user+assistant 拼接渲染后)分词的**硬上限,超长从尾部截断**。三个要点:①限的是整条序列不只 prompt;②我们的 assistant 在末尾 → 样本超长时**被砍的正是 loss 区**(max_length=4096 冒烟 loss=0 事故的根源),SpecForge 不报警;③它同时是显存主旋钮(激活 ∝L)。与 convert 的 MAXLEN 配对使用:数据端先按同值分桶,训练端就零截断。

## 九、关键来源

- SpecForge:[repo](https://github.com/sgl-project/SpecForge) · [训练文档](https://github.com/sgl-project/SpecForge/blob/main/docs/basic_usage/training.md) · [qwen3-32b config](https://github.com/sgl-project/SpecForge/blob/main/configs/qwen3-32b-eagle3.json) · [USP PR#425](https://github.com/sgl-project/SpecForge/pull/425) · [8×24GB 实证 issue#380](https://github.com/sgl-project/SpecForge/issues/380) · [vLLM 直载实证 issue#42508](https://github.com/vllm-project/vllm/issues/42508)
- speculators:[repo](https://github.com/vllm-project/speculators) · [在线训练教程(--from-pretrained 续训)](https://docs.vllm.ai/projects/speculators/en/latest/) · [数据配方文章(跨蒸馏+50K自蒸馏,3-4 epoch)](https://developers.redhat.com/articles/2026/07/06/smarter-data-generation-faster-speculator-training)
- 域适配证据:[arXiv 2503.07807(2K-19K 见效/域偏移致崩)](https://arxiv.org/abs/2503.07807) · [Draft-OPD 2605.29343(+23% AL)](https://arxiv.org/abs/2605.29343) · [OSD ICML'24](https://arxiv.org/abs/2310.07177) · [Baseten 实操(数据量/lr 表)](https://www.baseten.co/blog/how-to-train-custom-eagle-3-heads-for-speculative-decoding/) · [NVIDIA NeMo recipe(冻结配方)](https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/eagle-speculative-decoding)
- 长上下文:[LongSpec 2502.17421](https://arxiv.org/abs/2502.17421) · [SpecExtend 2505.20776](https://arxiv.org/abs/2505.20776) · EAGLE-3.1(vllm.ai/blog/2026-05-26-eagle-3-1)
- 现成 head 出处:[RedHat model card(ShareGPT+UltraChat)](https://huggingface.co/RedHatAI/Qwen3-32B-speculator.eagle3) · [AngelSlim(语料未公开)](https://huggingface.co/AngelSlim/Qwen3-32B_eagle3) · 官方 EAGLE repo 无 Qwen3 head(README TODO 未勾)
