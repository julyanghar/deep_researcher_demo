# EAGLE-3 域内训练数据(summary 分析场景)

> 目标:在现成 eagle3 head 基础上继续训一个专打"研究摘要/分析"场景的 draft head,后端 vLLM。完整方案见 [../../exp-docx/eagle-spec-decode/eagle3-domain-training-plan.md](../../exp-docx/eagle-spec-decode/eagle3-domain-training-plan.md)。

## 文件

**数据范围(2026-07-12 第三版:local 合并集)**:新自产轨迹 **`eagletraj_local_r1`**(q41-100 共 60 题,Ceager+题间并发3+REPORT_PARALLEL,summary 路径非并行)+ **TRASH 里全部旧 local run**(q1-40)。排除:online 模式(online100*)、`SUMMARY_PARALLEL` 分节 section 调用、**非标准 system prompt**(自校验,挡掉 l3v2/l0v2 等 prompt 实验 run)。**长度用 Qwen3 真 tokenizer 计数**(旧字符估算低估中文 token 一倍,已纠)。

| 文件 | 条数 | 用途 |
|---|--:|---|
| `data/summary_train_main.jsonl` | **4,067** | 主训练集(≤8192 token;zh 3492/en 558/mix 17)——第五版 2026-07-13,含 DRGym local 增广 460 条 |
| `data/summary_train_longtail.jsonl` | 20 | >12288,忽略 |
| `data/summary_heldout.jsonl` | 46 | 验收池(15 个题面完全隔离,禁进训练) |
| `data/summary_all_raw.jsonl` | 4,810 | 无损备份(含原始 messages) |
| `data/manifest.json` | — | 统计+held-out 题面清单 |
| `convert_harvest_to_eagle3.py` | — | 转换脚本(确定性,可复现) |

⚠️ **MAXLEN=12288**(真 tokenizer 分布:中位 7,247、p95 11,000,12288 覆盖 99.6%)→ 训练 `--max-length 12288`;12K 序列 4×48GB 紧张时降 batch/梯度检查点/offline。数据量 4,744 在域适配见效区间(2K-19K)中段。**复核全绿**:section=0 / 重复=0 / 非标准 system=0 / held-out 泄漏=0。

每行格式(SpecForge conversations 风格):
```json
{"id":"c1_van40_q1_4","language":"zh","conversations":[
  {"role":"system","content":"Compress the provided search results..."},
  {"role":"user","content":"<research_question>...</research_question><search_results>..."},
  {"role":"assistant","content":"中国中产阶层在教育、职业..."}]}
```

## 数据怎么来的

源 = `eval/results/drbench/{eagletraj_local_r1,TRASH/*}/q*/harvest.jsonl` 里 `tag=RESEARCH_SUMMARY_TEXT` 记录(31 个 local run)。
漏斗:**新 984 + 旧存量 → 排除分节/非标准system/去重+过滤 4,810 → 主集 4,744**(+ 20 长尾 + 46 held-out)。生成物证:`~/modify-code-runs/eagletraj/`(status/server/run 日志 + SUFFIX_TRAJ 109 万行)。

- **排除分节 ✓**:messages 含 `section_to_write`/`<outline>` 的剔除(只留非并行整段 summary);
- **去重 ✓**:按 user message 内容 md5(每条含不同 search_results,条条独立);
- **质量过滤 ✓**:`finish_reason=stop` 且 output≥50 字符;
- **题面隔离 ✓**:按 `<research_question>` 分组,md5 排序留 15 题做 held-out,训练集**零泄漏**(已复核);
- **on-policy ✓**:assistant = **Qwen3-32B temp=0 自产**(harvest 的 content)——EAGLE-3 要求的"用 target 自产回复训练"天然满足,省掉数据重生成步。

## 能不能直接拿去训练?

**数据内容层面:就绪**(去重/过滤/on-policy/题面隔离/格式规整都做完,逐行 JSON 零坏行)。

**训练前的框架适配闸**:

- ✅ **闸①(部署端)加载兼容——已验证通过(2026-07-11 兼容冒烟)**:下载 SpecForge 训的现成 Qwen3-32B eagle3 head(`Zhihu-ai/Zhi-Create-Qwen3-32B-Eagle3`,架构 `LlamaForCausalLMEagle3`),在**我们的 vLLM 0.18** 上 `--speculative-config method=eagle3` **零转换直接加载成功**、spec decode 端到端运行(draft/accept 非零)。→ **SpecForge 训出来的产物一定能被我们的 vLLM 服务**,这端无风险。(物证:`~/modify-code-runs/eagle3-test/server_specforge_smoke.log`;顺带:该通用写作域 head 在我们 held-out summary 上接受率仅 ~9.5%,又一佐证"通用 head 域错配"。)
- ✅ **闸②a schema——已验证通过(2026-07-11,读 SpecForge 源码+实测)**:`specforge/data/parse.py:106-116` 同时认 `role/content`(我们用的)和 sharegpt 的 `from/value`;`:179-189` 若首条是 system 就**用数据里的 system**(忽略模板默认)——我们的 `conversations:[{system,user,assistant}]` **零改动直接被吃**。
- ✅ **闸②b template 对齐——已验证通过(2026-07-11,Qwen3 tokenizer 实测)**:SpecForge parse.py 用 `tokenizer.apply_chat_template`(与 vLLM serving 同一方法)。实测一条 held-out:`train_render.startswith(serving_prompt)=True`,assistant 段(loss 区)与 serving 生成字节对齐,non-thinking 的空 `<think></think>` 块两边一致。**训练必须用 `--chat-template qwen3-instruct`(=enable_thinking=false),不是 qwen3-thinking。**

**结论:三道闸全绿(加载/schema/template),数据 `summary_train_main.jsonl` 直接可训,无需任何改动。** 唯一硬约束:训练 template 用 `qwen3-instruct`。

## 下一步(方案文档 §五-六)

```bash
# SpecForge online 模式,warm-start AngelSlim head
torchrun --nproc_per_node=4 scripts/train_eagle3.py \
  --target-model-path /data/yilin/huggingface/Qwen3-32B \
  --draft-model-config configs/qwen3-32b-eagle3.json \
  --ckpt-dir /data/yilin/huggingface/Qwen3-32B-eagle3-angelslim \
  --train-data-path data/summary_train_main.jsonl \
  --chat-template qwen3-instruct --max-length 12288 \
  --learning-rate 3e-5 --num-epochs 3 --output-dir ./out
```

验收:`summary_heldout.jsonl` 回放(复用 `~/modify-code-runs/eagle3-test/measure_speedup.py`),对预注册判据 **P0 接受率≥15.1%(保本)/ P1≥20%(胜 suffix 的 summary 段)/ P2≥35%(论文级)**。先在 held-out 上测现成 AngelSlim head 拿基线,再测训练后 head,同批 prompt 对比。

## 数据第五版(2026-07-13):DRGym local 增广

- 源:`drgym_local232_r1`(232 英文题 local 重放,top-10 与部署一致,3 迭代×3 researcher×3 subquery,SKIP_FINAL_REPORT)——**online 轨迹(drgym_online400_r1)只用于建缓存池,照 online100 v1 先例不进训练集**;
- 贡献 **2,988 条**(去重后),tok 中位 10,085 / p90 11,640:**≤8192 仅 460 条(15%)进当前主集;≤12288 覆盖 98%(2,937 条)**——这批增广的主力在 longtail 等待 12288 训练配置(二期 assistant-only 裁剪后 TP4 可跑);
- **heldout 锁定机制**:`data/heldout_topics_locked.json` 固定一期 15 个 drbench topic,新增数据不再扰动评测集(convert 带缺失断言);
- 全量账:dedup 7,798 | main(≤8192)4,067(zh 3492/en 558/mix 17)| longtail 3,685 | heldout 46;防线全绿:section=0 / dup=0 / badsys=0 / 泄漏=0;
- 另:DRGym 168 题因 DashScope 欠费未跑(online 建池阶段),题面清单可随时补(断点续)。
