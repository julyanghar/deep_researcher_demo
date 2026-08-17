# 复现資源指针(审计补录,2026-08-16)

备份审计发现的"只存在本机、一丢就断"的复现依赖,在此落账。配环境与跑实验的完整步骤见
[REPRODUCE.md](REPRODUCE.md)。

## 外部权重的精确 revision(来自本机 HF hub refs,权重本体已删/可重下)

| 用途 | HF repo id | revision(训练/实验当时) |
|---|---|---|
| target model(训练+serving) | `Qwen/Qwen3-32B` | `9216db5781bf21249d130ec9da846c4624c16137` |
| EAGLE-3 warm-start | `AngelSlim/Qwen3-32B_eagle3` | `74789e1a6a4b705a71d040315d911b58fb2a80ef` |
| 对照用第三方 head | `RedHatAI/Qwen3-32B-speculator.eagle3` | `dc84fe7ff1db31efa824776f49c141fc8195eb47` |
| 对照用第三方 head | `Zhihu-ai/Zhi-Create-Qwen3-32B-Eagle3` | `c29548eab1ed02c07a9c53fabd97703d60f25f5f` |

本机目录名与 repo 的映射:`/data/yilin/huggingface/Qwen3-32B-eagle3` = RedHatAI 那个,
`…-eagle3-angelslim` = AngelSlim 那个(md5 已核:phase2 的 config.json = AngelSlim config)。

## phase2-12k-3ep 训练的精确代码版本

- 框架:SpecForge 本地 fork,**训练用的是 commit `77f4a0f`**(在分支
  `yilin-trim-mem-probes` 上,已推 https://github.com/julyanghar/SpecForge )。
  注意 pr-trim-a-v3(=PR #705)的 scripts/ 里**没有** train_eagle3.py,复现训练要 checkout 77f4a0f。
- 启动脚本:[train/Eagle3/run-artifacts/run_retrain.sh](train/Eagle3/run-artifacts/run_retrain.sh)
  (含 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.85` 与
  torchrun 参数,这两样别处没有记录)。
- 完整 79 项训练参数:`training_state.pt` 的 `args` 字段(HF model repo 内);训练日志
  [train/Eagle3/run-artifacts/](train/Eagle3/run-artifacts/)。
- 已知损失(落账,补不回):`train/Eagle3/runs/` 三个中间 checkpoint 目录已清空
  (ep1_8k 对照臂权重永久丢失);`training_state.pt` 顶层无 optimizer 状态(续训只能 warm-start)。
- 环境快照注意:SpecForge backup-env 分支的 env/ 是 **2026-08-16 现拍**,与 7 月训练时有漂移
  (如 flash-attn 版本;训练时实际 fallback 到 flex_attention,见 retrain.log 的 UserWarning)。

## 数据谱系关键点

- phase2 训练数据 7,681 条中 **49.5% 来自 `eval/results/drbench/TRASH` 的 30 个 run**、38% 来自
  `drgym_local232_r1`;heldout 46 条中 43 条来自 TRASH。这两处原始 harvest 都已进 HF dataset
  (`drbench_trash.tar.zst` / `benchmarks_results.tar.zst`),删本地前先确认 HF 可下。
- 数据转换脚本 [train/Eagle3/convert_harvest_to_eagle3.py](train/Eagle3/convert_harvest_to_eagle3.py)
  全机唯一副本,已 force-add 进本仓库(原被 `.gitignore` 的 `train/` 整目录排除)。
- DRGym 评测的题集对齐文件(40/160 题 id 与实验产物严格对齐,来源不可考,丢了对不上):
  `eval/DeepResearchGym/queries/{researchy_queries_sample_doc_click_100.jsonl,gym160_cached_notrun.jsonl}`
  已 force-add;KPR gold(`eval/DeepResearchGym/key_point/` 2000 个 json)在 HF dataset
  `benchmark_gold.tar.zst`。

## 评委配置真值(三处曾不一致,以此为准)

blend-router-e2e 复评实际用的评委(`~/tmp/blend_e2e/judge.env`,tar 内有脱敏副本):
`JUDGE_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai`、
`JUDGE_MODEL=gemini-3.1-flash-lite`;embedding 检索用 `gemini-embedding-001`。
(仓库 example.env 里的 dashscope+deepseek 是旧默认,不是实验用的。)

## HF 备份仓库

- 权重+二期数据:https://huggingface.co/julyanghar/Efficient-DRAgent
- 大数据(私有):https://huggingface.co/datasets/julyanghar/Efficient-DRAgent-data
  (search_cache 三 tar、benchmark 结果、TRASH/saved 存档、benchmark_gold、
  modify-code-runs 物证、tmp 物证、LMCache ignored 物证;逐 tar sha256 见 `tars/SHA256SUMS`)
