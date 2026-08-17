# phase2-12k-3ep:二期域训 EAGLE-3 权重 + 训练数据(自包含)

二期最终产出:Qwen3-32B 的 summary 域训 EAGLE-3 draft head,TP4@12288 + 双裁剪,3 epoch。

## 内容

| 文件 | 说明 |
|---|---|
| `model.safetensors` | 权重本体(1.4G,= runs/qwen3-32b-summary-12k-trim-r1/epoch_2_step_5760) |
| `config.json` | 训练时的 draft config |
| `config_deploy.json` | 部署用 config(= AngelSlim config,避开新 transformers rope 解析差异) |
| `training_state.pt` | 优化器/scheduler 状态(续训用) |
| `data/train_main.jsonl` | **训练数据**,7,681 条(指纹 3a1078d2,zh 3935/en 3725/mix 21,≤12288 token) |
| `data/heldout.jsonl` | 评测集(46 条,15 个 drbench topic,题面隔离,从未进训练) |
| `data/heldout_topics_locked.json` | 锁定的 heldout topic 清单(保证一期/二期评测可比) |
| `data/manifest.json` | 数据统计 |

## 谱系

- **起点**:AngelSlim/Qwen3-32B_eagle3(腾讯开源第三方 head,warm-start,非本产出);
- **一期**:TP4@8192,3,607 条,2 epoch(见 runs/qwen3-32b-summary-8k-tp4-r1);
- **二期(本目录)**:TP4@12288,7,681 条(监督 ×2.08),3 epoch,双裁剪省 7.05GiB/卡解锁 12288。

## 训练配置(复现)

```
--target-model-path /data/yilin/huggingface/Qwen3-32B
--ckpt-dir /data/yilin/huggingface/Qwen3-32B-eagle3-angelslim   # warm-start
--train-data-path data/train_main.jsonl
--chat-template qwen3-instruct
--target-model-backend sglang --tp-size 4 --sglang-mem-fraction-static 0.44
--max-length 12288 --ttt-length 3 --attention-backend flex_attention
--shard-target-output --trim-loss-positions --trim-prompt-rows   # 双裁剪
--learning-rate 2e-5 --num-epochs 3 --warmup-ratio 0.05 --max-grad-norm 0.5
--draft-accumulation-steps 16 --save-interval 120
```

训练框架:SpecForge 本地 fork(四补丁 ropebuf/chunk-acc/nocompile/ckpt-norm + 位置维裁剪 A+B-i),见 exp-docx/eagle-spec-decode/phase2-trimming-work.md。

## 部署(vLLM 零转换)

```
vllm serve /data/yilin/huggingface/Qwen3-32B --tensor-parallel-size 4 \
  --speculative-config '{"method":"eagle3","model":"<本目录>","num_speculative_tokens":3}'
```
(部署时用 config_deploy.json 替换 config.json,或直接指向本目录——已含可加载 config)

## 训练指标

逐 epoch 训练 acc:0.44 → 0.47 → 0.48;全程 0 OOM。

## 评测结果

见 exp-docx/paper-submission/experiment-data-summary.md。核心:DRGym 英文长 prompt 上干净 held-out **1.303×**(碾压 suffix 0.976× 净减速);DRBench 中文短 prompt 上 1.11%/18.4%(输 suffix)。
