# 实验设置

← 返回 [experiment-data-summary.md](../experiment-data-summary.md)

## 模型与框架

| 项 | 配置 |
|---|---|
| Target | Qwen3-32B(TP4,bf16) |
| Draft | EAGLE-3,warm-start Tencent AngelSlim/Qwen3-32B_eagle3 |
| 训练框架 | SpecForge(online 模式,sglang target),本地 fork + 位置维裁剪 |
| 部署/评测 | vLLM 0.18,`num_speculative_tokens=3`,`--enforce-eager`,温度 0 |
| 评测方法 | heldout summary prompt replay(spec_decode_replay.py),逐题 TTFT+decode 计时 |

## 指标定义

- **接受率** = accepted / drafted tokens(vLLM `/metrics` 全局差分);
- **加速比** = vanilla / arm 逐题墙钟(total_s)中位,vanilla 为无投机绝对基准;
- **逐位置接受率** a1/a2/a3 = 第 1/2/3 个草稿 token 的接受率(vLLM SpecDecoding metrics)。

## 训练配置(两代)

| 代 | max_len | 数据 | 有效监督 | epoch | 显存/卡 |
|---|--:|--:|--:|--:|--:|
| 一期 | 8192 | 3,607 条 | 1.91M token | 2 | 基线 |
| 二期 | 12288 | 7,681 条(zh/en 均衡) | 3.98M token | 3 | 双裁剪省 7.05GiB,解锁 12288 |

裁剪细节见 [05-memory-efficiency.md](05-memory-efficiency.md)。

## 两个 workload(关键:形态差异)

| 数据集 | 语言 | 题数/样本 | prompt 中位长度 | 说明 |
|---|---|---|--:|---|
| **DRBench** | 中文 | 46 条 summary | ~11K 字符 | 块级检索,中文 |
| **DRGym** | 英文 | 672 条 summary(50 题×~13) | **~40K 字符**(≈10K token) | local top-10 检索,英文长 prompt |

DRGym 样本 prompt 是 DRBench 的 **~4 倍长**。两数据集**语言和长度两变量同时不同**(见 [06-caveats.md](06-caveats.md))。评测样本均为题面隔离的 held-out(DRGym 另含训练集内子集,已分层标记)。
