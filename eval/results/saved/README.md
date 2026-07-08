# eval/results/saved/ 早期归档

suffix 投机解码实验之前(约 2026 年 6 月上旬)的运行归档,多为 KV 复用 / blend / native 的早期对比
和规模化跑批。留档参考,不是当前主线;需要清理磁盘时优先从这里回收。

| 子目录 | 大致内容 |
|---|---|
| `ab32b_native_10` / `ab_blend_10` | Qwen3-32B native vs blend 的 10 题 A/B 对比 |
| `ab_meta` | A/B 元数据 + gpu_monitor.csv + 内嵌更早的 RUN_32b_v1 等(含自己的 README) |
| `deepsearchqa_50_qwen` / `deepsearchqa_50_qwen_flash_new` | DeepSearchQA 50 题评测(qwen / qwen-flash 评委);含 metrics.json / predictions.jsonl / failures.jsonl |
| `RUN_32b_v1` / `RUN_32b_vs_30b` | 32B 单跑 / 32B vs 30B 对比 |
| `RUN_3way` / `RUN_3way_small` | 三路对比(orig / p1 / p15 …) |
| `RUN_real50` / `RUN_real50_big` / `RUN_real50_normal` | real 50 题不同规模档 |
| `RUN_pipeline_0615` / `RUN_sync_0614` | 0615 流水线 / 0614 同步跑批 |
| `search_cache_pure` / `search_cache_van` | 搜索缓存的 pure / vanilla 变体 |

> 这些早于当前 suffix/税/检索主线,细节不再逐一考据;如需精确口径查对应日期的 `exp-docx` 或 `~/tmp` 旧日志。
