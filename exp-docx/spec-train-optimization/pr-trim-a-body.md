<!--
PR: julyanghar/SpecForge:pr-trim-a  ->  sgl-project/SpecForge:main
Title: feat: add --trim-loss-positions (loss-masked position trimming for online EAGLE3 training)
开 PR 链接: https://github.com/julyanghar/SpecForge/pull/new/pr-trim-a
以下为 GitHub PR 正文，直接复制粘贴（本注释块不用贴）。
-->

## Motivation

On prompt-heavy training data (agent traces, RAG, long-context SFT), most positions are masked out of the loss — only the assistant/supervised positions contribute. Yet online EAGLE3 training still materializes the vocab-wide teacher `target_p`, the draft `logits`, and the per-position loss over the **full** sequence at every TTT step, so the majority of that compute and memory is spent on positions that are multiplied by zero.

`--trim-loss-positions` computes the teacher `target_p`, the draft `logits`, and the loss only at supervised (loss-masked) positions. It is **mathematically equivalent** to the full-length path — the mean loss denominator is rescaled from `n_sup` back to the full length — and it saves memory and compute proportional to the masked fraction.

This is a port of the loss-end position-trimming idea (already present in TorchSpec) to SpecForge's TTT loop + precomputed teacher `target_p` pipeline. The genuinely SpecForge-specific parts are the sliding-window teacher-table compaction (`_build_trim_pack`) and the mean-denominator rescale. It only touches the **loss end**; the draft backbone still runs full-length here (backbone-internal trimming is future work, best discussed as an RFC first).

## Modifications

- `specforge/core/eagle3.py`:
  - `OnlineEagle3Model` gains a `trim_loss_positions` flag. When active (and eligible), it builds a compact teacher table over supervised rows + their k-step sliding window (`_build_trim_pack`), runs the backbone full-length, then gathers only supervised rows into `compute_logits` and `_acc_and_loss`.
  - `_acc_and_loss` gains `loss_scale` / `full_positions` so the trimmed mean is rescaled to the full-length semantics (`loss_scale = n_sup / full_len`).
  - New helpers `_build_trim_pack`, `_compute_target_p_eager`, `_compute_position_mask_at`.
  - Eligibility gate: online path with `batch==1`, `lk_loss_type is None`, at least one supervised position, and non-VLM; otherwise it falls back to the existing full-length path (byte-identical when off).
- `scripts/train_eagle3.py`: `--trim-loss-positions` CLI flag, wired into the `OnlineEagle3Model` construction.
- `tests/test_runtime/test_equiv_trim_loss_positions.py`: GPU equivalence test — the same online forward with trimming off vs. on must produce matching per-step losses (bf16 tolerance). GPU-only, matching the other online EAGLE3 equivalence tests in that directory.

No change to `llama3_eagle.py` / `flex_attention.py` (the backbone runs full-length under trimming).

## Related Issues

None known.

## Accuracy Test

This is an equivalence-preserving change, so it does not affect trained-model accuracy. Correctness is verified by `test_equiv_trim_loss_positions`: with a prompt-heavy loss mask, the per-step training losses with `--trim-loss-positions` on vs. off match within bf16 tolerance (relative error ≤ 5e-3). The off-path is byte-identical to `main` (the flag defaults to off).

## Benchmark & Profiling

The teacher `target_p` and draft `logits`/loss tensors are the largest per-step activations, all `∝ sequence_length × vocab`. Trimming makes them `∝ n_sup × vocab`, so the saving scales with the unsupervised fraction.

Measured on Qwen3-32B (TP4, `max_length=8192`, `flex_attention`, TTT=3) on prompt-heavy data where ~92% of positions are masked out of the loss (supervised fraction ≈ 3–18% per sample):

| metric (torch `max_memory_allocated`, per card) | off | `--trim-loss-positions` | saved |
|---|--:|--:|--:|
| peak training activation | 39.1–41.2 GiB | 36.5–38.0 GiB | **≈ 2.3–3.4 GiB/card** |

The saving is larger on more prompt-heavy data and at longer `max_length` (it is `∝ masked_positions`), and smaller on evenly-supervised data. Because it is a strict activation/compute reduction with equivalent output, it does not change inference latency/throughput. (This is a specific-config data point; the effect scales with the masked fraction, so a standard ShareGPT-shaped workload would show a proportionally smaller saving.)

## Checklist

- [x] Format your code according to the [Code Formatting with Pre-Commit](https://docs.sglang.ai/references/contribution_guide.html#code-formatting-with-pre-commit).
- [x] Add unit tests as outlined in the [Running Unit Tests](https://docs.sglang.ai/references/contribution_guide.html#running-unit-tests-adding-to-ci).
- [ ] Update documentation / docstrings / example tutorials as needed, according to [Writing Documentation](https://docs.sglang.ai/references/contribution_guide.html#writing-documentation-running-docs-ci).
- [ ] Provide throughput / latency benchmark results and accuracy evaluation results as needed, according to [Benchmark and Profiling](https://docs.sglang.ai/references/benchmark_and_profiling.html) and [Accuracy Results](https://docs.sglang.ai/references/accuracy_evaluation.html).
- [ ] For reviewers: If you haven't made any contributions to this PR and are only assisting with merging the main branch, please remove yourself as a co-author when merging the PR.
- [ ] Please feel free to join our Slack channel at https://sgl-fru7574.slack.com/archives/C09784E3EN6 to discuss your PR.
