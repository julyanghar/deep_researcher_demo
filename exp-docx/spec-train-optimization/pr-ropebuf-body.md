<!--
PR: julyanghar/specforge-yilin:pr-ropebuf  ->  sgl-project/SpecForge:main
Title: fix: rebuild non-persistent rotary buffers after from_pretrained (warm-start loss=NaN)
以下为 GitHub PR 正文，直接复制粘贴（本注释块不用贴）。
-->

## Motivation

Warm-starting draft training from a checkpoint (e.g. `--ckpt-dir`, or `--resume` from a saved run) immediately diverges to `loss=NaN`.

Root cause: the rotary embedding buffers `inv_freq`, `cos_cached`, and `sin_cached` are registered with `persistent=False` (in `specforge/modeling/draft/llama3_eagle.py`), so they are **not** stored in the checkpoint `state_dict`. When the draft model is loaded via `AutoEagle3DraftModel.from_pretrained(...)`, transformers' meta-device / `low_cpu_mem_usage` loading path leaves these buffers **uninitialized** — NaN once the model is moved to GPU — and every RoPE application produces NaN, so training diverges from the first step. The cold-start path (`from_config`) constructs the rotary module normally and is unaffected.

## Modifications

- `scripts/train_eagle3.py` (`build_draft_model`): after `from_pretrained` and before `.cuda()`, re-run `_init_rope()` on every module that owns a rotary embedding, which rebuilds the non-persistent buffers. `_init_rope` already exists, so no model-code change is needed. One added log line, guarded to print once.
- `tests/test_modeling/test_draft/test_llama3.py`: two regression tests —
  - `test_rotary_buffers_absent_from_state_dict` documents the root cause (the three buffers are absent from `state_dict`).
  - `test_rotary_buffers_rebuilt_after_from_pretrained` does a real `save_pretrained` → `from_pretrained` round-trip, applies the `_init_rope` rebuild, and asserts the buffers are finite and identical to a freshly-initialized reference. (Equality is asserted rather than observing NaN, because CPU-only CI does not reliably reproduce the uninitialized-memory NaN the fix prevents on GPU.)

## Related Issues

None known. (Reproducible on any warm start / resume of a draft model via `from_pretrained`.)

## Accuracy Test

Not applicable to inference accuracy — this changes the training script (`build_draft_model`), not model / kernel / architecture code. The behavioral effect is a correctness fix: warm-start training goes from `loss=NaN` (unusable) to normal convergence. The added unit test verifies the rotary buffers are finite and correct after a checkpoint round-trip.

## Benchmark & Profiling

Not applicable. The fix is a one-time buffer rebuild at model-load time (before training starts); it has no steady-state throughput or latency impact.

## Checklist

- [x] Format your code according to the [Code Formatting with Pre-Commit](https://docs.sglang.ai/references/contribution_guide.html#code-formatting-with-pre-commit).
- [x] Add unit tests as outlined in the [Running Unit Tests](https://docs.sglang.ai/references/contribution_guide.html#running-unit-tests-adding-to-ci).
- [ ] Update documentation / docstrings / example tutorials as needed, according to [Writing Documentation](https://docs.sglang.ai/references/contribution_guide.html#writing-documentation-running-docs-ci).
- [ ] Provide throughput / latency benchmark results and accuracy evaluation results as needed, according to [Benchmark and Profiling](https://docs.sglang.ai/references/benchmark_and_profiling.html) and [Accuracy Results](https://docs.sglang.ai/references/accuracy_evaluation.html).
- [ ] For reviewers: If you haven't made any contributions to this PR and are only assisting with merging the main branch, please remove yourself as a co-author when merging the PR.
- [ ] Please feel free to join our Slack channel at https://sgl-fru7574.slack.com/archives/C09784E3EN6 to discuss your PR.
