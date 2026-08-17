<!--
回复 #705 maintainer(jiapingW)的 SP 兼容性要求。
状态:✅ 定稿(2026-08-01,benchmark 数字已回填)。
发布前提:先把 trim-usp-dev 的 SP-native 实现 push 到 pr-trim-a(等用户在 review-report 交接闸点头),再贴本回复。
证据链:~/modify-code-runs/eagle3-trim-usp/(blueprint + run-0..10 日志 + review-report.md)。
以下整段贴到 #705 评论区(本注释块不用贴)。
-->

Great call asking about sequence parallelism — I dug in, and the honest answer is: **as previously written, trim was NOT compatible with USP** (it crashed, and in configurations where it didn't crash it would have been silently wrong). I've now made it SP-native, verified equivalence on 4-rank USP including pure ring-4, and measured the gains. Details below.

## What was wrong

The trim branch bypassed the USP adapter's `step_view`, so with `attention_backend=usp` it fed the backbone this rank's **whole local buffer** (own chunk + the `ttt_length` overlap tail) instead of the chunk:

1. the backbone input length disagreed with the collator's `position_ids` (and with other ranks' ring/Ulysses block sizes) — on current code this fails fast with a shape error in `torch.cat` (verified on a 4-rank fixture);
2. supervised positions in the overlap tail would have emitted loss **rows** on this rank while also being the next rank's own rows — cross-rank double counting;
3. the loss was normalized by `local_len` instead of `usp_chunk_size`, breaking the per-rank-denominator convention that DDP + `accumulation_steps ×= sp_size` relies on.

And nothing guarded the combination — `usp + trim_loss_positions` silently took that path.

## The fix (SP-native trim)

The backbone now runs **exactly the full path's inputs** under USP (chunk-sliced hidden/input_ids/attention-mask, adapter-owned position_ids); trimming applies only at the teacher/logits/loss stage:

- overlap-tail positions may act as **teachers** for own-chunk rows at deeper TTT steps (that's what the tail is for), but never emit loss rows themselves: `rows_j = {s − j : s ∈ sup, j ≤ s, s − j < usp_chunk_size}`;
- the loss denominator is `usp_chunk_size`, matching the full path's mean-over-chunk semantics per rank;
- a step whose row set comes out empty (e.g. all local supervision sits in the tail at depth 0) is padded with one zero-masked dummy row, so it contributes exactly zero loss while kernel launches and collectives stay aligned across ranks — including the mixed case where some ranks trim and others (no local supervision) fall back to the full path.

Each (teacher-position, TTT-depth) loss term is computed exactly once globally.

## Equivalence evidence (4 ranks, real offline pipeline)

Per-step losses, trim ON vs OFF on identical inputs, via the actual offline strategy path (`TargetHead.preprocess` → collator with `use_usp_preprocess` → forward), on **both** `ulysses2 × ring2` and **pure `ring4`**, across four mask patterns chosen to be adversarial:

| mask pattern | why it's there | result |
|---|---|---|
| all positions supervised | trivial-equality boundary: trim must select every chunk row; any row/denominator error is naked, no tolerance can hide it | max diff ≤ 2.4e-7 (1–2 ulp fp32; the smallest possible discrete error is ~5 orders larger) |
| ~30% random supervised | realistic | ≤ 1e-3 rel (bf16 GEMM-shape noise) |
| supervision straddling every rank boundary | overlap-tail-as-teacher on every rank | pass |
| supervision only in one rank's overlap tail | dead steps + two ranks with zero supervision (mixed trim/full ranks — the collective-alignment hazard) | pass, no hangs |

Backward is covered too: total grad-norm trim vs full agrees within 5e-3 (calibrated against flash-attn's own backward nondeterminism floor). As a fixture sanity check, the mean over ranks of full-path per-step losses reproduces a single-GPU full-sequence reference on the un-sharded sample.

A negative control run — the same experiment against the previous trim code — fails immediately (the shape error above), so the test discriminates.

## Gains at ring 4 (4× RTX 6000 Ada 48 GB, Llama3-8B-shaped: hidden 4096, target vocab 128256, draft vocab 32000, TTT 7, ~30% supervised)

| global len | per-rank chunk | full peak GB/rank | trim peak GB/rank | saving | full step (s) | trim step (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 8k | 2k | 8.22 | 7.54 | 0.68 | 0.431 | 0.385 (−10.7%) |
| 16k | 4k | 13.60 | 12.17 | 1.43 | 0.939 | 0.813 (−13.4%) |
| 32k | 8k | 24.37 | 21.45 | 2.92 | 2.193 | 1.960 (−10.6%) |
| **64k** | **16k** | **OOM** (all 4 ranks) | **40.32** | — | OOM | 5.025 |

And the same sweep at **ring 8** (8 GPUs):

| global len | per-rank chunk | full peak GB/rank | trim peak GB/rank | saving | full step (s) | trim step (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 64k | 8k | 24.37 | 21.61 | 2.76 | 2.903 | 2.726 (−6.1%) |
| **128k** | **16k** | **OOM** (all 8 ranks) | **40.32** | — | OOM | 8.200 |

Takeaways:

- **Trim moves the trainable-context frontier on this hardware: ring4/64k and ring8/128k both OOM on the full path and both run under trim** (40.3 GB/rank in each case — identical per-rank chunk, which also cross-checks the sharding math: full at ring8/64k reproduces full at ring4/32k to the hundredth of a GB).
- The saving is additive with SP as expected (both attack the same `[seq × vocab]` activations along orthogonal axes) and scales linearly with length; trim is also consistently *faster* per step (~6–13%), since the teacher projection and draft lm_head run on ~30% of positions instead of all of them.

Equivalence was verified at world size 8 as well (pure ring-8 and ulysses2×ring4, same four masks — all pass), and beyond the differential tests the branch carries an oracle suite where expected outputs are derived independently of the implementation: hand-written literal row/keep tables for the per-step selection (including the overlap-tail bound and dead steps), an analytic closed form for the loss (zero logits + normalized teachers ⇒ each masked row contributes exactly `ln(draft_vocab)`), and a NumPy re-derivation of per-step row counts / teacher token ids / step losses straight from the raw feature files. The committed test file (`tests/test_runtime/test_equiv_trim_usp.py`) keeps the CPU-only golden-table layer running in CI even on hosts without four GPUs.

Caveats, stated plainly: these are 48 GB cards (not 80 GB), the benchmark is the raw module without FSDP (the delta we measure — vocab-sized loss-stage activations — is not sharded by FSDP anyway), and features are synthetic (memory/latency don't depend on tensor contents). The saving scales with `1 − supervised_fraction`, so prompt-heavy long-context data benefits more. One more disclosure: the full-path baseline runs with `compact_teacher` **off** (the default), and the two features are currently mutually exclusive (the compact branch takes precedence). `compact_teacher` chunks the teacher's full-vocab fp32 transient over the vocab axis, but the full-length draft logits (+ their grad copies) and the `[chunk × draft_vocab]` teacher tables stay whole — those are what trim removes — so trim's saving should remain largely additive; we have not measured the `full + compact_teacher` combination at 64k.

The branch also picks up two robustness fixes that fell out of the verification: a dead-step NaN edge on the single-GPU path (a mask whose supervision is unreachable at deep TTT steps used to produce a mean over zero rows), and `training.trim_loss_positions` is now documented in `examples/configs/README.md` (the recipe-README guard).

Happy to adjust scope if you'd rather land the single-GPU part first and the USP part as a follow-up.
