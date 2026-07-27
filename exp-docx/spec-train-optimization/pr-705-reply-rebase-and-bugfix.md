<!--
#705 回复(v2 已 force-push 到 pr-trim-a,commit ba20730)。
整段贴到 #705 评论区（本注释块不用贴）。
-->

Rebased onto current `main` (force-push) — and while doing so I found and fixed a real bug in my own patch. Details below.

## The rebase

The unified-runtime refactor moved both files this PR touched: `specforge/core/eagle3.py` → `specforge/algorithms/eagle3/model.py`, and `scripts/train_eagle3.py` was removed, so the CLI flag is now a `TrainingConfig` field (`training.trim_loss_positions`) wired through `model_providers.build_eagle3_model`.

## The bug I found while porting

The original patch assumed the supervised-row set is shared across all TTT steps, with the teacher sliding (row `p` fixed, teacher at `p + j`). That is backwards.

The loop applies `padding(..., left=False)` to `position_mask` / `loss_mask` once per step, so at step `j` the mask seen at row `p` is `mask[p + j]`, while `step_view` slices the padded teacher so row `p` is supervised by the teacher at `p + j`. A row therefore contributes at step `j` iff `p + j` is supervised — i.e.

```
step j:  rows = {s - j : s in sup, s >= j},  teacher/mask pinned at those s
```

The **rows** shift down by one per step; the teacher positions stay put. My original code had it inverted (rows fixed, teacher sliding), which is correct at step 0 and wrong from step 1 onward.

Measured on a small fixture, old code vs the full-length path:

| mask | bf16 max rel. err | fp32 max rel. err |
|---|--:|--:|
| all-supervised | 25% | 25% |
| half-masked | 9.0e-3 | 9.2e-3 |

fp32 and bf16 give the *same* error, so it was a semantic bug, not numerical noise. (Step 0 matched exactly in every case, which is the signature of the per-step shift.)

Why my earlier checks missed it: my equivalence self-check compared the trimmed path against a reference that used the *unshifted* full-length mask — the reference shared the same wrong assumption, so both sides agreed. A self-check whose reference shares the bug can't detect it. The end-to-end test I had did compare against the real full path, but on data where the shift only moved a contiguous block's boundary by one position, so the difference stayed under the tolerance.

## After the fix

Rows now follow `sup - j` and the teacher is evaluated once at `sup` (no sliding window needed, so the pack is simpler than before). Verified against the full-length path:

| mask | bf16 | fp32 |
|---|--:|--:|
| all-supervised | 0.0 | 0.0 |
| half-masked (prompt-heavy) | 1.1e-7 | 1.1e-7 |

Plus 27 existing eagle3 runtime tests and 50 config tests pass; the off path (flag defaults to off) is unchanged.

On the memory numbers I quoted earlier: those were measured on the pre-refactor code path, with the old row selection. The fix changes *which* rows are selected, not how many — the old version used `n_sup` rows at every step, the new one uses `|{s in sup : s >= j}|`, which is at most `n_sup` and differs by at most `k` entries — so the memory characteristics are unchanged and the reported saving still holds (if anything it is now marginally conservative). Happy to re-measure on the unified runtime if you'd like a number from the current code path.

Sorry for shipping that first version. Flagging it now rather than after you'd spent review time on it.
