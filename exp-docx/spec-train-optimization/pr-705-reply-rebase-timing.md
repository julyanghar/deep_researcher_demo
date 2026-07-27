<!--
回复 PR #705:上游 unified-runtime 重构把本 PR 碰的两个文件都删了。
以下整段贴到 #705 评论区（本注释块不用贴）。
-->

Heads-up: this PR now conflicts because the unified-runtime refactor moved both files it touches — `specforge/core/eagle3.py` (`OnlineEagle3Model` now lives in `specforge/algorithms/eagle3/model.py`) and `scripts/train_eagle3.py` (removed; training options are now `TrainingConfig` fields in `specforge/config/schema.py`).

The change itself is unaffected by the refactor — the code was moved, not redesigned, and the underlying issue is unchanged: on prompt-heavy data the teacher `target_p` / draft `logits` / loss are still materialized over the full sequence even though only supervised positions contribute. So the port is mechanical: the same insertion points exist in the new `algorithms/eagle3/model.py`, and the CLI flag becomes a `TrainingConfig` field.

I'm rebasing onto the new layout now and will push shortly.

One question on timing, since I'd rather not churn: the training entrypoint has moved twice recently (`train_eagle3.py` → dataflow launcher → unified runtime). Is the `algorithms/` + `config/schema.py` layout stable enough to target now, or would you prefer I wait until things settle? Happy to hold if another move is imminent.
