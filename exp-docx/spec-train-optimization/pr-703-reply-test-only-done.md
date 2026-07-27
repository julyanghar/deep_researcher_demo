<!--
回复 PR #703：确认已按方案(2)"留测试"落地并推送。
背景：maintainer 让"修 conflict + 给 loss=NaN 例子"→ 上一条评论已解释 conflict 是 modify/delete、
重构已从结构上修掉 bug、附了 NaN 复现，并给出方案(1)关/(2)只留测试。用户选(2)。
本条=后续跟进，宣告 test-only 版已推。

⚠️ 对抗性事实核查修订记录（wf_5bf42321）：
   原稿有一句 overclaim——"这两个测试会在有人把 live model 换回 from_pretrained 时失败"。
   实际两个测试都只用直接构造函数、根本不碰 from_pretrained/训练 runtime，
   runtime 回退到 from_pretrained-as-live 时它们照样绿。已改成如实说明：
   测试锁的是两个"属性级不变量"(persistent=False + load 不污染)，真正防 bug 的是
   runtime 设计(from_config + strict=False)——两回事，不能混。

⚠️ 语气假设：本稿写成"接着上一条 options 评论的跟进"。若你还没发那条 options 评论，
   告诉我，我改成自包含版本。

下面整段贴到 #703 评论区（本注释块不用贴）。
-->

Following up — I've gone with option (2): the PR is now just the regression tests, with the obsolete fix dropped.

`scripts/train_eagle3.py` no longer exists on `main` (removed in the unified-runtime refactor), so the fix has nowhere left to live — and that same refactor is what makes the bug unreachable now: warm-start builds the live model with `from_config` and only ever loads weights via `load_state_dict(strict=False)`, so the `from_pretrained`-as-the-live-model path that caused it is gone. (I walked through this in the previous comment.)

So the diff is now a single file — `tests/test_modeling/test_draft/test_llama3.py`, `+56` lines — pinning the two properties that safe path relies on:

- **`test_rotary_buffers_absent_from_state_dict`** — the rotary buffers (`inv_freq`, `cos_cached`, `sin_cached`) are registered `persistent=False`, so a checkpoint's `state_dict()` never contains them. Fails if anyone ever flips them to `persistent=True`.
- **`test_warm_start_preserves_rotary_buffers`** — build the model normally (so `_init_rope` gives it valid buffers), then load a buffer-less checkpoint with `load_state_dict(strict=False)`; asserts the live model's rotary buffers stay finite and unchanged. Fails if the load path ever starts clobbering them.

These are unit-level — they construct the model directly rather than driving the full training runtime, so they guard the invariant rather than integration-testing the warm-start path. But they lock down exactly the two things that, if either silently regressed, would bring back the uninitialized-buffer `loss=NaN`.

Ready to merge — or to close if you'd rather not carry the tests. Either's fine with me; thanks for the review!
