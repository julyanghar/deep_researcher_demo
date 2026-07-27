# PR 草稿:ropebuf bugfix

- 分支:`julyanghar/specforge-yilin:pr-ropebuf` → `sgl-project/SpecForge:main`
- 开 PR 链接:https://github.com/julyanghar/specforge-yilin/pull/new/pr-ropebuf
- 净改动:2 文件(scripts/train_eagle3.py +18 / tests +51),本地 8 测试全过
- 定位:纯 bugfix、零新颖性争议、独立(和 A/B 无重叠)

---

## Title

`fix: rebuild non-persistent rotary buffers after from_pretrained (warm-start loss=NaN)`

## Body(直接贴到 GitHub PR)

### Problem

Warm-starting draft training from a checkpoint (e.g. `--ckpt-dir`, or `--resume` from a saved run) immediately diverges to `loss=NaN`.

Root cause: the rotary embedding buffers `inv_freq`, `cos_cached`, `sin_cached` are registered with `persistent=False` (see `specforge/modeling/draft/llama3_eagle.py`), so they are **not** part of the checkpoint `state_dict`. When the draft model is loaded via `AutoEagle3DraftModel.from_pretrained(...)`, transformers' meta-device / `low_cpu_mem_usage` loading path leaves these buffers **uninitialized** — NaN once the model is moved to GPU — and every forward that uses RoPE produces NaN.

This only affects the checkpoint-load path (`from_pretrained`); cold starts via `from_config` construct the rotary module normally and are unaffected.

### Fix

In `build_draft_model`, after `from_pretrained` and before `.cuda()`, re-run `_init_rope()` on every module that owns a rotary embedding, which rebuilds the non-persistent buffers:

```python
draft_model = AutoEagle3DraftModel.from_pretrained(...)
for module in draft_model.modules():
    if hasattr(module, "_init_rope"):
        module._init_rope()
draft_model = draft_model.cuda()
```

One file change; `_init_rope` already exists, so no model-code change is needed.

### Test

`tests/test_modeling/test_draft/test_llama3.py` adds two tests:
- `test_rotary_buffers_absent_from_state_dict` — documents the root cause: the three buffers are absent from `state_dict` (persistent=False).
- `test_rotary_buffers_rebuilt_after_from_pretrained` — real `save_pretrained` → `from_pretrained` round-trip, then the `_init_rope` rebuild; asserts the buffers are finite and identical to a freshly-initialized reference. (Equality is asserted rather than a NaN check, since CPU-only CI does not reliably reproduce the uninitialized-memory NaN that the fix prevents on GPU.)

### Notes

Independent of any feature work; safe to merge on its own.
