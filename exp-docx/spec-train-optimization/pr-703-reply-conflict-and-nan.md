<!--
回复 PR #703 的 maintainer 提问("help fix the conflicts" + "give an example that can occur loss=NaN")。
以下整段贴到 #703 评论区（本注释块不用贴）。
复现脚本单独附:repro_rope_nan.py（可作为 gist 或直接贴代码块）。
-->

Thanks for looking at this!

**On the conflict** — it isn't a textual conflict I can rebase away: this PR patches `scripts/train_eagle3.py`, which was removed in d5883d5 ("refactor: consolidate training on unified runtime"). So the fix as written no longer has a home.

**More importantly, I believe that refactor already makes this bug impossible**, so this PR may simply be obsolete. On current `main`:

- `build_eagle3_draft` creates the live model with `AutoDraftModel.from_config(...)`, so `__init__` runs `_init_rope()` and the rotary buffers are correctly initialized.
- `warm_start_draft_model` only does `model.load_state_dict(state, strict=False)`, and `state` comes from a *temporary* `from_pretrained` model whose `state_dict()` — precisely because these buffers are `persistent=False` — contains no rotary buffers at all. That temporary model is then `del`'d.

So the live model's correctly-initialized buffers are never overwritten. The old code was vulnerable only because the meta-loaded model **was** the live model.

**The NaN example you asked for.** This exercises the old path (a `from_pretrained` model used directly as the live model). Output from a single GPU run:

```
=== reference (from_config / __init__) ===
  inv_freq: finite=True    cos_cached[0]=1.000000    sin_cached[0]=0.000000

=== checkpoint state_dict ===
  rotary buffers present: []          # persistent=False -> never saved

=== after from_pretrained(...).cuda() ===
  inv_freq:    finite=True  matches_reference=False  first=2.8698592549372254e-42
  cos_cached:  all zeros    (should be cos(0)=1)
  sin_cached:  all zeros

=== forcing a cos/sin rebuild from the corrupted inv_freq ===
  (any sequence longer than the bogus cached length triggers _set_cos_sin_cache)
  cos: finite=False  nan=4164/266752
  sin: finite=False  nan=4164/266752
  forward output: finite=False  nan=8192
  loss = nan
```

Note that `inv_freq` came back as `-1.13e+27` on one run and `2.87e-42` on the next — it is uninitialized memory, so the symptom is nondeterministic: sometimes the garbage looks benign (RoPE silently degenerates to cos=0 / sin=0, i.e. positional information is destroyed but nothing crashes), and sometimes it produces outright NaN once the cache is rebuilt for a longer sequence. That nondeterminism is what made this hard to track down in the first place.

Repro script:

<details>
<summary><code>repro_rope_nan.py</code></summary>

```python
import tempfile

import torch
from transformers import LlamaConfig

from specforge.modeling.draft.llama3_eagle import LlamaForCausalLMEagle3

cfg = LlamaConfig(
    hidden_size=1024, intermediate_size=2048, num_hidden_layers=1,
    num_attention_heads=8, num_key_value_heads=2, vocab_size=4096,
    draft_vocab_size=1024, max_position_embeddings=2048,
    rms_norm_eps=1e-5, tie_word_embeddings=False,
)
d = tempfile.mkdtemp()

ref = LlamaForCausalLMEagle3(cfg)
ref_rot = ref.midlayer.self_attn.rotary_emb
for n in ("inv_freq", "cos_cached", "sin_cached"):
    t = getattr(ref_rot, n)
    print(f"ref {n}: finite={torch.isfinite(t).all().item()} first={t.flatten()[0].item():.6f}")

sd = ref.state_dict()
print("rotary buffers in state_dict:",
      [k for k in sd if any(k.endswith(b) for b in ("inv_freq", "cos_cached", "sin_cached"))])

ref.save_pretrained(d)
loaded = LlamaForCausalLMEagle3.from_pretrained(d).cuda()
got_rot = loaded.midlayer.self_attn.rotary_emb
for n in ("inv_freq", "cos_cached", "sin_cached"):
    t = getattr(got_rot, n)
    print(f"loaded {n}: finite={torch.isfinite(t).all().item()} "
          f"matches_ref={torch.allclose(t.float().cpu(), getattr(ref_rot, n).float().cpu())} "
          f"first={t.flatten()[0].item()}")

# force a cos/sin cache rebuild from the corrupted inv_freq
long_len = int(got_rot.max_seq_len_cached) + 16
dummy = torch.zeros(1, 1, long_len, cfg.hidden_size // 8, device="cuda")
with torch.no_grad():
    cos, sin = got_rot(dummy, seq_len=long_len)
print(f"after rebuild: cos finite={torch.isfinite(cos).all().item()} nan={torch.isnan(cos).sum().item()}")

x_emb = torch.randn(1, 8, cfg.hidden_size, device="cuda")
x_hid = torch.randn(1, 8, cfg.hidden_size * 3, device="cuda")
attn = torch.ones(1, 8, device="cuda")
with torch.no_grad():
    out = loaded(inputs_embeds=x_emb, hidden_states=x_hid, attention_mask=attn)
    loss = loaded.compute_logits(out).float().log_softmax(-1).mean()
print(f"forward finite={torch.isfinite(out).all().item()}  loss={loss.item()}")
```

</details>

**How would you like to proceed?** As I see it:

1. Close this PR as obsoleted by the refactor.
2. Keep only the regression test, retargeted at the invariant that actually makes the new design safe: that the rotary buffers are absent from `state_dict()`, and that a warm start leaves the live model's buffers intact. That would catch a future change that goes back to using a `from_pretrained` model as the live one.

I'm happy with either — (2) if you think that invariant is worth pinning down, (1) otherwise.
