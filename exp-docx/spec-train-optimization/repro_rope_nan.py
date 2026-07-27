"""Minimal repro: what happens to non-persistent rotary buffers across a
save_pretrained -> from_pretrained round-trip, on GPU.

Run: python repro_rope_nan.py
"""

import tempfile

import torch
from transformers import LlamaConfig

from specforge.modeling.draft.llama3_eagle import LlamaForCausalLMEagle3

cfg = LlamaConfig(
    hidden_size=1024,
    intermediate_size=2048,
    num_hidden_layers=1,
    num_attention_heads=8,
    num_key_value_heads=2,
    vocab_size=4096,
    draft_vocab_size=1024,
    max_position_embeddings=2048,
    rms_norm_eps=1e-5,
    tie_word_embeddings=False,
)

d = tempfile.mkdtemp()

# 1) reference model built the normal way (__init__ runs _init_rope)
ref = LlamaForCausalLMEagle3(cfg)
ref_rot = ref.midlayer.self_attn.rotary_emb
print("=== reference (from_config / __init__) ===")
for n in ("inv_freq", "cos_cached", "sin_cached"):
    t = getattr(ref_rot, n)
    print(f"  {n}: finite={torch.isfinite(t).all().item()} cos(0)-ish={t.flatten()[0].item():.6f}")

# 2) what actually lands in the checkpoint
sd = ref.state_dict()
present = [k for k in sd if any(k.endswith(b) for b in ("inv_freq", "cos_cached", "sin_cached"))]
print(f"\n=== checkpoint state_dict ===\n  rotary buffers present in state_dict: {present}  (persistent=False -> not saved)")

ref.save_pretrained(d)

# 3) the meta-device / low_cpu_mem_usage load path, then .cuda()
loaded = LlamaForCausalLMEagle3.from_pretrained(d)
loaded = loaded.cuda()
got_rot = loaded.midlayer.self_attn.rotary_emb
print("\n=== after from_pretrained(...).cuda()  [the OLD live-model path] ===")
bad = False
for n in ("inv_freq", "cos_cached", "sin_cached"):
    t = getattr(got_rot, n)
    fin = torch.isfinite(t).all().item()
    same = torch.allclose(t.float().cpu(), getattr(ref_rot, n).float().cpu())
    print(f"  {n}: finite={fin} matches_reference={same} first={t.flatten()[0].item()}")
    if not fin or not same:
        bad = True

# 4) does it poison a forward?
x_emb = torch.randn(1, 8, cfg.hidden_size, device="cuda", dtype=torch.float32)
x_hid = torch.randn(1, 8, cfg.hidden_size * 3, device="cuda", dtype=torch.float32)
attn = torch.ones(1, 8, device="cuda")
with torch.no_grad():
    out = loaded(inputs_embeds=x_emb, hidden_states=x_hid, attention_mask=attn)
print(f"\n  forward output finite={torch.isfinite(out).all().item()}  nan_count={torch.isnan(out).sum().item()}")

# 5) the NaN path: any sequence longer than the (bogus) cached length makes the
#    rotary module recompute cos/sin FROM the corrupted inv_freq.
print("\n=== forcing a cos/sin cache rebuild from the corrupted inv_freq ===")
long_len = int(got_rot.max_seq_len_cached) + 16
dummy = torch.zeros(1, 1, long_len, cfg.hidden_size // 8, device="cuda")
with torch.no_grad():
    cos, sin = got_rot(dummy, seq_len=long_len)
print(f"  after rebuild: cos finite={torch.isfinite(cos).all().item()} "
      f"nan={torch.isnan(cos).sum().item()}/{cos.numel()}")
print(f"                 sin finite={torch.isfinite(sin).all().item()} "
      f"nan={torch.isnan(sin).sum().item()}/{sin.numel()}")

# 6) and what a loss computed on top of that looks like
with torch.no_grad():
    out2 = loaded(inputs_embeds=x_emb, hidden_states=x_hid, attention_mask=attn)
    logits = loaded.compute_logits(out2)
    loss = logits.float().log_softmax(-1).mean()
print(f"\n  forward after rebuild: finite={torch.isfinite(out2).all().item()} "
      f"nan={torch.isnan(out2).sum().item()}")
print(f"  toy loss on those logits = {loss.item()}")

print(f"\n=== VERDICT: rotary buffers broken after from_pretrained? {bad} ===")
