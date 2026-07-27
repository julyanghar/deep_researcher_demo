<!--
B 级 RFC 草稿 —— 发成 sgl-project/SpecForge 的 GitHub Issue(不是 PR）。
目的：在投 backbone-internal 裁剪的代码 PR 之前，先探 maintainer 口风。
把 <A-PR#> 替换成你 A 级 PR 的实际编号。以下从标题下面开始贴。
-->

# [RFC] Backbone/TTT-internal position trimming for online EAGLE3 training

## TL;DR

Follow-up to the loss-end trimming PR (#<A-PR#>, `--trim-loss-positions`). That PR trims only the loss end; the draft **backbone still runs full-length** at every TTT step. This RFC proposes pushing loss-mask-aware trimming **into the draft backbone and the TTT unroll**, so that training compute and activation scale with the number of supervised tokens instead of the full sequence length. It is equivalence-preserving and orthogonal to gradient checkpointing (#669). Before writing the code PR, I'd like to know whether maintainers are open to backbone-internal changes here.

## Motivation

On prompt-heavy training data (agent/RAG traces, long-context SFT) the supervised (loss-masked) tokens are a small minority — in our summary-domain data ~92% of positions are masked out of the loss. Loss-end trimming (#<A-PR#>) removes the vocab-wide logits/loss on those positions, but the draft backbone still does full-length attention + MLP for all of them across all TTT steps, even though the masked positions' outputs never reach a live gradient.

## The idea

Two levels, both gated behind flags and defaulting off, building on `--trim-loss-positions`:

- **B-i (`--trim-prompt-rows`)**: in TTT steps 2..k, forward only the supervised **rows**. The masked (prompt) rows are needed only as step-1 KV context, so they are dropped from steps 2..k entirely (compact attention mask + absolute RoPE positions).
- **B-ii (`--trim-step1`)**: in step 1, the masked rows compute only their **K/V** (which supervised rows and later steps attend to) and skip their `q`/attention-output/`o_proj`/MLP.

### Why it's correct

Under EAGLE3's TTT mask, cross-position attention reads only step-1's K/V; steps 2+ are diagonal-private. So a masked row's step-1 attention-output / MLP, and its entire presence in steps 2..k, terminate in the loss-masked (zeroed) loss — they are autograd dead-ends, contributing exactly zero gradient. Their K/V, however, **are** attended by supervised rows, so those stay full-length. Because RMSNorm is row-independent, computing K/V from the full-length normed input and `q` from its supervised-row subset is bit-identical (up to bf16) to the full step.

We verified equivalence with an in-place dual-compute check (same run, same weights, fp64): worst relative loss error 2.6e-3 (< 5e-3), and per-step loss with the flags on vs off matches.

## Measured impact (Qwen3-32B, TP4, `max_length=8192`, ~92% masked)

Peak training activation, `torch.max_memory_allocated`, per card:

| config | peak | vs untrimmed |
|---|--:|--:|
| untrimmed | 39.1 GiB | — |
| A (`--trim-loss-positions`) | 36.9 GiB | −2.3 |
| A + B-i (`--trim-prompt-rows`) | 33.6 GiB | **−5.5** |

B-i cuts the peak by a further ~3.3 GiB on top of A. B-ii additionally saves ~2.7 GiB/card of **forward-resident** activation at `max_length=12288` (it does not lower the backward peak — that is the recompute axis, see below). Together, A+B-i let us run `max_length=16384` domain training on a single 4×48GB node without USP/offline, which previously OOM'd.

## Relationship to existing work

- **Loss-end trimming (#<A-PR#>)**: this RFC is the backbone-internal continuation of it.
- **`--draft-gradient-checkpointing` (#669)**: orthogonal axis. Checkpointing *recomputes* activations in the backward pass; this *eliminates* the compute for masked positions in the forward pass. They stack — and our measurements show the position axis alone does not lower the backward peak, so the two are complementary rather than competing.
- **Position chunking (#673)**: chunks the peak but keeps total compute; this reduces total compute.

## What I'm asking

1. Are maintainers open to backbone-internal position trimming in the online TTT loop (behind a default-off flag), or is loss-end trimming the preferred boundary?
2. If open: should B build directly on the A-level PR (stacked), and would you prefer B-i and B-ii as one PR or two?
3. Any concerns about the compact flex-attention mask path or the TTT-mask assumptions above that we should address up front?

I have a working, equivalence-tested implementation and can open the PR once there's a direction. Thanks!
