# P2 — attention is a 4× LONG-CONTEXT lever (not ~8%); the cheap fix is ruled out, a flash kernel is needed

Date: 2026-06-15. P2 was scoped as "fused attention, ~8% of the decode token." Profiling at real context
lengths reframes it entirely.

## The real finding: decode collapses with context
| context | tok/s |
|--------:|------:|
| ~8      | 54.5  |
| ~1024   | 27.6  |
| ~3072   | 13.7  |

**Decode is 4× slower at 3072 context than at 8** — the attention dominates as context grows. The
short-context benchmark (where all the arc's tok/s numbers were measured) *masks* this: it's the largest
real-world decode lever, far bigger than the ~8% the short-context profile suggested.

## It is NOT the GQA expansion (cheap fix ruled out, measured)
tinygrad's SDPA does `repeat_interleave` to expand 8 kv-heads → 32 (the GQA broadcast). I implemented a
GQA-native attention (group queries by kv head, broadcast k/v — no expansion), gated `GQA_ATTN`:
- **Correct**: byte-identical tokens to SDPA.
- **Not faster**: ctx8 53.5, ctx1024 25.6, ctx3072 12.3 — equal-to-slightly-slower than SDPA. **Default off.**
So the GQA expansion is handled efficiently (broadcast, not materialized); it is not the bottleneck.

## It is kernel inefficiency (needs a fused flash kernel)
At ctx 3072 the added cost is ~52 ms/token ≈ 1.4 ms/layer for: ~25M MACs (q@k + softmax + @v) and ~12 MB
KV read. At 60 TFLOPS that compute is ~0.4 µs; at 400 GB/s that read is ~31 µs. So 1.4 ms/layer is **neither
compute- nor memory-bound** — it is the SDPA decomposition (materialized [32, Tc] scores + a softmax pass +
separate @v, as several small low-occupancy kernels per layer) plus launch overhead. The fix is a single
**fused flash-attention decode kernel** (online softmax, no materialized scores, GQA-aware, occupancy-tuned)
replacing the ~5 SDPA sub-kernels per layer. That is a real build (correctness-critical: online softmax +
causal mask + GQA), not a reshape.

## Status
- **Lever identified and quantified**: long-context attention, 4× slowdown at 3072 — the biggest remaining
  real-world decode lever (the short-context arc numbers don't show it).
- **Cheap fix ruled out**: GQA-native reshape is exact but not faster (gated `GQA_ATTN`, default off).
- **Remaining**: a fused flash-attention decode kernel — substantial, the right next build for long-context.

## Honest note
All the arc's headline tok/s (23 → 60.9) are at SHORT context, where the GEMV weight-read dominates. At long
context, attention dominates and is 4× inefficient — so the practical decode speed for long prompts is gated
by P2, not by the GEMV work the rest of the arc optimized. This is the single most impactful remaining lever
for real usage, and it needs the flash kernel.

Repro: `/tmp/p2.py`-style context sweep (`generate` at increasing prompt lengths, decode tok/s); `GQA_ATTN=1`
for the exact-but-not-faster GQA-native variant.
