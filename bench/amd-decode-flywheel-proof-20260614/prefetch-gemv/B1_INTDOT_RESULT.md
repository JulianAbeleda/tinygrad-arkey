# B1/R1: in-graph int-dot is a DECISIVE NEGATIVE — the per-kernel GEMV is occupancy-bound, not compute-bound

Date: 2026-06-15. Hypothesis (R1): the prior "int-dot e2e == fp e2e" null was a false null, masked by the
Q6_K `ffn_down` fallback eating 59% of the token. With Q6_K now fast (clean 53.5 baseline), the int-dot win
should surface. **It does not. The revisit hypothesis is wrong — the null is real and structural.**

## Measurements (RX 7900 XTX, full clock, vs the clean 53.5 baseline)
| config | tok/s (repeats) | verdict |
|---|---|---|
| baseline (fp Q4_K GEMV, Q6_K on) | 49.4, 53.4 | — |
| int-dot unamortized (old D1) | 52.6 | null |
| int-dot + amortized quant (old E0) | 52.9, 53.6 | null |
| split-K ×4 on attn (occupancy) | 51.8, 53.9, 54.9 | within noise |
| split-K ×8 | 52.2 | within noise |
| split-K ×4 + int-dot | 50.5 | within noise |
| horizontal fusion (`Q4K_FUSE`) | 44.2 | **hurts** |

Run-to-run noise is **±3 tok/s (~6%)** even at forced clock. **No per-kernel GEMV lever clears it.**

## Why (the diagnosis, DEBUG=2 in-graph per-kernel)
The in-graph int-dot GEMV `q4k_q8_1_vdot_builtin_partial_4096_4096` runs at **28.5 µs** vs the fp
`q4k_gemv_partial_4096_4096` at **32 µs** — only **1.12×**, not the **1.36×** the standalone numbers (int-dot
76% vs fp 56%) predict. Both are ~31–34% of peak in-graph. The same int-dot kernel hits 64% when
*amortized over 200 reps standalone* — so the gap is **single-shot occupancy**, not compute:
- the attn GEMV is 4096 rows / LOCAL 64 = **64 workgroups** for one launch — too few to fill 96 CUs / hide
  latency. The ffn GEMV (12288 rows = 192 wg) reaches 54%; the small attn GEMV only 34%.
- int-dot's compute advantage is invisible because the kernel is **not compute-bound** in-graph — it is
  occupancy/latency-bound. And the q8 activation quant (`q8_1_bias_pack` ×108/token) adds cost that cancels
  the tiny GEMV gain → net null.
- split-K (more workgroups) is the *right* direction but the partial-sum overhead cancels the occupancy
  gain at batch-1 → within noise. Fusion makes one bigger-but-worse kernel → hurts.

## Conclusion — this CLOSES the "faster per-kernel GEMV" line
At batch-1, the per-layer GEMVs are small and single-shot; their in-graph ceiling is **~50–55% of peak —
already ≈ llama.cpp's 57%.** We are *at* the per-kernel ceiling. No per-kernel lever (int-dot, split-K,
fusion) beats it above noise. The standalone 76% is real but **cannot translate** because the bottleneck
in-graph is occupancy of a single small launch, not the dot-product math.

**This also down-grades revisits R3/R4/R5** (fusion, per-kernel opts, k/v coverage): they are per-kernel GEMV
levers too, so they are likely within-noise for the same reason — not because Q6_K masked them, but because
the per-kernel GEMV is near its batch-1 ceiling. (R6, the policy/flywheel re-score, is still worth doing; R7,
the structural-wall audit, is partly answered here: the e2e *is* occupancy-bound at the per-kernel level —
that part of the old "structural wall" was right; what was wrong was attributing the *whole* gap to it
instead of to Q6_K coverage.)

## What this means for "beyond llama"
Beating llama (57%) is **not** reachable through a faster per-kernel GEMV — we're already at that ceiling.
It requires the structural levers that change the work, not the kernel:
- **B3 (read fewer bytes)** — per-tensor sub-4-bit; the GEMV stays occupancy-bound but moves less data.
- **B2 (overlap)** — hide the 48% non-GEMV behind the weight stream so the token is max(), not sum().
- **B5 (multi-token / speculative)** — amortize the weight read across tokens.
- **B4 (sparse-KV attention)** — reduce the attention read.
None of these are per-kernel GEMV optimizations. That is the redirect.

## Measurement caveat
The ±6% e2e noise floor means any lever smaller than ~6% is unresolvable with this harness (per-token
`generate()` wall timing). Resolving small e2e effects needs device-metric timing + many reps + thermal
control — but the levers worth chasing now are all >6%, so this is not blocking.

Repro: `Q4K_VDOT=1 Q4K_VDOT_AMORT=1` (int-dot), `Q4K_FUSE=1` (fusion) vs baseline; DEBUG=2 for the in-graph
`q4k_q8_1_vdot_builtin_partial` vs `q4k_gemv_partial` per-kernel us.
