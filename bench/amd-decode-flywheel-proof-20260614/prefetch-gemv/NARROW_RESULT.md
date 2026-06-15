# Narrowed: the e2e bottleneck is the DEFAULT-CODEGEN GEMV kernel (12% peak), not gaps, not the clock

Date: 2026-06-15. Follow-on to PERLAYER_RESULT (kernel saturates standalone; e2e 12% even at full clock).
This run profiles a real decode token per-kernel to resolve Fork A (GEMVs fast, overhead/gaps dominate) vs
Fork B (the GEMV itself is slow in-graph). Method: `extra/qk_decode_profile.py` (PROFILE graph parse, with
merged-interval gap computation) + `DEBUG=2` ground-truth per-kernel GB/s on one forward.

## Result: Fork B

**Gaps:** within one token's 286-kernel graph, merged busy == span → **0% GPU idle.** The GPU is 100% busy.
The ~44 ms token is not launch gaps. (Fork A is dead.)

**The GEMV itself, in-graph (DEBUG=2 ground truth):**
```
r_toks_64_16_4_16_2_2_2_32   267 us   380 GFLOPS   ~105 GB/s   (the 12288x4096 FFN GEMV)
```
267 us for a 25 MB Q4_K weight read = **~105 GB/s = ~12% of the 859 GB/s peak.** The 380 GFLOPS confirms
the shape (2·12288·4096 = 100 MFLOP / 267 us). This matches the e2e effective 12% exactly — the token is
these GEMVs, running slow.

**The same op, competent standalone kernel (PERLAYER_RESULT, full clock, cold):**
| kernel for the IDENTICAL 12288×4096 GEMV | GB/s | % peak |
|---|---:|---:|
| tinygrad default codegen (in-graph)      |  105 |  12%  |
| our fp-dequant standalone                |  482 |  56%  |
| our v_dot4 int-dot standalone            |  686 |  80%  |

**The default tinygrad-generated Q4_K GEMV is 4.5–6.5× slower than achievable for the identical operation.**
The e2e wall is the *kernel codegen quality*, with the GPU fully busy executing a bad kernel — not the
clock, not Amdahl, not launch gaps.

## Why this is consistent with the D1/E0 null (and what it leaves open)
D1/E0 wired our fast kernel and saw e2e-neutral (30 = 30). Two readings remain, and the FIX experiment
distinguishes them:
- (a) **kernel-source**: the bad codegen is the whole story; a competent in-graph kernel runs at its
  standalone 56–80% and e2e jumps. D1/E0 was neutral only because its quant overhead (D1 unamortized, 7×/
  layer) canceled the kernel win.
- (b) **in-graph execution**: even our good kernel collapses to ~12% in-graph (single-shot occupancy / quant
  coupling) — the D1/E0 null is then intrinsic, and standalone bandwidth never transfers.

The standalone-vs-in-graph gap (686 vs 105 for the same op) is large enough that (a) is the leading
hypothesis, but it is NOT yet proven: our standalone number is 200-rep-amortized + warm, the in-graph kernel
is single-shot. The fix experiment must measure our kernel's **in-graph single-shot** bandwidth directly.

## The fix experiment (next)
Land a competent Q4_K GEMV in the decode graph and measure the in-graph kernel's GB/s (not just e2e tok/s):
1. Wire the fp_prefetch / v_dot4 kernel as the Q4_K linear (the E-phase path, amortized q8 quant: one quant
   feeding q/k/v and one feeding gate/up — 4 quants/layer, not 7).
2. Under DEBUG=2, read the replacement kernel's in-graph GB/s.
   - in-graph ≈ 480–680 (its standalone rate) → hypothesis (a): codegen was the wall; e2e should jump toward
     llama.cpp. Ship it.
   - in-graph ≈ 105 (collapses) → hypothesis (b): in-graph single-shot execution caps it; the lever moves to
     occupancy (bigger workgroups / persistent / fused multi-row) — the mmvq structural difference.
3. Either way the target is now concrete and measurable: get the in-graph GEMV from 105 GB/s toward 480+.

Repro: `DEV=AMD Q4K_PRIMITIVE=1 DEBUG=2 ... | grep r_toks_64_16_4` for the in-graph GEMV rate;
`extra/qk_decode_profile.py` for the gap analysis. Flags default-off.
