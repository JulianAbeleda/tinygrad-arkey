# Phase A0 RESULT (2026-06-15): amortized quant works but doesn't help -- the INT-DOT KERNEL is the
# e2e bottleneck (occupancy), not the quant. fp (58) is the decode ceiling. Phase A premise corrected.

A0: amortized the q8_1 quant via a cache keyed by x_vec UOp (q/k/v share attn-input, gate/up share
ffn-input), dispatched the shared linears to the BARRIER-FREE q4k_q8_1_intdot_partial_kernel.

## Results
- Cache: hits=36, misses=126, hit_rate=0.22 -- amortization WORKS (q/k/v + gate/up share one quant).
- End-to-end decode: 28 tok/s / 136 GB/s -- IDENTICAL to D0 (per-linear quant), and HALF of fp's 278.
  Amortizing the quant changed NOTHING.

## The corrected diagnosis (D0 mis-attributed the 28)
D0 blamed its 28 tok/s on the per-linear quant launches. A0 disproves it: with the quant amortized
(22% cache hits) the e2e is STILL 28. So the quant was never the bottleneck -- **the barrier-free
int-dot KERNEL is**, at 136 GB/s e2e (half of fp's 278). It does NOT pipeline (standalone 242 ->
e2e 136, the OPPOSITE of fp's 173 -> 278) because its int accumulators (dot + qsum REGs x 8 groups =
~16 int registers) raise register pressure -> lower occupancy -> the latency-bound small-GEMV decode
regime can't overlap consecutive GEMVs.

## The unified, final decode conclusion (proven from every angle)
BOTH int-dot structures lose end-to-end to fp, for the SAME reason -- occupancy, not compute:
- fused-LDS coop (409 GB/s standalone): LDS+barrier cap occupancy -> e2e 24.
- barrier-free global (242 GB/s standalone): int-accumulator register pressure caps occupancy -> e2e 28.
fp (173 standalone, 278 e2e, 58 tok/s) WINS end-to-end because its simple fp-FMA accumulator has LOW
register pressure -> HIGH occupancy -> it PIPELINES. In occupancy/latency-bound small-GEMV decode,
**occupancy (kernel simplicity) beats compute efficiency (the int-dot)** -- the int-dot's standalone
win is an e2e mirage in every structure.

So the decode ceiling on tinygrad/AMD is fp at 58 tok/s (56% of llama.cpp). llama.cpp's int-dot wins
e2e because its hand-asm mmvq is OCCUPANCY-efficient (DP4A packs 4 int8/register, fewer accumulator
regs) -- an occupancy-optimized hand kernel tinygrad's generated int-dot is not. Closing the gap needs
an occupancy-efficient int-dot (DP4A-packed, minimal registers) = hand-written AMD asm (the Writer),
which tinygrad's primitives + codegen do not produce. The Phase A premise (quant placement was the
wall) was REFUTED by its own A0: the wall is the int-dot kernel's occupancy. This is the honest,
final, multiply-confirmed end-state of the decode investigation.
