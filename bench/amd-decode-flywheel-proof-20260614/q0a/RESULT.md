# Phase Q0a RESULT (2026-06-15): the int-dot ~81 is a MICROBENCH ARTIFACT; fp (58) remains best.

The bet: fuse the q8_1 quant into the int-dot GEMV (one kernel) to capture the D0 microbench ~81 tok/s
end-to-end. Built `q4k_q8_1_fused_intdot_kernel` (kept in `extra/q4_k_gemv_primitive.py` as a
documented negative): phase 1 quantizes x -> q8_1 into LDS (barrier), phase 2 int-dots from LDS.

## Results
- Correctness: PASS (rel_err 0.0073 vs fp, both parts=1/4 -- the expected ~0.7% int8 error, matches X0).
- Standalone microbench (ffn_gate): 10.1 Q4-GB/s -- ~24x SLOWER than the separate-quant int-dot's 242.
- End-to-end decode: 6 tok/s (vs fp 58, D0 separate-quant 28). Catastrophic.

## Why -- the fused-staging lowering wall (recurring)
The phase-1 quant PROLOGUE is not hoisted/parallelized by tinygrad's lowering: it is replicated per
OUTPUT ROW (rows=12288 for ffn_gate) instead of computed once per workgroup. Same wall as W2's dequant
prologue and G0''. Each row re-quantizes all of x -> ~24x slowdown. Correct, but the lowering won't
hoist the cooperative LDS fill.

## The honest verdict: the int-dot ~81 ceiling does NOT exist end-to-end
The D0 microbench (242 GB/s, +40%) assumed a FREE pre-quantized activation. End-to-end the quant must
be paid, and BOTH strategies lose to fp:
- separate quant kernel (D0): per-linear launch overhead -> 28 tok/s.
- fused quant prologue (Q0a.2): prologue replicated per row -> 6 tok/s.
So fp (58 tok/s, 56% of llama.cpp) REMAINS the best decode kernel. The int-dot/q8_1 "+40%" was a
microbench artifact that excluded the unavoidable activation-quant cost. Q0a's pre-registered ~75-81
gate FAILS.

## Net for the whole decode thread
Every codegen-reachable decode lever is now probed and NEGATIVE end-to-end: DP4A (D0), latency-hiding
(L0), lossy-quant-at-int8 (X0 weak), and now the fused int-dot (Q0a). fp at 58 (56% of llama.cpp) is
the honest ceiling of the expressible primitives. The residual gap to llama.cpp is the cross-layer /
hand-asm rungs (Mirage OSDI'25 for the search way; the Writer/hand-kernel otherwise) -- exactly as the
"why" analysis predicted. The build did its job: it disproved a microbench-implied win by paying the
cost the microbench excluded.
