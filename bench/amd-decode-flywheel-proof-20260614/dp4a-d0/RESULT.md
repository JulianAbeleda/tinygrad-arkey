# Phase D0 RESULT -- the DP4A ceiling probe (2026-06-15): gate NOT cleared. Do not build Phase D.

Pre-registered D0 gate: a HAND-WRITTEN int8/DP4A decode GEMV must approach ~llama.cpp (~90-104 tok/s)
to justify building the DP4A codegen vocabulary (D1-D4). If it plateaus well below, DP4A is not the
lever -> rescope/stop (roofline discipline; don't build a compiler feature for the wrong bottleneck).

## Evidence
Baselines: current fp decode = 58 tok/s / 278 GB/s; llama.cpp = 104 tok/s / ~470-500 GB/s.

Per-tensor microbench (`extra/q8_1_q4k_bench.py`, device Q4-GB/s = compressed-weight bandwidth):
  kernel                         ffn_gate   attn_q
  float (current)                173        75
  vdot   (explicit v_dot4 asm)    35        21     <- WORST: asm volatile blocks scheduling/pipelining
  intdot (int8 MAC, compiler)    242        82     <- best, +40% over fp on ffn_gate
  vdot_parallel                  140        50
Best int8 variant (intdot) = ~242 GB/s = ~50% of llama.cpp's ~470-500.

End-to-end decode (intdot wired into model.py behind Q4K_INTDOT, then reverted):
  28 tok/s / 135 GB/s  -- REGRESSED below fp (58), because the per-layer q8_1 activation quant is an
  unfused extra kernel and at batch-1 its launch overhead dominates.

Optimistic extrapolation (if the quant were fused, scaling the microbench ratio by the fp
micro->e2e factor): intdot e2e ~ 278 * (242/173) ~ 390 GB/s ~ 81 tok/s.

## Verdict: gate NOT cleared
Even the optimistic, properly-implemented int8 ceiling is ~81 tok/s (~78% of llama.cpp) -- an
improvement over 58 but NOT parity (104). And critically, the EXPLICIT DP4A instruction (`v_dot4`,
exactly what Phase D would teach the codegen) is the SLOWEST variant; the modest win comes from the
int8 ACTIVATION (fewer bytes), not the dot instruction.

## Why (the diagnosis, consistent with M0)
Decode is MEMORY/occupancy-bound (M0). DP4A accelerates COMPUTE (the dot product). Speeding up compute
on a memory-bound kernel does not help -- which is exactly why explicit DP4A is slowest and int8
barely helps. llama.cpp's advantage is its MEMORY-side engineering (access patterns, occupancy, q8_1
activation packing that reduces bytes), NOT the dot instruction per se.

## Decision
Do NOT build the DP4A codegen vocabulary (Phase D D1-D4). It would optimize the wrong thing (compute)
on a memory-bound kernel. If the decode gap is pursued, the lever is int8-ACTIVATION + occupancy/memory
access -- and even that appears to ceiling around ~81 tok/s (~78% of llama.cpp), not parity, and needs
a fused quant. Honest caveat: a from-scratch, occupancy-tuned, quant-fused int8 GEMV might do better,
but that is itself a large hand-written-kernel effort (the thing the search philosophy wants to avoid),
and the memory-bound diagnosis says the dot instruction is not where the bottleneck lives.

The D0 gate did its job: it caught the wrong lever with a cheap probe before any compiler change.
