# Phase L0 RESULT (2026-06-15): latency-hiding is NOT the binding constraint. Do not build Phase L.

Pre-registered L0 gate: hand-probe occupancy / prefetch / double-buffer on the decode GEMV; build the
subsystem only if one moves decode toward ~90-104 tok/s. If none moves it -> latency-hiding isn't the
constraint -> STOP (roofline discipline).

## Probes run (end-to-end decode tok/s; baseline fp = 58 / 278 GB/s, llama.cpp = 104 / ~470-500)
- L0a OCCUPANCY via existing parts/LOCAL knobs: FLAT (device GB/s ~82-86 across parts 1->16;
  re-confirms M0's full-opt-space flatness). Existing occupancy/tiling/ILP levers exhausted.
- L1 OCCUPANCY-FORCING (patched HIPRenderer to emit amdgpu_waves_per_eu(N), end-to-end):
    WAVES_PER_EU=2 -> 30.0 tok/s | =4 -> 29.7 | =6 -> 30.0 | =8 -> 21.3
  Forcing higher occupancy REGRESSES decode (~halves it). The compiler's DEFAULT occupancy is already
  optimal; more waves -> register spills / less ILP -> worse. So decode is NOT occupancy-starved.
- L2 PREFETCH (ILP proxy): the UPCAST/UNROLL/parts knobs that feed LLVM independent loads are FLAT
  (M0 + L0a). LLVM (the HIP path's scheduler) already extracts the available ILP; kernels also already
  overlap END-TO-END (278 GB/s aggregate vs ~85-173 per-kernel). Prefetch-via-ILP is exhausted.

## Verdict: STOP -- do not build Phase L (the latency-hiding subsystem)
None of the accessible latency-hiding levers move decode toward parity; forcing occupancy makes it
WORSE. The signal that occupancy-forcing regresses is the key one: an occupancy/latency-STARVED kernel
would SPEED UP with more waves -- this one slows down. So the decode GEMV is not bound by hideable
memory latency; it is bound by what the compiler already balances (occupancy + LLVM scheduling) plus
the Q4_K DEQUANT ALU cost (~3862 vector ops/kernel, M0) competing with memory throughput. A
memory-latency-hiding subsystem (async copy / double-buffer / waves control) would optimize a
constraint that is not binding here.

## Where the decode gap actually lives (best current understanding)
Both scoped decode levers now probed and both NEGATIVE: DP4A (D0, compute -- wrong axis, explicit
v_dot4 was slowest) and latency-hiding (L0 -- not the constraint; occupancy-forcing regresses). The
residual ~2x gap to llama.cpp is most consistent with: (a) the Q4_K dequant ALU cost per byte on this
GPU, and (b) possibly kernel COUNT / fusion (decode issues ~252 small GEMV launches/token; llama.cpp
fuses more). Neither is addressed by the DP4A or latency-hiding vocabularies. Honest conclusion: decode
PARITY with llama.cpp is not reachable via the codegen-vocabulary levers we scoped; the gap is a
dequant-cost + kernel-structure problem, and the realistic ceiling (D0) is ~81 tok/s (~78%), not 104.

The L0 probe did its job: a cheap, reversible renderer patch caught a non-binding constraint before
building a major codegen subsystem.
