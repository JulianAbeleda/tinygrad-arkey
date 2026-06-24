# Phase Mi -- Mirage (cross-layer superoptimization) probe + RESULT

Date: 2026-06-15
Question: does Mirage's cross-layer search (the frontier we identified for the decode 58->104 gap) have
a win on our setup? Scope = what trying it means; RESULT = the feasibility probe.

## Mi0 -- feasibility (make-or-break, ran FIRST): BLOCKED on three independent grounds.

**(1) Hardware: Mirage is CUDA/NVIDIA-only; this is an AMD machine.** Mirage (OSDI'25, github
mirage-project) runs its multi-level search BY measuring candidate kernels on an NVIDIA GPU and EMITS
CUDA/Triple. This box: `nvidia-smi` not found, no CUDA toolkit, no `nvcc` -> Mirage cannot even build
its CUDA C++ extensions, let alone run its on-device search. Device is AMD gfx1100 (ROCm/HIP). So
Mirage literally cannot run here, and even if run elsewhere its CUDA output does not execute on AMD.

**(2) Codegen: its wins don't port to tinygrad's lowering.** Mirage's value is cross-layer search
PLUS its own efficient fused-kernel codegen. To use it on AMD/tinygrad you would hand-port the
optimizations it discovers (custom fused kernels, e.g. fused norm+GEMV, multi-GEMV). But tinygrad's
lowering has REPEATEDLY failed to express efficient fused custom kernels -- the fused-staging wall hit
at W2 (dequant prologue not hoisted) and Q0a (quant prologue replicated per row, ~24x slowdown). A
hand-ported Mirage fusion would hit the SAME wall. Mirage's wins rely on its codegen, which we cannot
use on this target.

**(3) Prize: limited on this target anyway.** Mirage's win-class is cross-layer FUSION / kernel-count
reduction. Sized on our decode: per-kernel microbench is 85-173 GB/s but END-TO-END aggregate is 278
GB/s -- i.e. tinygrad's JIT ALREADY pipelines/overlaps the ~252 kernel launches (end-to-end is faster
per-tensor than the isolated microbench). So launch/fusion headroom is already largely captured by the
JIT. The remaining 58->104 gap is per-kernel BANDWIDTH UTILIZATION (278 vs llama.cpp's ~470 GB/s) --
the dequant-instruction-count problem (Q), already probed NEGATIVE end-to-end -- not a launch-fusion
problem Mirage would fix.

## Verdict: Mirage yields NO win on our AMD/tinygrad target.
Its value is real but lives on NVIDIA/CUDA with its own codegen -- a different target. On this setup it
is triple-blocked (no CUDA hardware; its fusions don't port through tinygrad's lowering; and its
win-class has little on-target prize since the JIT already pipelines launches). The cross-layer search
DIRECTION is the right frontier (and we validated its lower rung -- the learned loop -- in N1/N2), but
realizing it for AMD/tinygrad would require an AMD-targeting cross-layer superoptimizer that does not
exist, not Mirage.

## The honest end-state of the decode investigation
Every codegen/search-reachable decode lever is now exhausted and NEGATIVE end-to-end:
- schedule/occupancy/ILP (M0, L0): flat / regress.
- DP4A compute (D0): wrong axis.
- int-dot fusion (Q0a): microbench artifact; fused-staging wall.
- lossy-quant at int8 (X0): weak home; uniform int8 is a constant.
- cross-layer fusion / Mirage (Mi0): hardware + codegen + prize all blocked.
fp at **58 tok/s (56% of llama.cpp, 32% of HBM peak)** is the honest ceiling of what tinygrad's
expressible primitives + search reach on AMD decode. The residual gap to llama.cpp (104) is the
hand-written-AMD-kernel direction -- the "Writer" the search philosophy is built to avoid, and exactly
what llama.cpp is. This is consistent with the whole investigation's thesis-boundary finding: the
decode optimum lives in the cross-layer / hand-asm space that single-layer search + tinygrad's lowering
do not reach. The positive result of the program remains the loop on the SCHEDULE axis for the BATCHED
regime (N1/N2, 33-98% of peak); single-stream quantized decode parity on AMD/tinygrad is, on this
evidence, not reachable without hand kernels.
