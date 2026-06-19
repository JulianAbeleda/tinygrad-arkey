# Performance frontier exhaustion — current source-of-truth

This is the checkpoint after the llama residual audit, q8/MMVQ lifecycle audit, spec-verify breakdown, prefill
weight-reuse redirect, hand-LDS WMMA refutation, external-BLAS ceiling run, pure-tinygrad WMMA sweep, and Tensile
primitive extraction through TPE-4. It answers: what has been exhausted by primitive, what is still measurable only
with better tools, and what remains a deep build rather than a bounded kernel tweak.

## Verdict table

| frontier | status | evidence | what remains |
|---|---|---|---|
| tinygrad-vs-llama decode gap | **explained / bounded space exhausted** | per-role delta audit: summed ceilings ~+27-30% ~= the whole gap; residual sits behind q8/full-MMVQ lifecycle | no cheap decode kernel; only deep/lossy q8 lifecycle or sub-gate stacking |
| llama decode MMVQ residual | **open measurement, not a build target** | fresh llama d0 trace: 85.6% MMVQ; source shows RDNA3-specific `sudot4`, Q4_K/Q6_K unpack/affine/reduction costs; no spill evidence | needs working gfx1100 per-role counters/ATT to prove a role-specialized opportunity |
| q8/RMSNorm lifecycle | **deferred behind codegen capability** | llama: q8 3.57%, RMSNorm 4.60%; tinygrad Q8L-0/1 pass but Q8L-2 kills store-group expressibility | only reopen with an LDS-reduction multi-output custom-kernel capability; expected decode EV ~3-4% |
| spec decode as shortcut | **closed** | verify T>1 cost is distributed across Q4_K, Q6_K, attention, and lost T==1 fast paths | only broad batched-forward/prefill-class work, not one verify kernel |
| 8B prefill quant-weight reuse | **closed** | PREFILL_V2 realizes fp16 weights and uses WMMA; no in-forward quant dequant to amortize | VRAM-frugal 14B/32B policy only, outside current 8B scope |
| hand-LDS WMMA prefill | **refuted** | hand-LDS WMMA 41.5 TFLOPS vs default 40.8 TFLOPS, 1.02x; IC-served on gfx1100 | do not reopen as "add LDS tiling" |
| external BLAS prefill ceiling | **measured; HIP-runtime bridge closed; Lane B TPE-4 PASS for ffn_gate/up** | hipBLASLt 69.8 TFLOPS on ffn_gate/up, 1.71x tinygrad; EBT-1 proves HIP runtime and tinygrad HCQ/KFD are mutually exclusive in one process; TPE-4 launches the rocBLAS Tensile ffn_gate/up primitive through HCQ at 66.91 TFLOPS, correct/no-copy/no-HIP | TPE-5 shape matrix for ffn_down + attn_q/o, then weighted pp512 model; codegen/Tensile-class rewrite only after deciding external-artifact policy |
| pure tinygrad prefill WMMA issue | **refuted as bounded config sweep** | POWN-1 best = 42.0 TFLOPS, same plateau; more waves, bigger tiles, BK32, noLDS all regress | only a deeper codegen/assembly/Tensile-class rewrite, not a scoped knob build |
| prefill attention | **deferred / phase-specific** | pp512 llama/tinygrad residual is matmul-first; reuse-free flash-prefill was 170-760x slower | long-prompt-only audit if attention dominates; needs real LDS/register flash primitive |
| host/runtime launch overhead | **refuted for current decode** | tinygrad W==D/host-sync ~0; llama HIP graphs explain its own launch-boundary handling | only GPU-work removal or explicit graph-boundary primitive, not "Python overhead" |
| NVIDIA / RTX 5090 portability | **separate backend audit** | not part of gfx1100 primitive exhaustion | audit CUDA backend/library boundaries separately before transferring conclusions |

## What is actually left

Only two material things remain that are not already closed, refuted, or sub-gate:

1. **Tensile extraction through HCQ — TPE-5 PASS, now at the TPE-6/policy gate:** the library ceiling exists, EBT-1
   killed in-process HIP-runtime interop, TPE-4 proved one extracted rocBLAS Tensile primitive runs through tinygrad
   HCQ at mature-backend speed, and **TPE-5 proved it generalizes** — ffn_gate/up 66.8, ffn_down 68.9 (StreamK, no
   workspace), attn_q/o 58.9 TFLOPS, all correct/stable/no-workspace/no-layout-copies from one code object, weighted
   ~**1.40× full pp512** (~95% of llama). The remaining gates are TPE-6 (one-block transfer + minimal runtime helper)
   and the external-artifact policy decision — no longer a kernel question.
2. **Better llama MMVQ counters:** useful for research completeness, but locally blocked by gfx1100 counter-tool
   support. Current source/trace evidence does not justify a build.

Everything else is shipped, refuted, below the Amdahl gate, or requires a new deep codegen/Tensile-class capability
rather than a bounded primitive edit.

## Practical conclusion

The project has exhausted the bounded primitive explanations for why llama.cpp benchmarks above this tinygrad fork.
The remaining difference is not one missing instruction, one scheduler knob, or one fusion. It is the lifecycle of
complete performance primitives:

- decode: q8 activation format + native dot4 + packed MMVQ scheduler, with tinygrad blocked by q8 lifecycle/codegen
  economics;
- prefill: dense WMMA/GEMM issue quality, where LDS tiling and the bounded pure-tinygrad config sweep are refuted,
  while external BLAS proves a higher ceiling;
- long prompt: separate attention locality, only relevant when the prompt regime makes it large.

After POWN-1 and EBT-1, there is no remaining bounded no-deps prefill kernel route and no direct HIP-runtime bridge.
After TPE-4 and TPE-5, the remaining performance route is no longer speculative and is proven to generalize: Tensile
extraction through HCQ keeps mature-backend speed across the three high-share prefill roles (~1.40× weighted pp512,
~95% of llama), gated now only by TPE-6 one-block transfer and the external-artifact policy decision — or resting at
PREFILL_V2.
