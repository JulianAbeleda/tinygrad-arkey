# Route A result — dependency-free RDNA3 WMMA asm GEMM: PROVEN VIABLE, but naive (≈13 TFLOPS), A2 is the expert grind

## What was achieved (A0 + A1)
- **A0 (gate-zero):** the existing hand-asm GEMM `amd_asm_matmul.py` is **fp32 scalar FMA, not WMMA** (40.7 TFLOPS, architecturally below the tensor-core path) — wrong algorithm. The WMMA hand-asm kernel `rdna4_asm_matmul.py` targets RDNA4/gfx1200 and **hung gfx1100** (different WMMA operand layout). The RDNA3 DSL *does* support `v_wmma_f32_16x16x16_f16`.
- **A1 (the milestone — SHIPPED as `extra/gemm/rdna3_wmma_matmul.py`):** a **correct, dependency-free RDNA3 WMMA assembly GEMM**, built via tinygrad's assemble→ELF backend with **zero LLVM**. The RDNA3 WMMA operand layout was nailed (A/B = 8 VGPR/16 fp16, C = 8 VGPR fp32, B stored transposed for contiguous columns). Single-tile RMSE 0.0002; full TM×TN tiled GEMM + K-loop RMSE 0.0002, **CORRECT**. This **proves Route A's path works end to end** — tinygrad can emit working WMMA assembly for gfx1100 with no external dependency.

## The honest number
| kernel | TFLOPS | who schedules |
|---|---:|---|
| our RDNA3 WMMA asm (naive, A1) | **≈13** (steady; 19.28 was a warm-clock outlier) | tinygrad (hand) |
| tinygrad LLVM warmstart | ~48 | LLVM |
| Tensile (rocBLAS) | 66 | hand-tuned asm (AMD) |

The naive hand kernel is **~13 — 3.7× *below* LLVM's 48.** It's correct but unoptimized: a full `s_waitcnt(0)` barrier every K-iteration fully serializes loads and WMMAs, single wave, no prefetch/overlap, no occupancy tuning.

## A2 binding constraint (named, per principles)
- TM/TN occupancy sweep: smaller tiles are *worse* (TM=TN=2→14, vs TM=TN=4→13–19), so the bottleneck is **per-iteration load↔compute serialization amortized over WMMAs**, not occupancy.
- The fix is **software pipelining** (prefetch next-K fragments while computing current WMMAs). But at TM=TN=4 the **128 accumulator VGPRs leave no room to double-buffer** the 64 fragment VGPRs (2×64 + 128 + temps > 256). Smaller tiles fit a pipeline but lose amortization. **This VGPR-vs-amortization tension is the binding constraint** — exactly the register-allocation + instruction-scheduling problem that LLVM solves automatically (→48) and Tensile solves by expert hand-allocation (→66).
- Caveat: software-pipelining was **refuted as Infinity-Cache-served on gfx1100** (CG-R1), so even a correct pipeline may gain less here than on a bandwidth-bound GPU — the realistic dependency-free ceiling is uncertain.

## Verdict
- **Route A is PROVEN VIABLE** (A1): the dependency-free WMMA-asm path works, the hard RDNA3 layout problem is solved, and a correct kernel exists and is committed. This is a real, durable capability.
- **Completing A2 (beat LLVM's 48, chase Tensile's 66) is a multi-day expert-asm-scheduling project**, not a remaining-session task: it means hand-implementing software pipelining + occupancy balancing + instruction scheduling within a tight VGPR budget — replicating by hand what LLVM does automatically. Starting from a naive 13, with hang-prone iteration and noisy measurement, and an IC-served caveat capping the upside.
- **Funding decision (research mode):** the next layer is named (hand-scheduling/pipelining within VGPR budget). Fund it only if a dependency-free >48 prefill matmul is worth multi-day expert-asm work with an uncertain ceiling — vs. the existing options: PREFILL_V2 (~80% llama, shipped) or the external Tensile `.co` (1.41× llama, dependency). The honest expected outcome of full A2 is "maybe match/modestly beat LLVM's 48", not a clear path to 66.

## Files / provenance
`extra/gemm/rdna3_wmma_matmul.py` (A1 kernel, committed 6c0c65fb2). A0/backend map:
`amd-tensile-class-codegen-scope-20260619.md`. IC-served caveat: CG-R1
(`prefill-codegen-pipeline-redo-result`). Measurements this session (DEBUG=2 min-over-runs, fair).
