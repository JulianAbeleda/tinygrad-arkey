# WHY tinygrad's FMA matmul isn't at rocBLAS quality — exhaustive search + PMC diagnosis

User: "do an exhaustive search on our own and see why it's not at rocBLAS quality." Done. Defined both bars
(matmul-quality-definitions), exhaustively searched tinygrad's FMA opt space, PMC-localized the binding axis.

## Exhaustive search (586 FMA configs, UPCAST x LOCAL x GROUP x UNROLL)
tinygrad's BEST non-TC FMA config = **11.2 TFLOPS** (UM4 UN8 LM4 LN8 GK4 UN4). vs WMMA 42, rocBLAS 66-77.
- GROUP (LDS reduce-stage) gives ~NOTHING: 11.2 (GK4) vs 11.1 (GK0). tinygrad's LDS doesn't help the FMA path.
- So the entire OPT SPACE tops at 11 -> not a tuning problem; a codegen-capability problem.

## PMC diagnosis (best-FMA vs WMMA, GL2C/SQ counters)
| | L2 hit | VALU instrs (dynamic) | VALU/busy | bound by |
|---|---:|---:|---:|---|
| best-FMA (11.2) | **92%** | **1.69e9** | 3.47 | **VALU-ISSUE** (not memory: L2 hit 92%) |
| WMMA (42) | 77% | 1.67e7 | 0.08 | WMMA-unit + occupancy (VALU idle) |

- **FMA path is VALU-instruction-issue-bound:** 92% L2 hit -> operands ARE cache-served (not memory-bound); it
  issues **100x more VALU instructions** than WMMA (1.69e9 vs 1.67e7) because scalar FMA = ~1 madd/inst/lane vs
  WMMA's 16x16x16 = 4096 madds/inst. ~1.69e9 ≈ 2x the minimum useful-FMA count -> dense issue (~0.75 inst/
  SIMD-cycle) but capped by the VALU instruction throughput. At 11 TFLOPS = ~18% of the ~61 fp32-acc FMA roofline.
- **WMMA path is NOT issue-bound** (VALU idle, 0.08) -- it packs madds into few instructions; capped at 42 by the
  RDNA3 WMMA-unit throughput + single-wave occupancy (the exhausted WMMA frontier).

## So WHY isn't tinygrad at rocBLAS quality? (per-axis, both paths lose for DIFFERENT reasons)
rocBLAS (66) uses packed-FMA at ~ROOFLINE via: dense/efficient FMA issue (near dual-issue, minimal overhead) +
operand-LDS-staging + software-pipelining + shape-tuned occupancy. tinygrad:
- **FMA path (11):** VALU-issue-bound at ~18% of roofline -- emits scalar v_fma with ~2x instruction overhead and
  no packing/dual-issue density; GROUP-LDS gives nothing (it's reduce-staging, not operand-staging). The codegen
  doesn't produce a dense, dual-issued, operand-staged FMA inner loop.
- **WMMA path (42):** packs madds (issue-light) but the RDNA3 WMMA unit + single-wave occupancy cap it at 42
  (below the FMA roofline rocBLAS reaches; raising occupancy regressed -- the exhausted frontier).
- Neither path has software-pipelining (linearizer pin).

## Verdict (the precise, measured gap)
It is NOT a config-tuning gap (exhaustive search tops at 11) and NOT a memory gap (L2 hit 92%). It is a
**codegen-capability gap on the FMA inner loop**: tinygrad cannot emit a dense, dual-issue-packed, operand-LDS-
staged, software-pipelined FMA matmul (rocBLAS/Tensile's kernel). Reaching rocBLAS quality = building that
Tensile-class FMA codegen (instruction packing + operand-staging + pipelining + occupancy) -- a multi-month core
capability, not a bounded arc -- OR vendoring the external .co. tinygrad's practical best stays WMMA 42 (~47% llama).

## Files
matmul-quality-definitions, why-tensile-works-fma-not-wmma. Sweep /tmp/fma_sweep.py, PMC /tmp/fma_pmc.py.
