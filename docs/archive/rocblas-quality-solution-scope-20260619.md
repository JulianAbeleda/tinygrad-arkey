# SOLUTION SCOPE — reaching rocBLAS quality (42 -> 66), reprioritized by the PMC diagnosis

## The diagnosis reframes the solution (key)
The FMA path is **VALU-ISSUE-bound** (L2 hit 92% = NOT memory-bound; 1.69e9 VALU instr, ~2x the 805M minimum,
issuing ~0.75 inst/SIMD-cycle). Therefore:
- **Operand-LDS-staging is NOT the FMA lever** (cache already serves operands at 92% hit). DEPRIORITIZED.
- **Software-pipelining is NOT the FMA lever** (issue-bound, not latency-bound). DEPRIORITIZED.
- The FMA levers are **instruction density**: (1) remove the ~2x instruction overhead, (2) dual-issue (VOPD).
- rocBLAS uses `v_fma_mix_f32` (1 madd/inst, same as tinygrad would) -> its 6x is ISSUE EFFICIENCY (dual-issue +
  minimal overhead + occupancy keeping the VALU fed), NOT a different instruction or operand-staging.

## The hard ceiling math (why bounded codegen has LOW value)
FMA path bounded levers: overhead-removal (~2x: 11->22) + dual-issue (~2x: ->44) ≈ **WMMA's 42**. So even fully
successful bounded FMA codegen only reaches ~tinygrad's EXISTING WMMA 42 -- it does NOT beat the current best.
Beating 42 -> 66 needs the FULL recipe (overhead + dual-issue + occupancy + the last ~1.5x of scheduling) on either
path = the multi-month codegen. **There is no bounded codegen win that beats 42.**

## The three real options
### A. Full Tensile-class FMA codegen (dependency-free, multi-month)
Build: dense dual-issue-packed FMA instruction-selection + minimal-overhead inner loop + occupancy balancing +
(later) operand-staging. The real path to ~66 in pure tinygrad. HIGH effort (core renderer + scheduler), HIGH risk
(broad test surface), and the occupancy sub-piece already regressed in POWN. Multi-month.

### B. Vendor the external kernel (.co) + FIX in-model transfer  <-- RECOMMENDED FIRST
hipBLASLt/Tensile already has the 66-77 TFLOPS kernel, PROVEN, on disk (/opt/rocm/lib/rocblas/library). The ONLY
blocker is the prior "doesn't transfer in-model" (0.999x, commit ebcca2a5b). **BUT that 0.999x was measured BEFORE
we understood the clock non-reproducibility (manual-DPM erratic, same config 570-1551) -- it is SUSPECT.** So:
- **P0 (cheap, decisive): RE-MEASURE external-Tensile/hipBLASLt in-model speedup CLEANLY in AUTO mode** (now known
  reproducible: 1449/1449/1448). If it transfers (e.g. >=1.3x e2e) -> vendoring the .co IS the bounded solution to
  ~66. If confirmed ~1.0x -> P1.
- **P1 (if no transfer): diagnose WHY** the isolated-66 -> in-model-1.0x. Candidates: shape mismatch (in-model
  prefill matmul shape differs from the isolated bench), dispatch/overhead per call, the Tensile kernel not actually
  selected for our shape. PMC the in-model Tensile matmul vs isolated. This localizes a fixable integration bug vs a
  real wall.
- Dependency cost: bundling a vendored .co / linking hipBLASLt (the user's deps-policy call). The in-model HCQ route
  exists (PREFILL_TENSILE_GEMM research flag, extra/qk_tensile_*).

### C. Accept 42 (ship concrete-KV 1.24x)  <-- the pragmatic rest
tinygrad WMMA 42 = ~47% llama; concrete-KV byte-identical 1.24x is the dependency-free shippable win. Past 42 is a
project (A) or a dependency (B).

## Recommendation
**Do B-P0 first** (re-measure external-Tensile transfer in AUTO mode) -- cheap, decisive, and the prior "doesn't
transfer" verdict is contaminated by the clock confound we since uncovered. It's the only path with an EXISTING
66-TFLOPS kernel; if it transfers, it's the bounded solution. Bounded pure-tinygrad codegen (the v_fma_mix /
dual-issue tweaks) is LOW value (ceiling ~42, doesn't beat current). Option A (full codegen) only if B is refused
on dependency grounds and the team commits multi-month.

## P0 deliverable
Clean AUTO-mode interleaved A/B: in-model prefill pp512 with external-Tensile-GEMM ON vs OFF (PREFILL_TENSILE_GEMM
+ the qk_tensile_inmodel route), best-of, reproducible. Verdict: transfers (>=1.2x -> pursue B) or not (-> P1 diagnose).

## Files
why-tinygrad-fma-not-rocblas-quality, matmul-quality-definitions. External route: extra/qk_tensile_*.py,
prefill-tensile-land-result (the suspect 0.999x). rocBLAS .co: /opt/rocm/lib/rocblas/library/.
