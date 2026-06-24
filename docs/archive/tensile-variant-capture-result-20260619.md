# RESULT — Per-variant Tensile capture UNBLOCKED the audit; measured primitive attribution (gfx1100, 2026-06-19)

Executes `tensile-variant-capture-scope-20260619.md` C0–C5. **The variant-ablation is UNBLOCKED and run: we now
launch arbitrary Tensile variants on tinygrad buffers and have a MEASURED per-primitive attribution — replacing the
retracted recited list ("SW-pipeline + dense-issue + occupancy").**

## Tooling built (the unblock)
- `extra/qk_tensile_solution_sweep.cpp` — host-only rocBLAS: `rocblas_gemm_ex_get_solutions` for the gateup shape
  (m=512,n=12288,k=4096, HHS) → **159 solutions**, dispatch each via `rocblas_gemm_algo_solution_index`.
- `extra/qk_tensile_kernarg_capture_all.cpp` — LD_PRELOAD shim capturing EVERY distinct dispatched kernel's kernarg
  (keyed by symbol, not first-per-role) → **145 variant kernargs** to `/tmp/kernargs_all.jsonl`.
- `extra/qk_tensile_variant_ablation.py` — per-variant HCQ launcher using the captured kernarg (correct grid from the
  captured workitems/local) + correctness + PMC. **9/9 representative variants launch CORRECT** (rel_err 3.5e-4).
- **Why it works:** rocBLAS computes each variant's workgroup→tile mapping kernarg; we capture it; tinygrad HCQ then
  launches the variant on its own buffers (no HIP at measure-time). This is the counterfactual machine the audit needed.

## Measured ablation matrix (gateup GEMM, GRBM_GUI_ACTIVE cycles; lower = faster)
| MT | LDSB | GRBM cycles | SQ_WAIT (stalls) | DRAM rdreq | SQ_WAVES | LDS bytes |
|---|---|---:|---:|---:|---:|---:|
| 64x64x32 | 0 | **2,294,048** | 1.20e9 | 2.49M | 6144 | 25088 |
| 64x64x16 | 0 | 2,579,763 | 2.15e9 | 3.41M | 6144 | 12544 |
| 128x128x16 | 0 | 3,824,504 | 1.99e9 | 4.46M | 1536 | 25088 |
| 128x128x16 | 1 | 3,835,635 | 1.77e9 | 4.62M | 1536 | 8704 |
| 32x32x32 | 1 | 4,173,099 | 6.58e9 | 1.69M | 24576 | 4352 |
| 32x32x32 | 0 | 4,267,271 | 3.75e9 | 1.73M | 24576 | 12544 |
| 32x32x16 | 0 | 4,487,799 | 9.34e9 | 5.75M | 24576 | 6272 |
| 32x32x16 | 1 | 5,264,570 | 11.1e9 | 8.60M | 24576 | 2176 |
| 64x64x16 | 1 | **7,495,322** | 16.7e9 | 8.18M | 6144 | 4352 |
| (ref) tinygrad WMMA | — | **~12,500,000** | 14.2e9 | 29.8M | 3072 | 0 |
- LDSB0 = double-buffered (more LDS, software-pipelined); LDSB1 = single-buffer (less LDS, serialized). Confirmed by
  the LDS bytes (LDSB0 ≈ 2× LDSB1).

## MEASURED attribution (replaces the recited list)
1. **Integrated scheduling quality is the DOMINANT factor.** **Every Tensile variant (2.29–7.5M cycles) beats
   tinygrad WMMA (12.5M)** — even the worst-scheduled one (64x64x16 single-buffer, no pipeline) is **1.7× faster**;
   the best (64x64x32 double-buffer) is **5.5×**. So the win is NOT any single bolt-on primitive — baseline
   well-scheduled Tensile already wins. (Consistent with A3: the primitives don't transfer to tinygrad individually.)
2. **Software-pipelining (LDSB0 double-buffer) is a tile-DEPENDENT multiplier, not a universal lever:** same-tile
   LDSB0 vs LDSB1 = **2.9× at 64x64x16**, 1.17× at 32x32x16, **1.00× at 128x128x16**, 0.98× at 32x32x32. So
   double-buffering matters most at the 64×64 tile and is ~irrelevant at 128×128. (My earlier flat "software-pipeline
   is the lever" was too strong.)
3. **Occupancy has an OPTIMUM, not monotonic:** the fastest tile is **64×64** (2.3–2.6M, 6144 waves), NOT the
   max-occupancy 32×32 (24576 waves, 4.3–5.3M) nor the big-tile 128×128 (1536 waves, 3.8M). "More waves = better" is
   refuted; there is a sweet spot balancing occupancy vs operand reuse.
4. **K-depth 32 > 16** (64×64 LDSB0): 1.12×.
5. **DRAM + stalls track cycles** (the slow 64x64x16-LDSB1 has 8.2M DRAM / 16.7e9 stalls; the fast 64x64x32-LDSB0 has
   2.5M / 1.2e9) — confirming the bottleneck is memory-stall-bound, and LDS staging/pipelining reduces both.
6. **RETRACTED stays retracted:** "dense issue" (VALU-equal). The "shape" hypothesis (earlier) is also moot — this is
   all one shape.

## Notable
- rocBLAS's chosen/shipped kernel for this shape is **128×128 LDSB0 (3.82M)**, but isolated, **64×64×32 LDSB0
  (2.29M) is 1.67× faster** here. Isolated-PMC ≠ in-model (rocBLAS tunes for the full dispatch/occupancy context, and
  PMC perturbs), so this is not a "rocBLAS picked wrong" claim — but it flags that the in-model 1.84× is from the
  128×128 variant, with headroom in principle.

> **⚠ CAVEAT (2026-06-20):** the Tensile-vs-Tensile ablation below is internally consistent and stands. But the
> **"vs tinygrad WMMA (~12.5M cycles)" discriminator** rests on an UNVALIDATED tinygrad baseline (6.5–46.5 TFLOPS
> across methods). And the implied "scheduling-quality, not a primitive" reading is RETRACTED: BEAM (never on in
> prod) emits a correct LDS-WMMA at 46.5 TFLOPS, so tinygrad CAN schedule LDS — the production gap is at least partly
> opt-selection (shipped warmstart is hand-picked no-LDS), not a codegen incapacity. See `prefill-tensile-DEFINITIVE`
> validation caveat.

## Verdict on "how do you know the primitives"
Now ANSWERED with measurement: **the dominant factor is integrated scheduling quality (every variant beats tinygrad,
even un-pipelined); software-pipelining adds up to 2.9× but is tile-dependent (≈0 at 128×128); occupancy has an
optimum at 64×64.** This is a measured sensitivity table, not a recited list. The dependency-free implication is
unchanged: tinygrad can't emit ANY of these well-scheduled variants (A3), so the lever stays the vendored `.co` or a
Tensile-class codegen capability.

## Audit-tooling status (corrected)
We DO now have enough tooling to audit Tensile across variants: solution-sweep capture (`qk_tensile_solution_sweep` +
the all-symbols shim) + per-variant HCQ launch (`qk_tensile_variant_ablation`) + PMC scoreboard. The earlier
"BLOCKED" verdict (`tensile-variant-ablation-result`) is now SUPERSEDED — the blocker was the per-variant kernarg,
which the solution-sweep capture provides.
