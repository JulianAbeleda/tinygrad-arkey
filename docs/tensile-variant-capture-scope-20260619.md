# SCOPE — Per-variant Tensile kernarg capture → unblock the variant-ablation audit (gfx1100, 2026-06-19)

Unblocks `tensile-variant-ablation-result-20260619.md` (BLOCKED: only the one rocBLAS-selected kernel launches; its
kernarg is bound to it). Builds the tooling to make ANY Tensile variant launchable on tinygrad-owned buffers, so we
can ablate one structural knob at a time and MEASURE which primitive drives the prefill win.

## Why the audit is blocked, and why this fixes it
- **Blocker:** launching a Tensile kernel from tinygrad HCQ needs that kernel's exact 128-byte kernarg — which holds
  per-kernel workgroup→tile mapping fields (magic-number division constants / WGM) that only rocBLAS knows how to
  compute. We captured ONE (the default solution); reusing it for any other variant misroutes workgroups → MMU fault.
- **Fix:** force rocBLAS to dispatch EVERY solution for the gateup shape and capture each variant's kernarg. rocBLAS
  computes the correct mapping fields for each; we record them. Then tinygrad HCQ can launch any captured variant
  (no HIP at measure-time) → the variant-ablation is unblocked.
- **Why it audits:** with per-variant launch, we run the SAME PMC scoreboard (cycles / DRAM / stalls / occupancy /
  VALU) while varying ONE knob → a measured counterfactual we currently lack. Specifically:
  - `LDSB0` vs `LDSB1` at fixed 128×128 tile = isolate **software-pipelining / double-buffering** (tile, grid,
    occupancy all held constant — the cleanest single-primitive ablation).
  - tile `128→64→32` = isolate **occupancy ↔ LDS-reuse** tradeoff.
  - K-depth `16→32` = isolate **pipeline/block depth**.
  This converts "Tensile is a black box; the win is scheduling quality (inferred from A3)" into "the win is X% from
  staging, Y% from occupancy, Z% from pipelining (measured)" — or proves the gap is invariant to these knobs
  (→ confirms it's holistic scheduling quality, now with Tensile-side evidence, not just A3's tinygrad-side one).

## Existing pieces to reuse
- `extra/qk_tensile_kernarg_capture.cpp` — LD_PRELOAD shim hooking `hipModuleGetFunction` (fn→symbol) +
  `hipExtModuleLaunchKernel` (dump kernarg+geometry). Currently **deduped to first-solution-per-role**.
- `extra/qk_prefill_blas_ceiling.cpp` — host-only rocBLAS driver (compiles, `/tmp/qk_ceiling` exists); already calls
  `rocblas_gemm_ex` for these shapes. Confirms host-only rocBLAS avoids the split HIP/rocBLAS device-toolchain issue.
- `rocblas_gemm_ex_get_solutions` + `rocblas_gemm_algo_solution_index` — present in this rocBLAS (rocblas-beta.h).
- `extra/qk_tensile_variant_ablation.py` — the per-variant HCQ launcher + correctness + PMC (built; currently reuses
  the single kernarg).
- `extra/qk_tensile_hcq_launch.py` `NamedAMDProgram` / `kd_offset` — load any symbol from the `.co`.

## Phase plan
### C0 — Solution-sweep host driver (`extra/qk_tensile_solution_sweep.cpp`)
For the gateup shape (M=512,N=12288,K=4096, HHS = fp16 in / fp32 compute): `rocblas_gemm_ex_get_solutions` → for each
valid index, `rocblas_gemm_ex` with `rocblas_gemm_algo_solution_index` (dispatch that variant once). Host-only rocBLAS.

### C1 — Extend the capture shim
Capture EVERY distinct dispatched symbol (key by symbol, drop the per-role dedup) → JSONL
`{symbol, kernarg_bytes(128), global, local, num_workgroups, M,N,K}` to `bench/qk-tensile-ablation/kernargs.jsonl`.

### C2 — Build + run capture
Compile shim (`g++ -shared -fPIC`) + driver (host-only rocBLAS link, the qk_ceiling recipe). Run
`LD_PRELOAD=shim ./sweep` → capture N variants' kernargs for the gateup shape. Verify ≥1 each of the ablation knobs
(LDSB0/LDSB1, tiles, K-depth) is present.

### C3 — Wire per-variant launch
Extend `qk_tensile_variant_ablation.py`: look up the captured kernarg for the requested symbol (instead of reusing the
one), substitute tinygrad buffer VAs, set the captured grid, launch via HCQ, correctness-gate. Confirm ≥2 distinct
variants now launch correctly (the original blocker is cleared).

### C4 — The ablation sweep
For each correct variant: PMC scoreboard (2 passes) + cycle timing + disasm (LDS bytes, ds_load, barriers, wmma/fma).
Build the matrix variant → {tile, K-depth, LDSB, lds_B, waves} × {cycles, DRAM, stalls, occupancy, VALU}.

### C5 — Attribution + verdict
Read sensitivities (LDSB0 vs LDSB1 = pipeline; tile = occupancy/LDS; K-depth = pipeline depth). Output the measured
"which primitive, how much" — or the evidenced "gap is invariant to these knobs → scheduling quality." Correct
`prefill-tensile-DEFINITIVE-source-of-truth` §4/§5 with the MEASURED attribution.

## Gates / hazards
- Each variant launch in its own subprocess (an MMU fault wedges the device).
- Verify per-variant CORRECTNESS before trusting counters.
- PMC perturbs timing → cycle/count RATIOS only; ≤8 counters/pass.
- Confounded knobs (tile moves occupancy AND LDS) → attribute at the granularity the data supports.
- If `rocblas_gemm_ex_get_solutions` returns few/no alternatives for this shape, capture falls back to whatever
  rocBLAS dispatches across a few problem sizes that hit different tiles (documented as a partial result).
- Split-toolchain: keep the driver HOST-ONLY rocBLAS (no `__global__` kernels) per the qk_ceiling precedent.

## Deliverables
- `docs/tensile-variant-capture-result-20260619.md`; `extra/qk_tensile_solution_sweep.cpp`; extended
  `qk_tensile_kernarg_capture.cpp` + `qk_tensile_variant_ablation.py`; `bench/qk-tensile-ablation/{kernargs.jsonl,
  ablation_matrix.json}`. Measure-only, no route/default change.

## Definition of done
≥3 distinct gateup variants launchable+correct via HCQ with captured kernargs, measured with the PMC scoreboard,
yielding a defensible per-knob attribution (or the evidenced "scheduling-quality, knob-invariant" verdict). Unblocks
Tensile audit beyond the single kernel.
