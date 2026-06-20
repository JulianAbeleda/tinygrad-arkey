# RESULT — Tensile variant-ablation audit: BLOCKED at launch (gfx1100, 2026-06-19)

Executes `tensile-variant-ablation-scope-20260619.md` P0/P1. **Verdict: the variant-ablation experiment is BLOCKED —
only the ONE rocBLAS-selected kernel (whose kernarg we captured) launches; every other variant faults. So we cannot
ablate Tensile by selecting variants, and my earlier "yes, the 580 variants are a built-in ablation matrix" claim was
too optimistic.** Driver: `extra/qk_tensile_variant_ablation.py`.

## P0 — variant space (clean, done)
580 kernels, all `MI16x16x16` (WMMA — confirms the in-model kernel uses WMMA, not FMA-only). Varying knobs: tile
`{32,64,128}` × K-depth `{16,32}`; **`LDSB0`(360) vs `LDSB1`(220) = the double-buffer / software-pipeline knob**;
`WGM{4,8}`; `AMAS{0,3}`. Invariant: GSU1 (no split-K), `PGR1`/`PLR1` (all prefetch), `WG32_4_1` (all 128 threads/wg,
4 waves). So the *intended* clean ablation (LDSB0 vs LDSB1 at fixed tile) is well-defined.

## P1 — launch + correctness: BLOCKED
- Reusing the captured 128-byte kernarg + per-tile grid, **only the exact selected kernel runs** (rel_err 3.5e-4);
  **8/8 sibling 128×128 variants (LDSB0/LDSB1, WGM4/8, AMAS0/3) MMU-fault** (same fault VA, and a fault wedges the
  device — so each variant must run in its own subprocess).
- Cause: the captured kernarg is **bound to the one kernel**. Tensile passes per-kernel workgroup→tile mapping
  fields (magic-number division constants / WGM mapping) in the kernarg; reusing the selected kernel's kernarg for
  any other variant misroutes workgroups → out-of-bounds → fault. Even same-tile (128×128) siblings fault, so it is
  NOT a simple grid recompute.
- The `.kd` `kernarg_size` reads 0 for these kernels (metadata-only), so per-variant kernarg layout can't be derived
  from the descriptor either.

## What this means for "auditing Tensile"
| capability | status |
|---|---|
| Static disasm of ANY variant's ISA | ✅ works (but per-variant body extraction is fiddly — 580 names share huge common substrings; needs symbol-range, not substring, matching) |
| PMC + cycles on the ONE selected kernel | ✅ works (already measured: memory-stall-bound, 6.6× DRAM, 6.5× stalls, VALU-equal) |
| **Variant-ablation (launch+measure other variants)** | ❌ **BLOCKED** — kernarg bound to one kernel |
| SQTT per-wave instruction/stall trace on the ONE kernel | exists (heavyweight), untested here — the remaining deep-audit path, but gives no counterfactual |

**Unblocking variant-ablation requires per-variant kernarg CAPTURE:** a HIP program that forces each rocBLAS solution
index (`rocblas_gemm_ex` + `rocblas_gemm_ex_get_solutions`) for the gateup shape and LD_PRELOAD-captures each kernel's
kernarg — a separate build (and runs into the split HIP/rocBLAS toolchain). Not a quick win.

## Honest verdict on "which primitives make Tensile win"
With current tooling we **cannot** attribute the win to a single primitive by ablation. The defensible statements,
in decreasing certainty:
1. **MEASURED (one kernel):** the gap is **memory-stall-bound, not compute** — VALU-equal, 6.6× more DRAM reads, 6.5×
   more stalls on the WMMA side. (solid)
2. **STRUCTURAL (disasm):** Tensile's kernel is LDS-staged (24.5 KB), 4-wave/wg, double-buffered (LDSB), WMMA+FMA. It
   *has* the primitives. (solid)
3. **COUNTERFACTUAL (the only one we have — A3, not Tensile):** when those primitives (double-buffer SW-pipeline +
   occupancy + bank-pad) were built into tinygrad's own kernel on this exact shape, they were **net-negative** (6 vs
   32 TFLOPS). So the win is NOT "having the primitives" — it is the **integrated instruction-scheduling quality**
   that LLVM/Tensile produce and tinygrad's renderer does not. (this is the strongest available answer, and it is a
   holistic/negative one — NOT a single-primitive attribution)
4. **RETRACTED:** "dense issue" (VALU-equal refutes it); and the implication that variant-ablation could cleanly
   isolate the primitive (blocked).

So: "the primitives that make Tensile win" is partly the **wrong question** — the same primitives in tinygrad make it
slower. The honest answer is *scheduling quality*, and we know that from A3's counterfactual, not from a Tensile
ablation (which we can't run).

## If we want to push further (options, none free)
- **Per-variant kernarg capture** (force rocBLAS solution selection + capture) → unblocks the LDSB / tile / K-depth
  ablation → a real measured sensitivity. Separate HIP-side build.
- **SQTT trace on the selected kernel** → per-wave stall-reason histogram (mem-wait vs barrier vs dep) → localizes
  the stall, but no counterfactual ("if I removed the pipeline…").
- **Accept the current answer:** memory-stall-bound + the A3 counterfactual = "scheduling quality, not a bolt-on
  primitive," and stop. The shippable lever is unchanged (vendored Tensile `.co`, byte-identical, 1.84×).
