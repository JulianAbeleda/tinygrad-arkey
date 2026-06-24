# BB-5a.10 PTM-3 — Native Candidate Scope: `software_pipelined_k_loop`

Date: 2026-06-20

Inputs: PTM-1 (`bb5a10_ptm1_…result.json`), PTM-2 decision, `codegen_oracle.json`.

Verdict:
`PTM3_SCOPED_SOFTWARE_PIPELINED_K_LOOP_BUILD_IS_CODEGEN_WALL`

**Scope only. This does NOT build a kernel.** It defines the single chosen row's contract and gates so that
*if* the codegen capability is ever funded, the work is pre-specified and measured under the PTM-1 harness.

## The one row

`software_pipelined_k_loop` — overlap the next K-block's global loads with the current block's WMMA, via
double-buffered global→(LDS or register)→WMMA prefetch (Tensile PGR1/PLR1/DepthU16). Macro-tile (128×128×16)
and the WMMA fragment (16×16×16) are already at parity per the oracle; the only delta is K-loop scheduling.

## Dataflow contract
- K-loop processes DepthU=16 per iteration; while WMMA consumes block *i*, the global loads for block *i+1*
  are already in flight (prefetch issued before the WMMA, waited on after).
- Operand staging may use LDS **only as part of this overlap** (not standalone — that's closed). The
  load→stage→WMMA register handoff must be explicit and dependency-correct.
- The blocker, stated plainly: tinygrad's linearizer cannot hoist a global load across the loop RANGE, so
  the prefetch cannot be expressed today (BEAM-hang / linearizer-RANGE class). This is the capability gap,
  not a parameter tweak.

## Gates (all measured under the PTM-1 harness — interleaved, one clock, best-of-N)
1. **Correctness:** sampled-tile relative RMSE ≤ 0.05 vs `a@b` reference (the existing `sample_correctness`
   in the P8 scripts).
2. **Resource:** scratch/private = 0; VGPR within occupancy budget (no spill — prior "more-acc → 11 TFLOPS
   spill" is the failure mode to avoid).
3. **Performance:** best TFLOPS must **exceed tinygrad's authority** in the *same interleaved run* (i.e.
   beat the `authority_tinygrad` row, ~53 this clock / ~42 nominal). Catching up to authority is not a win —
   PTM-1 showed tinygrad already achieves it via LLVM. Stretch target = Tensile ~66. Report with sclk
   provenance and the authority row measured in the same process (never a stored cross-session number).

## Why this is a direction, not a build order
Per MEMORY (POWN, Route-A A1/A2/A3, both why-tensile docs): the dependency-free hand-ASM SW-pipeline was
already attempted and capped at ~24-32 TFLOPS — below LLVM, far below Tensile. Achieving ≥ authority
requires a genuine renderer/linearizer capability (cross-RANGE load hoisting + dense dual-issue scheduling),
a multi-day-to-month effort with high uncertainty whose ceiling is likely ~LLVM (WMMA-capped), not Tensile.
**PTM-3 records the contract; it does not authorize the build.** Funding that capability is a separate
project decision.

## What this leaves for the user (PTM-4 / PTM-5)
- **PTM-4 (your call):** the *only* near-term path to ~66/87%-llama is the external rocBLAS Tensile `.co`
  (dependency policy). PTM-1 makes this sharper: dependency-free, tinygrad already sits at its own authority
  (~42 nominal / ~47% llama); the SW-pipeline build is walled. So the realistic fork is **accept the
  vendored `.co` (→ ~87% llama) vs rest at authority (~47%)**.
- **PTM-5:** decode transfer stays blocked (`ROADMAP_ONLY`, max movement 14.087µs < 30µs).

## Next

Stop. PTM-4 is the user's dependency-policy decision; PTM-5 is blocked. No further kernel work without
funding the SW-pipelined-K-loop codegen capability.
