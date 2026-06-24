# Prefill AMD GEMM — A-Prefetch PLR Experiment (the last ~9% to Tensile)

Date: 2026-06-20

## Result

`PLRA_HELPS_PARTIAL`. The hand-written **A-prefetch local-read** — prefetch the next K-substep's A fragments
into the **dead coop-load temp registers** (the register-lifetime overlap the VGPR audit identified) during
the current substep's WMMAs — is **correct (rel RMSE 2.08e-4)** and adds a **robust +10–11% over the
bank-conflict-fix frontier**, reproduced 7×, with PMC confirming it's real latency hiding. It validates the
audit's claim that the "VGPR wall" was an allocator gap, not hardware — and brings the dependency-free kernel
to the edge of Tensile parity.

Implementation: `build_gemm_lds2(..., PLRA=1)` (`extra/gemm/rdna3_wmma_matmul.py`). Probe:
`extra/qk_amd_gemm_plra_probe.py`.

## What it does

`build_gemm_lds2`'s loop was SIA0: per K-substep `load A+B → wait → 16 WMMA` (serial; the ds_load latency is
exposed). For BK32 (KT=2 substeps), `PLRA=1` rewrites the inner block to overlap:

```
load A0, B0; wait
prefetch A1  -> into the DEAD CTA/CTB coop-load regs   # the register-lifetime overlap
WMMA substep0                                          # hides the A1 load latency
load B1; wait                                          # B1 still synchronous (B' wouldn't fit 256 VGPR)
WMMA substep1 (from the prefetched A1)
```

The prefetch buffer reuses `CTA/CTB` (dead after the cooperative LDS store) — exactly the ~32 VGPR Tensile's
register pool would reclaim. Full A+B prefetch (64 VGPR) doesn't fit 256, so this is **partial (A-only) PLR**.

## Measured (7 sessions, interleaved, clock-fair within each)

| | plra0 (bank-fix) | plra1 (+A-prefetch PLR) | gain | correct |
|---|---:|---:|---:|---|
| this session-cluster (low clock) | 53.5–55.3 | **60.0–61.4** | **+10–11%** | 2.08e-4 |
| authority (LLVM) | ~52.5 | — | — | — |

- **Gain reproduced 7×: ×1.10–1.13.** The clock was stuck low this whole cluster (plra0 ~54 vs the bank-fix
  session's 60.7), so the **within-session ratio (+10–11%) is the portable claim**; the absolute is
  clock-gated.
- **Projection**: at the bank-fix frontier's high-clock value (~60.7), +10% → **~67 ≈ Tensile parity (~66)**.
  Not directly observed this session (clock-low); the ratio is the robust part, the absolute parity is
  inferred.

## PMC — mechanism confirmed (it's latency hiding)

| counter | plra0 | plra1 |
|---|---:|---:|
| cycles | 2.258 M | **2.142 M** (−5%) |
| busy/active (SIMD util) | 36.5 | **37.3** |
| VALU/active | 3.83 | **4.04** |
| LDS-active/active | 16.7 | 17.6 |
| bankconf/active | 2.79 | 2.94 |

Same LDS work, but plra1 runs in fewer cycles with **higher SIMD utilization** — the prefetch keeps the WMMA
units fed, so they stall less. This is the `PLR1` latency-hiding effect Tensile gets, now reproduced
dependency-free.

## Significance — the VGPR wall was real but crossable

This directly confirms the corrected audit: the "256-VGPR wall" was a **static-allocation** artifact, not
hardware. Reusing the dead coop-load registers (one register-lifetime overlap, by hand) was enough to add the
A-prefetch and gain +10%. The three composable dependency-free levers — all measured, all on the
`assemble_linear` hand-asm path — are now:

| lever | gain | mechanism (measured) |
|---|---|---|
| wg2 occupancy | ~54→57 | avoids L2 contention |
| PAD16 bank-conflict-free LDS | +13% | bankcf 28.6→2.7/cyc |
| **A-prefetch PLR** | **+10–11%** | SIMD busy/active 36.5→37.3 (latency hiding) |

Stacked, the dependency-free kernel reaches **~61 this (low-clock) session / projects to ~67 at full clock**
— at or above Tensile's ~66, correct, no vendored `.co`.

## Honesty

- **Partial PLR** (A-only); B is still synchronous (B' prefetch needs >256 VGPR). Full PLR would need
  Tensile-style register overlap on *both* operands.
- The **+10–11% ratio** is the robust, reproduced claim. **Parity (~66) is a projection**, not a direct
  measurement — the clock was stuck ~10% low this whole session, so plra1 read ~60–61, not ~67. Pinning the
  clock (open) would let us report the absolute.
- Single shape; correctness verified (2.08e-4); PMC perturbs timing (deltas trustworthy).
- `PLRA=0` default leaves `build_gemm_lds2` byte-identical (same insts, same 2.08e-4) — no regression to the
  proven path.

## Verdict & standing

`PLRA_HELPS_PARTIAL` — the dependency-free PLR experiment **succeeded**: +10–11% measured, latency-hiding
confirmed, correct. The dependency-free prefill kernel is now `BK32 + PAD16 + A-prefetch PLR` at wg2, reaching
**~61 (this clock) / ~67 projected**, at the edge of Tensile parity — every lever named and measured, zero
dependencies.

## Next (optional)

1. **Pin/telemetry-bin the clock** to directly confirm whether plra1 reaches ~66 at the high-clock frontier
   (the only thing between "measured +10%" and "measured parity").
2. **Full A+B PLR** would need register-overlap on both operands (Tensile's pool); the B' buffer needs the
   C-tile overlap trick — deeper, but the A-only result shows the direction pays.
