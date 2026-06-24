# Prefill AMD GEMM — Clock-Pinned Measurement (corrects the parity projection)

Date: 2026-06-20

## Why

The PLR experiment reported "+10–11%, projects to ~67 ≈ Tensile parity" — but the projection used a
non-reproducible auto-boost reading (plra0 = 60.7 in one session). `rocm-smi --setperflevel high` pins the
clock, so we can replace the volatile best-of-N + projection with a **reproducible** absolute. It works
without root.

## Result — pinned high, reproduced 5×

| config (pinned high) | TFLOPS | spread |
|---|---:|---:|
| LLVM authority | 53.3 | ±0.2 |
| BK32 + PAD16 (bank-fix) | 56.0 | ±0.3 |
| **BK32 + PAD16 + A-prefetch PLR** | **61.1** | ±0.4 |
| PLR gain | **+9%** | reproducible |

Pinning **collapsed the variance** (auto: 54–61 volatile → pinned: 61.1 ± 0.4). Live sclk during compute
hits ~2170 MHz (near the 2251 peak), so this *is* near-peak — the kernel is reproducibly ~61.

## What this corrects

- **The "~67 / Tensile-parity projection" is retracted.** That relied on a one-off auto-boost reading
  (plra0 = 60.7); at the *reproducible* pinned clock, plra0 = 56.0, so plra1 = 61.1, **not ~67**.
- **The +9–11% PLR gain stands** (reproducible, PMC-confirmed latency hiding). It is real; the over-optimism
  was only in the absolute projection, which the auto-boost inflated.

## Honest standing (reproducible)

- Dependency-free prefill GEMM = **61.1 ± 0.4 TFLOPS** at pinned high clock, correct (2.08e-4).
- **+15% over the LLVM authority** (53.3) — clean, reproducible, clock-controlled.
- **~92% of Tensile (~66)** — NOT parity at a stable clock. (Caveat: Tensile's ~66 may itself include
  auto-boost; a clock-matched Tensile number would need running the `.co` in the same pinned interleave,
  which we don't. Anchored to the authority: Tensile ≈ +24% over authority, ours ≈ +15% → ~93% of Tensile's
  authority-relative advantage.)

## The three composable dependency-free levers (final, reproducible)

| lever | reproducible effect | mechanism (measured) |
|---|---|---|
| wg2 occupancy | avoids the wg4 L2-contention dip | L2 hit 56→64 |
| PAD16 bank-conflict-free LDS | 53→56 (~+5% here; +13% in the volatile session) | bankcf 28.6→2.7/cyc |
| A-prefetch PLR | 56→61 (+9%) | SIMD busy/active 36.5→37.3 (latency hiding) |

(The per-lever percentages shift between the volatile and pinned sessions; the *ordering and mechanisms* are
stable, the pinned absolutes are the trustworthy ones.)

## Verdict

Clock-pinning did its job: it made the result **reproducible** and **corrected an over-optimistic
projection**. The dependency-free arc lands at a **reproducible ~61 TFLOPS, +15% over the LLVM authority,
~92% of Tensile** — Tensile-class, correct, zero dependencies, every lever named and measured. Full Tensile
parity (~66) is **not** reached at a stable clock; closing the last ~8% needs full A+B PLR (Tensile's
both-operand register overlap) or the vendored `.co`.

## Honesty / housekeeping

- `rocm-smi --setperflevel high` is the reproducibility tool; reset to `auto` after measuring to restore
  default GPU state.
- Single shape; the pinned numbers are reproduced 5× (±0.4); authority is the clock anchor.
- This supersedes the "edge of parity / ~67 projected" framing in the PLR doc with the reproducible ~61.
