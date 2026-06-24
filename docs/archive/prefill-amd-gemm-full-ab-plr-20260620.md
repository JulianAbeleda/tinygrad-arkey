# Prefill AMD GEMM — Full A+B PLR (built, correct, but dominated)

Date: 2026-06-20

## Result

`FULL_AB_PLR_LOSES_TILE_COST`. Full A+B prefetch (`PLRAB`) is **implemented and correct (rel RMSE 2.08e-4)**,
but it **loses to A-only PLR at the full 4×4 tile** — reproduced 3×, `full/A-only = 0.913`. The dependency-free
optimum stays **A-only PLR at 128×128 (~61 pinned)**. The dependency-free route to the last ~8% via PLR is
**closed**; only the vendored `.co` reaches Tensile.

Implementation: `build_gemm_lds2(..., PLRAB=1)`. Probe: `extra/qk_amd_gemm_plrab_probe.py`.

## The VGPR budget (why full PLR forces a smaller tile)

Full A+B PLR needs a **2nd A+B fragment buffer** (`WM·8 + WN·8` VGPR) live during compute, on top of
`FA + FB + ACC`:

| tile | `10 + FA(32) + FB + ACC + 2nd-buf` | fits 256? |
|---|---|---|
| 4×4 (128×128) | `10 + 32 + 32 + 128 + 64 = 266` | **no (−10)** — assert rejects (VGPR 300 w/ control) |
| 4×3 (128×96) | `10 + 32 + 24 + 96 + 56 = 218` | yes |

So full PLR is impossible at 4×4 and only fits by shrinking the tile. A-only PLR (one 32-VGPR buffer) was the
most that fit at 4×4.

## Measured (pinned clock, interleaved, reproduced 3×)

| config | tile | TFLOPS | note |
|---|---|---:|---|
| **wn4_plra1** (A-only PLR) | 4×4 | **61.0–61.6** | dependency-free best |
| wn3_plra0 (bank-fix, no PLR) | 4×3 | 58.5 | smaller-tile baseline |
| wn3_plrab1 (**full A+B PLR**) | 4×3 | 55.7–56.2 | 0.913× the 4×4 A-only |
| authority (LLVM) | — | 53.1–53.3 | |

Two findings:

1. **Full A+B @4×3 (56) < A-only @4×4 (61).** The WN=3 reuse loss outweighs the full latency hiding.
2. **Full A+B PLR even regresses its own 4×3 tile (58.5 → 56).** The 2nd buffer pushes VGPR usage to ~248,
   dropping VGPR-limited occupancy from ~8 → 6 waves/SIMD; that occupancy loss exceeds the prefetch gain.

So the 2nd-buffer cost (smaller tile **and** lower occupancy) dominates the benefit. Full prefetch is the wrong
trade on this 256-VGPR file.

## Why A-only @4×4 is the optimum

A-only PLR keeps the full 128×128 tile (max WMMA reuse) **and** full occupancy (one 32-VGPR buffer reusing the
dead coop-load temps, no net VGPR growth), hiding the larger of the two operand latencies (A). Full A+B can
only be bought by giving up tile reuse or occupancy — and both cost more than the extra B-latency hiding. This
is the same reuse↔register tension that capped BK-depth, now confirmed for prefetch depth.

## Standing — the dependency-free arc is complete

| kernel | clock-matched TFLOPS |
|---|---:|
| LLVM authority | ~53 |
| **ours (dependency-free optimum): BK32 + PAD16 + A-only PLR @4×4** | **~61 pinned / ~92% of Tensile** |
| Tensile `.co` (vendored dep) | ~62 |

The dependency-free optimum is `build_gemm_lds2(BK=32, PAD=16, PLRA=1)` at the 4×4 tile, wg2. **Full A+B PLR
does not improve it** — the only dependency-free route to the last ~8% is closed by the VGPR budget. Reaching
Tensile requires either the vendored `.co` (traceable, ~62, the declined dependency) or a fundamentally
different register schedule (Tensile's full pool that overlaps C-accumulators with fragments — not expressible
as a knob on this kernel).

## Honesty

- Full A+B PLR is correct (2.08e-4) and reproduced; it simply doesn't win — an honest negative.
- `wn3_plra1` (A-only at 4×3) is unbuildable (dead coop temps = 28 < WM·8 = 32 for WN=3) — not relevant.
- Single shape; pinned-clock interleaved, reproduced 3×; `PLRAB=0`/`PLRA=0` defaults leave the proven kernel
  byte-identical.

## Verdict

`FULL_AB_PLR_LOSES_TILE_COST` — full A+B prefetch is built and correct but dominated by A-only PLR at the full
tile. The dependency-free GEMM rests at **~61 pinned (~92% of Tensile, +15% over the LLVM authority)**; PLR
depth, like BK depth, hits the reuse↔register wall. The remaining ~8% is the vendored `.co` or a
register-pool rewrite — neither a knob.
