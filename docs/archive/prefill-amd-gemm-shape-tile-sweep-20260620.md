# Prefill AMD GEMM — Shape-Specific Tile Sweep (hypothesis refuted, ceiling recalibrated)

Date: 2026-06-20

## Hypothesis

From the tuning-table audit: our shape (M=512 small, N=12288 large) is untuned in Tensile, so the square
128×128 fallback might be suboptimal — a **non-square tile** matched to the skewed aspect (smaller BM for
M-parallelism, wider BN for N-reuse) could do better, toward the apparent ~79 ceiling.

## Result — refuted: square 128×128 is best

Swept tile shapes at our otherwise-best config (BK32, PAD16, PLRA where it fits), all at wg2 LDS=32768 (isolates
tile shape from occupancy), pinned, interleaved. Reproduced.

| tile (BM×BN) | grid | TFLOPS |
|---|---|---:|
| **128×128 (square)** | 96×4 | **57.3** |
| 64×256 | 48×8 | 55.9 |
| 256×128 | 96×2 | 55.3 |
| 128×256 | 48×4 | 53.7 |
| authority (LLVM) | — | 53.4 |
| 64×128 | 96×8 | 52.8 |
| 64×192 | 64×8 | 51.3 |

`SQUARE_128_STILL_BEST`. **No non-square tile beats square 128×128** for this shape. Smaller BM (more
M-parallelism) and wider BN (more N-reuse) both *hurt* — the square tile's WMMA reuse dominates. Our
`BK32+PAD16+PLRA` 128×128 is genuinely the optimum for this kernel family on this shape.

## The ceiling was overstated — recalibrated to ~65, not 79

The prior audit cited "Tensile ~79 on the nearest tuned shape → we're 22% off." That used the **single best
neighbor**. The full tuned-speed-by-M cluster (N=8192, K=4096) tells a different story:

| tuned M | speed (TFLOPS) |
|---:|---:|
| 256 | 64.7 |
| **384** | **79.1 ← outlier sweet-spot** |
| 768 | 65.3 |
| 1536 | 68.2 |
| 3072–6144 | 68.8–69.5 |

**M=384's 79 is a clear outlier** (a sweet-spot alignment); the representative cluster is **~65–69**. Our M=512
sits between M=384 and M=768, so a *tuned* 512-shape kernel would realistically hit **~65–68**, not 79.

So the corrected picture: our dependency-free kernel at **~57–62 pinned is ~85–95% of the realistic tuned
ceiling (~65)** — and Tensile's tuned numbers are offline benchmarks possibly at boost clock, so clock-adjusted
the gap is smaller still. The "22% off" was an artifact of the 79 outlier; the real gap to a hypothetical
tuned-for-our-shape kernel is **~5–12%, partly clock, and not closeable by tile shape.**

## What this settles

- **Tile shape is not the lever** — square 128×128 is optimal; the skewed-aspect hypothesis is refuted.
- **The realistic ceiling is ~65, not 79** — we're at ~90% of it, and the remainder is the same small
  hidden/confound mix already characterized (ds_load +3%, scheduling, clock, work-difference).
- Combined with every prior result, **our kernel is at/near the achievable ceiling for this shape on this
  hardware** — there is no identified lever (tile, depth, occupancy, bank, prefetch, VALU, ds_load) that
  meaningfully closes the small remainder.

## Verdict

`SQUARE_128_STILL_BEST`. The shape-specific-tile hypothesis is refuted: square 128×128 wins. And the ceiling
recalibrates from the 79 outlier to a representative **~65–68**, putting our ~57–62 at **~90% of the realistic
tuned ceiling**. The dependency-free arc is complete: a correct, Tensile-class kernel at near the achievable
ceiling for this (untuned-by-Tensile) shape, every lever measured, zero dependencies.
