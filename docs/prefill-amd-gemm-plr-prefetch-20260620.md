# Prefill AMD GEMM — PLR / Prefetch (the last ~60→66)

Date: 2026-06-20

## Result

`PLR_DBUF_NO_PARITY_RESTS_AT_TENSILE_CLASS`. The available prefetch lever cannot close the last gap to
Tensile. The dependency-free frontier **rests at Tensile-class ~60 TFLOPS** (BK32 + PAD16, wg2); full Tensile
parity (~66) needs intra-substep PLR that overflows the VGPR file at the productive tile.

Probe: `extra/qk_amd_gemm_plr_probe.py` → `bench/.../amd_gemm_plr_result.json`.

## What was tested

`build_gemm_lds2` has a double-buffer prefetch lever (`DBUF`): prefetch the next K-block's global loads + LDS
writes while computing the current block (Tensile's `PGR`). Stacked on the PAD16 bank fix, at controlled
occupancy. Question: does prefetch + bank-fix + wg2 reach parity (≥64)?

| config | TFLOPS (best, reproduced) | wg/CU | note |
|---|---:|---:|---|
| bk32_pad16_dbuf0 | ~56–60 | 2 | bank-fix frontier |
| bk32_pad16_dbuf1 | ~59 | 1 | + prefetch, but LDS forces wg1 |
| bk16_pad16_dbuf1 | ~51 | 2 | prefetch + fix + wg2, but low density |
| bk16_pad16_dbuf0 | ~47 | 2 | BK16 fix only |
| bk16_pad32_dbuf1 | ~49 | 2 | BK16 deeper pad + prefetch |
| pad0_dbuf1 (earlier) | 58.1 | 2 | prefetch helps **+8%** when it fits wg2 |
| authority (LLVM) | ~52.5 | — | global-direct |

## Why prefetch can't close it — the 3-way budget squeeze

The win needs four things at once, and **64 KB LDS + 256 VGPR can't hold all four**:

1. **BK32 compute density** (deeper K-block → ~55 base vs BK16's ~42).
2. **PAD16 bank-conflict-free layout** (the ~11× conflict cut → +13%).
3. **wg2 occupancy** (the L2-contention sweet spot).
4. **A second LDS buffer** for prefetch overlap (`DBUF`).

- `DBUF` *does* help (+8% on `pad0_dbuf1` at wg2) **when it fits**. But BK32+PAD16+DBUF = 40960 B > 32768 →
  **forces wg1**, and the lost occupancy ≈ cancels the prefetch: dbuf1@wg1 vs dbuf0@wg2 is a **tie at ~56–60**,
  and the ordering **flips with clock across sessions** (one session dbuf0 won, another dbuf1) — so neither is
  a robust win.
- Dropping to **BK16** to fit prefetch+fix+wg2 (24576 B) **sacrifices too much compute density** (~47–51 ≪
  BK32). Net loss.

So the four requirements are mutually exclusive on this hardware budget via these knobs.

## The real wall — intra-substep PLR is VGPR-blocked

True Tensile `PLR1` overlaps the *next fragment read* with the *current WMMA* **inside** the K-block (no
second LDS buffer needed). That needs the next fragments live in registers while the current ones compute:
**2× fragment VGPRs (128) + accumulators (128) > 256**. At the productive 4×4 tile it overflows; a smaller
tile loses the reuse that made it fast. Tensile fits `PLR1` at 256 VGPR via its specific fragment/accumulator
register schedule — reproducing that is a separate, deep register-allocation project, not a knob on
`build_gemm_lds2`.

## Honesty

- All within-run interleaved, reproduced; the dbuf1↔dbuf0 ordering is **clock-dependent** (±~6%), so it is
  reported as a **tie**, not a gain — the robust claim is "no combo reaches parity (≥64)."
- Single prefill shape, best-of-N, clock-volatile absolute TFLOPS.

## Verdict & where this rests

`PLR_DBUF_NO_PARITY_RESTS_AT_TENSILE_CLASS`. The dependency-free path is **done at Tensile-class ~60 TFLOPS**
(BK32 + PAD16, wg2) — correct, ~14% over the LLVM authority, with the full mechanism named and measured. The
residual to Tensile (~66) is **intra-substep PLR**, walled by the 256-VGPR budget at the productive tile.

## Final standing of the arc

| milestone | TFLOPS | how |
|---|---:|---|
| global-direct hand-asm | ~29 | baseline |
| LLVM authority | ~53 | tinygrad's own WMMA |
| BK32 single-buffer (wg4 default) | ~49 | L2-contention-limited |
| + wg2 occupancy | ~57 | L2 sweet spot |
| **+ PAD16 bank-conflict fix** | **~60.7** | **crosses Tensile-class** |
| Tensile selected | ~66 | + `PLR1` register-scheduled prefetch (VGPR-walled for us) |

**Dependency-free, correct, Tensile-class, beating tinygrad's LLVM authority by ~14% — and every step named
and measured.** The last ~9% to Tensile is a register-allocation wall, or the vendored `.co`.
