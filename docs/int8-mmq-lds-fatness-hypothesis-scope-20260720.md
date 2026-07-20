# Scope: is LDS fatness the real cause of the int8-MMQ wedge? + int8-MMQ target

Date 2026-07-20. Follows the llama-vs-tinygrad prefill gap analysis and the exhaustive tile/pack/wait audit (llama Q4_K MMQ vs ours).

## Background (from the audit)
- The 14B prefill gap is entirely **GEMM kernel efficiency**: llama ~61 TFLOP/s vs tinygrad direct-packed ~11 (~5.6× ≈ the 364→1837 tok/s gap). Direct-packed uses **no WMMA** (scalar FMA); llama uses **int8 WMMA** (`v_wmma_i32_16x16x16_iu8`).
- Waits: **identical** (both single-buffered, 4 `__syncthreads`/superblock, no cp.async on AMD) — not a differentiator.
- Tile: essentially the same (128×128, K32, 8 warps, 16×16×16 fragment).
- **Packing is the lever:** llama keeps weights+activations int8 → int8 tensor cores (~2× fp16 WMMA), and folds the Q4_K `−dmin·min` correction via the pre-summed activation (DS4 trick), no 2nd dot.
- Our earlier **generated int8-MMQ was abandoned** because it wedged the GPU at scale (SQ type-2 + MES REMOVE_QUEUE reset at ~64 workgroups). It used **57,856 B LDS** (q4 full-K panel 38912 + q8 18432 + ids 512). llama's int8-MMQ uses **~43 KB** (staged per-superblock).

## The int8-MMQ target (what to build if the approach is viable)
Match llama's structure (see the tile/pack/wait audit): int8 weights (raw nibbles in LDS) + int8-quantized activations (block_q8_1_mmq, DS4 = half2(d,sum)), `v_wmma_i32_16x16x16_iu8`, DS4 min-correction folded via `dsB.y·dmA.y`, **lean per-superblock ~43 KB LDS** (not the fat 57 KB full-K-resident layout), bank-conflict padding (`K%8==4`), single-buffered with the 4-barrier/superblock cadence.

## The hypothesis under test
- **H1 (LDS fatness):** the 57 KB LDS caused the wedge; reducing LDS toward llama's 43 KB (or lower) removes it.
- **H1 is suspect on the numbers:** gfx1100 CU mode ≈ 64 KB LDS/CU, so **57 KB → 1 wg/CU AND 43 KB → 1 wg/CU** — same occupancy. If LDS-driven occupancy were the cause, 43 KB wouldn't help. And llama's 43 KB int8-MMQ runs the full 14B grid fine (measured 1837 tok/s).
- **H2 (kernel/dispatch bug):** the wedge is specific to OUR int8 port (or the tinygrad HCQ/PM4 dispatch of a many-workgroup int8-WMMA kernel), not LDS size. llama's int8 at the same tile/LDS class does not wedge.

## Experiment (this scope's test)
Controlled LDS sweep on real gfx1100: a synthetic minimal WMMA kernel with **tunable LDS** (allocate K bytes, barrier + a WMMA K-loop, write output), dispatched at increasing workgroup counts (32 / 64 / 128 / 256 / 544). Sweep LDS ∈ {16, 32, 43, 48, 57, 64} KB. For each (LDS, grid): clean or wedge (SQ type-2 + reset)? Guarded, single GPU lane.
- **If the wedge threshold tracks LDS** (appears above some KB) → H1, LDS is the cause → build the lean 43 KB int8-MMQ.
- **If the wedge is LDS-independent** (same grid threshold regardless of LDS, or no synthetic wedge at all) → H2 → the wedge is our kernel/dispatch, and the lean-LDS rebuild alone won't fix it; the real fix is elsewhere (kernel bug / dispatch path), and int8-MMQ is still the throughput target but needs the bug found.

Result decides whether "rebuild int8-MMQ lean" is the path or whether we chase a dispatch/kernel bug.

## RESULT (2026-07-20): H1 REFUTED — LDS fatness is NOT the wedge cause
Synthetic tunable-LDS sweep (256-thread/8-wave workgroups, DEFINE_LOCAL K bytes, full-buffer write + barrier + readback, real AMDISARenderer + HCQ dispatch), 6 LDS sizes × 5 grids = **30/30 CLEAN**, including the **exact original wedge condition (57,856 B @ 64 wg)** and the maximal (65,536 B @ 544 wg). 0 wedges, GPU healthy throughout, journal clean. Occupancy classes 4/2/1 wg-per-CU all ran identically.

**Verdict: H2.** The wedge does not track LDS size or occupancy. Rebuilding int8-MMQ lean (43 KB) will NOT by itself remove the wedge. Caveat: the synthetic kernel had **no WMMA** (minimal variant), so this rules out LDS/occupancy/barrier/dispatch-shape but leaves the real kernel's **WMMA (`v_wmma_i32_16x16x16_iu8`) traffic or memory-access pattern** as the remaining suspect.

**Broader read:** llama's int8-MMQ (same tile/pack, WMMA, 544 wg) runs correct+fast; our *generated* int8-MMQ wedges and our *generated* fp16-dequant produces wrong multi-wave numbers. Tile/pack/waits now all match llama. So the root cause is a **generated-codegen bug in multi-wave WMMA execution**, not LDS/occupancy/tiling/waits. That is the thing to chase (or route around by using a kernel that already works — llama.cpp, or the hand kernel which is correct in real full-model runs).
