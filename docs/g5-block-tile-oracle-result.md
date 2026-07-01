# G=5 Block Tile Resource Oracle Result

**Date:** 2026-07-01  
**Kernel:** `flash_block_tiled_xlane_score_pv_tile_whole_cache_40_128`  
**Model:** Qwen3-14B (Hq=40, Hkv=8, G=5)  
**Tool:** `extra/qk_g5_resource_oracle.py` — patches `HIPCompiler.compile`, parses AMDHSA ELF `.note` msgpack, counts instructions from `llvm-objdump-17` AMD ISA disasm.

## Static resource metrics

| Metric | Value |
|--------|-------|
| VGPR count | 91 |
| SGPR count | 25 |
| Scratch bytes | **0** |
| LDS bytes | 8192 |
| Wavefront size | 32 (wave32) |
| Max threads / workgroup | 160 (WARPS=5 × LANES=32) |

## Instruction-level metrics

| Metric | Value |
|--------|-------|
| Total instructions | 1610 |
| Math ops (v_fma_f32, v_mul_f32, v_dot2acc_f32_f16, ds_bpermute_b32) | 141 |
| DS (LDS) instructions | 130 |
| Global load/store | 28 |
| Barriers (s_barrier) | 1 |
| Branches | 3 |
| **Instruction bloat ratio** | **11.4×** |

## Key findings

**No register spilling.** `scratch_bytes=0`, `vgpr_spill_count=0`. The 91 VGPRs fit in gfx1100 SIMD register file (512 VGPRs per lane × 32 lanes = 16384 per SIMD; 5 wavefronts × 91 VGPRs × 32 = 14560 ≈ 89% VGPR occupancy, 1 workgroup per SIMD = 31% wave occupancy).

**Not barrier flooding.** 1 `s_barrier` total (the cooperative K+V preload fence). The static analysis estimate of 8 was wrong.

**Instruction bloat.** 1610 total instructions for 141 math ops = 11.4× ratio (threshold >10×). The surrounding address computation, bounds checking, loop control, and intermediate moves dominate the instruction mix. With only 3 branches, most of the NB=8-iteration preload+compute loop is unrolled.

**LDS staging.** 8192 bytes = `2 × TK × Hd × sizeof(f16)` = `2 × 16 × 128 × 2`. Both K and V tiles are staged. The 130 DS instructions are 8 preload iterations × ~8 K-loads + ~8 V-loads + score ops = plausible.

## Timing

| | Baseline (gqa_coop_vec) | G=5 block tile |
|-|------------------------|----------------|
| Per-workgroup time | 27 µs | 2090 µs |
| Raw ratio | 77.4× | |
| Work-adjusted ratio | 19.4× (G=5 handles 4× positions/WG) | |

## BoltBeam classification

`LDS_OR_MEMORY_OVERHEAD` — work-adjusted 19.4× slowdown with `scratch_bytes=0`; baseline LDS bytes not captured (baseline kernel construction failed on `UOp.CAST dtypes.weakint` during oracle run).

The instruction-level signal (`bloat_ratio=11.4×`) is consistent: excess non-math instructions inflate LDS and global memory issue latency per math operation.

## Ruled out

- **REGISTER_SPILL** — `scratch_bytes=0`, `vgpr_spill_count=0`  
- **BARRIER_FLOOD** — 1 barrier (prior estimate of 8 was wrong)  
- **LAYOUT_MISMATCH** — WARPS=G=5 is intentional (not an off-by-one)

## Open question

Does staging only K into LDS (not V) reduce the overhead enough? K-only = 4096 bytes LDS (half), one preload stage instead of two. But the instruction bloat (11.4×) is structural — it comes from the tinygrad code generator, not K+V vs K-only. K-only staging remains BLOCKED until measured.

The native ISA path (Phase I) is the other route — it bypasses tinygrad's code generator entirely.

## Artifacts

- `bench/g5-block-tile/compiler_pathology_v1_dynamic.json` — flat oracle output
- `bench/g5-block-tile/compiler_pathology_v1_ingest.json` — kernels-list format for BoltBeam ingest
- `bench/g5-block-tile/compiler_pathology_v1_normalized.json` — BoltBeam `boltbeam.normalized_evidence.v1` output
