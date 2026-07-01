# G=5 Block Tile Flash Kernel — Result

Date: 2026-07-01. Model: Qwen3-14B-Q4_K_M (Hq=40, Hkv=8, G=5). Hardware: gfx1100.

## What was done

`flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel` had a hard `if G != WARPS: raise ValueError`
(line 963) with `WARPS=4` hardcoded. 14B has G=Hq/Hkv=40/8=5, so it was always rejected.

Fix: changed `WARPS = 4` to `WARPS = G` (parameterized). Added `flash_decode_g5_block_tile` function in
`extra/qk_flash_decode.py` that directly invokes the kernel through the sliced-KV path (l_route=L=128,
grid=Hkv×ceildiv(ctx,L)). Updated model.py with `DECODE_FLASH_BLOCK_TILE_G5=0` gate.

## Results

### Attempt 1: whole_cache routing
- **Correctness**: PASS
- **W==D ctx512**: baseline 50.2 tok/s → 10.0 tok/s (**-80.1%**)
- **Kernel**: flash_partial_coop_vec_40_128=27µs vs block tile=1915µs (**71× slower**)
- **Grid**: 8 × 6 = 48 WGs (l_route=96 from target_s=48 default)

### Attempt 2: sliced-KV routing (flash_decode_g5_block_tile)
- **Correctness**: PASS
- **W==D ctx512**: 9.5 tok/s (**-81.1%**)
- **Kernel**: ~2090µs (**78× slower**)
- **Grid**: 8 × 4 = 32 WGs (l_route=128, ceildiv(512,128)=4 splits)

**Verdict: G5_REFUTED_WD_REGRESSION** (both routing strategies refuted)

## Root cause

The per-workgroup kernel cost is the bottleneck, not occupancy. The original diagnosis (whole_cache
under-occupies the GPU) was wrong — the sliced path gives the same WG count as baseline flash_partial but
is still 78× slower per WG.

The block tile kernel reads both K and V into LDS (vs baseline which uses L2-cached V reads), doubling
global memory traffic. Cross-warp accumulation for G=5 GQA groups (WARPS-1=4 cross-warp reads per head
vs 3 for G=4) adds overhead. The net effect: 40 layers × 2090µs = 83.6ms flash attention overhead vs
baseline total step time of ~20ms.

## What stays

- `WARPS = G` parameterization — correct, no regression on 8B (G=4 unchanged)
- `DECODE_FLASH_BLOCK_TILE_G5=0` gate + `flash_decode_g5_block_tile` function — default-off
- Correctness test: `extra/qk_g5_block_tile_correctness.py`

## Reopen condition

Reopen only if:
1. Native RDNA3 ISA block tile achieves per-kernel time <100µs (vs 2090µs current), or
2. Profiling reveals a specific tinygrad codegen pathology (scalar loads, register spilling, excessive
   LDS barriers) with a targeted fix reaching <50µs per kernel call
