# G=5 Block Tile Flash Kernel — Result

Date: 2026-07-01. Model: Qwen3-14B-Q4_K_M (Hq=40, Hkv=8, G=5). Hardware: gfx1100.

## What was done

`flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel` had a hard `if G != WARPS: raise ValueError`
(line 963) with `WARPS=4` hardcoded. 14B has G=Hq/Hkv=40/8=5, so it was always rejected.

Fix: changed `WARPS = 4` to `WARPS = G` (parameterized). Removed the ValueError check. Added
`DECODE_FLASH_BLOCK_TILE_G5=0` gate in model.py that enters `flash_decode_attention_whole_cache`
with `DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE=1` and `DECODE_ATTN_BLOCK_TILE=1`.

## Result

- **Correctness**: PASS — token-identical to baseline at ctx=512 (rel_rmse=0)
- **W==D ctx512**: baseline 50.2 tok/s → G5 10.0 tok/s (**-80.1%**)
- **Kernel timing**: baseline `flash_partial_coop_vec_40_128` = 27µs; block tile = 1915µs (**71× slower**)
- **Verdict: G5_REFUTED_WD_REGRESSION**

## Root cause

`flash_decode_attention_whole_cache` reads the full MAXC=4608 KV cache regardless of actual context.
At ctx=512 with L=96 (default DECODE_ATTN_FUSED_XLANE_SCORE_PV_S=48):

- `s_route = ceildiv(512, 96) = 6` splits → grid = `Hkv × s_route = 8 × 6 = 48` workgroups
- Baseline `gqa_coop_vec` at ctx=512: grid = `8 × 48 = 384` workgroups (sliced to actual ctx)
- Block tile is 8× under-occupied at ctx=512

The block tile was designed for full-MAXC context use where s_route≈smax=48. At ctx=512, it collapses
to 6 splits. Higher split counts (`DECODE_ATTN_FUSED_XLANE_SCORE_PV_S=384`) made it worse (10.6 tok/s)
because L shrinks to 12 < TK=16, causing NB=1 (just one very thin block per workgroup).

## What stays

- `WARPS = G` (parameterized) in the kernel — correct, no regression on 8B (G=4 unchanged)
- `DECODE_FLASH_BLOCK_TILE_G5=0` gate — default-off, correct, no impact unless set
- Correctness test: `extra/qk_g5_block_tile_correctness.py`

## Reopen condition

`decode_flash_block_tile_g5_native_context`: route the G=5 block tile through the standard flash path
(sliced KV to actual context length), not `flash_decode_attention_whole_cache`. Requires a
flash_partial variant that accepts `(q, k_slice, v_slice)` with grid `Hkv × ceildiv(ctx, L)` matching
baseline occupancy. The kernel structure (LDS staging, online softmax, G=5 warp layout) is proven
correct — only the invocation architecture needs to change.
