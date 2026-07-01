# PR5 — Prologue Range LDS Staging Track: Final Report

Date: 2026-07-01. Model: Qwen3-14B-Q4_K_M, gfx1100.

## Track summary

| Phase | Verdict | Key finding |
|-------|---------|-------------|
| PR0 | CLASSIFICATION_CORRECTED | LDS0 said EMITTER_BLOCKED (prologue_range_uop missing); PR0+PR1 show the UOp pattern already exists in block tile |
| PR1 | SEARCH_SPACE_INCOMPLETE | Block tile pattern is expressible; 14B G=5 ≠ block tile's G==WARPS==4 constraint |
| PR2 | DEFERRED | Prerequisite: G=5 block tile kernel |
| PR3 | DEFERRED | Prerequisite: G=5 block tile kernel |
| PR4 | DEFERRED | Prerequisite: G=5 block tile kernel |
| PR5 | LEDGER_UPDATE | BoltBeam updated with corrected classification |

## Classification evolution

| Phase | Label | Reason |
|-------|-------|--------|
| EB5 | PRIMITIVE_MISSING | LDS-alloc UOp missing (WRONG) |
| LDS0 | EMITTER_BLOCKED | Corrected to: prologue range structure missing (PARTIALLY CORRECT) |
| PR0/PR1 | SEARCH_SPACE_INCOMPLETE | The UOp pattern exists (block tile for G=4); 14B needs G=5 variant |

## What the code actually shows

1. **K LDS staging works** (qk_flash_decode.py:214,253): per-token K staging via REDUCE loop + barrier + LDS read — already in production for pall kernels.

2. **K+V LDS staging works for G=4** (`flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel`, lines 954–1067): TK=16 cooperative stage of both K and V into LDS with barrier before compute. This IS the prologue pattern, expressed as a staged REDUCE within the main block REDUCE.

3. **14B G=5 is hard-blocked** (line 963): `if G != WARPS: raise ValueError(...)`. 14B has Hq=40, Hkv=8, G=5. WARPS=4. No G=5 variant exists.

4. **Per-token V staging is NOT_BENEFICIAL**: d=LOCAL in flash_partial_coop_vec already gives coalesced V reads; per-token LDS has zero reuse and adds overhead.

## Reopen condition (precise)

Build `flash_block_tiled_g5_score_pv_kernel` or equivalent:
- WARPS=5 (or interleaved G=5 warp mapping)
- TK=16 cooperative staging of K and V into LDS (8KB per workgroup — fits 64KB budget easily)
- Online softmax + d-sharded PV accumulation for G=5 GQA groups
- Gate: `DECODE_ATTN_BLOCK_TILE_G5=0` (default-off)
- No handwritten HIP (generated UOp path, same as 8B block tile)
- Once built: PR2 (microgate) → PR3 (prototype) → PR4 (W==D) can proceed

BoltBeam candidate: `decode_flash_v_prologue_lds_stage`
Evidence ref: `bench/prologue-range-lds-staging/pr1_latest.json`
Status: `search-space-incomplete`

## Expected W==D if unblocked

Upper bound: 6.69% (E_49152_32_3 eliminated) + V-read acceleration (LDS vs L2).
The block tile for 8B already demonstrates the mechanism end-to-end.
