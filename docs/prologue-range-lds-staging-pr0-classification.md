# PR0 — Corrected Classification: EMITTER_BLOCKED → SEARCH_SPACE_INCOMPLETE (14B G=5 shape)

Date: 2026-07-01. Model: Qwen3-14B-Q4_K_M, gfx1100.

## What changed

| Phase | Classification | Reason |
|-------|---------------|--------|
| EB5 | PRIMITIVE_MISSING | Incorrectly claimed LDS-alloc UOp missing |
| LDS0 | EMITTER_BLOCKED | Correctly refuted PRIMITIVE_MISSING; named "prologue range UOp" as blocker |
| PR0 (this doc) | SEARCH_SPACE_INCOMPLETE | Prologue-range UOp exists (block tile proves it); 14B's G=5 shape is blocked by block-tile's G==WARPS==4 constraint |

## What AddrSpace.LOCAL and UOp.barrier can already do

The primitives are NOT missing. Evidence (from `extra/qk_flash_decode.py`):

- **K LDS staging** (lines 214, 253): `klds = UOp.placeholder((Hd,), dtypes.half, N, addrspace=AddrSpace.LOCAL)` with staged store + `bar = UOp.barrier(UOp.group(kstage))` — in production, already used.
- **Both K and V LDS staging** (lines 975–1011, `flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel`): cooperative load of TK=16 tokens of K and V into LDS, then barrier, then main REDUCE. This IS the prologue-range pattern. It is expressible in the current UOp DSL.
- **Barrier lowering** (`tinygrad/renderer/cstyle.py:370`): `__builtin_amdgcn_fence + __builtin_amdgcn_s_barrier` — correct AMD gfx1100 barrier.

## What is actually missing for 14B

The block tile kernel (`flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel`) hard-blocks on:

```python
G = Hq // Hkv   # 14B: G = 40 // 8 = 5
WARPS = 4
if G != WARPS: raise ValueError(f"block tile expects G=={WARPS}, got {G}")
```

14B's GQA ratio G=5 doesn't fit the block tile's WARPS=4 warp layout. The block tile architecture assumes one warp per GQA group, with 4 warps = 4 GQA groups per KV head. For 14B with G=5, this structure doesn't map.

The block tile IS available for the 8B model (Hq=32, Hkv=8, G=4), gated by DECODE_ATTN_BLOCK_TILE + DECODE_ATTN_GENERATED_WHOLECACHE + DECODE_ATTN_AMDGCN_TILE=0.

## Why per-token V staging doesn't help

`flash_partial_coop_vec_kernel` (line 1229): `d = UOp.range(W, 2, AxisType.LOCAL)` — the head dimension d is already a LOCAL (workgroup-thread) axis. Adjacent d-lanes read adjacent V elements → coalesced global reads are already maximally efficient. Adding per-token LDS staging writes V to LDS then reads back with zero reuse — overhead only.

## Reopen condition (precise)

Build a G=5 variant of the block tile kernel for 14B (Qwen3-14B: Hq=40, Hkv=8, G=5, Hd=128). This requires:
1. WARPS=5 (or a different warp layout that accommodates G=5, e.g. 2×5 warp grid)
2. Per-block staging of TK tokens of K and V into LDS (TK × Hd × 2 bytes × 2 = TK × 512 bytes)
3. Online softmax with d-sharded PV across 5 GQA groups per KV head
4. Barrier between staging REDUCE and compute REDUCE

Once a G=5 block tile variant exists:
- Gate under `DECODE_ATTN_BLOCK_TILE_G5=0` (default-off)
- Correctness: rel_rmse ≈ 0, token-identical
- W==D: projected upper bound 6.69% (E_49152_32_3) + V-read acceleration from LDS
- This is the candidate: `decode_flash_v_prologue_lds_stage`
