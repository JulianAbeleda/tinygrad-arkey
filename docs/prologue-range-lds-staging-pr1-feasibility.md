# PR1 — Emitter Feasibility Audit: V LDS Staging in flash_partial (14B)

Date: 2026-07-01. Model: Qwen3-14B-Q4_K_M (Hq=40, Hkv=8, G=5, Hd=128), gfx1100.

## Three angles audited

### Angle 1: Per-token V staging inside the j REDUCE loop

NOT_BENEFICIAL.

`flash_partial_coop_vec_kernel` uses `d = UOp.range(W, 2, AxisType.LOCAL)` where W=Hd+1=129.
Each workgroup has 129 threads; thread d reads V[kvh, t, d] for each j-iteration t. Adjacent
lanes access adjacent memory → coalesced 256-byte loads → near-peak bandwidth utilization.

Adding LDS for per-token V staging (copy to LDS, barrier, read from LDS) has zero data reuse:
each LDS value is written once and read once per j iteration. The overhead of LDS write + barrier
exceeds any gain. This matches the LDS0 assessment.

### Angle 2: Full-split prologue (pre-j staging of all L tokens)

SEARCH_SPACE_INCOMPLETE — blocked by G=5 shape.

The per-block cooperative staging pattern (stage TK tokens → barrier → compute) IS expressible
in the current UOp DSL. Proof: `flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel`
(qk_flash_decode.py:954) stages TK=16 tokens of both K and V into LDS with exactly this pattern.

However, the block tile kernel hard-rejects 14B's shape:
```python
G = Hq // Hkv   # 14B: 40 // 8 = 5
WARPS = 4
if G != WARPS: raise ValueError(f"block tile expects G=={WARPS}, got {G}")
```

The block tile's warp layout (LANES=32, WARPS=4, THREADS=128) assumes G=WARPS=4 (one warp per
GQA group). 14B has G=5. There is no existing variant for G=5. Building one requires:
- New warp layout (e.g. WARPS=5, THREADS=160 or a hybrid WARPS=4 with interleaved G=5 groups)
- New staging shape: TK × Hd × 2 for each of K and V
- New online softmax structure for G=5 groups per KV head
- Careful LDS sizing: TK=16 × 128 × 2 bytes × 2 (K+V) = 8192 bytes → fits 64KB LDS budget

This is a NEW KERNEL FUNCTION, not a UOp primitive addition or a parameter change.

### Angle 3: Block tile for 8B (G=4 — reference)

EXISTS AND WORKS for 8B (Hq=32, Hkv=8, G=4).

The 8B model's block tile (DECODE_ATTN_BLOCK_TILE + DECODE_ATTN_GENERATED_WHOLECACHE) binds at
model.py:1113 for Hq=32, Hkv=8. It stages K and V in LDS with TK=16. This demonstrates the
capability works end-to-end on AMD gfx1100 for G=4. It cannot be reused for G=5 without
structural changes.

## PR1 verdict: SEARCH_SPACE_INCOMPLETE

| angle | verdict |
|-------|---------|
| Per-token V staging inside j loop | NOT_BENEFICIAL (coalesced loads, no reuse) |
| Full-split prologue (block tile pattern) | SEARCH_SPACE_INCOMPLETE (G=5 not supported, G=4 8B only) |
| Cross-kernel LDS | GRAPH_LIFETIME_BLOCKED (LDS workgroup-scoped, confirmed from LDS0) |

**PR1_SEARCH_SPACE_INCOMPLETE**

The UOp DSL CAN express the V LDS staging pattern (as proven by the 8B block tile). The gap is
a G=5 block tile variant for 14B. This is new kernel design work, not a UOp primitive addition.

## LDS sizing feasibility for a G=5 block tile

If TK=16, Hd=128, K+V:
- LDS per workgroup: 2 × TK × Hd × 2 bytes = 2 × 16 × 128 × 2 = 8192 bytes = 8KB
- gfx1100 LDS budget: 64KB per CU
- Occupancy: 64KB / 8KB = 8 workgroups per CU max → not LDS-limited
- Grid = Hkv × S = 8 × 5 = 40 workgroups (at ctx512, FLASH_L≈102, S=5) → well within 96 CUs

## What would unlock PR2-PR4

Build `flash_block_tiled_g5_score_pv_kernel` (generated UOp, no handwritten HIP):
- WARPS=5 or interleaved G=5 warp mapping within THREADS
- TK=16 staging loop for K+V into LDS
- Online softmax + d-sharded PV for G=5 GQA groups
- Gate under `DECODE_ATTN_BLOCK_TILE_G5=0`
- BoltBeam candidate: `decode_flash_v_prologue_lds_stage`

This is the precise reopen condition. PR2-PR4 are DEFERRED pending this kernel design.
