# GP1 — Primitive Gap Analysis

Date: 2026-07-01

## Oracle data (from bench/g5-block-tile/compiler_pathology_v1.json)

| metric | value |
|--------|-------|
| VGPR | 91 |
| scratch_bytes | 0 |
| LDS bytes | 8192 (K+V both staged) |
| barrier_count | 1 |
| static_inst_count | 1610 |
| math_op_count | 141 |
| bloat ratio | 11.4× |
| BoltBeam class | LDS_OR_MEMORY_OVERHEAD |

## What the current kernel generates

`flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel` stages BOTH K and V into
LDS:
- `ksh` = TK×Hd fp16 = 16×128×2 = 4096 bytes
- `vsh` = TK×Hd fp16 = 16×128×2 = 4096 bytes
- Total LDS = 8192 bytes

The staging loop iterates `STAGES = ceildiv(TK*Hd, THREADS)` times per thread to
cooperatively fill both tiles from global cache. For G=5:
- THREADS = LANES×WARPS = 32×5 = 160
- STAGES = ceildiv(16×128, 160) = ceildiv(2048, 160) = 13

So 13 iterations × 2 tiles (K+V) = 26 global loads + LDS stores per staging phase,
plus address arithmetic, bounds checks, and the barrier.

The 1610 instructions break down approximately:
- ~13 staging steps × 2 (K+V) × ~6 instructions each = ~156 for staging
- ~NB=8 blocks × TK=16 tokens × (RP=4 dot-product iterations + PV accumulation + softmax) ≈ 8×16×15 ≈ 1920 instructions for reduce body (inflated by unrolled address calc)
- Rest: output write, loop control, prologue init

The instruction bloat (11.4×) primarily comes from the V staging path — address
calculations and global→LDS transfer for V data that the baseline already has L2-warm
from E_49152_32_3.

## Gap: V staging is redundant given E_49152_32_3

E_49152_32_3 writes current-token V into the KV cache, which immediately warms L2 for
the subsequent cache read. The generated kernel then stages V from L2→LDS — an
intermediate hop that:
1. Increases LDS usage from 4096→8192 bytes (uses 2× the LDS budget)
2. Adds ~780 instructions for V staging (~half the bloat)
3. Doubles the STAGES loop iterations

V staging into LDS provides NO bandwidth benefit over the warmed-L2 path — the L2
already holds V from E_49152_32_3.

## Generic primitive: `staging` parameter

The minimum generic primitive is a `staging` parameter to the existing kernel
function:

```python
flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel(
    Hd, Hq, Hkv, MAXC, L, S, Tc,
    staging="KV_BOTH"  # new parameter: "K_ONLY" | "KV_BOTH"
)
```

When `staging="K_ONLY"`:
- `ksh` allocated and filled from global (same as KV_BOTH)
- `vsh` NOT allocated; barrier fires after kstore only
- V reads in the PV accumulation loop go to `cache[1, 0, kvh, t, d]` (global, L2-warm)

When `staging="KV_BOTH"` (current default):
- Both ksh and vsh allocated and filled (existing behavior, byte-identical)

This is:
- Parameterized over G, Hd, TK (not specialized to 14B geometry)
- Generated UOp path (no handwritten ISA)
- BoltBeam candidate grammar axis: `staging ∈ {K_ONLY, KV_BOTH}`

## Projected effect

Expected LDS: 4096 bytes (K-only, halved from 8192).
Expected instruction reduction: ~780 instructions removed (~half the bloat), from ~1610→~830.
New bloat ratio: ~830/141 ≈ 5.9× (halved, but still >5×).
Expected kernel speedup: difficult to predict precisely, but V staging elimination
removes both LDS allocation pressure and ~half the instruction count. The remaining
bloat (K staging + address calc + softmax) is likely still significant vs baseline's
27µs/WG. The W==D gate will measure the actual gain.

## Classification

Verdict: **REACHABLE_NOW** — `staging=K_ONLY` is implementable as a UOp parameter
change to the existing generated kernel. No new primitive required. No handwritten
kernel. Unblocked by oracle confirmation of LDS_OR_MEMORY_OVERHEAD.

## What remains blocked

The 5.9× remaining bloat after K-only staging is not addressed by this primitive.
Further reduction would require:
1. Vectorized global loads for K staging (global_load_b64/b128 instead of b16 per element)
2. Better address arithmetic in the UOp lowering
3. Or: AMDISARenderer direct lowering of the block tile (bypasses LLVM, explicit control)

These are separate levers (Phase I native W==D baseline is the next AMDISARenderer track).
