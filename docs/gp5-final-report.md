# GP5 Final Report — Generated ISA Primitive Track

Date: 2026-07-01

## Result: GP4_PASS_TIER_A

The K_ONLY staging parameter for the G=5 block tile kernel delivers Tier A gains
at ctx512 and ctx2048 on 14B decode.

## Full GP track results

| Phase | Verdict | Finding |
|-------|---------|---------|
| GP0 | PASS | flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel is generated via UOps; no handwritten ISA |
| GP1 | REACHABLE_NOW | staging=K_ONLY removes V→LDS staging (8KB→4KB LDS); V reads from global (L2-warm) |
| GP2 | IMPLEMENTED | staging parameter added to kernel fn; DECODE_FLASH_BLOCK_TILE_G5_KONLY=0 flag in model.py |
| GP3 | GP3_PASS_MICROGATE | rel_rmse=0.00e+00 for G=5 at ctx64/512 and G=4 at ctx128 |
| GP4 | GP4_PASS_TIER_A | ctx512 +3.9 tok/s (+7.8%); ctx2048 +6.9 tok/s (+14.7%) |
| GP5 | SEE BELOW | BoltBeam updated; rollback=DECODE_FLASH_BLOCK_TILE_G5_KONLY=0 |

## W==D results (14B, gfx1100)

| ctx | Baseline | K_ONLY | Delta | % |
|-----|----------|--------|-------|---|
| 128 | 52.1 tok/s | 52.2 tok/s | +0.1 | flat (no flash) |
| 512F | 49.9 tok/s | 53.8 tok/s | **+3.9** | **+7.8%** |
| 2048F | 46.9 tok/s | 53.8 tok/s | **+6.9** | **+14.7%** |

Kernel count: progs 7→6 at ctx512 (one kernel eliminated).

## Mechanism

E_49152_32_3 (the BENEFICIAL_CACHE_WARM kernel, 6.69% GPU @ ctx512) writes current-token
V into the KV cache, which immediately warms L2. The K_ONLY variant exploits this: instead
of staging V from global→LDS (redundant given the L2 warmth), it reads V directly from
global cache (L2-hit path). This:
- Halves LDS usage: 8192→4096 bytes per workgroup
- Removes ~780 instructions (V staging loop) from the 1610-instruction kernel
- Eliminates the vsh placeholder allocation
- Fires one fewer kernel per decode step

The gain increases with context length because longer contexts have more flash splits,
each benefiting from the eliminated V staging.

## Oracle path closed

The LDS_OR_MEMORY_OVERHEAD classification from the BoltBeam oracle (scratch=0, LDS=8192)
directly indicated the K_ONLY fix. The classification chain was:
EB5 PRIMITIVE_MISSING → LDS0 EMITTER_BLOCKED → PR1 SEARCH_SPACE_INCOMPLETE →
G5 LDS_OR_MEMORY_OVERHEAD (oracle) → **GP1 REACHABLE_NOW → GP4 TIER_A**

## Flags

- `DECODE_FLASH_BLOCK_TILE_G5=1` (required, default-off)
- `DECODE_FLASH_BLOCK_TILE_G5_KONLY=1` (required for K_ONLY, default-off)
- Rollback: set either flag to 0

## Promotion recommendation

Both flags should be default-ON candidates for 14B (Hq=40/Hkv=8/gfx1100). The gain is
large enough (Tier A) to justify promotion. Token correctness is exact (rel_rmse=0).

The `staging` parameter is generic — works for any G (tested G=4 and G=5). 32B (Hq=64,
Hkv=8, G=8) is the next test target before promoting the G generalization.
