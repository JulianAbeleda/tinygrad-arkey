# Cooperative-staging LaneMap (M2) — built, proven, composes → generated tile 2.35× (2026-06-27)

Scope lineage: `docs/decode-coalesced-load-primitive-scope-20260626.md` →
`docs/layout-codegen-full-scope-20260625.md` P1.2 (LaneMap).

## Verdict

`COOPERATIVE_STAGE_LANEMAP_COMPOSES__GENERATED_TILE_2P35X__STILL_FAR_FROM_HAND_ASM`

The M2 cooperative-staging LaneMap primitive is built, proven on a tile-independent proving ground, and
applied to the generated decode block tile, where it makes the GLOBAL cache staging vectorize
(`global_load_d16=0` → `float4`/`dwordx4`). It **composes** with the coalesced-load primitive (M1-side) and
the shipped recurrence-unroll/list scheduler (`SCHED_UNROLL`/`SCHED_LIST`): the three foundation layers stack,
numerically correct, to **~2.35× on the isolated generated tile**. This is foundation progress (the pure-search
codegen layers compose), **not** a promotable kernel — the generated tile is still ~56× off the hand-asm owned
tile; the residual is the deep instruction-scheduling/codegen wall (Tier 2/3, comgr/LLVM-owned for Track A).
Isolated timing only — **no in-model W==D claimed**.

## Built

- `extra/qk_cooperative_stage_lanemap.py` — `CooperativeStageLaneMap(total, threads, width)`: the first-class,
  validated thread→element map for cooperative contiguous staging. T threads load `total` contiguous elements,
  each owning a contiguous `width`-chunk (`chunk = stage*threads + tid`), so the per-thread `width` axis is a
  **unit-stride LOOP axis** the coalesced-load primitive promotes to a vector load. The bridge-independent,
  reusable analogue of the GEMV's `Q4KGateUpLaneMap` for cooperative staging.
- Applied (opt-in `DECODE_STAGE_COALESCE=<W>`, default-off byte-identical) to
  `flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel` (`extra/qk_flash_decode.py`): the one-element-per-
  thread cache staging becomes a per-thread W-chunk staging.
- **Safety fix to `extra/qk_coalesced_load_lowering.py`:** never coalesce an axis that indexes a REG store
  (an accumulator/carry axis). Principled (only pure-load axes widen) and it unblocked the composition: the V
  LDS load axis `dd` also indexes `acc[dd]`, and promoting it under recurrence-unroll's `dd`-duplication +
  carry-threading broke numerics (`A: COALESCED+UNROLL → FAIL`). Excluding accumulator axes drops the (neutral)
  V-LDS promotion and makes the full stack correct.

## Proven (proving grounds, tile-independent)

- `test/external/test_cooperative_stage_lanemap.py` (3/3): `validate` rejects ragged/odd configs; the lane
  map's element index is **unit-stride in `w`** (statically coalescable) and strided in the stage axis;
  end-to-end on AMD the staging renders a `float4` GLOBAL load and is numerically exact — **masked and
  unmasked** (the `t<Tc` index mask is constant within a 4-element chunk, so the coalescing predicate holds).
- `test/external/test_coalesced_load_lowering.py` (5/5) still green after the accumulator-axis safety fix.

## Composition ladder (isolated tile, `extra/qk_decode_block_tile_isolated_timing.py`, stable reps)

| config | ctx=512 | ctx=4096 | vs baseline |
|---|---|---|---|
| baseline generated tile | 1.024 ms | 7.289 ms | — |
| L1 staging-coalesce (`DECODE_STAGE_COALESCE=4 COALESCED_LOAD_LOWERING=1`) | 0.918 ms | 6.578 ms | −10% |
| L2 + `SCHED_UNROLL=8 SCHED_LIST=1` | 0.437 ms | 3.142 ms | **−57% (~2.35×)** |
| owned (hand-asm reference) | 0.008 ms | 0.031 ms | — |

Microgate `BLOCK_TILE_MICROGATE_PASS` for L1 and L2 (all 4 cases, `max_abs` unchanged 1.526e-05); default-off
byte-identical.

## What this proves / doesn't

- **Proves:** the pure-search codegen foundation layers — cooperative-staging LaneMap (M2) → coalesced-load
  lowering (`OptOps.COALESCE` realization) → recurrence-unroll + list scheduler — are real, default-off,
  proving-ground-validated, and **compose** to a correct 2.35× on a machine-generated kernel. The GEMV's
  "representation ≠ speed" wall is partly answered here: because the attention staging is a contiguous
  bandwidth load (not a packed gather+dequant), the LaneMap + the codegen lowering it lacked do convert to real
  speed.
- **Doesn't:** close the gap to the hand-asm owned tile (~56× remains). That residual is the deep
  instruction-scheduling/clause/waitcnt layer comgr/LLVM owns for Track A — the documented Tier-2/3 ceiling. No
  in-model W==D win is claimed (isolated timing is not promotion authority).

## Next

- W==D the L2 stack in-model (`extra/qk_decode_runtime_overhead.py`) to see how much of the 2.35× transfers
  (expected: still far below the owned default — this is foundation, not a promotion candidate yet).
- Attribute the ~56× residual: is it comgr not pipelining the generated structure (→ Track-B asm scheduler), or
  remaining structural deltas (cross_lane count 20 vs 5, LDS size)? That is the next foundation question.
