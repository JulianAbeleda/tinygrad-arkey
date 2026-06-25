# P2.1 LaneMap-aware gpudims rescope

Date: 2026-06-25
Task: `docs/layout-codegen-full-scope-20260625.md` TASK P2.1
Verdict: `P2_1_KILL_RESCOPE`

## Goal checked

P2.1 asked for the owned Q4_K decode GEMV thread map to become scheduler-expressible:

- `lane = block_group*8 + word_col`
- `word_col = lane % 8`
- `block_group = lane // 8`
- `word_col` makes eight adjacent packed Q4_K word loads consecutive
- `block_group` splits the K-block REDUCE work across one wave

The intended target is the owned kernel in `extra/q4_k_gemv_primitive.py`:

- `lane = UOp.special(32, "lidx0")`
- `bg = lane // 8`
- `lane4 = lane % 8`
- `blk = bg * bpb + lblk`
- `total = warp_reduce_sum(acc[0], lane, 32)`

## What blocks a small `add_gpudims` change

`tinygrad/codegen/gpudims.py:add_gpudims` currently treats local/global ranges as hardware dimensions, but explicitly skips `AxisType.REDUCE` when substituting ranges:

```python
if r.arg[1] == AxisType.REDUCE: continue
```

That skip is not just a missing case. A REDUCE range is both:

1. An index variable used in address expressions.
2. An ended/control-flow range attached to `Ops.REDUCE` semantics.

For P2.1, substituting `bg -> lidx0 // 8` would make the packed-word address expression look like the owned kernel, but it would not by itself preserve the reduction. The compiler would still need to know that partial sums from different lanes are semantically one reduction and must be combined across the wave.

The owned kernel does that explicitly with `warp_reduce_sum(acc[0], lane, 32)`. A pure scheduler expression needs an equivalent semantic rewrite, not just a gpudims substitution.

## Why this is the kill condition

The P2.1 kill condition says to re-scope if splitting a REDUCE range across hardware lanes within one wave needs a structural rewrite of `get_grouped_dims` larger than the rest of the roadmap.

That is the case here. The required change is broader than `get_grouped_dims`:

- `get_grouped_dims` can create `special(lidx0)` and split/mod it.
- `add_gpudims` can substitute RANGE variables in address expressions.
- But neither owns the semantic lowering of `Ops.REDUCE` into a cross-lane reduction.
- Replacing a REDUCE range with `lidx0//8` without rewriting the REDUCE would risk a silent wrong kernel.

This violates the P1.1/P2.1 rule: silent wrong layout/reduction is worse than no IR.

## What remains true

The coalescing part is expressible:

```python
lane = UOp.special(32, "lidx0")
lane4 = lane % 8
word_idx = base + 4 + (grp // 2) * 8 + lane4
```

For any 8-lane subgroup, `lane4` gives stride-1 packed-word offsets. The blocker is not address algebra. The blocker is preserving the reduction semantics when `block_group = lane//8` is a REDUCE split.

## Rescoped P2.1 target

The correct next task is not "teach `gpudims` to substitute REDUCE axes." The next task is:

> Add a first-class lane-partitioned reduction form that can bind a REDUCE shard to hardware lanes and lower the required cross-lane combine explicitly.

Minimum shape:

- Represent a lane partition as metadata/object, likely using `LaneMap`:
  - `lane_extent = 32`
  - `word_col = lane % 8`
  - `block_group = lane // 8`
- Allow address algebra queries to consume that map for coalescing proof.
- Lower the corresponding reduction into a warp reduction, reusing `extra/qk_warp_reduce_lowering.py` / `extra/amd_warp_reduce.py` semantics.
- Only after that should `add_gpudims` substitute the REDUCE shard to `lidx0//8`.

## Recommended next execution task

Create a new narrow primitive before retrying P2.1:

`P2.1a LanePartitionReduce`

Goal:

- Given a value indexed by `(block_group, word_col)` where both are derived from one hardware lane, prove and lower the reduction across the whole wave.

Gate:

- Structural: no raw `Ops.REDUCE` remains for the lane-partitioned axis after lowering.
- Source: emits `special(lidx0)` and `ds_bpermute`/warp-reduce path.
- Layout: Q4_K packed-word INDEX has `lane4=lidx0%8` and `is_coalesced(index, lidx0)` for the 8-lane subgroup.
- Correctness: synthetic Q4_K row result matches the owned warp kernel/reference within the existing reassociation tolerance.

Until that exists, P2.1 should remain paused rather than landing a partial `gpudims` substitution.
