# Shared-attention query-row ownership microvariant: blocker record

Date: 2026-07-23

## Result

No executable query-row or output-ownership microvariant was added.  The
current gfx1100 primitive cannot lower either experiment while reducing its
allocated live PV state.  A register-number remap would be misleading: it
would preserve the same physical C-fragment and PV accumulator ownership.

Production defaults and generic WMMA are unchanged.

## Exact current ownership

`amd_gfx1100_q16_grid_hd128_loop_attention` emits one wave32 workgroup for
each `(q_head, q_tile)` and runs eight QK plus eight PV WMMA operations per KV
tile.  Each PV operation returns `float.vec(8)` and the loop state owns all
eight blocks:

```text
acc blocks: 8
lanes/block: 8 fp32
live PV values/wave: 64 fp32
fixed PV C lease: v8..v71 (64 VGPRs)
m/l lease: v72..v87 (16 VGPRs)
```

The baseline captured allocation is approximately 254 VGPRs.  This record
does not claim an allocation delta because no candidate executable exists.

## Why a smaller query-row wave cannot be a microvariant

The WMMA result ABI is fixed to `float.vec(8)` per lane.  Its row mapping is
`row = 2*e + (lane >> 4)`, so one instruction produces ownership for all 16
query rows.  Giving a wave only 8 rows would require it to compute and retain
the full result fragment, then discard half.  Therefore:

```text
requested rows/wave: 16 -> 8
allocated QK C:       unchanged
allocated PV C:       unchanged
allocated PV state:   unchanged (64 fp32)
```

It fails the experiment's admission premise before timing: independent
logical output ownership alone does not reduce live physical state.

## Why output-dimension slicing is not representable today

A four-block diagnostic would reduce the semantic persistent PV state from
64 to 32 fp32 values per wave, but requires an ABI extension across all of
these fixed-eight contracts:

- `AMDLoopStateSpec` only permits `acc.block in [0, 8)`.
- `AMDAttentionOutputDrainSpec` requires `blocks == 8` and eight PV sources.
- AMD ISA lowering fixes `acc` to `v8 + block*8` and drains all 64 values.
- HIP lowering also assumes exactly eight accumulator fragments.
- Capture validation requires eight PV WMMA roles and exactly 16 WMMA sites.

Changing only the scheduler loop would fail lowering or silently retain the
other four accumulator blocks.  Changing all five contracts is a new
versioned accumulator-slice ABI, not a small gated microvariant.

## Required follow-on design

Implement an explicitly experimental `acc_slice_v1` descriptor with:

1. `hd_block_start` and `hd_block_count`, initially `(0, 4)` or `(4, 4)`.
2. Slice-aware state validation and fixed-C allocation, reserving only the
   selected `8 * hd_block_count` PV registers.
3. Slice-aware HIP and ISA output drain addresses using
   `hd_block_start * 16`.
4. A separate proof/capture schema that records the slice, its 8 QK plus
   `hd_block_count` PV WMMA roles, and independently measured VGPR allocation.
5. A diagnostic two-launch composition or deliberate QK/softmax recomputation
   policy, since one slice cannot produce the complete Hd128 output.

Only after a compiled candidate reports fewer allocated VGPRs and higher
calculated residency should replay timing be considered.  The expected
semantic PV-state delta for a four-block slice is `64 -> 32 fp32` values per
wave; the physical VGPR delta is intentionally unclaimed until compiler
allocation evidence exists.
