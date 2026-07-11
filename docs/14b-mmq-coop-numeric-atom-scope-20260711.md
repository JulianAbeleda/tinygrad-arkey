# 14B MMQ Cooperative Numeric Atom Scope

Purpose: finish the next real R4 blocker after the lowered store-owner proof.
The target is not route promotion. The target is the smallest emitted tinygrad
backend that combines numeric Q4_K x Q8_1 DS4 compute with the llama-style
single-owner writeback contract.

## Current Ground Truth

Already done:

```text
DS4 layout/reference/formula are correct.
sudot4 primitive exists.
direct DS4 warp and dot4x4 atoms pass bounded correctness.
R3 LDS skeleton stages DS4 q8 values through LOCAL memory and a barrier.
R4 lowered store-owner trace passes as fragmented AMD ISA proof:
  16x16x256, 8 fragments, 256 gated global stores, 256 unique owners.
R5 report exists but is non-promotable because no emitted cooperative numeric
backend has a bounded win.
```

Still blocked:

```text
q4k_q8_1_mmq_amd_ds4_coop_tile_atom_v0 has no emitted numeric kernel.
run_bounded_harness still raises blocked_numeric_compute for that backend.
R6 route binding remains illegal.
```

## Minimal Candidate

Name:

```text
q4k_q8_1_mmq_amd_ds4_coop_tile_atom_v0
```

First bounded shape:

```text
M=16, N=16, K=256, activation_layout=mmq_ds4
```

The first candidate may be one-wave/one-fragment for 16x16. It must still use
the same owner metadata convention as the lowered R4 proof and must emit an
actual AMD/tinygrad custom kernel, not a reference wrapper.

## Implementation Path

1. Start from `_q4k_q8_1_bounded_ds4_lds_skeleton_kernel` in
   `extra/qk/mmq_q4k_q8_atom.py`.
2. Add a new coop backend constant/entrypoint for
   `AMD_DS4_COOP_TILE_BACKEND_ATOM_ID`.
3. Emit a bounded 16x16x256 kernel that:
   - stages DS4 q8 values through LOCAL memory and a barrier,
   - computes Q4_K x DS4 numeric values using the existing formula helpers,
   - writes outputs with `arg=("store_owner", owner_tuple)` so AMD ISA proof
     rows can join back to owner coverage,
   - keeps `production_dispatch_changed=False`.
4. Add a source-hash helper for the coop atom.
5. Wire `run_bounded_harness` for `AMD_DS4_COOP_TILE_BACKEND_ID` only after the
   new entrypoint passes bounded correctness.
6. Update machine-search status from `blocked_numeric_compute` to searchable or
   evidence-only PASS only when the emitted backend passes.

## Required Tests

Minimum new or updated tests:

```text
test_mmq_q4k_q8_atom.py:
  coop atom source hash exists
  coop atom matches DS4 reference on AMD when available
  lifecycle marks shared_memory_staging=True, bounded_only=True,
  production_dispatch_changed=False

test_mmq_bounded_harness.py:
  AMD_DS4_COOP_TILE_BACKEND_ID no longer raises blocked_numeric_compute once
  the emitted coop atom passes
  harness report status PASS for 16x16x256

test_mmq_machine_search.py:
  cooperative_multi_wave_tile status only changes if bounded harness can run
  R6 remains blocked unless R5 reports a bounded emitted coop win
```

## Stop Conditions

Stop and record the exact blocker if any of these happen after a concrete
lowering attempt:

```text
AMD ISA renderer spills on the smallest numeric 16x16x256 kernel and no smaller
numeric slice preserves the owner/compute contract.

The kernel compiles but numeric correctness fails and the failure cannot be
localized to Q4_K decode, DS4 indexing, min correction, or store ownership.

The kernel can only pass by using a reference wrapper or by changing production
dispatch.

The store-owner proof rows disappear or no longer cover every output exactly
once.
```

## Non-Goals

```text
Do not bind the route.
Do not change default route from direct_packed.
Do not claim 14B whole-prefill improvement.
Do not optimize beyond the first correct emitted cooperative numeric atom.
```
