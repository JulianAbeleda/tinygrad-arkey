# NV Q6 online-hypothesis causal A/B synthesis

## Decision

Four proposed structural levers were tested against the authoritative
`M=512, N=4096, K=12288`, 170-owner Q6_K route under a serialized GPU lock.

Only final-participant reduction produced a repeatable causal timing win. It
recovers `15.744 us` median and wins all 31 alternating rounds, but cannot be
promoted because the parent broad route still fails the direct-reference
correctness tolerance. Contiguous Q8 publication and lifetime separation are
rejected as standalone levers. Single-body reuse remains inconclusive because
tinygrad cannot currently express the required nested runtime segment loop
while preserving the 170-CTA geometry.

## Results

| hypothesis | reflected in pinned llama | exactness | measured result | verdict |
| --- | --- | --- | --- | --- |
| one reusable K/segment body | yes: one tile-processing K loop and accumulator lifetime | fixed-up output bit-exact to control | `(170,2)` screen `413.696 us` vs `284.320 us`; static IMMA `256` vs `512`, LDL/STL `129/194` vs `251/377` | `INCONCLUSIVE_GEOMETRY`; needs nested runtime RANGE in 170 CTAs |
| final-participant reduction | no for this geometry: pinned llama has a separate Stream-K fixup kernel; this is a CUTLASS-derived alternative | bit-exact to current deterministic fixup | `295.936 us` vs `311.936 us`, `15.744 us` median recovery, wins `31/31`, removes `16,777,216 B` partial traffic | `INVEST`, blocked from promotion by parent-route reference error |
| producer/consumer lifetime separation | indirectly: llama's normalized binary has much lower local traffic | allclose passes; strict segmented error `6.866e-4` | `356.032 us` min, `356.896 us` median; stack `72 B`, LDL/STL `20/20` | `REJECT` as standalone lever |
| contiguous Q8 publication | yes: `block_q8_1_mmq` is pretransposed for contiguous shared copy | allclose passes | `325.440 us` min, `325.984 us` median; zero local traffic, but `384 IMMA` and 12 barriers | `REJECT` as implemented |

## Corrected llama interpretation

The pinned llama source explicitly declares `mul_mat_q_stream_k_fixup` and
selects a tile-count launch that skips fixup only when tile efficiency is at
least 90 percent. For 128 output tiles against the approximately 170-SM
geometry, that condition is not met. Therefore final-participant reduction is
a measured improvement over both our broad fixup and llama's high-level
scheduler choice, not an imitation of the pinned llama path.

The pinned source directly supports the other two structural observations:

- Q8 data is pretransposed into `block_q8_1_mmq` so it can be copied to shared
  memory contiguously.
- The MMQ tile processor owns one accumulator array and iterates K inside one
  physical processing structure.

Neither property is sufficient in isolation. The A/Bs show that preserving
the exact publisher expansion, barrier count, and 170-CTA scheduling geometry
is part of the causal package.

## What to build next

1. Fix nested runtime `RANGE` lowering in `simplify_merge_adjacent`. The
   current failure is a `KeyError` involving `UOp(Ops.END)` when the outer
   segment range contains the inner K256 epoch range.
2. Re-run the one-body candidate with the original `(170,1)` grid. Require
   exact output, the normalized single body (`256 IMMA`), and a same-process
   timing win. A `(170,2)` result is not admissible evidence for this gate.
3. Integrate final-participant reduction only after the parent main route is
   direct-reference exact. Its measured ceiling is already established:
   approximately `15.7 us` on this shape.
4. Do not invest further in generic spill elimination or generic contiguous
   Q8 publication. Any new attempt must preserve llama's normalized producer
   counts and barrier topology, rather than merely improving registers or
   local-memory counts.

## Evidence

- `docs/task_workflow/input/nv-q6-oracle-single-body-experiment-20260831.md`
- `docs/task_workflow/input/nv-q6-final-participant-fixup-decision-20260831.md`
- `docs/task_workflow/input/nv-q6-binary-ab-verdict-20260831.md`
- `docs/task_workflow/input/nv-q6-llama-normalized-oracle-contract-20260831.md`

