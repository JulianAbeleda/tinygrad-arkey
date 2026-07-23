# Next shared-attention resource primitive

The measured four-block accumulator slice changes allocation from `254` VGPR
to `244/245` VGPR.  This proves PV state is relevant but falsifies the idea
that halving it alone will cross the occupancy threshold.

## Chosen next variant: split QK/state from PV output slices

Do not reduce to one/two PV blocks first.  It would save at most the remaining
PV C lease while retaining the QK fragment, softmax state, fragment-load
temporaries, and address lifetime in the same allocation interval.

Instead use two dependent kernels per `(q_head,q_tile,kv tile sequence)`:

1. `score_state_v1`: QK WMMA plus online `(m,l)` recurrence. It writes final
   `m,l` for 16 rows to a compact explicit temporary.
2. `pv_slice_v1`: recomputes QK/softmax for one four-block output slice using
   the saved final row normalization contract, owns only four PV C fragments,
   and writes its disjoint Hd64 output half.

The two PV slices run independently. QK recomputation is deliberate; it
removes coexistence of QK C, loop-carried m/l updates, and PV C from the PV
kernel rather than merely renumbering them.

## Pseudocode

```text
score_state(q, k):
  m=-inf; l=0
  for kv_tile:
    score = QK_WMMA(q, k[kv_tile])
    m, l = online_update(score, m, l)
  store(m, l)

pv_slice(q, k, v, m, l, output_block_base):
  acc[4]=0
  for kv_tile:
    score = QK_WMMA(q, k[kv_tile])
    p = exp(score - m) / l
    for state_block in 0..3:
      acc[state_block] += PV_WMMA(p, v[kv_tile, output_block_base+state_block])
  store(acc, output_block_base)
```

## Affected contracts

- Versioned `AMDScoreStateSpec` owning only m/l output and its explicit
  scratch layout.
- `AMDPackedFragmentLoopSpec` role separation for score-state and PV-slice.
- New state-load primitive for m/l in the PV slice; no implicit spill/private
  memory is permitted.
- Slice capture schema must record one score-state pass and two disjoint PV
  passes, including QK recomputation in each PV pass.
- Existing full ABI/default remains unchanged and fail-closed.

## Expected resource result

The PV slice should remove loop-carried m/l writes and QK C from the same
allocation interval as its four PV C fragments. The admission target is
`<= 192 VGPR` per PV slice, versus `244/245` for the current four-block
variant. This is a target, not a claim.

## Gates

- Full Hd128 output numeric error against fp32 reference at the existing
  attention tolerance.
- Two PV stores cover exactly blocks `[0,8)` with no overlap.
- State scratch is explicit LDS/global ABI, zero private bytes, zero spills.
- Captured QK/PV role counts prove score-state plus two recomputing PV passes.
- Promote only if compiled allocation is `<=192` VGPR or calculated residency
  demonstrably increases. Otherwise record the negative result; do not replay.
