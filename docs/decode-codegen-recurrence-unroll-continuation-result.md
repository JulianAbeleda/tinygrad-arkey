# Decode codegen recurrence-unroll continuation result

Date: 2026-06-26

Prompt: `docs/decode-codegen-scheduler-capability-continuation-codex-prompt.md`

## Verdict

The requested recurrence classifier was partially implemented, but the real decode block-tile correctness oracle still does **not** pass under `SCHED_UNROLL=2`.

Current label:

`SEARCH_BLOCKED_BY_CODEGEN__RECURRENCE_REWIRE`

Do not run timing/W==D with `SCHED_UNROLL` yet. The correctness oracle fails before codegen reaches a valid kernel.

## What was changed

File touched:

- `extra/qk_codegen_recurrence_unroll.py`

Classifier additions:

- Added root-buffer detection for `INDEX(AFTER(...))` chains.
- Added true-carry classification: only re-thread an `AFTER` if its loaded value feeds a `STORE` back to the same underlying root buffer.
- Added explicit per-iteration re-init exclusion: `AFTER(..., r)[idx].store(CONST)` is not a carry, even if the same root has an inner accumulator later.
- Preserved multi-range `AFTER` structure when replacing the selected range: `AFTER(X, a, r)` becomes `AFTER(X, a, replacement)`, not `AFTER(X, replacement)`.
- For vector carries, attempted to thread through the matching previous-copy store/inner-END rather than the whole copied store chain.

## Correctness gate result

Command:

```bash
DEV=AMD JIT=1 SCHED_UNROLL=2 PYTHONPATH=. python3 extra/qk_decode_attention_block_tile_microgate.py
```

Result: still fails before numeric comparison.

Observed gate output reports `SEARCH_BLOCKED_BY_CODEGEN__MULTI_WARP_NOT_EXPRESSED`, but the more accurate label for this run is `SEARCH_BLOCKED_BY_CODEGEN__RECURRENCE_REWIRE` because the failure is inside `SCHED_UNROLL` CFG reconstruction, not the base multi-warp tile.

Failure point:

```text
tinygrad/codegen/late/linearizer.py:86
assert y.src[1] not in x.backward_slice_with_self
```

## Exact blocker signature

A direct CFG diagnostic on the transformed graph reports:

```text
RECURRENCE_REWIRE_BLOCKER
parent: END closes axis=6 type=REDUCE size=2
predecessor: RANGE axis=6 type=REDUCE size=2
successor: END closes axis=6 type=REDUCE size=2
successor_src1: RANGE axis=6 type=REDUCE size=2
successor_src1_in_predecessor_backward_slice: True
sibling_order:
  END closes axis=6 type=REDUCE size=2 depcount 1 ranges [(3, 'REDUCE', 4), (1008, 'REDUCE', 8), (0, 'GLOBAL', 8), (1, 'GLOBAL', 2)]
```

Interpretation:

- Axis 1008 is the new unrolled outer token range (`tt` unrolled by 2: original 16 -> 8).
- Axis 6 is the inner `rp` fdot2 accumulator range.
- The original `dotp` re-init is no longer the only issue; the transformed graph still leaks the inner `rp` END into a self/sibling nesting relationship.
- The remaining bug is the interaction between the outer recurrence re-thread and the nested inner accumulator closure.

## What this means

The hard-core canonical recurrence unroll remains useful, but the decode tile still needs one more structural rule before timing is valid:

- The unroller must preserve the inner accumulator (`dotp` over `rp`) entirely inside each unrolled copy.
- No dependency used to re-thread `acc/den/mx` may make the inner `rp` END appear as a sibling of itself in CFG nesting.
- The current carry-state selector is still too coarse or selects an END whose backward slice includes the inner `rp` range in a way CFG cannot order.

## Next concrete fix

Add a safer per-carry state selector:

1. For each true carry `a`, find the store back to the same root **at the selected outer recurrence level**.
2. Exclude stores/ENDs whose closure range is an inner range (`rp`, axis 6) unless that range is also fully closed before the carry dependency is attached.
3. Prefer the post-update scalar/vector state for `acc/den/mx` only:
   - `acc`: the `END(acc_store, dd)` over axis 7, not any END involving `rp`.
   - `den`: the scalar den store after acc update.
   - `mx`: the scalar mx store after den update.
4. Add a debug mode to `extra/qk_codegen_recurrence_unroll.py` that prints selected range, true carries, re-inits, inner ranges, and selected carry dependency node per carry. This should be committed as a generic diagnostic because it is now the fastest way to make progress.

Stop condition remains correctness-only: do not proceed to isolated timing until `BLOCK_TILE_MICROGATE_PASS` under `SCHED_UNROLL=2`.
