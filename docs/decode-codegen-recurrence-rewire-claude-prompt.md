# Claude prompt: finish recurrence rewire for generated decode block tile

You are in `/home/ubuntu/tinygrad-arkey` on AMD gfx1100. The goal is the generic codegen scheduling capability, not a one-off attention-kernel shortcut.

## Goal

Make `SCHED_UNROLL=<U>` correctness-clean on the generated decode block tile so cross-iteration ILP can be exposed to the default-off list scheduler (`SCHED_LIST=1`).

The specific required oracle is:

```bash
DEV=AMD JIT=1 SCHED_UNROLL=2 PYTHONPATH=. python3 extra/qk_decode_attention_block_tile_microgate.py
```

It must print:

```text
BLOCK_TILE_MICROGATE_PASS
```

Do not run isolated timing, W==D, or promotion gates until this correctness oracle passes.

## Strategic context

This is the durable machine-search/codegen foundation:

- Layer 1 already exists: `extra/qk_codegen_list_scheduler.py`, enabled by `SCHED_LIST=1`.
- Layer 2 already exists in canonical form: `extra/qk_codegen_recurrence_unroll.py`, enabled by `SCHED_UNROLL=<U>`.
- The canonical single-accumulator recurrence unroll works.
- The generated decode block tile still fails because its recurrence shape is more complex.

The point is not to hand-restructure the attention kernel. The point is to teach codegen how to safely unroll manual REG recurrence loops so BubbleBeam/FutureSight can eventually search over the unroll/schedule decision.

## Current status

Recent work partially implemented the recurrence classifier in:

- `extra/qk_codegen_recurrence_unroll.py`

Current result doc:

- `docs/decode-codegen-recurrence-unroll-continuation-result.md`

The classifier now attempts to distinguish:

- true read-modify-write carries,
- per-iteration constant re-inits,
- multi-range `AFTER` nodes,
- matching carry-state dependencies.

But the real decode block-tile oracle still fails before numeric comparison.

## Current failure

Command:

```bash
DEV=AMD JIT=1 SCHED_UNROLL=2 PYTHONPATH=. python3 extra/qk_decode_attention_block_tile_microgate.py
```

Failure point:

```text
tinygrad/codegen/late/linearizer.py:86
assert y.src[1] not in x.backward_slice_with_self
```

The microgate reports `SEARCH_BLOCKED_BY_CODEGEN__MULTI_WARP_NOT_EXPRESSED`, but that label is too broad for this run. The accurate label is:

```text
SEARCH_BLOCKED_BY_CODEGEN__RECURRENCE_REWIRE
```

because the base block tile passes without `SCHED_UNROLL`; the failure is specifically introduced by recurrence unroll CFG reconstruction.

## Exact blocker signature

A direct diagnostic on the transformed graph reports:

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

- Axis `1008` is the new unrolled outer token range. Original `tt` size 16 becomes size 8 for `SCHED_UNROLL=2`.
- Axis `6` is the inner `rp` fdot2 accumulator range.
- The unroller is still making the inner `rp` END appear as a sibling/dependency of itself after outer recurrence rewrite.
- This means the current carry-state selector is still too coarse or it is selecting an END whose backward slice includes the inner `rp` range in a way CFG cannot order.

## Important model of the tile recurrence

In `extra/qk_flash_decode.py`, the generated block-tile kernel has nested recurrence structure:

- Outer block loop `b`.
- Token loop `tt` over TK=16. This is the loop we are trying to unroll.
- Inner dot loop `rp`. This builds `dotp` for one token.
- Then the online softmax recurrence updates `acc`, `den`, and `mx` across `tt`.

The key distinction:

- `dotp.after(b, tt)[0].store(0.0)` is a per-token re-init, not a carry.
- `dotp.after(rp)` is an inner accumulator for the q.k dot and must stay intra-copy.
- `acc.after(tt)`, `den.after(tt)`, and `mx.after(tt)` are true carries and must be re-threaded across unrolled copies.

## What must be fixed next

Fix `extra/qk_codegen_recurrence_unroll.py` so the selected carry dependency for each true carry does not drag the inner `rp` END into the outer `tt` recurrence CFG.

Concrete target:

1. Add or improve debug output behind an env var, for example `SCHED_UNROLL_DEBUG=1`, printing:
   - selected unrolled range,
   - all `AFTER(..., r)` nodes,
   - which are true carries,
   - which are re-inits,
   - which inner ranges are duplicated,
   - for each true carry, the exact selected previous-copy dependency node.

2. For each true carry, select the post-update state at the correct outer recurrence level:
   - `acc`: should depend on the `END(acc_store, dd)` closure over the `dd` loop, not on any END involving `rp`.
   - `den`: should depend on the scalar den store after the acc update.
   - `mx`: should depend on the scalar mx store after the den update.

3. Preserve inner accumulator isolation:
   - The `rp` range and `dotp` stores must be duplicated per unrolled copy.
   - No carry dependency used for `acc/den/mx` should make the inner `rp` END participate as a sibling of itself in `CFGContext`.

4. Re-run only the correctness oracle first:

```bash
DEV=AMD JIT=1 SCHED_UNROLL=2 PYTHONPATH=. python3 extra/qk_decode_attention_block_tile_microgate.py
```

Only after `BLOCK_TILE_MICROGATE_PASS`, run timing sweep:

```bash
DEV=AMD JIT=1 SCHED_UNROLL=2 SCHED_LIST=1 PYTHONPATH=. python3 extra/qk_decode_block_tile_isolated_timing.py
DEV=AMD JIT=1 SCHED_UNROLL=4 SCHED_LIST=1 PYTHONPATH=. python3 extra/qk_decode_block_tile_isolated_timing.py
DEV=AMD JIT=1 SCHED_UNROLL=8 SCHED_LIST=1 PYTHONPATH=. python3 extra/qk_decode_block_tile_isolated_timing.py
```

## Guardrails

- Do not hand-restructure `extra/qk_flash_decode.py` to make the tile faster.
- Do not bypass CFG or relax the assertion.
- Do not materialize intermediate tensors.
- Keep everything default-off behind `SCHED_UNROLL` / `SCHED_LIST`.
- Do not claim progress unless the correctness oracle passes.
- Do not claim performance progress unless isolated timing actually moves.

## Success criteria

Minimum success:

```text
BLOCK_TILE_MICROGATE_PASS under SCHED_UNROLL=2
```

Then performance success:

```text
SEARCH_PROGRESS__RECURRENCE_UNROLL
```

Only if isolated block-tile timing improves under `SCHED_UNROLL=<U> SCHED_LIST=1` versus the current generated baseline.

If the classifier cannot be made correct, record the exact blocker node under:

```text
SEARCH_BLOCKED_BY_CODEGEN__RECURRENCE_REWIRE
```
