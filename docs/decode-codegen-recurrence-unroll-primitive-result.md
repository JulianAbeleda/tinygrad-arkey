# Recurrence-aware loop-unroll primitive — built + verified on the canonical recurrence (2026-06-26)

The foundational codegen scheduling primitive (`docs/decode-codegen-recurrence-unroll-primitive-scope.md`)
is implemented and the hard core (AFTER-chain reconstruction) is verified correct. Default-off.

## Built

- `extra/qk_codegen_recurrence_unroll.py` — `unroll_recurrence(sink, U)`: finds a REDUCE range with a
  recurrence carry, makes a fresh outer range `r2` of size N/U, and replicates the per-iteration store-chain
  U times, mapping `r → r2*U+u` in indices and **re-threading every `AFTER(X, r)` loop-carry** (copy 0 →
  `AFTER(X, r2)`; copy u → copy u−1's store) — the exact obstacle from the layer-2 investigation. Inner
  ranges nested inside the loop are duplicated per copy (fresh ids) so the copies are independent.
- Hook: `tinygrad/codegen/__init__.py` `full_rewrite_to_sink`, env-gated `SCHED_UNROLL=<U>` (AMD), run on
  the raw `ast` before the recurrence is lowered; added to the `to_program` cache key. Default codegen
  (SCHED_UNROLL unset) is byte-identical.

## Verified (correctness)

Tiny single-accumulator recurrence reduce (`out[h] = Σ_j in[h*8+j]` via a REG accumulator + `.after(j)`):
**U=0/2/4 all numerically correct** → the AFTER-chain reconstruction is correct on the canonical recurrence.
This is the hard core proven: a recurrence loop can be scalar-unrolled with the loop-carry re-threaded.

## The attention-tile generalization (precisely characterized, not yet handled)

`extra/qk_decode_attention_block_tile_microgate.py` under `SCHED_UNROLL=2` does NOT yet pass — it trips the
control-flow assertion `CFGContext` `tinygrad/codegen/late/linearizer.py:86`
(`assert y.src[1] not in x.backward_slice_with_self`, the sibling-range ordering TODO). The tile's
recurrence is more complex than the canonical case in three ways v1 does not yet handle:
1. **Per-iteration re-inits vs true carries.** `dotp.after(b, tt)[0].store(0.0)` is a multi-range `AFTER`
   that is a per-iteration *re-init* (stores a constant), NOT a loop-carry; only `acc/den/mx` are genuine
   recurrences. v1 rewires *all* `AFTER(_, tt)`, including the re-init — wrong. Fix: only rewire an
   `AFTER(X, r)` whose read feeds the value that is stored back to X (a true recurrence), leaving re-inits
   (constant stores) to replicate per copy unchanged.
2. **Multi-range `AFTER`.** `dotp.after(b, tt)` carries over two ranges; the rewire must preserve the other
   ranges (drop/replace only `tt`), not collapse to `after(r2)`.
3. **Nested inner accumulators** (the `rp` dot loop has its own `AFTER(dotp, rp)`), which must be left
   intra-copy and not confused with the `tt` recurrence.

These are well-defined: the next increment is a recurrence classifier (true-carry vs re-init vs
inner-accumulator) feeding the same AFTER-chain reconstruction that already works. The tiny kernel + the
block-tile microgate are the correctness oracles; `SCHED_LIST` (layer 1) + the isolated timing are the
perf oracles (the timing is still flat at 7023 µs and moves only once a correct unroll exposes the ILP).

## Status

- AFTER-chain reconstruction primitive: BUILT + VERIFIED on the canonical recurrence (`SCHED_UNROLL`).
- Layer 1 list scheduler: BUILT + VERIFIED (`SCHED_LIST`).
- The two compose: unroll exposes cross-iteration ILP, the scheduler interleaves it. Both default-off.
- Remaining for the decode tile: generalize the primitive to the tile's multi-accumulator recurrence
  (the classifier above). Label: `SEARCH_PROGRESS__RECURRENCE_UNROLL_CANONICAL_VERIFIED__TILE_GENERALIZATION_DEFINED`.

Honest accounting: the hard core (re-threading a recurrence carry through an unroll) is done and proven —
that was the thing the whole layer-2 investigation flagged as the obstacle. The decode tile's specific
recurrence shape needs the carry/re-init classifier to apply it; that is the focused next increment, with
the structure and oracles in place.
