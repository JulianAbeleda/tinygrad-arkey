# Codegen scheduling capability — layer 1 (list scheduler) built + measured (2026-06-26)

First real component of the long-term codegen scheduling capability
(`docs/decode-codegen-scheduler-capability-scope.md`), built and verified, with an empirical result that
precisely localizes the next component.

## Built (real, generic, default-off)

- `extra/qk_codegen_list_scheduler.py` — a latency-aware list scheduler over the linearized UOp list. It
  partitions the list into basic blocks delimited by structural ops (RANGE/END/BARRIER/DEFINE/…),
  latency-reorders ops WITHIN each block (loads/`ds_bpermute` get high latency; it issues independent ready
  ops to fill latency shadows, tie-breaking to the original loads-early/stores-late order), and never moves
  an op across a structural boundary. **Correctness-preserving by construction:** it emits a valid
  topological order of the same UOps and respects the loop-nesting invariant.
- Hook: `tinygrad/codegen/late/linearizer.py` — env-gated `SCHED_LIST=1`, default-off (default codegen is
  byte-identical).

## Verified

- Correctness: matmul+relu, reduce, exp+add match NumPy with `SCHED_LIST=1`.
- Real target: `extra/qk_decode_attention_block_tile_microgate.py` → `BLOCK_TILE_MICROGATE_PASS` with
  `SCHED_LIST=1` (max_abs 1.5e-5). The scheduler is correct on the decode block tile.
- (An earlier naive whole-list reorder broke reductions — it violated loop nesting; the basic-block scope is
  the provably-safe fix. Recorded because it is the exact constraint any tinygrad scheduler must respect.)

## Measured (isolated block-tile timing, ctx4096)

| SCHED_LIST | generated block tile |
|---|---:|
| 0 (off) | 7023 µs |
| 1 (on)  | 7075 µs |

**No improvement.** This is the key result.

## Finding — within-block scheduling is necessary but NOT sufficient

The block-tile hot loop is a serial chain: per token, load → fdot2 → cross-lane reduce (`ds_bpermute`
ladder) → online-softmax update (`m/l/acc`) → PV. **Within one iteration's basic block these ops are
mutually dependent — there is no independent work to interleave into the cross-lane/recurrence latency
shadow.** The independent work lives in the NEXT iteration, behind the online-softmax recurrence. So a
list scheduler (which reorders within a block) cannot move the number — confirmed (7023→7075).

The binding capability is therefore **cross-iteration software pipelining / unrolling**: transform the
recurrence reduction loop so multiple iterations' independent work (loads, dots, the cross-lane reduce that
does not depend on the running state) coexists in one block, where the list scheduler (now built) then
interleaves it. This is layer 2 and it is harder: a naive `AxisType.UNROLL` of a recurrence is blocked
because tinygrad's expander vectorizes UNROLL axes, and a serial `m/l/acc` recurrence cannot be vectorized
— so layer 2 needs a **recurrence-aware loop pipelining transform** (a software-pipeline prologue/steady/
epilogue, or a scalar-unroll that keeps the recurrence serial while overlapping the independent prologue
of the next iteration), not a vectorizing unroll.

## Status / next

- Layer 1 (list scheduler): DONE, verified, committed, default-off. Permanent infrastructure — layer 2
  builds on it (it is the consumer of the ILP layer 2 exposes).
- Layer 2 (recurrence-aware software pipelining / scalar unroll): the precisely-localized next component.
  The harness is now in place (the scheduler + the isolated-timing method + the block-tile microgate) to
  build and verify it safely.
- This converts `SEARCH_BLOCKED_BY_CODEGEN__SCHEDULING` from a hypothesis into a measured, two-layer
  capability gap: list scheduling solved-and-insufficient; cross-iteration software pipelining is the
  binding lever. Label: `SEARCH_PROGRESS__CODEGEN_SCHEDULER_LAYER1` (list scheduler built; layer-2 SWP is
  next).

Honest scope note: layer 2 (a recurrence-aware software pipeliner) is the hard core of the capability and a
real compiler sub-project; it is the right long-term build (not dodged), and the verified layer-1 scheduler
is the foundation it stands on.
