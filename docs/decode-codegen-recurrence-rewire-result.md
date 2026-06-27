# Recurrence rewire fixed — generated decode block tile unrolls correctly (2026-06-26)

Prompt: `docs/decode-codegen-recurrence-rewire-claude-prompt.md`

## Verdict

`SEARCH_PROGRESS__RECURRENCE_UNROLL`

The recurrence-rewire blocker is resolved. The required correctness oracle passes, and isolated
block-tile timing improves substantially and reproducibly under `SCHED_UNROLL=<U> SCHED_LIST=1`.

```bash
DEV=AMD JIT=1 SCHED_UNROLL=2 PYTHONPATH=. python3 extra/qk_decode_attention_block_tile_microgate.py
# -> BLOCK_TILE_MICROGATE_PASS
```

Also `BLOCK_TILE_MICROGATE_PASS` at `SCHED_UNROLL=4` and `SCHED_UNROLL=8`; base (no `SCHED_UNROLL`) and the
canonical single-accumulator recurrence remain correct (no regression).

## Root cause

`extra/qk_codegen_recurrence_unroll.py` had two defects that surfaced only on the block tile's nested
recurrence (the canonical single-accumulator case has neither, so it already worked):

1. **Inner ranges were never duplicated.** Detection used
   `inner_ranges = [u for u in final_state.toposort() if u.op is Ops.RANGE and r in u.ranges]`.
   An inner RANGE's own `.ranges` is just `{itself}` — it never contains the outer unrolled range `r`, so
   the list was always empty. The inner dot loop `rp` (axis 6) and acc-update loop `dd` (axis 7) were left
   shared across the U copies, so each copy's `END(_, rp)`/`END(_, dd)` nested as a **sibling of itself** in
   `CFGContext` → `assert y.src[1] not in x.backward_slice_with_self` (`linearizer.py:86`). This was the
   `RECURRENCE_REWIRE_BLOCKER` (axis 6 END inside axis 6 END).

2. **The per-iteration re-init register was shared.** `dotp` (REG 235) is reset to `0.0` at the top of each
   `tt` iteration (`dotp.after(b, tt)[0].store(0.0)`), accumulated over `rp`, and read out. Its non-carry
   `AFTER` collapsed to one shared `AFTER(dotp, b, r2)` for all copies, so the reset ran **once**; copy 1's
   dot product would accumulate on top of copy 0's residue → wrong numerics (and serialised the copies).

## Fix

`extra/qk_codegen_recurrence_unroll.py`, `_unroll_one_range`:

1. **Correct inner-range detection.** Duplicate (fresh axis ids per copy) exactly the ranges whose `END`
   lives inside `final_state` **and whose END body depends on `r`** (`r in e.ranges`) — i.e. loops
   re-executed every `r` iteration: `rp`, `dd`. One-time prologue loops reachable via the carry chain
   (acc init `za` axis 2, LDS staging `st` axis 4) have their END in `final_state` but their body does NOT
   depend on `r`, so they correctly stay shared.

2. **Per-copy private re-init registers.** Roots of non-carry `AFTER`s that are `DEFINE_REG`/`DEFINE_LOCAL`
   (here `dotp`) get a fresh register per copy; the non-carry `AFTER` is rebuilt onto the private register
   (keeping `r -> r2` for the ordering slot, not the index expr `r2*U+u`). Each copy now resets and
   accumulates in its own register — correct numerics and real cross-copy ILP for the dot products.

The true carries `acc/den/mx` (REG 232/234/233) stay single shared accumulators, re-threaded across copies
via `_last_store_to_root` (acc through copy u-1's duplicated `dd` END, den/mx through their scalar stores) —
unchanged and verified by the debug trace.

3. **`SCHED_UNROLL_DEBUG=1`** prints selected range, true carries, re-inits (and which are duplicated),
   duplicated inner ranges, and the per-carry previous-copy dependency node.

For the canonical case both new sets are empty, so the transform reduces to the prior (verified) behaviour —
no regression by construction, confirmed by `extra/qk_canonical_recurrence_check.py` at U=0/2/4.

## Correctness evidence

- `BLOCK_TILE_MICROGATE_PASS` at U=2/4/8; max_abs identical to baseline (e.g. 1.526e-05).
- Exact equality at the isolated-timing shape (MAXC=4608, L=86, Tc=512): U=0 vs U=8 output is
  **bit-for-bit identical (max|diff| = 0.0)** — the speedup is output-preserving, not work-skipping.
- Base path (no `SCHED_UNROLL`) unchanged; canonical recurrence PASS at U=0/2/4.

## Performance (isolated block-tile tile kernel, `extra/qk_decode_block_tile_isolated_timing.py`)

Median GPU `tm` (DEBUG=2), 3 reps, σ < 0.5%:

| config                    | ctx=512   | ctx=4096  |
|---------------------------|-----------|-----------|
| baseline (no unroll)      | 1.024 ms  | 7.32 ms   |
| `SCHED_LIST=1` only       | 1.027 ms  | 7.31 ms   | (flat — no ILP exposed yet; correct control)
| `SCHED_UNROLL=2 SCHED_LIST=1` | 1.038 ms | 7.37 ms | (flat/slightly worse below the latency-hiding threshold)
| `SCHED_UNROLL=4 SCHED_LIST=1` | 0.714 ms | 5.18 ms | (−30% / −29%)
| `SCHED_UNROLL=8 SCHED_LIST=1` | **0.539 ms** | **3.89 ms** | (**−47% / −47%, ~1.88x**) |

The token loop is latency-bound on the per-token LDS warp-reduce; unrolling + list scheduling overlaps copy
u+1's dot product into copy u's reduce-latency shadow. `SCHED_LIST` alone is flat (nothing to interleave),
confirming the win comes from unroll-exposed ILP, not the scheduler reordering the serial baseline.

## Status / next

- Default-off preserved (`SCHED_UNROLL` / `SCHED_LIST`); base codegen byte-identical.
- This is the durable capability: codegen can now safely scalar-unroll a nested manual-REG recurrence loop
  (true-carry re-thread + inner-range duplication + private re-init registers), so BubbleBeam/FutureSight can
  search over the unroll factor `U`. Next: let the search choose `U` per kernel and wire the gate, then
  consider unrolling the outer `tt`+`b` jointly.
