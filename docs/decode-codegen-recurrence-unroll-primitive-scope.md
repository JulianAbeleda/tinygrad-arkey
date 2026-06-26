# Recurrence-aware loop-unroll primitive (AFTER-chain reconstruction) — scope (2026-06-26)

The foundational codegen primitive for the scheduling capability: correctly scalar-unroll a REDUCE loop by
factor U, re-threading the `Ops.AFTER(acc, range)` loop-carry across the U copies, so the copies' independent
prologues (loads + fdot2 + cross-lane reduce) coexist in one basic block where the layer-1 list scheduler
(`SCHED_LIST`, built) interleaves them into the recurrence latency shadow. Default-off, generic, no
materialization, parameterized by U (later a searchable knob). Layer 1 + this = the durable foundation.

## The transform (precise)

Given a sink with a REDUCE `Ops.RANGE` `r` (size N, the loop var) and a manual REG-accumulator recurrence
(stores reading `X.after(r)` loop-carries, closed by an `END` over `r`) — as in
`flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel` (`extra/qk_flash_decode.py:957-967`) — unroll by
U (first version requires U | N):
1. New range `r2` of size N/U, same `AxisType` (REDUCE).
2. For `u in 0..U-1`, build copy `u` of the body with the loop *index* `r` mapped to `r2*U + u`
   **in index/value expressions only** (via `substitute`, the `x.substitute({var: var.const_like(i)})`
   idiom, `tinygrad/schedule/multi.py:11`).
3. **Re-thread the recurrence (the primitive's hard core):** every loop-carry read `X.after(r)` (X a REG
   accumulator) is rewired:
   - copy 0 → `X.after(r2)` (the value carried from the previous outer iteration),
   - copy u>0 → copy u−1's corresponding STORE result.
   Substitute must NOT touch the `AFTER`'s range arg (that would make `acc.after(const)`, the exact bug in
   `docs/decode-codegen-swp-layer2-investigation.md`); the rewire is a separate, explicit step.
4. The `END` over `r` becomes `END` over `r2`, reading copy U−1's final store.

## Why this is the primitive (not the loop-split quick win)

Generic over any loop-carried dependency (the AFTER-rewiring machinery is reusable for online-softmax,
scans, IIR, the prefill-GEMM K-loop); it is the layer SWP is built from (SWP = unroll + reschedule +
prologue/epilogue); it composes with the layer-1 scheduler (its consumer); it is a pure transform (no
intermediate buffer / HBM cost); U is a single searchable knob. Loop-split is kernel-shape-specific, adds a
materialized score buffer, and does not generalize — rejected.

## Build + verify plan (verification-first; do NOT scale before the tiny case is correct)

1. **Understand the representation:** dump the pre-linearize UOp graph of a *tiny* manual REG-accumulator
   recurrence-reduce kernel (op / `.ranges` / `AFTER` src) to fix the exact node shapes the transform
   rewrites. (`extra/qk_codegen_recurrence_unroll.py` recon mode.)
2. **Implement `unroll_recurrence(sink, U)`** in `extra/qk_codegen_recurrence_unroll.py`; hook env-gated
   `SCHED_UNROLL=<U>` in `tinygrad/codegen/__init__.py` near the other opt-in passes (after the opt stage,
   before expander), added to the `to_program` cache key (`:255`). Default-off.
3. **Verify on the tiny kernel FIRST:** numeric == NumPy for U∈{1,2,4} (U=1 must be identity); dump shows U
   independent prologues in one block. Only then proceed.
4. **Verify on the real target:** `extra/qk_decode_attention_block_tile_microgate.py` →
   `BLOCK_TILE_MICROGATE_PASS` with `SCHED_UNROLL` (the correctness oracle for the recurrence rewire).
5. **Measure:** isolated block-tile timing with `SCHED_UNROLL=4 SCHED_LIST=1` vs baseline (currently flat
   7023 µs). The number moves iff the unroll exposes real ILP that layer 1 interleaves — the live oracle.
6. Then route gate clean + ISA-vec gate + W==D toward baseline.

## Pass/fail labels

- `RECURRENCE_UNROLL_TINY_PASS` / `_FAIL__NUMERIC` — the tiny-kernel correctness gate (step 3).
- `RECURRENCE_UNROLL_MICROGATE_PASS` — block-tile numeric correct under unroll (step 4).
- `SEARCH_PROGRESS__RECURRENCE_UNROLL` — unroll + `SCHED_LIST` moves the isolated timing (the capability is
  real); continue to W==D + generality (prefill) + searchable-U (Phase 3).
- `SEARCH_BLOCKED_BY_CODEGEN__RECURRENCE_REWIRE` — if the AFTER-chain reconstruction cannot be made correct
  on the manual-REG recurrence shape (record the exact node it breaks on; this is a real finding, not a
  stop on the capability).

## Constraints

Default-off (env-gated + cache key); shipped default route + GEMVs byte-for-byte unchanged; correctness
first, tiny-kernel before model, microgate before timing; pure transform (no materialization); do not edit
`tinygrad/runtime/autogen/**`; do not hand-restructure the attention kernel. Iterative compiler work against
the tiny-kernel + microgate + isolated-timing harness — build it right, not fast.
