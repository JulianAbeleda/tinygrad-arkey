# Codex task prompt — continue the codegen scheduling capability (foundation + remaining work)

Copy below the line into Codex. Context: `docs/decode-codegen-scheduler-capability-scope.md`,
`docs/decode-codegen-recurrence-unroll-primitive-result.md`,
`docs/decode-codegen-list-scheduler-result.md`. This is the long-term machine-search enabler — a generic
codegen instruction-scheduling/software-pipelining capability so the *machine* produces latency-hidden
kernels. Build the foundation properly; do NOT take quick wins (no loop-split, no hand-restructuring the
attention kernel). Everything default-off, correctness-first.

---

Repo `/home/ubuntu/tinygrad-arkey` (AMD gfx1100; hardware present, `DEV=AMD JIT=1 PYTHONPATH=.`).

## What is already built and VERIFIED (committed, default-off — do not regress)

Two foundational primitives, both env-gated, default codegen byte-identical:

1. **Layer 1 — latency-aware list scheduler** (`extra/qk_codegen_list_scheduler.py`, hooked in
   `tinygrad/codegen/late/linearizer.py` under `SCHED_LIST=1`). Basic-block-scoped (correctness-preserving
   by construction); reorders independent ops to fill load/`ds_bpermute` latency shadows. Verified:
   matmul/reduce/elementwise + the block-tile microgate correct. It is the *consumer* of cross-iteration
   ILP — alone it moves nothing (a serial loop body has no independent work).

2. **Layer 2 — recurrence-aware loop-unroll primitive** (`extra/qk_codegen_recurrence_unroll.py`, hooked in
   `tinygrad/codegen/__init__.py` `full_rewrite_to_sink` under `SCHED_UNROLL=<U>`, on the raw `ast` before
   the recurrence lowers; in the `to_program` cache key). It unrolls a REDUCE loop by U and **re-threads
   each `AFTER(X, r)` loop-carry** across the U copies (copy 0 → `AFTER(X, r2)`; copy u → copy u−1's store),
   duplicating inner ranges per copy. VERIFIED CORRECT on the canonical single-accumulator recurrence
   (`out[h]=Σ_j in[h*8+j]`, U=0/2/4). The hard core (re-threading a recurrence carry through an unroll)
   works.

The two compose: unroll exposes cross-iteration ILP, the scheduler interleaves it.

## The remaining work, in strict order

### STEP 1 (do this) — generalize the unroll primitive to the decode tile's recurrence
`extra/qk_decode_attention_block_tile_microgate.py` under `SCHED_UNROLL=2` currently trips the CFG
assertion `tinygrad/codegen/late/linearizer.py:86` (sibling-range ordering), because v1 mis-classifies the
tile's `AFTER` nodes. Add a **recurrence classifier** to `_unroll_one_range`
(`extra/qk_codegen_recurrence_unroll.py`) that, for each `AFTER(X, …)` over the loop range `r`, decides:
- **true carry** — the read feeds the value stored back to X (read-modify-write: e.g. `acc[dd].store(acc.after(tt)[dd]*corr + …)`, and `den`, `mx`). **Re-thread these** (existing logic).
- **per-iteration re-init** — `AFTER(X, …, r)` immediately consumed by a `STORE` of a constant (e.g.
  `dotp.after(b, tt)[0].store(0.0)`). **Do NOT re-thread**; replicate per copy unchanged.
- **inner accumulator** — `AFTER(X, inner_range)` where `inner_range != r` (e.g. the `rp` dot loop's
  `AFTER(dotp, rp)`). Leave intra-copy; it is duplicated with the inner range.
- **multi-range AFTER** — `AFTER(X, a, …, r)`: when re-threading, **preserve the other ranges**, replace
  only `r`'s role (do not collapse to `after(r2)` and drop `a`).

The recon (exact node shapes) is in `docs/decode-codegen-recurrence-unroll-primitive-result.md`; the
representation is: in-loop carry = `AFTER(X, r)`, per-iter result = `END(store_chain, r).src[0]`,
post-loop = `AFTER(X, END)`. Keep the transform pure (no materialization), generic, default-off.

### STEP 2 — verify correctness (oracles, in order)
- Tiny canonical kernel must STAY correct (regression guard) — the U=2/4 reduce test in
  `docs/decode-codegen-recurrence-unroll-primitive-result.md`.
- `DEV=AMD JIT=1 SCHED_UNROLL=2 PYTHONPATH=. python3 extra/qk_decode_attention_block_tile_microgate.py`
  → `BLOCK_TILE_MICROGATE_PASS` (numeric correct under unroll). This is the correctness oracle for the
  recurrence rewire on the real tile.

### STEP 3 — measure (the live perf oracle)
Isolated per-kernel timing of the block tile (eager `custom_kernel` + DEBUG=2, see
`docs/decode-block-tile-codegen-result.md` Part A) at ctx4096, comparing:
`SCHED_UNROLL=0` (7023 µs baseline) vs `SCHED_UNROLL=4 SCHED_LIST=1`. **The number moves iff the unroll
exposes real ILP that layer 1 interleaves.** Sweep U∈{2,4,8}. If it drops materially → the capability is
real (`SEARCH_PROGRESS__RECURRENCE_UNROLL`).

### STEP 4 — close decode + prove generality
- Route gate clean + ISA-vec gate `ISA_VEC_AUTHORITATIVE_PASS` under `SCHED_UNROLL SCHED_LIST`.
- W==D toward baseline 82.4/103.5/101.8/94.6:
  `DEV=AMD JIT=1 DECODE_ATTN_GENERATED_WHOLECACHE=1 DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE=1 DECODE_ATTN_BLOCK_TILE=1 V_DOT2_LOWERING=1 SCHED_UNROLL=4 SCHED_LIST=1 PYTHONPATH=. python3 extra/qk_decode_runtime_overhead.py`.
- **Generality proof (required):** the same `SCHED_UNROLL`+`SCHED_LIST` moves the prefill-GEMM hot loop —
  the test that this is a capability, not a kernel hack.

### STEP 5 — make it a searchable codegen decision (Phase 3)
Lift U (and the unroll/schedule choice) into BubbleBeam/FutureSight so the *machine* selects it; decode
attention + prefill GEMM are the anchor cases the search must reproduce. This is the end state.

## Labels (the only abort is correctness/regression, never cost)

- `RECURRENCE_UNROLL_MICROGATE_PASS` — STEP 2 numeric correct under unroll.
- `SEARCH_PROGRESS__RECURRENCE_UNROLL` — STEP 3 isolated timing moves; continue.
- `SEARCH_PROGRESS__CODEGEN_SCHEDULER` — STEP 4 W==D toward baseline + generality on prefill.
- `SEARCH_BLOCKED_BY_CODEGEN__RECURRENCE_REWIRE` — only if a *correct* classifier still cannot apply the
  reconstruction on the tile's recurrence (record the exact node it breaks on). Even then: keep building
  the capability; do not drop it.

## Constraints

Default-off (env-gated + cache key already wired for `SCHED_UNROLL`/`SCHED_LIST`); shipped default route +
q4k GEMVs byte-for-byte unchanged; correctness-first, tiny-kernel + microgate before timing; pure transform
(no materialization); do NOT loop-split or hand-restructure the attention kernel for speed (off-principle —
the foundation is the generic codegen capability, not a faster hand-written kernel); do NOT edit
`tinygrad/runtime/autogen/**`. Bracketed-prefix commits (`[codegen]`, `[nn]`) with gate verdicts. Do NOT
claim a step worked unless the correctness oracle passes AND (for perf steps) the isolated timing or W==D
actually moved.

Deliverable: the recurrence classifier landing `BLOCK_TILE_MICROGATE_PASS` under `SCHED_UNROLL`, the
isolated-timing before/after (U sweep), and — if it moves — W==D + the prefill generality proof; or, if a
correct classifier still can't apply, the exact `RECURRENCE_REWIRE` blocker node. Build the foundation
right, not fast.
