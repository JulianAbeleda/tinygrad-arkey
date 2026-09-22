# Predecessor-conditioned C0/C2/C3 result (Q branch, warp-coop projection)

> **SUPERSEDED 2026-08-23:** This artifact did not contain an installed P arm,
> used fresh zero-filled FFN allocations in C3, did not validate C2/C3 output
> SHAs after those arms, and did not retain both reported raw sessions. The
> corrected exact live-prefix reverse-bracket result is
> `nv-r-predecessor-conditioned-exact-result-20260823.md`.

Commit: `6570abc025514273faa100c66b979e531585a1e1`
Tool: `extra/llm_research/decode/nv_r_predecessor_conditioned_live.py`
Evidence: `docs/task_workflow/evidence/nv-r-predecessor-conditioned-live-20260823/`
GPU: RTX 5090, SM 2850 MHz, mem 14001 MHz, locked, fresh process per run

## Question

How much of the Q-projection command-interval residual `R` is set by the
installed production predecessor (cache state), versus the stress-flush
cache-sensitivity ceiling?

The prior harness measured a stress-flush cold-hot sensitivity of
`+6.13..6.53 us` for `q4k_warp_coop_q8_dp4a_partial_4096_4096`. That is a
ceiling. This gate measures the fraction the actual production predecessor
realizes.

## Tooling correction

- [MEASURED] The earlier live tool reported `PROVIDER_MISMATCH` because it
  read the target input VA from `buf_va[0]`. The capture shows the target
  cubin receives `[output, weight, input] = [65536, 9437184, 8192]`; its input
  is `buf_va[2]`, which exactly equals the provider output `buf_va[0]`.
- [MEASURED] Replay on the corrected argument order is bit-exact:
  `replay_output_checksum == production_output_checksum == f957b653...` in every
  run. `provider_output_idx = 0`.
- [MEASURED] `fill_kernargs` QMDs are now built once per dispatch, not aliased
  across in-flight launches, matching the corrected harness discipline.

## Measured arms

One model load, one fresh process, same cloned live buffers for all three arms:

- `C0` isolated target on cloned input (weight warmed by repeated launch).
- `C2` provider norm -> target (immediate 24 KB predecessor).
- `C3` FFN gate/up (56.6 MB weights) + FFN down (41.3 MB weight) -> provider
  -> target (full local ~98 MB prefix that precedes the provider in the
  production schedule).

Medians over 24 retained samples after 4 warmup, two locked sessions:

| Session | C0 us | C2 us | C3 us | C2-C0 | C3-C0 | C3-C2 |
|---------|-------|-------|-------|-------|-------|-------|
| 1       | 6.096 | 5.712 | 7.200 | -0.384 | +1.104 | +1.488 |
| 2       | 6.128 | 5.696 | 7.200 | -0.432 | +1.072 | +1.504 |

- [MEASURED] The distributions separate: C3 min `7.040..7.072 us` is above C0
  max `6.240..6.272 us` in both sessions.
- [MEASURED] Production output checksum `f957b653...` is identical across all
  three runs, so the replay is deterministic.

## Interpretation

- [INFERRED] The immediate predecessor (24 KB provider norm) contributes
  approximately zero to cache-state: `C2 - C0` is negative by `~0.4 us`.
- [INFERRED] The ~98 MB FFN prefix adds `+1.07..1.50 us` to the Q projection.
  Against the `+6.13..6.53 us` stress-flush ceiling, the actual production
  predecessor realizes roughly 17-23% of the stress sensitivity.
- [INFERRED] Therefore "cache state explains all of `R`" is not supported at
  the production-predecessor level for the Q branch. The stress flush is an
  upper bound, not a production mechanism. The predecessor-conditioned
  cache-state share of the Q branch is modest and bounded at roughly 1.5 us.

## Not yet measured

- [UNMEASURED] Whether the ~17-23% fraction generalizes to the K/V, O, flash,
  gate/up, and down rows. This is the next C0-C3 extension.
- [UNMEASURED] The residual `R` beyond the ~1.5 us predecessor share (dispatch
  tail, kernel body, or occupancy). A wait-exit / admission trace is still
  required to close `P - C` into named observables.
- [UNMEASURED] Full-token wall recovery. No `240 tok/s` recovery is booked from
  this result.

## Next gate

Extend the same live-replay partition to the K/V cooperative projection and
the O projection, where the ledger rows are larger, then run the K/V
two-queue scheduler microgate and the token-SHA reverse wall bracket.
