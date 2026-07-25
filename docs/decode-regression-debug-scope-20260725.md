# Scope: debug the decode throughput regression (2026-07-25)

## Measured facts

8B Q4_K_M, ctx512, `extra/qk/decode/decode_runtime_overhead.py` (`tinygrad.decode.fixed_depth.v2`),
`--nmeas 40 --reps 5`, auto clock, `flash` route, fp16 KV, `flash_decode_threshold=512`:

| code state | date | tok/s | per-token |
|---|---|---:|---:|
| `dbec46337` | 07-16 | **114.64** | 8.72 ms |
| `dbec46337`, as recorded on 07-16 | 07-16 | 113.94 | -- |
| `f345d4f8d` | 07-21 | **98.49** | 10.15 ms |
| `dfbaddbdf` (HEAD) | 07-25 | **95.11** | 10.50 ms |

14B, same harness: 68.90 (07-16) -> 60.29 (today) at ctx512; 59.57 -> 53.48 at ctx4096.

**The machine is not the cause.** 07-16 code re-run today reproduces its own 07-16 number to within 0.6%,
on the same GPU within the same hour as the slow HEAD measurement. Configuration recorded in both artifacts
is identical: same `runtime_settings`, same model `identity_sha256`, same route, 948 vs 949 captured programs.

**There are two segments, and they may have different causes:**
- 07-16 -> 07-21: 114.64 -> 98.49 (**-14.1%**), 373 commits
- 07-21 -> 07-25: 98.49 -> 95.11 (**-3.4%**), 514 commits

Do not assume one root cause explains both.

## Phase 1 -- isolate (IN PROGRESS)

`git bisect run` over `dbec46337..f345d4f8d` in an isolated worktree, threshold 106 tok/s
(good = 114.64, bad = 98.49; the gap is 16 points, so the threshold is not near either side).
Script: `$CLAUDE_JOB_DIR/tmp/bisect_test.sh`, `--reps 3` for speed.

Steps observed so far: 115.38 (good), 98.22 (bad). The two clusters are cleanly separated, which means the
threshold is not producing ambiguous verdicts.

**Caveat to respect:** the decode core is deleted at `45cfc399c` (07-21 10:33), which is *outside* this
narrowed window but inside the full one. The second segment (07-21 -> today) cannot bisect natively -- the
harness must be injected at each step, holding the instrument constant while the code under test varies.

## Phase 2 -- confirm

Isolating a commit is not proof. Required before accepting it:
1. Re-run the suspect and its parent at `--reps 5` (bisect uses 3), interleaved, warm.
2. Revert the suspect's change on top of HEAD and re-measure. If throughput returns to ~114, the attribution
   holds. If it does not, the bisect found a threshold crossing rather than the cause.

## Phase 3 -- mechanism

Only after Phase 2. What to compare between the fast and slow states, cheapest first:
- **Program count / route**: both states report `flash` and ~948 programs, so gross route selection is
  probably unchanged -- but confirm `programs_per_token_by_route` and the captured JIT program count exactly.
- **Per-token wall breakdown**: `host_sync_pct_of_wall` and `host_sync_residual_ms` are already in the
  artifact. If host-sync share grew, the regression is host-side, not kernel-side.
- **Emitted kernel diff**: dump the decode kernels in both states and diff. A changed kernel is a codegen
  regression; identical kernels point at scheduling, dispatch, or memory layout.
- **PMC counters** on the decode dispatch in both states (`extra/qk/prefill/prefill_boltbeam_trace.py`
  requires sudo; restore `power_dpm_force_performance_level` to `auto` afterwards).

## Phase 4 -- fix

Depends entirely on Phase 3. Do not pre-commit to an approach.

## Gates for any fix

- 8B decode ctx512 back to ~114 tok/s, `--reps 5`, interleaved against HEAD, warm reps only.
- 14B decode ctx512 back toward 68.9.
- Prefill unchanged: 8B pp512 ~3700 tok/s (prefill was NOT part of this regression; do not trade it away).
- Unit suite failure-set equality (currently 50 failed).
- Token parity unchanged.
- `cat /sys/class/drm/card*/device/power_dpm_force_performance_level` reads `auto` before trusting a timing.

## Why this went unnoticed for 9 days

`extra/qk/bench.py --decode` invoked `extra/qk/decode_runtime_overhead.py`, which `45cfc399c` deleted while
leaving the caller intact, so the canonical decode benchmark failed with file-not-found from 07-21 until it
was restored on 07-25. `test/unit/test_measurement_authority.py` now fails if any path `bench.py` dispatches
to does not exist.
