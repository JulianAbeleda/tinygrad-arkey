# QK Harness Validation

Date: 2026-06-12

Purpose: validate the manifest-checked QK policy harness on real AMD runs and
publish one matrix over the current 8B, 14B, and 32B artifacts.

## Results

Initial matrix:

| model | status | reference | explicit tok/s | generated tok/s | gain | A/B | note |
|---|---|---|---:|---:|---:|---:|---|
| 8B | `accept` | explicit | `49.35` | `53.49` | `8.41%` | true | stable generated-policy win |
| 14B | `needs-rerun` | explicit | `22.72` | `34.75` | `52.93%` | true | generated window stayed unstable after top-up |
| 32B | `accept` | generic | `3.44` | `4.16` | `20.98%` | true | existing capped-policy artifact; generic-baseline caveat |

Rerun matrix:

| model | status | reference | explicit tok/s | generated tok/s | gain | A/B | note |
|---|---|---|---:|---:|---:|---:|---|
| 8B | `accept` | explicit | `49.35` | `53.49` | `8.41%` | true | stable generated-policy win |
| 14B rerun | `accept` | explicit | `22.76` | `39.61` | `74.02%` | true | stable generated-policy win on clean rerun |
| 32B | `accept` | generic | `3.44` | `4.16` | `20.98%` | true | existing capped-policy artifact; generic-baseline caveat |

Matrix artifacts:

- `matrix-summary.json`
- `matrix-summary.md`
- `matrix-summary-rerun.json`
- `matrix-summary-rerun.md`

## Harness Checks

- Fresh 8B and 14B runs wrote `manifest.json`.
- All stage status files were emitted.
- 8B completed with `status=accept`.
- 14B completed with `status=needs-rerun`, which is the correct harness verdict
  because `generated3` was more than 10% below the latest-window mean after the
  allowed top-up runs.
- `--reuse` succeeded on both 8B and 14B without rerunning heavy stages.
- A changed-argument `--reuse` smoke on 8B failed against the manifest as
  expected.
- The 14B rerun passed under the original harness rule. No decision-metric
  change was needed.

## Interpretation

The harness behavior is validated. The initial 14B result is a real example of
the harness refusing an exciting but unstable speedup; the targeted rerun shows
that instability was transient measurement noise, not a reason to weaken the
decision rule.

Use the rerun matrix as the current 8B/14B/32B summary. The next work should
stay at the harness level unless a storage change naturally enables a cleaner
32B comparison.

Stop rule preserved: do not resume kernel search from this track, and do not
chase a third scaling point by perfecting 32B.
