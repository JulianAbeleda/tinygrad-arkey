# 14B Harness Instability Analysis

Date: 2026-06-12

Input artifact: `bench/qk-harness-20260612/14b/`.

## Verdict

`measurement-noise`, with no harness rule change yet.

The failing generated run recovered by the final quarter, but it was slow for
the first three quarters of the 128-token benchmark. That is too much of the
measured decode window to ignore by switching the decision metric to `last32`.

## Evidence

The run failed because the latest generated decision window was
`generated3/generated4/generated5`, and `generated3` was more than 10% below the
window mean on full-run average.

Per-quarter tok/s:

| run | avg | tokens 1-32 | 33-64 | 65-96 | 97-128 | last16 |
|---|---:|---:|---:|---:|---:|---:|
| generated1 | `39.38` | `39.09` | `39.58` | `40.38` | `38.47` | `37.98` |
| generated2 | `33.93` | `40.27` | `27.45` | `30.15` | `37.85` | `37.41` |
| generated3 | `23.75` | `17.65` | `17.07` | `21.14` | `39.12` | `38.69` |
| generated4 | `39.88` | `40.81` | `41.34` | `38.87` | `38.50` | `38.62` |
| generated5 | `40.62` | `41.57` | `41.45` | `40.30` | `39.18` | `38.92` |

The profile does not point to a new kernel-quality regression:

- batched explicit: `23.28 tok/s`;
- batched generated: `41.85 tok/s`;
- profile outliers: `0` for both batched rows.

## Rule Decision

Do not change the decision metric yet.

Changing to `avg_drop1` would not fix this case. Changing to `last64` would also
not fix it, because `generated3` was still slow across the third quarter.
Changing to `last32` would accept the run, but that would hide a collapse that
lasted roughly 96 tokens in a 128-token benchmark.

The right next action is one targeted 14B rerun under the existing rule. If the
rerun stabilizes, treat the original as noise. If it fails the same way, keep
14B generated policy marked unstable until the harness can explain the source of
the long early-run collapse.

## Rerun Result

Artifact: `bench/qk-harness-20260612/14b-rerun/`.

The targeted rerun passed under the unchanged rule:

- explicit: `22.76 tok/s`;
- generated: `39.61 tok/s`;
- gain: `74.02%`;
- A/B: `true`;
- status: `accept`;
- generated stability reasons: none.

Conclusion: the original `generated3` collapse was transient measurement noise,
not a harness-rule bug. Keep the existing full-run average stability rule.
