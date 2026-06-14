# AMD Decode Flywheel Phase 4 Shadow v0

Blind static-stage shadow: the cost model predicted/ranked a fresh batch of
candidates on untouched dominant Q4_K tensors BEFORE any GPU run (frozen in
`predictions.jsonl` / `freeze.json`), then was scored against the same
baselines after the deterministic generators produced outcomes.

- conclusion: `shadow_inconclusive_no_live_candidate_in_fresh_batch_model_underperforms_prior`
- shadow gate met: `False`
- fresh rows: `6`
- fresh label distribution: `{'construction_blocked': 1, 'diagnostic_only': 1, 'reject': 1, 'tie': 3}`

## Metrics (fresh batch)

| model | accuracy | macro-F1 | false accept | p@1 | p@3 | ndcg | first-live |
|---|---:|---:|---:|---:|---:|---:|---:|
| `xgboost` | 0.167 | 0.071 | 0.000 | 0.000 | 0.000 | 0.000 | n/a |
| `mechanism_prior` | 0.833 | 0.667 | 0.000 | 0.000 | 0.000 | 0.000 | n/a |
| `simple_family_heuristic` | 0.833 | 0.667 | 0.000 | 0.000 | 0.000 | 0.000 | n/a |
| `reject_all` | 0.167 | 0.071 | 0.000 | 0.000 | 0.000 | 0.000 | n/a |

## Gate

- met: `False`
- blocker: macro_f1 not above mechanism_prior
- blocker: ranking not above mechanism_prior
- blocker: no live candidate in fresh batch (ranking undefined)

## Interpretation

v0 is blind static-stage, instance-level generalization (new tensors, same
mechanism families). A failed gate is a real result under the Phase 4 stop
rule: the model stays documentation-only and does not steer execution order.
