# AMD Decode Flywheel Cost Model

This Phase 3B artifact tests the learned-cost-model version of kernel triage.
It uses only pre-result candidate/context features and scores on the same
family-split holdout as the Phase 2 baselines.

- conclusion: `cost_model_shadow_candidate`
- backend request: `auto`
- xgboost available: `True`
- train rows: `98`
- holdout rows: `38`
- feature policy: `pre_result_analytical_context_v0`
- feature count: `232`

## Backends

- `xgboost`: `ok` (xgboost `3.2.0`, rank score `ranker`)

## Metrics

| model | accuracy | macro-F1 | false accept | p@3 | ndcg |
|---|---:|---:|---:|---:|---:|
| `reject_all` | 0.237 | 0.077 | 0.000 | 0.000 | 0.170 |
| `mechanism_prior` | 0.684 | 0.479 | 0.000 | 0.167 | 0.253 |
| `simple_family_heuristic` | 0.684 | 0.479 | 0.000 | 0.167 | 0.253 |
| `xgboost` | 0.868 | 0.873 | 0.000 | 0.250 | 0.500 |

## Leakage Audit

- raw ids used as categorical features: `False`
- target/result fields used: `False`
- excluded fields: `id, candidate_id, label, reason, retry, evidence, source_files, split, family_order, status, gain, gain_pct, candidate_gbs, current_gbs, decision, correctness_ok, ab_match_result`

## Interpretation

XGBoost is the right off-the-shelf backend for the larger version of this
test, but the feature extractor and holdout are the load-bearing pieces.
This artifact does not test novel mechanism proposal; it only tests whether
structured pre-result features can triage or rank candidate experiments
better than deterministic priors.
