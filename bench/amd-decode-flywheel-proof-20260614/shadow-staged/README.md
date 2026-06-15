# AMD Decode Flywheel Phase 4.1 Cost-Aware Staged Shadow

Reframes shadow mode to the actual flywheel value: wasted GPU reduction. A
pre-microbench gate is valid only if it keeps 100% of live candidates; its
value is how many expensive microbench experiments it can safely skip (skip
everything scored below the lowest-scored live candidate). Predictions were
frozen before any microbench.

- conclusion: `cost_model_gate_beats_prior_reduces_wasted_experiments`
- fresh candidates: `16` | live: `2`
- label distribution: `{'construction_blocked': 8, 'raw_accept_unconfirmed': 2, 'reject': 1, 'tie': 5}`
- model beats prior: `True`

## Experiments saved at 100% live-recall

| gate | experiments run | saved vs run-all | live recall |
|---|---:|---:|---:|
| `run_all` | 16 | 0 | 1.00 |
| `mechanism_prior` | 16 | 0 | 1.00 |
| `simple_family_heuristic` | 16 | 0 | 1.00 |
| `xgboost` | 2 | 14 | 1.00 |

## Interpretation

The flywheel-relevant question is whether any cheap pre-result gate avoids
wasted microbench runs without dropping a real winner, and whether the learned
model does this better than the deterministic mechanism prior. The prior
winning is still a decisive result: the deterministic tool carries the flywheel
and the learned model adds no value at the current feature set.
