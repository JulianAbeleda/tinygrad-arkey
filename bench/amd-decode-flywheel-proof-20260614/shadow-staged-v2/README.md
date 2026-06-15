# AMD Decode Flywheel Phase 4.2 Cost-Aware Staged Shadow (Minimal-Gate Ablation)

Wasted-GPU reduction in shadow. A pre-microbench gate is valid only if it keeps
100% of live candidates; its value is how many microbench experiments it can
safely skip (skip everything scored below the lowest-scored live candidate).
Predictions were frozen before any microbench. The ladder finds the SIMPLEST
deterministic gate that captures the signal; the model ships only if it strictly
beats the role x mechanism lookup.

- conclusion: `cost_model_strictly_beats_role_mechanism_prior_ship_the_model`
- fresh candidates: `40` | live: `7` | live patterns: `3`
- label distribution: `{'construction_blocked': 20, 'raw_accept_unconfirmed': 7, 'reject': 1, 'tie': 12}`
- simplest sufficient gate: `xgboost` | model beats role x mechanism prior: `True`

## Experiments saved at 100% live-recall (gate ladder)

| gate | experiments run | saved vs run-all | live recall |
|---|---:|---:|---:|
| `run_all` | 40 | 0 | 1.00 |
| `mechanism_prior` | 40 | 0 | 1.00 |
| `role_mechanism_prior` | 40 | 0 | 1.00 |
| `simple_family_heuristic` | 40 | 0 | 1.00 |
| `xgboost` | 17 | 23 | 1.00 |

## Per (role x mechanism) cell (n / live / dead-skipped by model | role_mech | mech)

| cell | n | live | model | role_mech | mech |
|---|---:|---:|---:|---:|---:|
| `attn_q x direct_output` | 7 | 3 | 0 | 0 | 0 |
| `attn_q x reduce_unroll` | 7 | 0 | 7 | 0 | 0 |
| `attn_q x row_upcast` | 7 | 3 | 0 | 0 | 0 |
| `attn_q x two_dim_local` | 7 | 0 | 7 | 0 | 0 |
| `ffn_gate x direct_output` | 3 | 0 | 3 | 0 | 0 |
| `ffn_gate x reduce_unroll` | 3 | 0 | 3 | 0 | 0 |
| `ffn_gate x row_upcast` | 3 | 1 | 0 | 0 | 0 |
| `ffn_gate x two_dim_local` | 3 | 0 | 3 | 0 | 0 |

## Interpretation

If the role x mechanism lookup matches the model, ship the lookup and keep the
model documentation-only -- a cheap deterministic gate reducing wasted GPU is a
flywheel win. The model is only worth shipping if it strictly beats the lookup,
i.e. it captures signal beyond the (role x mechanism) cell.
