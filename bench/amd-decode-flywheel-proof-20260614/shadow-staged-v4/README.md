# AMD Decode Flywheel Phase 4.2 Cost-Aware Staged Shadow (Minimal-Gate Ablation)

Wasted-GPU reduction in shadow. A pre-microbench gate is valid only if it keeps
100% of live candidates; its value is how many microbench experiments it can
safely skip (skip everything scored below the lowest-scored live candidate).
Predictions were frozen before any microbench. The ladder finds the SIMPLEST
deterministic gate that captures the signal; the model ships only if it strictly
beats the role x mechanism lookup.

- conclusion: `inconclusive_insufficient_live_candidates_or_patterns`
- fresh candidates: `32` | live: `4` | live patterns: `3`
- label distribution: `{'construction_blocked': 16, 'raw_accept_unconfirmed': 4, 'reject': 5, 'tie': 7}`
- simplest sufficient gate: `xgboost` | model beats role x mechanism prior: `True`

## Experiments saved at 100% live-recall (gate ladder)

| gate | experiments run | saved vs run-all | live recall |
|---|---:|---:|---:|
| `run_all` | 32 | 0 | 1.00 |
| `mechanism_prior` | 32 | 0 | 1.00 |
| `role_mechanism_prior` | 32 | 0 | 1.00 |
| `simple_family_heuristic` | 32 | 0 | 1.00 |
| `xgboost` | 16 | 16 | 1.00 |

## Per (role x mechanism) cell (n / live / dead-skipped by model | role_mech | mech)

| cell | n | live | model | role_mech | mech |
|---|---:|---:|---:|---:|---:|
| `attn_q x direct_output` | 3 | 1 | 0 | 0 | 0 |
| `attn_q x reduce_unroll` | 3 | 0 | 3 | 0 | 0 |
| `attn_q x row_upcast` | 3 | 0 | 0 | 0 | 0 |
| `attn_q x two_dim_local` | 3 | 0 | 3 | 0 | 0 |
| `ffn_gate x direct_output` | 5 | 1 | 0 | 0 | 0 |
| `ffn_gate x reduce_unroll` | 5 | 0 | 5 | 0 | 0 |
| `ffn_gate x row_upcast` | 5 | 2 | 0 | 0 | 0 |
| `ffn_gate x two_dim_local` | 5 | 0 | 5 | 0 | 0 |

## Interpretation

If the role x mechanism lookup matches the model, ship the lookup and keep the
model documentation-only -- a cheap deterministic gate reducing wasted GPU is a
flywheel win. The model is only worth shipping if it strictly beats the lookup,
i.e. it captures signal beyond the (role x mechanism) cell.
