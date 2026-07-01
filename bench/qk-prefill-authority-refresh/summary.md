# Prefill P1 authority baseline refresh

**Verdict:** PREFILL_P1_PASS_AUTHORITY_BASELINE_PINNED

## Whole-prefill tok/s (synced authority)
| arm | ctx512 | ctx1024 | ctx2048 | ctx4096 | route kernels |
|---|---|---|---|---|---|
| current_default | 4441.5 | 4243.8 | 3850.2 | 3242.7 | 30 |
| eightwave_off | 5615.9 | 5371.0 | 4772.1 | 3873.2 | 30 |
| pipe_tm2_tn2 | 4432.8 | 4248.8 | 3846.1 | 3241.7 | 30 |

## pipe_tm2_tn2 re-validation vs current_default
| ctx | current | pipe_tm2_tn2 | Δ% |
|---|---|---|---|
| 512 | 4441.5 | 4432.8 | -0.2 |
| 1024 | 4243.8 | 4248.8 | 0.1 |
| 2048 | 3850.2 | 3846.1 | -0.1 |
| 4096 | 3242.7 | 3241.7 | -0.0 |

sanity (current_default@512 >= 3000): True (got 4441.5); route attribution non-empty: True; 8192 unsupported (max_context=4608).