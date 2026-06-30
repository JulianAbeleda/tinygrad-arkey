# Prefill P1 authority baseline refresh

**Verdict:** PREFILL_P1_PASS_AUTHORITY_BASELINE_PINNED

## Whole-prefill tok/s (synced authority)
| arm | ctx512 | ctx1024 | ctx2048 | ctx4096 | route kernels |
|---|---|---|---|---|---|
| current_default | 3596.9 | 3504.8 | 3251.7 | 2818.9 | 30 |
| eightwave_off | 3489.5 | 3408.0 | 3171.7 | 2766.4 | 30 |
| pipe_tm2_tn2 | 4294.6 | 4097.6 | 3711.9 | 3139.9 | 30 |

## pipe_tm2_tn2 re-validation vs current_default
| ctx | current | pipe_tm2_tn2 | Δ% |
|---|---|---|---|
| 512 | 3596.9 | 4294.6 | 19.4 |
| 1024 | 3504.8 | 4097.6 | 16.9 |
| 2048 | 3251.7 | 3711.9 | 14.2 |
| 4096 | 2818.9 | 3139.9 | 11.4 |

sanity (current_default@512 >= 3000): True (got 3596.9); route attribution non-empty: True; 8192 unsupported (max_context=4608).