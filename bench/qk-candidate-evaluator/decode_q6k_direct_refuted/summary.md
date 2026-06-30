# PMS-R2 Candidate Evaluator -- decode_q6k_direct_refuted

Verdict: **PMS_R2_PASS_EVALUATOR_REPLAYS_KNOWN_DECISIONS**

Tier: **REFUTED_REGRESSION** (disposition `refute`); reproduces artifact `AMD_ISA_Q6K_DIRECT_SPEED_REGRESSION` -> decision_reproduced=**True**

Baseline: `decode_q6k_coop_shipped` | authority: `decode_wd` (bench/amd-isa-backend-q6k-direct-speed/latest.json)

Default contract: opt_in_or_forced | rollback: {'Q6K_DIRECT_ROUTE': '0'}

| ctx | baseline tok/s | candidate tok/s | delta % | token_match | route_bound | cand spread % |
|---:|---:|---:|---:|:--:|:--:|---:|
| 512 | 103.63 | 97.35 | -6.06 | True | True | 50.15 |
| 1024 | 101.68 | 95.76 | -5.82 | True | True | 47.04 |
| 2048 | 99.21 | 94.19 | -5.06 | True | True | 44.13 |
| 4096 | 94.5 | 89.99 | -4.77 | True | True | 42.65 |

Speed: median -5.44% / worst -6.06% / best -4.77%.
Correctness: {'correct': True, 'gate': 'token_match_all_ctx'}.
Route-bound all ctx: True; no hidden fallback: True.
