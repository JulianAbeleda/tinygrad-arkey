# PMS-R2 Candidate Evaluator -- decode_q4k_g3_generated

Verdict: **PMS_R2_PASS_EVALUATOR_REPLAYS_KNOWN_DECISIONS**

Tier: **SPEED_EQUIVALENT_PASS** (disposition `promote`); reproduces artifact `AMD_ISA_G3_PROMOTION_PASS_SPEED_EQUIVALENT` -> decision_reproduced=**True**

Baseline: `decode_q4k_owned_warp` | authority: `decode_wd` (bench/amd-isa-backend-g3-weight-promotion/latest.json)

Default contract: default_on_no_flag | rollback: {'BUBBLEBEAM_FUTURESIGHT': '0'}

| ctx | baseline tok/s | candidate tok/s | delta % | token_match | route_bound | cand spread % |
|---:|---:|---:|---:|:--:|:--:|---:|
| 512 | 103.79 | 103.93 | 0.135 | True | True | 54.16 |
| 1024 | 101.98 | 102.04 | 0.059 | True | True | 50.34 |
| 2048 | 99.56 | 99.74 | 0.181 | True | True | 48.95 |
| 4096 | 94.83 | 94.44 | -0.411 | True | True | 46.06 |

Speed: median 0.097% / worst -0.411% / best 0.181%.
Correctness: {'correct': True, 'gate': 'token_match vs owned/default at every ctx'}.
Route-bound all ctx: True; no hidden fallback: True.
