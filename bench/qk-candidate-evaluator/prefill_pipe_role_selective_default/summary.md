# PMS-R2 Candidate Evaluator -- prefill_pipe_role_selective_default

Verdict: **PMS_R2_PASS_EVALUATOR_REPLAYS_KNOWN_DECISIONS**

Tier: **PROMOTE_TIER_B** (disposition `promote`); reproduces artifact `ROLE_SELECTIVE_PASS_BEATS_GLOBAL` -> decision_reproduced=**True**

Baseline: `prefill_pipe_global_rollback` | authority: `prefill_whole` (bench/qk-prefill-pipe-role-selective/latest.json)

Default contract: default_on_no_flag | rollback: {'PREFILL_PIPE_ROLE_SELECTIVE': '0'}

| ctx | baseline tok/s | candidate tok/s | delta % | token_match | route_bound | cand spread % |
|---:|---:|---:|---:|:--:|:--:|---:|
| 512 | 4292 | 4434 | 3.3 | True | True | 0.0 |
| 1024 | 4092 | 4236 | 3.5 | True | True | 0.1 |
| 2048 | 3708 | 3846 | 3.7 | True | True | 0.1 |
| 4096 | 3083 | 3192 | 3.5 | True | True | 0.1 |
| 8192 | 2461 | 2532 | 2.9 | True | True | 0.1 |

Speed: median 3.5% / worst 2.9% / best 3.7%.
Correctness: {'correct': True, 'gate': 'correct_equivalent (logit fingerprint match: argmax+sum)'}.
Route-bound all ctx: True; no hidden fallback: True.
