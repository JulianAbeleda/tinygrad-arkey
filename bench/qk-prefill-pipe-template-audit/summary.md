# PMS-R6 Prefill Pipe Role Policy Template Audit

Verdict: **PMS_R6_PASS_PREFILL_TEMPLATE_PROVEN**

Rule (fact-driven, not a shape if): `pipe_enabled = lds2_blas_ratio < 1.0`.

Policy reproduced from facts == code decision: **True** (derived excluded ['ffn_gate_up'] == code excluded ['ffn_gate_up']).

| role | M | N | K | tm | tn | lds2/BLAS | latency-bound | pipe (derived) | pipe (code) |
|---|---:|---:|---:|---:|---:|---:|:--:|:--:|:--:|
| attn_qo | 512 | 4096 | 4096 | 2 | 2 | None | True | True | True |
| attn_kv | 512 | 1024 | 4096 | 2 | 2 | None | True | True | True |
| ffn_down | 512 | 4096 | 12288 | 2 | 2 | None | True | True | True |
| ffn_gate_up | 512 | 12288 | 4096 | 2 | 2 | 1.07 | False | False | False |

Role-selective replay: raw `ROLE_SELECTIVE_PASS_BEATS_GLOBAL`, evaluator `PROMOTE_TIER_A` (reproduced=True). Rollback: PREFILL_PIPE_ROLE_SELECTIVE=0 -> prefill_pipe_global_rollback.
