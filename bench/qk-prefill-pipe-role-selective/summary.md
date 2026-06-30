# Role-selective prefill pipe vs global

**Verdict:** ROLE_SELECTIVE_PASS_BEATS_GLOBAL

promote role-selective as the new default (beats global above noise, correct, no regression).

noise bar (max run spread) = 0.3%; correctness equivalent = True

| ctx | old_lds2 | global_pipe | role_selective | rs vs old | rs vs global |
|---|---|---|---|---|---|
| 512 | 3593 | 4292 | 4434 | +23.4% | +3.3% |
| 1024 | 3492 | 4092 | 4236 | +21.3% | +3.5% |
| 2048 | 3259 | 3708 | 3846 | +18.0% | +3.7% |
| 4096 | 2779 | 3083 | 3192 | +14.8% | +3.5% |
| 8192 | 2266 | 2461 | 2532 | +11.7% | +2.9% |