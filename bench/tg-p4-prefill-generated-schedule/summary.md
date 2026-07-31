# TG-P4 Prefill Generated Schedule

Verdict: **TG_P4_PASS_PREFILL_GENERATED_SCHEDULE**

All instructions identical: True; role policy preserved: True

| role | out_f | in_f | family | expected | family_ok | insts_identical | n_insts |
|---|---|---|---|---|---|---|---|
| attn_qo | 4096 | 4096 | pipe | pipe | True | True | 246 |
| attn_kv | 1024 | 4096 | pipe | pipe | True | True | 246 |
| ffn_down | 4096 | 12288 | pipe | pipe | True | True | 246 |
| ffn_gate_up | 12288 | 4096 | lds | lds | True | True | 603 |
