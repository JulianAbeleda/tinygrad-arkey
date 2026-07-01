# TG-P9.0 Primitive Backlog

Verdict: **TG_P9_0_PASS_PRIMITIVE_BACKLOG_PINNED**

The two generated-codegen primitives 8B attention needs to match owned (from TG-P8):

| primitive | status | phase | fixes |
|---|---|---|---|
| live_tc_split_geometry | expected EMITTER_BLOCKED | TG-P9.1/9.2 | ctx512 over-launch (fixed S, per=ceildiv(Tc,S)) |
| split_preserving_attention_lse_combine | expected EMITTER_BLOCKED/PRIMITIVE_MISSING | TG-P9.3/9.4 | ctx4096 combine lifecycle (556us, binding) |
| owned_external_attention_route | FORBIDDEN_FINAL_DEFAULT | - | rollback/oracle only |

Reproduced TG-P8: ctx512 87.7% owned, ctx4096 95.9% owned (binding).

Promotion requires generated attention >=98% of owned at BOTH ctx512 and ctx4096.
