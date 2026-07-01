# TG-P6 Pure-Search Diagnostic Mode

Verdict: **TG_P6_PASS_PURE_SEARCH_DIAGNOSTIC_MODE**

| gate | result |
|---|---|
| fail_current | PASS (normal default is impure: attention owned HIP; violations=['decode_attention']) |
| pass_after | PASS (forcing generated attention makes all hot families pure) |
| explicit_rollback | PASS (a named rollback flag surfaces a violation naming the route + scope) |
| route_report | PASS (guard prints per-family route + provenance) |

Effective routes on a normal fast default run:

- **decode_q4k_gemv**: `decode_q4k_g3_generated` (machine_authored_generated) — pure
- **decode_q6k_gemv**: `decode_q6k_coop_generated` (machine_authored_generated) — pure
- **prefill_gemm**: `prefill_pipe_role_selective_generated` (machine_authored_generated) — pure
- **decode_attention**: `decode_attention_owned_two_kernel` (external_handwritten_kernel) — IMPURE
