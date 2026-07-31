# TG-P6 Pure-Search Diagnostic Mode

Verdict: **TG_P6_PASS_PURE_SEARCH_DIAGNOSTIC_MODE**

| gate | result |
|---|---|
| pass_current | PASS (normal generated default has no violations) |
| explicit_rollback | PASS (a named rollback flag surfaces a violation naming the route + scope) |
| route_report | PASS (guard prints per-family route + provenance) |

Effective routes on the current generated default run:

- **decode_q4k_gemv**: `decode_q4k_g3_generated` (machine_authored_generated) — pure
- **decode_q6k_gemv**: `decode_q6k_coop_generated` (machine_authored_generated) — pure
- **prefill_gemm**: `prefill_pipe_role_selective_generated` (machine_authored_generated) — pure
- **decode_attention**: `decode_flash_live_split_g4_8b_kvboth` (machine_authored_generated) — pure
