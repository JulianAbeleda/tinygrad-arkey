# Prefill Graph GEMM Coverage Definition - 2026-06-20

Verdict: `SCOPE_PREFILL_GRAPH_GEMM_COVERAGE_DEFINED`

Enough coverage for `PREFILL_GRAPH_GEMM=1` should be split by promotion level.

## Experimental Opt-In

The route is acceptable as an explicit fast path when all of these hold:

| gate | threshold | current status |
|---|---:|---|
| full in-model speedup | material vs `PREFILL_V2` | pass |
| one-role numeric correctness | rel RMSE <= `1e-2` | pass |
| sampled NLL smoke | max abs dNLL <= `0.01` | pass |
| corpus NLL degradation | max positive dNLL <= `0.01` | pass |
| greedy generation coverage | exact token match | pass |

This level allows benign numeric drift when it does not degrade true-token probability beyond the bound and does
not change greedy continuations.

Decision: `PASS_PREFILL_GRAPH_GEMM_EXPERIMENTAL_OPT_IN`. The existing `PREFILL_GRAPH_GEMM=1` flag is the approved
experimental route; no default behavior changes.

## Default-On

Default-on needs stricter evidence:

| gate | threshold |
|---|---:|
| repeated same-session perf | stable speedup, no sync artifact |
| corpus NLL degradation | max positive dNLL <= `0.01` |
| greedy generation | exact match across a larger prompt set |
| OOM behavior | no new load/run OOM beyond `PREFILL_V2` |
| fallback audit | unsupported shapes fall back cleanly |
| parity report | max abs dNLL reported even if not gating |

Strict absolute dNLL parity is useful as an implementation-drift report, but it should not be the only quality
gate if the question is user-visible quality. The quality gate should be degradation plus generation stability.

## Next Tool

`extra/qk_prefill_graph_gemm_generation_coverage.py` compares baseline `PREFILL_V2` and graph route continuations
in separate subprocesses. It uses 512-token prompts so the concrete prefill path and graph route are exercised,
then checks greedy generated token IDs for exact equality.

Result: `PASS_PREFILL_GRAPH_GEMM_GENERATION_COVERAGE` across 4 prompts x 8 generated tokens, with zero token
mismatches.
