# LUNA-052: Context-128 Candidate Review Gate

Verdict: `TOOL_FAILURE` (not `PASS`).

Dependency review: LUNA-051 cannot select a repair because its required
semantic and route captures are missing. LUNA-052 therefore rejects
implementation and keeps LUNA-053 through LUNA-055 `NOT_RUN`.

## Review result

| Gate | Result | Basis |
|---|---|---|
| UOp destination semantics | `TOOL_FAILURE` | No retained STORE ancestry or source indices. |
| Vector lane ownership | `TOOL_FAILURE` | No failing program to compare against 8B ctx128 and 14B ctx512. |
| KV/cache bounds | `INCONCLUSIVE` | The failing prefill linear and state-update owner are unobserved. |
| Genericity | `INCONCLUSIVE` | Could be packed-WMMA output fusion, direct-packed, or generic `Tensor.linear` lowering. |
| 8B impact | `TOOL_FAILURE` | Historical pass is not a same-capture positive control. |
| Fallback policy | `INCONCLUSIVE` | Source proves a packed decline can fall through, but no ctx128 selection log exists. |

## Candidate review decision

Do not modify the renderer, add `EXPAND_SSA`, materialize a vector temporary,
or alter route coverage. Each could make HIP compile while changing the
semantic state update. Do not add a 14B ctx128 fail-loud guard yet: it would be
an unsupported diagnosis unless LUNA-034 proves the direct-packed fallback is
selected. Existing explicit `TINYGRAD_PREFILL_PACKED_WMMA=0` rejection remains
the only source-proven unsafe-route guard.

## Exact future compile and numerical gates

Once the missing capture exists, a single candidate may proceed only if all
items below are recorded in its candidate artifact directory.

1. Compile-only: failing 14B ctx128 shape succeeds; its final STORE writes an
   addressable global/local destination with the expected lane cardinality.
2. Compile-only controls: 8B ctx128 and 14B ctx512 compile with identical
   route/code-object identity to their accepted baselines, or explicitly
   retained resource/code-object deltas.
3. Route control: a 14B ctx512 packed-WMMA selection and a permitted non-14B
   direct-packed selection are both positively logged; ctx128's actual branch
   is logged with every decline reason.
4. Numerical: deterministic temperature-zero token parity, finite outputs,
   and cache/KV bounds for ctx128. A compiling program alone is rejection.
5. Runtime acceptance: only after the preceding checks, run eight decode
   tokens for `"hi"`, one short sentence, and the required length matrix in
   separate locked processes.

## What must be rerun after ROCm tracing is restored

Run LUNA-030/031/034 first, with a known tinygrad kernel dispatch as collector
positive control. Retain the semantic UOp/HIP slice, route decision log, and
the two route controls above. Reopen LUNA-051 only after those artifacts have
matching branch/commit/model identities; then select one bounded repair layer.
