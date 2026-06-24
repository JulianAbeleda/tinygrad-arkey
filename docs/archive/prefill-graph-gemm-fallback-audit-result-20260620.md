# Prefill Graph GEMM Fallback Audit Result (Gate 3) - 2026-06-20

Verdict: `PASS_PREFILL_GRAPH_GEMM_FALLBACK_AUDIT`

Run:

```bash
DEV=AMD PREFILL_GRAPH_GEMM=1 PYTHONPATH=. python3 extra/qk_prefill_graph_gemm_fallback_audit.py
```

Structural audit of `route_pf16_graph_gemm` (mock `lin` objects; no model load). Every supported shape routes
and every unsupported / ineligible case returns `None` without misrouting, exception, or output allocation.

| case | expect | got | pass |
|---|---|---|---|
| valid gate/up shape, T=512 | tensor | tensor | ✓ |
| valid down shape, T=512 | tensor | tensor | ✓ |
| unsupported T=256 | None | None | ✓ |
| unsupported out_f not mult of 128 | None | None | ✓ |
| unsupported in_f not mult of 32 | None | None | ✓ |
| missing realized `_pf16_w` | None | None | ✓ |
| bias present | None | None | ✓ |
| role filter excludes role | None | None | ✓ |
| role filter includes role | tensor | tensor | ✓ |
| role filter set, lin has no role | None | None | ✓ |
| flag-gated call site (default unchanged when flag absent) | gated | gated | ✓ |

11/11 pass. No exception on any unsupported case. Default behavior unchanged when `PREFILL_GRAPH_GEMM` is
absent: the only call site (`tinygrad/llm/model.py` `_pf16`) is gated by `if PREFILL_GRAPH_GEMM and w is not
None:`, so with the flag off the route is never consulted.

## Notes

- The route's None-returns (role filter, missing `_pf16_w`, bias, `x.shape[-2] != 512`, shape not tile-divisible)
  all occur **before** any `Tensor.empty` output allocation, so an unsupported shape costs nothing and silently
  falls through to the normal `PREFILL_V2` fp16 matmul.
- Test hygiene: tinygrad's `getenv` is `lru_cache`d, so the audit calls `getenv.cache_clear()` after mutating
  `PREFILL_GRAPH_GEMM_ROLES` (otherwise the first cached read masks later role-filter cases — a test artifact,
  not a route bug).

Gate 3 is satisfied for default-on readiness: enabling the flag does not misroute unsupported shapes.
