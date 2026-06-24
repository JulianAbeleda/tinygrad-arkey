# Prefill Graph Route One-Role Timing Result - 2026-06-20

Verdict: `PASS_PREFILL_GRAPH_ROUTE_ONE_ROLE_TIMING_MATERIAL`

Run:

```bash
DEV=AMD PROFILE=1 GRAPH_ONE_KERNEL=1 PYTHONPATH=. python3 extra/qk_prefill_graph_route_one_role_timing.py
```

Result:

| row | median |
|---|---:|
| graph GEMM route | `1155.3us` |
| current tinygrad graph matmul | `2129.64us` |
| component speedup | `1.8434x` |
| projected full PREFILL_V2 speedup | `1.4826x` |

The graph route clears the materiality gate. Next gate is full-model route measurement.
