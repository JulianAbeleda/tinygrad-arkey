# Prefill Graph Route One-Role Correctness Result - 2026-06-20

Verdict: `PASS_PREFILL_GRAPH_ROUTE_ONE_ROLE_CORRECTNESS`

Run:

```bash
DEV=AMD PROFILE=1 GRAPH_ONE_KERNEL=1 PYTHONPATH=. python3 extra/qk_prefill_graph_route_one_role_correctness.py
```

Result:

| item | value |
|---|---:|
| rel RMSE vs tinygrad matmul | `0.0002077` |
| max abs | `0.0002508` |
| graph node | `prefill_graph_route_gemm_correctness` |
| graph node duration | `1319.4us` |

This closes one-role correctness for the real `512x12288x4096` gate/up-shape matmul.
