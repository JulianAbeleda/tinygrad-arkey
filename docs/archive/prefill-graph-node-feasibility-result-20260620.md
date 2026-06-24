# Prefill Graph Node Feasibility Result - 2026-06-20

Verdict: `PASS_PREFILL_GRAPH_NODE_FEASIBILITY`

Run:

```bash
DEV=AMD PROFILE=1 GRAPH_ONE_KERNEL=1 PYTHONPATH=. python3 extra/qk_prefill_graph_node_feasibility_probe.py
```

Result:

| item | value |
|---|---:|
| profile graph events | `1` |
| matching graph node | `prefill_graph_route_gemm` |
| graph node duration | `710us` |
| replay wall median | `0.956ms` |
| output sample | finite, nonzero |

This proves the dependency-free GEMM can be represented as an HCQGraph-captured node at the real
`M=512,N=12288,K=4096` PREFILL_V2 gate/up shape.
