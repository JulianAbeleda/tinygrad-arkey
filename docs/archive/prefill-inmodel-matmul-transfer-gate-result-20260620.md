# Prefill In-Model Matmul Transfer Gate Result - 2026-06-20

Verdict: `BLOCKED_PREFILL_MATMUL_TRANSFER_NEEDS_GRAPH_ROUTE`

Run:

```bash
PYTHONPATH=. python3 extra/qk_prefill_inmodel_matmul_transfer_gate.py
```

## Result

| item | value |
|---|---:|
| matmul share of PREFILL_V2 graph span | `71.15%` |
| required matmul speedup for `1.15x` full prefill | `1.2245x` |
| theoretical full speedup if isolated `78.6 TFLOPS` transfers | `1.4371x` |

The isolated GEMM win is large enough to matter if it transfers. The blocker is that the dependency-free
`78.6 TFLOPS` GEMM is not currently a graph-captured PREFILL_V2 node.

Decision: do not start another isolated GEMM microkernel. The only material prefill matmul move is a graph-route
transfer gate.
