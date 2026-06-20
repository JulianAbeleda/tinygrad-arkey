# Prefill Graph Route Transfer Scope - 2026-06-20

Verdict: `PASS_PREFILL_GRAPH_ROUTE_TRANSFER_SCOPE_READY`

The only material prefill path left is graph transfer:

```text
dependency-free 78.6 TFLOPS GEMM -> captured PREFILL_V2 graph node
```

Target shape and layout:

| item | value |
|---|---|
| role | one real PREFILL_V2 gate/up-style matmul |
| shape | `M=512, N=12288, K=4096` |
| input layout | `A[T,K]` |
| weight layout | `W[out,K]`, already natural PREFILL_V2 fp16 |
| output layout | `C[T,out]` |
| kernel | `build_gemm_lds2(BK=32, PAD=16, PLRA=1)` |

Phases:

1. graph-node feasibility at the real shape;
2. one-role correctness against the current PREFILL_V2 linear;
3. one-role graph timing;
4. full-bucket projection;
5. full PREFILL_V2 measurement.

Kill conditions:

- if the node is not captured in HCQGraph, block on runtime integration;
- if the node needs a transpose/copy to match PREFILL_V2 layout, block on layout route;
- if one-role timing does not project `>=1.15x` full prefill, stop prefill GEMM transfer.
