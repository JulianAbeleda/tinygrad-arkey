# Prefill Graph GEMM Drift Attribution Result - 2026-06-20

Verdict: `PASS_PREFILL_GRAPH_GEMM_DRIFT_ATTRIBUTION`

Run:

```bash
DEV=AMD PREFILL_V2=1 PYTHONPATH=. python3 extra/qk_prefill_graph_gemm_drift_attribution.py /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf
```

The probe scores the high-drift corpus rows with role-filtered graph routing:

`6:384,3:256,3:510,2:256,5:128,0:510`

## Summary

| variant | roles | mean dNLL | max abs dNLL | max positive dNLL | argmax mismatches |
|---|---|---:|---:|---:|---:|
| `attn_qkv` | q/k/v projections | `-0.002278` | `0.012493` | `0.008491` | `0` |
| `attn_output` | attention output | `-0.006408` | `0.023793` | `0.010317` | `0` |
| `attention_all` | all attention matmuls | `-0.005530` | `0.014244` | `0.003530` | `0` |
| `ffn_gateup` | gate/up | `-0.003954` | `0.020339` | `0.008295` | `0` |
| `ffn_down` | down | `-0.003810` | `0.020382` | `0.003533` | `0` |
| `ffn_all` | all FFN matmuls | `-0.002614` | `0.009887` | `0.007561` | `0` |
| `all` | attention + FFN | `-0.004607` | `0.017593` | `0.009443` | `0` |

## What This Means

The drift is distributed numeric drift from alternate fp16 GEMM paths, not one broken role:

- all variants keep greedy argmax stable on the focused rows,
- full graph routing reproduces the corpus maximum: `max_abs_dNLL = 0.017593`,
- `ffn_all` alone stays just inside the absolute threshold (`0.009887`),
- `attn_output`, `ffn_gateup`, and `ffn_down` individually can exceed the absolute threshold,
- combining roles is not additive; some drift cancels because later layers/norms/softmax reshape the perturbation.

The focused rows are mostly non-hit positions with high NLL. The one low-margin hit row (`0:510`, argmax margin
`0.052708`) still kept the same argmax under every variant.

## Boundary

This closes role/confidence attribution. It does not yet provide layer-by-layer hidden-state deltas or a
real-activation per-matmul tensor-diff ledger. Those are only needed if we want to reduce absolute drift rather
than accept degradation/generation gates for the experimental fast path.
