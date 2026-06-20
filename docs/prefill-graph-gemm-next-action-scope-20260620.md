# Prefill Graph GEMM Next Action Scope - 2026-06-20

Verdict: `PASS_PREFILL_GRAPH_GEMM_NEXT_ACTION_SCOPED_AND_EXECUTED`

The next action after the first one-window sampled quality pass was to expand the gate before any promotion
decision. The right near-term risk was not matmul correctness or performance; those were already banked. The
risk was that the single sampled position was too narrow to call the full in-model route quality-clean.

## Action

Run the VRAM-safe sampled NLL gate across two 512-token windows:

- baseline child: `PREFILL_V2=1`, graph route disabled,
- graph child: `PREFILL_V2=1 PREFILL_GRAPH_GEMM=1`,
- both score `model.logits(tokens, 0)[:, -2, :]` so only one vocab vector is realized per window,
- both keep `T=512` so the graph GEMM route remains exercised.

## Result

`PASS_PREFILL_GRAPH_GEMM_SAMPLED_QUALITY`

| window | baseline NLL | graph NLL | dNLL | argmax match |
|---:|---:|---:|---:|---:|
| `0` | `1.407725` | `1.407725` | `0.000000` | yes |
| `1` | `1.729449` | `1.729449` | `0.000000` | yes |

## Next Boundary

The graph route is no longer blocked on sampled quality. It remains default-off because broader corpus/perplexity
coverage is still useful hardening before a default-on policy change.
