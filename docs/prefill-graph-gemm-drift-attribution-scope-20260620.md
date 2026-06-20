# Prefill Graph GEMM Drift Attribution Scope - 2026-06-20

Verdict target: `PASS_PREFILL_GRAPH_GEMM_DRIFT_ATTRIBUTION`

The corpus-quality gate showed small numeric drift:

- mean dNLL: `-0.000784`,
- max positive dNLL: `0.009443`,
- max abs dNLL: `0.017593`,
- argmax mismatches: `0`.

The next question is why drift exists and which role owns it.

## Tool

`extra/qk_prefill_graph_gemm_drift_attribution.py`

It adds no default route change. The graph route now has an optional probe-only filter:

```bash
PREFILL_GRAPH_GEMM_ROLES=ffn_gate,ffn_up
```

The probe tags loaded model linears with `_prefill_graph_role`, then runs selected high-drift corpus rows through
baseline `PREFILL_V2` and role-filtered graph variants.

## Variants

| variant | roles |
|---|---|
| `attn_qkv` | `attn_q,attn_k,attn_v` |
| `attn_output` | `attn_output` |
| `attention_all` | `attn_q,attn_k,attn_v,attn_output` |
| `ffn_gateup` | `ffn_gate,ffn_up` |
| `ffn_down` | `ffn_down` |
| `ffn_all` | `ffn_gate,ffn_up,ffn_down` |
| `all` | all tagged roles |

## Scored Rows

Default pairs are the largest corpus-drift rows plus the old final-token smoke:

`6:384,3:256,3:510,2:256,5:128,0:510`

Each row records baseline NLL, graph NLL, dNLL, argmax, and argmax margin. This covers:

- role attribution,
- target probability drift,
- whether drift concentrates on low-confidence / non-hit positions.

Layer-by-layer hidden-state capture remains a later tool if role attribution is inconclusive.
