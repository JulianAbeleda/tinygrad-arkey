# Prefill Graph GEMM Corpus Quality Result - 2026-06-20

Verdict: `BLOCKED_PREFILL_GRAPH_GEMM_CORPUS_QUALITY`

Run:

```bash
DEV=AMD PREFILL_V2=1 PYTHONPATH=. python3 extra/qk_prefill_graph_gemm_corpus_quality.py /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --windows 8
```

The tool explicitly forces the logits call through prefill-v2 state, installs the warmstart table, disables decode
Q4_K linears, and runs baseline and graph route in separate subprocesses:

| child | env |
|---|---|
| baseline | `DEV=AMD PREFILL_V2=1` |
| graph | `DEV=AMD PREFILL_V2=1 PREFILL_GRAPH_GEMM=1` |

## Result

| item | value |
|---|---:|
| windows | `8` |
| score offsets | `128,256,384,510` |
| scored positions | `32` |
| baseline mean NLL | `2.538924` |
| graph mean NLL | `2.538140` |
| mean dNLL | `-0.000784` |
| max positive dNLL | `0.009443` |
| max abs dNLL | `0.017593` |
| argmax mismatches | `0` |
| retry failures | `0` |

## Gates

| gate | result |
|---|---:|
| baseline finite | pass |
| graph finite | pass |
| scored positions expected | pass |
| mean dNLL <= `0.002` | pass |
| max positive dNLL <= `0.01` | pass |
| max abs dNLL <= `0.01` | fail |
| argmax mismatches <= `0` | pass |
| child retry failures == `0` | pass |

## Largest Rows

| window | offset | baseline NLL | graph NLL | dNLL | baseline argmax | graph argmax | target |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `6` | `384` | `2.547465` | `2.529872` | `-0.017593` | `1431` | `1431` | `705` |
| `3` | `256` | `8.973960` | `8.959895` | `-0.014065` | `11050` | `11050` | `3717` |
| `3` | `510` | `5.275367` | `5.262686` | `-0.012681` | `374` | `374` | `4157` |
| `2` | `256` | `9.223421` | `9.232864` | `0.009443` | `7192` | `7192` | `66967` |

## Interpretation

This is not an OOM/tooling blocker and not an argmax-quality blocker. The route is finite, stable, and greedy
identical on all 32 sampled positions. It is blocked only by the stricter absolute-parity gate: some positions
move by more than `0.01` NLL in either direction.

The promotion decision now needs one policy call before default behavior changes:

- keep `max_abs_dNLL <= 0.01` if the goal is near-bitwise logit parity, in which case graph GEMM stays opt-in,
- or switch the promotion criterion to degradation-only (`max_positive_dNLL <= 0.01`) if lower NLL is allowed as
  benign numeric drift, in which case this run would pass the quality gate.

Until that policy is decided, `PREFILL_GRAPH_GEMM=1` should remain default-off.
