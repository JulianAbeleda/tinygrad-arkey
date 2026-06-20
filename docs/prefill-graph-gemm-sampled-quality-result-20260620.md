# Prefill Graph GEMM Sampled Quality Result - 2026-06-20

Verdict: `PASS_PREFILL_GRAPH_GEMM_SAMPLED_QUALITY`

Run:

```bash
DEV=AMD PREFILL_V2=1 PYTHONPATH=. python3 extra/qk_prefill_graph_gemm_quality_sampled.py /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --windows 2
```

Result:

| item | value |
|---|---:|
| windows | `2` |
| baseline mean sampled NLL | `1.568587` |
| graph GEMM mean sampled NLL | `1.568587` |
| mean dNLL | `0.000000` |
| max abs dNLL | `0.000000` |

Per-window rows:

| window | start | baseline NLL | graph NLL | dNLL | baseline argmax | graph argmax | target |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `0` | `0` | `1.407725` | `1.407725` | `0.000000` | `429` | `429` | `429` |
| `1` | `256` | `1.729449` | `1.729449` | `0.000000` | `264` | `264` | `12291` |

Gates:

| gate | result |
|---|---:|
| baseline finite | pass |
| graph finite | pass |
| `abs(dNLL) <= 0.01` | pass |

This closes the immediate VRAM-safe sampled quality gate for the full in-model graph GEMM route. It does not
replace a future full-window or corpus perplexity pass; it proves that real `T=512` graph-routed prefill windows
can be scored without materializing `(512, vocab)` logits and do not perturb the sampled teacher-forced
predictions.
