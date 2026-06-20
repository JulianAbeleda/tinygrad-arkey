# Prefill Graph GEMM Sampled Quality Result - 2026-06-20

Verdict: `PASS_PREFILL_GRAPH_GEMM_SAMPLED_QUALITY`

Run:

```bash
DEV=AMD PREFILL_V2=1 PYTHONPATH=. python3 extra/qk_prefill_graph_gemm_quality_sampled.py /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --windows 1
```

Result:

| item | value |
|---|---:|
| baseline sampled NLL | `1.407725` |
| graph GEMM sampled NLL | `1.407725` |
| mean dNLL | `0.000000` |
| max abs dNLL | `0.000000` |
| baseline argmax | `429` |
| graph argmax | `429` |
| target | `429` |

Gates:

| gate | result |
|---|---:|
| baseline finite | pass |
| graph finite | pass |
| `abs(dNLL) <= 0.01` | pass |

This closes the immediate VRAM-safe quality smoke for the full in-model graph GEMM route. It does not replace a
future full-window or corpus perplexity pass; it proves that a real `T=512` graph-routed prefill can be scored
without materializing `(512, vocab)` logits and does not perturb the sampled teacher-forced prediction.
