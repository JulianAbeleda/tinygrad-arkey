# Prefill Graph GEMM Generation Coverage Result - 2026-06-20

Verdict: `PASS_PREFILL_GRAPH_GEMM_GENERATION_COVERAGE`

Run:

```bash
DEV=AMD PREFILL_V2=1 PYTHONPATH=. python3 extra/qk_prefill_graph_gemm_generation_coverage.py /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf
```

The gate compares baseline `PREFILL_V2` and `PREFILL_GRAPH_GEMM=1` in separate subprocesses. Each prompt is a
real 512-token prefill, then greedy generation is compared by token ID.

## Result

| item | value |
|---|---:|
| prompts | `4` |
| prompt tokens | `512` |
| generated tokens per prompt | `8` |
| compared generated tokens | `32` |
| prompt mismatches | `0` |
| token mismatches | `0` |
| retry failures | `0` |

## Continuations

| prompt | generated text |
|---:|---|
| `0` | ` belief was meant to detect. But the` |
| `1` | ` that require a specific kind of soil and` |
| `2` | ` but is actually insurance.\n\nThe problem of` |
| `3` | ` the same memory, and the cost of` |

Baseline and graph route produced byte-identical token IDs for all rows.

## Interpretation

This closes the user-visible greedy-generation smoke for experimental opt-in promotion. The route still has
reported absolute NLL drift above `0.01` on the corpus-quality probe, so default-on remains a separate policy and
coverage decision.
