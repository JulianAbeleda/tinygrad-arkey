# Prefill Graph GEMM Full Route Result - 2026-06-20

Verdict: `PASS_PREFILL_GRAPH_GEMM_FULL_ROUTE_PERF_PASS_SAMPLED_QUALITY`

`PREFILL_GRAPH_GEMM=1` wires eligible `PREFILL_V2` fp16 matmuls through the dependency-free graph-capturable
RDNA3 GEMM. It is default-off and falls back to normal `PREFILL_V2` for unsupported shapes.

## Performance

Same-session comparison:

| route | tok/s | ms / 512 | notes |
|---|---:|---:|---|
| `PREFILL_V2=1` | `2593.2` | `197.4` | flag off, warmstart `apply=5` |
| `PREFILL_V2=1 PREFILL_GRAPH_GEMM=1` | `4895.9` | `104.6` | graph GEMM route, warmstart bypassed |

Against the banked production row:

| baseline | tok/s | graph route speedup |
|---|---:|---:|
| banked PREFILL_V2 | `2797` | `1.75x` |
| llama pp512 reference | `3020` | graph route is `1.62x` llama |

The route transfers the isolated GEMM win into model throughput: `104.6ms / 512` is effectively back at the
isolated-kernel throughput class.

The existing `qk_prefill_v2_measure.py` reports `gate_pass=false` for the graph route only because its historical
gate requires warmstart `apply>0`; this route bypasses warmstart by design. Performance is a pass.

## Quality

One-role numeric correctness already passes:

| gate | result |
|---|---:|
| one-role rel RMSE | `0.0002077` |
| one-role max abs | `0.0002508` |

Full-route teacher-forced NLL with the original full-window harness was attempted:

```bash
DEV=AMD PREFILL_V2=1 PREFILL_GRAPH_GEMM=1 PYTHONPATH=. python3 extra/qk_prefill_v2_nll_eval.py /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --windows 1
```

It blocked on VRAM in the current harness:

```text
Allocation of 8.00 MB failed on AMD. Used: 18.95 GB
```

A VRAM-safe sampled quality gate was then added and passed:

```bash
DEV=AMD PREFILL_V2=1 PYTHONPATH=. python3 extra/qk_prefill_graph_gemm_quality_sampled.py /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --windows 2
```

| item | value |
|---|---:|
| windows | `2` |
| baseline mean sampled NLL | `1.568587` |
| graph GEMM mean sampled NLL | `1.568587` |
| mean dNLL | `0.000000` |
| max abs dNLL | `0.000000` |

Sampled teacher-forced quality is now a pass. The remaining quality work is broader hardening: full-window or
corpus perplexity without retaining `(512, vocab)` logits.

## Decision

The prefill graph-route transfer is real and material. Performance and sampled quality are both banked. The next
decision is whether to keep this as an explicit opt-in route while building broader corpus-quality tooling, or to
promote behind a stronger non-default experimental flag.
