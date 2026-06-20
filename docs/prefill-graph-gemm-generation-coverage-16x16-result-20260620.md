# Prefill Graph GEMM Generation Coverage Result (Gate 2, 16×16) - 2026-06-20

Verdict: `PASS_PREFILL_GRAPH_GEMM_GENERATION_COVERAGE`

Run:

```bash
DEV=AMD PREFILL_V2=1 PYTHONPATH=. python3 extra/qk_prefill_graph_gemm_generation_coverage.py \
  /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --prompts 16 --max-new-tokens 16
```

Baseline `PREFILL_V2` and graph `PREFILL_V2 + PREFILL_GRAPH_GEMM` continuations generated in separate
subprocesses (each prompt is a 512-token window, exercising the concrete prefill path and graph route), then
greedy generated token IDs compared for exact equality.

| metric | value | threshold | pass |
|---|---:|---:|---|
| prompts | 16 | 16 | ✓ |
| generated tokens compared | 256 | 256 | ✓ |
| token mismatches | 0 | 0 | ✓ |
| prompt mismatches | 0 | 0 | ✓ |
| child retry failures | 0 | 0 | ✓ |

The graph route's distributed fp16 numeric drift (`max_abs_dNLL = 0.017593`, report-only) does **not** change
greedy continuations across the broader 16×16 sample — every generated token matches the baseline exactly. This
expands the prior 4×8 / 32-token coverage to 16×16 / 256 tokens with zero mismatches.

Gate 2 is satisfied for default-on readiness.
