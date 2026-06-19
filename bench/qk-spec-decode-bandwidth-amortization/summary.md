# Spec decode bandwidth amortization SDB-1/SDB-2 - 2026-06-19

Read-only analysis. No hardware execution or SPEC_DECODE route.

## SDB-1

| draft | K | accepted/pass | draft cost | current verify | current speedup R=0 | verify budget for 1.2x (R=0.2) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B-Q8_0 | 2 | 2.213 | 0.403 | 4.064 | 0.495 | 1.241 |
| Qwen3-0.6B-Q8_0 | 3 | 2.569 | 0.604 | 4.358 | 0.518 | 1.337 |
| Qwen3-0.6B-Q8_0 | 4 | 2.844 | 0.806 | 4.652 | 0.521 | 1.364 |
| Qwen3-1.7B-Q8_0 | 2 | 2.387 | 0.803 | 4.064 | 0.49 | 0.986 |
| Qwen3-1.7B-Q8_0 | 3 | 2.862 | 1.204 | 4.358 | 0.515 | 0.981 |
| Qwen3-1.7B-Q8_0 | 4 | 3.262 | 1.606 | 4.652 | 0.521 | 0.912 |
| Qwen3-1.7B-Q8_0 | 8 | 4.437 | 3.212 | 9.142 | 0.359 | 0.286 |

## SDB-2

- current T=5 verify: `58.96ms`
- one pass: `12.675ms`
- target for `<=1.5x`: `19.012ms`
- required cut: `39.948ms` (`0.678` of verify)
- classification: `project_level_batched_forward`

| component | share | candidate primitive | single sufficient? |
|---|---:|---|:--:|
| q4k_gemm | 0.316 | Q4_K batched weight-read reuse | False |
| q6k_gemm_lm_head | 0.166 | Q6_K/lm_head batched weight-read reuse | False |
| attention_reduces | 0.486 | short-block causal verify attention + reductions | False |
| elementwise_norm | 0.032 | norm/RoPE/SwiGLU/residual | False |

## Verdict

- Current spec remains non-viable because verify is too expensive.
- The PMU bandwidth framing is correct, but the missing target verify is project-level batched-forward work.
- SDB-3 should not start as a bounded kernel proof unless a credible T-cheap full-verify route is introduced.
