# Spec decode bandwidth amortization SDB-1/SDB-2 result - 2026-06-19

Purpose: execute the first two phases from `spec-decode-bandwidth-amortization-scope-20260619.md`:

- SDB-1: analytic viability model over current acceptance, draft speed, target speed, verify ladder, and runtime.
- SDB-2: verify-fastpath design audit across Q4_K, Q6_K/lm_head, attention/reduces, and elementwise.

This is read-only. It does not run hardware, route `SPEC_DECODE`, or change defaults.

Artifacts:

- `extra/qk_spec_decode_bandwidth_model.py`
- `bench/qk-spec-decode-bandwidth-amortization/model.json`
- `bench/qk-spec-decode-bandwidth-amortization/summary.md`

## SDB-1 result

Current verify is too expensive for every measured draft/K setting.

| draft | K | accepted/pass | draft cost | current verify | current speedup R=0 | verify budget for 1.2x with R=0.2 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B-Q8_0 | 2 | 2.213 | 0.403 | 4.064 | 0.495 | 1.241 |
| Qwen3-0.6B-Q8_0 | 3 | 2.569 | 0.604 | 4.358 | 0.518 | 1.337 |
| Qwen3-0.6B-Q8_0 | 4 | 2.844 | 0.806 | 4.652 | 0.521 | 1.364 |
| Qwen3-1.7B-Q8_0 | 2 | 2.387 | 0.803 | 4.064 | 0.490 | 0.986 |
| Qwen3-1.7B-Q8_0 | 3 | 2.862 | 1.204 | 4.358 | 0.515 | 0.981 |
| Qwen3-1.7B-Q8_0 | 4 | 3.262 | 1.606 | 4.652 | 0.521 | 0.912 |
| Qwen3-1.7B-Q8_0 | 8 | 4.437 | 3.212 | 9.142 | 0.359 | 0.286 |

Interpretation:

- The 0.6B draft is the viable draft family; 1.7B draft cost is too high.
- For the 0.6B K=4 case, current verify is `4.652x` one target pass.
- To clear `1.2x` with a small runtime allowance (`R=0.2`), verify must be `<=1.364x`.
- To clear a strong `1.5x`, verify must be near `<=0.89x` with `R=0.2`, which is unrealistic unless runtime and
  verify are both exceptionally good.

So the practical hard gate is not merely `<=1.5x`; for a robust `>=1.2x`, the 0.6B route needs roughly
`<=1.3-1.4x` verify after runtime overhead.

## SDB-2 result

Current T=5 verify:

| item | value |
|---|---:|
| one T=1 pass | `12.675ms` |
| current T=5 verify | `58.960ms` |
| target for `<=1.5x` one pass | `19.012ms` |
| required cut | `39.948ms` |
| required cut fraction | `67.8%` |

Directional component shares at T=5:

| component | share | directional real ms | candidate primitive | single sufficient? |
|---|---:|---:|---|:--:|
| Q4_K GEMM | 31.6% | 18.634 | Q4_K batched weight-read reuse | no |
| Q6_K/lm_head | 16.6% | 9.814 | Q6_K/lm_head batched weight-read reuse | no |
| attention/reduces | 48.6% | 28.640 | short-block causal verify attention + reductions | no |
| elementwise/norm | 3.2% | 1.873 | norm/RoPE/SwiGLU/residual | no |

No single component can cut enough. Even making attention/reduces fully free would not meet the `<=1.5x` verify
gate. Q4_K alone is also insufficient, which preserves the old refutation of the single-kernel Q4_K verify row.

## Verdict

SDB-1: `PASS_MODEL_BUILT`.

SDB-2: `NO_BOUNDED_SHARED_PRIMITIVE`.

The PMU bandwidth framing is correct: spec decode is the right class of algorithmic lever for HBM-bound decode.
But the current tinygrad verify path is not close, and the missing piece is not a bounded primitive. It is a
project-level **T-cheap batched-forward** route that must make Q4_K, Q6_K/lm_head, and attention/reduces cheap
together while preserving exact causal/KV semantics and low-sync accept/commit.

## Consequence

Do **not** start SDB-3 as a bounded kernel proof. SDB-3 only earns a build if a credible project-level T-cheap
batched-forward route is funded first.

Lifecycle state update:

- `decode_spec_verify_shortcut`: stays `closed`.
- `decode_spec_weight_amortization_lifecycle`: moves from `diagnostic` to `project_level`.
- `decode_spec_tcheap_verify_forward`: remains a generated legal row, but it is project-level, not a small kernel
  task.
