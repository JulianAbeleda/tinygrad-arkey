# Prefill P0 theoretical ceiling audit

**Verdict:** PREFILL_P0_PASS_CEILING_PINNED

Prefill is **COMPUTE-bound (GEMM FLOP-limited, M=512 high arithmetic intensity)**. GEMM = 85% of wall @ctx512; ceiling set by GEMM TFLOPS (tinygrad ~40.8 vs measured BLAS 51.8-76.7).

## Ceiling / current / candidate by ctx
| ctx | current | pipe_tm2_tn2 | ceiling (UB) | cur % ceil | cand % ceil | cand->ceil headroom |
|---|---|---|---|---|---|---|
| 512 | 3597 | 4253 | 5469.0 | 65.8% | 77.8% | 28.6% |
| 1024 | 3504 | 4037 | 5328.0 | 65.8% | 75.8% | 32.0% |
| 2048 | 3248 | 3659 | 4938.0 | 65.8% | 74.1% | 35.0% |
| 4096 | 2803 | 3110 | 4262.0 | 65.8% | 73.0% | 37.0% |
| 8192 | None | None | None | -% | -% | -% |

## Per-role GEMM floor (M=512)
| role | shape | BLAS TF | tg TF | speedup | time share |
|---|---|---|---|---|---|
| ffn_gate_up | 512x12288x4096 | 69.8 | 40.8 | 1.711x | 0.386 |
| ffn_down | 512x4096x12288 | 70.9 | 40.8 | 1.739x | 0.217 |
| qo_proj | 512x4096x4096 | 76.7 | 40.8 | 1.88x | 0.148 |
| kv_proj | 512x1024x4096 | 51.8 | 40.8 | 1.27x | 0.096 |

## Answers
- ceiling@512 ~= 5469.0 tok/s; current 3597 = 65.8% of it
- pipe_tm2_tn2 = 77.8% of ceiling -> 28.6% headroom remains (NOT near ceiling)
- dominant floor role: ffn_gate_up
- compute-bound (GEMM TFLOPS), not memory/launch
- pipe_tm2_tn2 +11-19% plausible/not-refuted; needs P1 authority + P6 long-context re-validation

## Caveats
- ctx512 ceiling uses MEASURED role shares (per_role_time_tax DIAGNOSTIC; the live tool OOM'd, fell back to prior artifact); P2 refreshes authoritative per-ctx shares
- ctx>=1024 ceiling is an UPPER BOUND (same GEMM speedup); real ceiling is lower as attention's non-GEMM share grows with ctx (consistent with candidate's declining delta)
- uses measured practical BLAS ceilings (51.8-76.7 TF), NOT the 122 TF WMMA marketing peak
- non-GEMM (attention/norm/rope/copy/launch) assumed held fixed at BLAS-lift; some of it is itself accelerable (P2/P3)