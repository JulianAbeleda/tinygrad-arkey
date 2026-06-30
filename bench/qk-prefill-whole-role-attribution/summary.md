# Prefill P2 whole-prefill role attribution

**Verdict:** PREFILL_P2_PASS_ROLE_ATTRIBUTION_PINNED

compute-bound (GEMM>=50%): True ({'0': 85.0, '3584': 57.4}); gate_up dominant: True; worst unknown: 0.0%

## Role wall-stack (% GPU time) + GEMM effective TFLOPS
| role | sp=0 % | sp=3584 % | eff TFLOPS@0 | BLAS | % of BLAS |
|---|---|---|---|---|---|
| ffn_gate_up | 35.29 | 23.65 | 75.4 | 69.8 | 108.0 |
| ffn_down | 24.47 | 16.57 | 54.3 | 70.9 | 76.6 |
| attn_qo | 16.91 | 11.54 | 52.4 | 76.7 | 68.4 |
| norm_rope_elementwise | 15.05 | 42.61 | — | — | — |
| attn_kv | 8.29 | 5.63 | 26.7 | 51.8 | 51.6 |