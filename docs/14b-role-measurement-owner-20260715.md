# 14B role measurement owner — 2026-07-15

Validation-only capture for the research candidates. No production route
selection was changed.

## Results

The staged Q6 candidate completed on the exact 14B `ffn_down` pp512 shape
`M=512, N=4096, K=12288`:

| candidate | route identity | median ms | tok/s-equivalent | kernels/programs | fallback |
|---|---|---:|---:|---:|---|
| staged Q6 combined | `staged_dequant_then_fp16_wmma` | 14.437 | 35,465 | 3 programs, 1 WMMA | false |
| staged Q6 contraction-only | `staged_dequant_then_fp16_wmma` | 5.404 | 94,743 | 3 programs, 1 WMMA | false |
| direct Q6 rollback comparator | `direct_packed` | 440.869 | 1,161 | 1 program, 0 WMMA | false |

The Q6 owner harness does not yet emit a numerical correctness comparison;
therefore correctness is recorded as **not captured**, not PASS.

The fused Q4 gate passed its bounded `M=16, N=16, K=256` correctness fixture:
fused route `fused_packed_q4`, 2 kernels, 0.0256 ms, relative RMSE
`1.17e-7`, and `fallback=false`. The exact full pp512 14B Q4 role-shape
attempt was stopped after 180 seconds without an artifact; no full-role Q4
timing or correctness result is published.

Commands:

```sh
PYTHONPATH=. pytest -q test/unit/test_q4k_fused_q4_correctness_gate.py \
  test/unit/test_q6k_14b_wmma_bench.py test/unit/test_q6k_wmma_route_gate.py \
  test/unit/test_role_shape_route_validation.py
PYTHONPATH=. python3 extra/qk/q4k_fused_q4_correctness_gate.py
PYTHONPATH=. python3 extra/qk/q6k_14b_wmma_bench.py
```
