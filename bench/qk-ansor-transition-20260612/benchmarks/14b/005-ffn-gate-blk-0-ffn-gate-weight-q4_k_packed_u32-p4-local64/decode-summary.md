# QK Decode Summary

| label | samples | avg tok/s | drop1 | last64 | last32 | last16 | stdev | min | max | Q4 install | Q6 install | storage MB | policy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `explicit1` | 128 | 38.59 | 38.84 | 36.51 | 39.41 | 39.18 | 6.51 | 6.05 | 43.34 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `explicit2` | 128 | 40.20 | 40.47 | 39.80 | 39.22 | 38.89 | 4.42 | 6.06 | 43.28 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `explicit3` | 128 | 38.56 | 38.81 | 39.06 | 38.84 | 39.16 | 6.55 | 5.86 | 43.30 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `generated1` | 128 | 28.88 | 29.06 | 28.19 | 27.38 | 27.97 | 2.85 | 6.02 | 30.80 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/005-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local64/policy.json` |
| `generated2` | 128 | 28.67 | 28.84 | 27.82 | 26.96 | 26.34 | 3.15 | 6.08 | 31.02 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/005-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local64/policy.json` |
| `generated3` | 128 | 28.70 | 28.88 | 28.20 | 27.43 | 27.05 | 2.62 | 5.95 | 30.90 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/005-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local64/policy.json` |
