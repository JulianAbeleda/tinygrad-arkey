# QK Decode Summary

| label | samples | avg tok/s | drop1 | last64 | last32 | last16 | stdev | min | max | Q4 install | Q6 install | storage MB | policy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `explicit1` | 128 | 49.16 | 49.47 | 45.20 | 43.94 | 37.38 | 9.81 | 9.93 | 57.99 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `explicit2` | 128 | 50.85 | 51.17 | 50.98 | 48.63 | 46.73 | 8.18 | 9.89 | 57.37 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `explicit3` | 128 | 50.39 | 50.71 | 45.74 | 41.90 | 34.49 | 9.87 | 9.88 | 57.99 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `explicit4` | 128 | 51.83 | 52.16 | 50.70 | 49.65 | 47.18 | 7.34 | 9.71 | 57.72 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `explicit5` | 128 | 53.37 | 53.71 | 52.43 | 51.82 | 51.48 | 4.79 | 9.94 | 57.64 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `explicit6` | 128 | 53.70 | 54.04 | 52.41 | 51.61 | 51.45 | 4.49 | 9.73 | 57.85 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `generated1` | 128 | 32.24 | 32.42 | 35.05 | 40.98 | 40.77 | 12.81 | 9.69 | 44.69 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/005-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local64/policy.json` |
| `generated2` | 128 | 41.87 | 42.12 | 41.34 | 40.91 | 40.62 | 3.25 | 9.96 | 44.92 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/005-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local64/policy.json` |
| `generated3` | 128 | 41.79 | 42.06 | 40.76 | 40.87 | 40.69 | 3.78 | 6.96 | 44.63 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/005-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local64/policy.json` |
| `generated4` | 128 | 41.68 | 41.93 | 40.93 | 40.10 | 40.46 | 3.78 | 9.86 | 44.64 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/005-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local64/policy.json` |
