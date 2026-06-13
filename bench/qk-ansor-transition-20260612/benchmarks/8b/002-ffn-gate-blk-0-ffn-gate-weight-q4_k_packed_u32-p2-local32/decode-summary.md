# QK Decode Summary

| label | samples | avg tok/s | drop1 | last64 | last32 | last16 | stdev | min | max | Q4 install | Q6 install | storage MB | policy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `explicit1` | 128 | 53.23 | 53.57 | 52.02 | 51.49 | 51.27 | 4.90 | 9.81 | 57.99 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `explicit2` | 128 | 53.38 | 53.73 | 51.89 | 51.08 | 50.94 | 4.61 | 9.87 | 57.51 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `explicit3` | 128 | 53.05 | 53.39 | 52.48 | 51.70 | 51.33 | 5.22 | 9.82 | 57.45 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `generated1` | 128 | 46.85 | 47.15 | 47.97 | 47.30 | 47.14 | 6.98 | 8.43 | 52.76 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/002-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local32/policy.json` |
| `generated2` | 128 | 48.91 | 49.22 | 47.74 | 47.42 | 47.30 | 4.39 | 9.83 | 52.62 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/002-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local32/policy.json` |
| `generated3` | 128 | 46.70 | 46.99 | 47.83 | 46.71 | 45.85 | 8.18 | 9.45 | 52.59 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/002-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local32/policy.json` |
