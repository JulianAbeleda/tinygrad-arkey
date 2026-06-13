# QK Decode Summary

| label | samples | avg tok/s | drop1 | last64 | last32 | last16 | stdev | min | max | Q4 install | Q6 install | storage MB | policy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `explicit1` | 128 | 53.61 | 53.95 | 52.40 | 51.37 | 51.18 | 4.46 | 9.82 | 57.74 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `explicit2` | 128 | 51.42 | 51.75 | 47.61 | 51.83 | 51.42 | 8.19 | 9.48 | 57.88 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `explicit3` | 128 | 49.71 | 50.02 | 47.25 | 41.34 | 51.50 | 10.30 | 9.93 | 57.76 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `generated1` | 128 | 51.69 | 52.01 | 50.56 | 49.84 | 49.49 | 4.13 | 9.87 | 55.58 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/006-ffn-down-blk-4-ffn-down-weight-q4_k_packed_u32-p1-local32/policy.json` |
| `generated2` | 128 | 51.37 | 51.70 | 49.96 | 49.49 | 49.56 | 4.46 | 9.76 | 55.40 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/006-ffn-down-blk-4-ffn-down-weight-q4_k_packed_u32-p1-local32/policy.json` |
| `generated3` | 128 | 50.14 | 50.48 | 48.35 | 49.47 | 49.13 | 6.21 | 7.55 | 55.11 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/006-ffn-down-blk-4-ffn-down-weight-q4_k_packed_u32-p1-local32/policy.json` |
