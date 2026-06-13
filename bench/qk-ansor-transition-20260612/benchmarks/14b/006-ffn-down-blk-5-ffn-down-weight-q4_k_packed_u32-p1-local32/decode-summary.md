# QK Decode Summary

| label | samples | avg tok/s | drop1 | last64 | last32 | last16 | stdev | min | max | Q4 install | Q6 install | storage MB | policy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `explicit1` | 128 | 39.92 | 40.18 | 38.17 | 38.39 | 38.83 | 4.81 | 6.07 | 43.39 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `explicit2` | 128 | 38.29 | 38.54 | 37.26 | 37.79 | 38.75 | 5.24 | 6.08 | 43.24 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `explicit3` | 128 | 40.28 | 40.55 | 39.19 | 38.65 | 38.11 | 3.89 | 6.08 | 43.28 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `generated1` | 128 | 39.40 | 39.66 | 38.62 | 38.13 | 38.01 | 3.29 | 6.08 | 42.15 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/006-ffn-down-blk-5-ffn-down-weight-q4_k_packed_u32-p1-local32/policy.json` |
| `generated2` | 128 | 37.34 | 37.59 | 38.29 | 38.12 | 37.58 | 6.23 | 5.83 | 42.03 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/006-ffn-down-blk-5-ffn-down-weight-q4_k_packed_u32-p1-local32/policy.json` |
| `generated3` | 128 | 39.17 | 39.43 | 38.68 | 38.08 | 37.59 | 3.55 | 5.92 | 41.95 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/006-ffn-down-blk-5-ffn-down-weight-q4_k_packed_u32-p1-local32/policy.json` |
