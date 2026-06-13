# QK Decode Summary

| label | samples | avg tok/s | drop1 | last64 | last32 | last16 | stdev | min | max | Q4 install | Q6 install | storage MB | policy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `explicit1` | 128 | 39.64 | 39.90 | 37.66 | 35.13 | 37.39 | 4.95 | 6.06 | 43.43 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `explicit2` | 128 | 39.08 | 39.34 | 39.43 | 39.47 | 39.22 | 5.74 | 6.07 | 43.50 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `explicit3` | 128 | 40.08 | 40.35 | 39.05 | 37.61 | 39.19 | 5.21 | 6.03 | 43.15 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `generated1` | 128 | 38.05 | 38.30 | 39.20 | 39.37 | 39.29 | 6.02 | 6.06 | 43.22 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/001-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p1-local64/policy.json` |
| `generated2` | 128 | 41.08 | 41.35 | 40.26 | 39.79 | 39.57 | 3.43 | 6.07 | 43.78 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/001-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p1-local64/policy.json` |
| `generated3` | 128 | 40.17 | 40.44 | 39.89 | 39.74 | 39.39 | 4.60 | 6.03 | 43.91 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/001-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p1-local64/policy.json` |
