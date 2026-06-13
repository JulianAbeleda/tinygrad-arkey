# QK Decode Summary

| label | samples | avg tok/s | drop1 | last64 | last32 | last16 | stdev | min | max | Q4 install | Q6 install | storage MB | policy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `explicit1` | 128 | 37.39 | 37.64 | 35.73 | 39.45 | 39.22 | 6.65 | 5.69 | 43.31 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `explicit2` | 128 | 40.60 | 40.87 | 39.77 | 39.13 | 38.87 | 3.43 | 6.02 | 43.47 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `explicit3` | 128 | 40.49 | 40.76 | 39.89 | 39.47 | 39.20 | 3.70 | 6.10 | 43.45 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `generated1` | 128 | 31.04 | 31.23 | 30.93 | 30.55 | 30.50 | 3.78 | 6.00 | 33.75 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/002-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local32/policy.json` |
| `generated2` | 128 | 31.75 | 31.95 | 31.23 | 30.74 | 30.34 | 2.53 | 6.06 | 33.67 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/002-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local32/policy.json` |
| `generated3` | 128 | 29.28 | 29.46 | 31.08 | 31.03 | 30.82 | 5.92 | 5.85 | 33.43 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/002-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local32/policy.json` |
