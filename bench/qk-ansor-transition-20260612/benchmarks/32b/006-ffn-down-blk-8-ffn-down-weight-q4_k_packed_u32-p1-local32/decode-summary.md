# QK Decode Summary

| label | samples | avg tok/s | drop1 | last64 | last32 | last16 | stdev | min | max | Q4 install | Q6 install | storage MB | policy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `explicit1` | 128 | 16.99 | 17.10 | 16.14 | 15.46 | 15.81 | 2.38 | 3.02 | 18.52 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `explicit2` | 128 | 17.67 | 17.78 | 17.43 | 17.33 | 17.23 | 1.39 | 3.15 | 18.56 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `explicit3` | 128 | 17.48 | 17.59 | 17.32 | 17.07 | 16.72 | 1.44 | 3.11 | 18.52 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `generated1` | 128 | 16.67 | 16.78 | 16.34 | 16.49 | 16.74 | 1.93 | 2.92 | 17.96 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/006-ffn-down-blk-8-ffn-down-weight-q4_k_packed_u32-p1-local32/policy.json` |
| `generated2` | 128 | 16.58 | 16.69 | 16.27 | 16.17 | 16.70 | 1.95 | 2.88 | 17.91 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/006-ffn-down-blk-8-ffn-down-weight-q4_k_packed_u32-p1-local32/policy.json` |
| `generated3` | 128 | 16.98 | 17.09 | 16.61 | 16.20 | 15.98 | 1.54 | 3.01 | 18.00 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/006-ffn-down-blk-8-ffn-down-weight-q4_k_packed_u32-p1-local32/policy.json` |
