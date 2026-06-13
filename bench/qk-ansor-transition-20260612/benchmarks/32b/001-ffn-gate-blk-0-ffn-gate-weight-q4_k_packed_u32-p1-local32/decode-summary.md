# QK Decode Summary

| label | samples | avg tok/s | drop1 | last64 | last32 | last16 | stdev | min | max | Q4 install | Q6 install | storage MB | policy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `explicit1` | 128 | 17.47 | 17.58 | 17.15 | 16.75 | 16.07 | 1.69 | 3.13 | 18.48 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `explicit2` | 128 | 16.38 | 16.49 | 15.22 | 13.17 | 8.94 | 3.61 | 3.13 | 18.55 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `explicit3` | 128 | 16.57 | 16.67 | 15.56 | 16.22 | 15.32 | 2.74 | 3.13 | 18.54 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `explicit4` | 128 | 16.88 | 16.99 | 16.45 | 17.08 | 17.20 | 2.46 | 3.15 | 18.43 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `explicit5` | 128 | 16.66 | 16.77 | 17.27 | 16.93 | 16.42 | 2.73 | 3.15 | 18.57 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `generated1` | 128 | 17.01 | 17.12 | 16.14 | 17.36 | 17.26 | 2.42 | 3.16 | 18.53 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/001-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p1-local32/policy.json` |
| `generated2` | 128 | 17.18 | 17.29 | 16.61 | 16.52 | 15.62 | 2.24 | 3.00 | 18.52 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/001-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p1-local32/policy.json` |
| `generated3` | 128 | 17.55 | 17.66 | 17.39 | 17.16 | 16.90 | 1.43 | 3.15 | 18.55 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/001-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p1-local32/policy.json` |
