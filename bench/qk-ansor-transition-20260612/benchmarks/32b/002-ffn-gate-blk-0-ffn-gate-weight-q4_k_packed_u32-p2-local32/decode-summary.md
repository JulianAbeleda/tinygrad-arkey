# QK Decode Summary

| label | samples | avg tok/s | drop1 | last64 | last32 | last16 | stdev | min | max | Q4 install | Q6 install | storage MB | policy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `explicit1` | 128 | 17.04 | 17.15 | 17.10 | 17.34 | 17.27 | 1.80 | 3.14 | 18.51 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `explicit2` | 128 | 17.59 | 17.70 | 17.27 | 16.90 | 16.84 | 1.46 | 3.16 | 18.51 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `explicit3` | 128 | 17.67 | 17.78 | 17.42 | 17.22 | 17.07 | 1.38 | 3.15 | 18.51 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `generated1` | 128 | 12.42 | 12.49 | 11.00 | 9.53 | 12.29 | 2.89 | 3.16 | 14.39 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/002-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local32/policy.json` |
| `generated2` | 128 | 13.20 | 13.28 | 12.87 | 12.34 | 11.71 | 1.52 | 3.16 | 14.27 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/002-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local32/policy.json` |
| `generated3` | 128 | 13.52 | 13.60 | 13.60 | 13.49 | 13.45 | 1.33 | 3.11 | 14.29 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/002-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local32/policy.json` |
| `generated4` | 128 | 12.43 | 12.51 | 11.24 | 10.18 | 9.72 | 2.43 | 3.16 | 14.34 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/002-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local32/policy.json` |
| `generated5` | 128 | 13.30 | 13.38 | 12.79 | 13.49 | 13.43 | 1.84 | 3.15 | 14.28 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/002-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local32/policy.json` |
| `generated6` | 128 | 13.63 | 13.71 | 13.55 | 13.43 | 13.42 | 0.98 | 3.15 | 14.21 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/002-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local32/policy.json` |
| `generated7` | 128 | 11.29 | 11.35 | 8.82 | 7.28 | 8.25 | 3.46 | 3.03 | 14.27 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/002-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local32/policy.json` |
