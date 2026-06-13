# QK Decode Summary

| label | samples | avg tok/s | drop1 | last64 | last32 | last16 | stdev | min | max | Q4 install | Q6 install | storage MB | policy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `explicit1` | 128 | 16.65 | 16.75 | 16.85 | 17.34 | 17.24 | 2.67 | 2.91 | 18.45 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `explicit2` | 128 | 17.25 | 17.36 | 16.62 | 16.23 | 15.28 | 2.08 | 3.10 | 18.49 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `explicit3` | 128 | 17.50 | 17.61 | 17.28 | 16.96 | 17.21 | 1.59 | 3.03 | 18.50 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `generated1` | 128 | 13.10 | 13.18 | 13.11 | 13.04 | 13.00 | 1.29 | 3.04 | 13.98 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/004-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local32/policy.json` |
| `generated2` | 128 | 12.87 | 12.95 | 12.96 | 12.74 | 12.82 | 1.18 | 3.13 | 14.01 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/004-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local32/policy.json` |
| `generated3` | 128 | 12.75 | 12.83 | 12.99 | 13.02 | 12.93 | 1.99 | 3.05 | 14.01 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/004-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local32/policy.json` |
