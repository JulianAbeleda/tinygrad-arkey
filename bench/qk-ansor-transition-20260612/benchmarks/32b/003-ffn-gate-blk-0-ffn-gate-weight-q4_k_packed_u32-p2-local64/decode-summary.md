# QK Decode Summary

| label | samples | avg tok/s | drop1 | last64 | last32 | last16 | stdev | min | max | Q4 install | Q6 install | storage MB | policy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `explicit1` | 128 | 17.01 | 17.12 | 16.42 | 16.98 | 17.25 | 2.33 | 3.14 | 18.52 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `explicit2` | 128 | 17.64 | 17.75 | 17.39 | 17.30 | 17.25 | 1.36 | 3.16 | 18.66 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `explicit3` | 128 | 17.70 | 17.82 | 17.49 | 17.32 | 17.25 | 1.35 | 3.14 | 18.52 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `generated1` | 128 | 13.27 | 13.35 | 13.03 | 12.77 | 12.96 | 1.37 | 3.10 | 14.31 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/003-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local64/policy.json` |
| `generated2` | 128 | 13.42 | 13.50 | 13.48 | 13.34 | 13.22 | 1.11 | 3.11 | 14.20 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/003-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local64/policy.json` |
| `generated3` | 128 | 13.52 | 13.61 | 13.29 | 13.26 | 13.12 | 1.13 | 3.14 | 14.21 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/003-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local64/policy.json` |
