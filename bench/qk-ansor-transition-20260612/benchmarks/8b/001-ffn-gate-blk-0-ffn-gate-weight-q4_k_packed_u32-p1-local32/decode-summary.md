# QK Decode Summary

| label | samples | avg tok/s | drop1 | last64 | last32 | last16 | stdev | min | max | Q4 install | Q6 install | storage MB | policy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `explicit1` | 128 | 53.39 | 53.73 | 51.96 | 51.30 | 51.24 | 4.64 | 9.84 | 57.42 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `explicit2` | 128 | 53.13 | 53.47 | 51.99 | 51.00 | 50.09 | 4.70 | 9.86 | 57.81 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `explicit3` | 128 | 53.30 | 53.64 | 51.92 | 50.95 | 50.26 | 4.80 | 9.80 | 57.54 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `generated1` | 128 | 52.48 | 52.82 | 51.17 | 50.78 | 50.55 | 4.68 | 9.84 | 57.06 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/001-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p1-local32/policy.json` |
| `generated2` | 128 | 52.71 | 53.05 | 51.27 | 50.72 | 50.05 | 4.56 | 9.80 | 57.14 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/001-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p1-local32/policy.json` |
| `generated3` | 128 | 52.51 | 52.84 | 51.12 | 50.15 | 50.56 | 5.01 | 9.63 | 56.71 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/001-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p1-local32/policy.json` |
