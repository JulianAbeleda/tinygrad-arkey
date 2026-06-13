# QK Decode Summary

| label | samples | avg tok/s | drop1 | last64 | last32 | last16 | stdev | min | max | Q4 install | Q6 install | storage MB | policy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `explicit1` | 128 | 52.11 | 52.45 | 52.30 | 51.58 | 51.25 | 6.11 | 9.61 | 57.76 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `explicit2` | 128 | 53.38 | 53.72 | 52.59 | 51.76 | 51.35 | 4.49 | 9.85 | 57.81 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `explicit3` | 128 | 51.44 | 51.76 | 52.03 | 51.62 | 51.45 | 7.10 | 9.88 | 57.76 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `generated1` | 128 | 27.12 | 27.26 | 20.50 | 24.65 | 33.16 | 12.59 | 9.83 | 45.28 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/004-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local32/policy.json` |
| `generated2` | 128 | 37.93 | 38.15 | 40.12 | 38.18 | 37.82 | 6.85 | 9.91 | 44.88 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/004-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local32/policy.json` |
| `generated3` | 128 | 41.76 | 42.01 | 40.46 | 39.18 | 39.42 | 3.74 | 9.97 | 45.38 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/004-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local32/policy.json` |
| `generated4` | 128 | 42.09 | 42.34 | 41.46 | 40.96 | 40.78 | 3.18 | 9.84 | 44.90 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/004-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local32/policy.json` |
