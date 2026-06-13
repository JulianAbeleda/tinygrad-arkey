# QK Decode Summary

| label | samples | avg tok/s | drop1 | last64 | last32 | last16 | stdev | min | max | Q4 install | Q6 install | storage MB | policy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `explicit1` | 128 | 17.43 | 17.54 | 16.98 | 16.45 | 16.13 | 1.84 | 3.10 | 18.50 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `explicit2` | 128 | 17.67 | 17.79 | 17.44 | 17.23 | 17.19 | 1.37 | 3.15 | 18.49 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `explicit3` | 128 | 17.03 | 17.14 | 16.25 | 15.08 | 17.27 | 2.36 | 3.07 | 18.50 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `generated1` | 128 | 16.41 | 16.51 | 14.97 | 17.34 | 17.27 | 3.46 | 3.12 | 18.52 | 384 | 64 | 0.00 | `bench/qk-ansor-transition-20260612/benchmarks/32b-confirm/001-ffn-gate-local32-p1/policy.json` |
| `generated2` | 128 | 16.94 | 17.05 | 16.16 | 16.79 | 17.22 | 2.72 | 3.16 | 18.62 | 384 | 64 | 0.00 | `bench/qk-ansor-transition-20260612/benchmarks/32b-confirm/001-ffn-gate-local32-p1/policy.json` |
| `generated3` | 128 | 17.59 | 17.71 | 17.44 | 17.29 | 17.25 | 1.40 | 3.06 | 18.58 | 384 | 64 | 0.00 | `bench/qk-ansor-transition-20260612/benchmarks/32b-confirm/001-ffn-gate-local32-p1/policy.json` |
