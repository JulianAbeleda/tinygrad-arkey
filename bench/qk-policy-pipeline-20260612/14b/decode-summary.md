# QK Decode Summary

| label | samples | avg tok/s | drop1 | last64 | last32 | last16 | stdev | min | max | Q4 install | Q6 install | policy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `explicit1` | 128 | 22.29 | 22.42 | 22.68 | 21.86 | 20.52 | 3.36 | 6.06 | 24.59 | 180 | 20 | `` |
| `explicit2` | 128 | 23.45 | 23.59 | 23.27 | 23.11 | 23.00 | 1.69 | 6.07 | 24.62 | 180 | 20 | `` |
| `explicit3` | 128 | 21.85 | 21.97 | 20.83 | 21.53 | 22.58 | 3.53 | 6.04 | 24.40 | 180 | 20 | `` |
| `generated1` | 128 | 38.56 | 38.82 | 37.06 | 34.62 | 36.09 | 5.72 | 6.10 | 43.32 | 240 | 40 | `bench/qk-policy-pipeline-20260612/14b/policy.json` |
| `generated2` | 128 | 30.32 | 30.51 | 22.77 | 26.46 | 31.37 | 11.00 | 6.11 | 43.29 | 240 | 40 | `bench/qk-policy-pipeline-20260612/14b/policy.json` |
| `generated3` | 128 | 40.43 | 40.70 | 39.31 | 39.14 | 39.02 | 3.50 | 6.08 | 43.32 | 240 | 40 | `bench/qk-policy-pipeline-20260612/14b/policy.json` |
| `generated4` | 128 | 39.61 | 39.87 | 37.80 | 35.26 | 38.62 | 4.68 | 6.08 | 43.21 | 240 | 40 | `bench/qk-policy-pipeline-20260612/14b/policy.json` |
| `generated5` | 128 | 39.93 | 40.19 | 39.11 | 37.87 | 38.42 | 3.90 | 5.98 | 43.22 | 240 | 40 | `bench/qk-policy-pipeline-20260612/14b/policy.json` |
