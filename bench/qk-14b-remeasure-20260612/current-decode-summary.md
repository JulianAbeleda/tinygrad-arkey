# QK Decode Summary

| label | samples | avg tok/s | drop1 | last64 | last32 | last16 | stdev | min | max | Q4 install | Q6 install | policy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `explicit1` | 128 | 23.27 | 23.40 | 22.99 | 22.74 | 22.90 | 1.71 | 6.04 | 24.35 |  |  | `` |
| `generated1` | 128 | 39.57 | 39.83 | 38.51 | 37.53 | 36.13 | 4.84 | 6.04 | 43.12 | 240 | 40 | `bench/qk-semantic-20260612/14b-full-level2-skip-stopped-policy.json` |
| `explicit2` | 128 | 23.18 | 23.32 | 22.72 | 22.12 | 21.27 | 2.05 | 6.04 | 24.40 |  |  | `` |
| `generated2` | 128 | 39.42 | 39.68 | 38.55 | 36.86 | 39.08 | 4.84 | 6.09 | 43.29 | 240 | 40 | `bench/qk-semantic-20260612/14b-full-level2-skip-stopped-policy.json` |
| `explicit3` | 128 | 23.36 | 23.50 | 23.10 | 23.02 | 22.93 | 1.72 | 6.08 | 24.39 |  |  | `` |
| `generated3` | 128 | 40.05 | 40.32 | 39.25 | 38.17 | 36.73 | 3.93 | 6.06 | 43.21 | 240 | 40 | `bench/qk-semantic-20260612/14b-full-level2-skip-stopped-policy.json` |
