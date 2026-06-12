# QK Decode Summary

| label | samples | avg tok/s | drop1 | last64 | last32 | last16 | stdev | min | max | Q4 install | Q6 install | policy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `explicit1` | 128 | 3.46 | 3.46 | 3.47 | 3.47 | 3.48 | 0.16 | 2.07 | 3.54 |  |  | `` |
| `explicit2` | 128 | 3.43 | 3.43 | 3.40 | 3.32 | 3.30 | 0.21 | 2.33 | 3.56 |  |  | `` |
| `explicit3` | 128 | 3.43 | 3.44 | 3.46 | 3.45 | 3.44 | 0.18 | 2.26 | 3.54 |  |  | `` |
| `generated1` | 128 | 4.13 | 4.14 | 4.02 | 4.01 | 3.95 | 0.35 | 2.77 | 4.34 | 112 | 32 | `bench/qk-policy-cap-20260612/32b-1536mb/policy.json` |
| `generated2` | 128 | 4.21 | 4.21 | 4.23 | 4.22 | 4.20 | 0.24 | 2.89 | 4.34 | 112 | 32 | `bench/qk-policy-cap-20260612/32b-1536mb/policy.json` |
| `generated3` | 128 | 4.15 | 4.16 | 4.20 | 4.19 | 4.25 | 0.31 | 2.83 | 4.35 | 112 | 32 | `bench/qk-policy-cap-20260612/32b-1536mb/policy.json` |
