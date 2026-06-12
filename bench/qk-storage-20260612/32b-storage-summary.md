# QK Decode Summary

| label | samples | avg tok/s | drop1 | last64 | last32 | last16 | stdev | min | max | Q4 install | Q6 install | storage MB | policy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `32b_static_cap` | 4 | 4.01 | 4.30 | 4.01 | 4.01 | 4.01 | 0.59 | 3.12 | 4.33 | 112 | 32 | 1526.25 | `bench/qk-policy-cap-20260612/32b-1536mb/policy.json` |
| `32b_runtime_cap` | 4 | 3.57 | 3.71 | 3.57 | 3.57 | 3.57 | 0.29 | 3.14 | 3.72 | 43 | 0 | 1535.62 | `bench/qk-policy-pipeline-20260612/32b/policy.json` |
