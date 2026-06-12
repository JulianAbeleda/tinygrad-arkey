# QK Decode Summary

| label | samples | avg tok/s | drop1 | last64 | last32 | last16 | stdev | min | max | Q4 install | Q6 install | storage MB | policy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `sidecar` | 4 | 45.80 | 57.77 | 45.80 | 45.80 | 45.80 | 23.96 | 9.86 | 58.11 | 162 | 18 | 3786.75 | `bench/qk-policy-pipeline-20260612/8b/policy.json` |
| `cap512` | 4 | 12.27 | 13.46 | 12.27 | 12.27 | 12.27 | 2.39 | 8.68 | 13.51 | 28 | 0 | 504.00 | `bench/qk-policy-pipeline-20260612/8b/policy.json` |
| `ondemand` | 4 | 2.87 | 0.55 | 2.87 | 2.87 | 2.87 | 4.65 | 0.54 | 9.84 | 162 | 18 | 708.75 | `bench/qk-policy-pipeline-20260612/8b/policy.json` |
