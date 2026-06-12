# QK Experiment Matrix

| path | model | status | ref | explicit tok/s | generated tok/s | gain % | % llama | A/B | policy MB | runtime MB |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `bench/qk-harness-20260612/8b` | `8B` | `accept` | `explicit` | 49.35 | 53.49 | 8.41 | 52.86 | True | 102.38 | 3786.75 |
| `bench/qk-harness-20260612/14b-rerun` | `14B` | `accept` | `explicit` | 22.76 | 39.61 | 74.02 | 60.20 | True | 186.33 | 7551.56 |
| `bench/qk-shared-storage-20260612/32b` | `32B` | `accept` | `explicit` | 11.15 | 17.23 | 54.56 | 55.94 | True | 295.08 | 0.00 |

## Summary

```json
{
  "accepted": 3,
  "accepted_mean_gain": 0.4566029232404473,
  "experiments": 3,
  "statuses": {
    "accept": 3
  }
}
```
