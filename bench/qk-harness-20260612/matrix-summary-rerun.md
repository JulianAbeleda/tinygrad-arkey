# QK Experiment Matrix

| path | model | status | ref | explicit tok/s | generated tok/s | gain % | % llama | A/B | policy MB | runtime MB |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `bench/qk-harness-20260612/8b` | `8B` | `accept` | `explicit` | 49.35 | 53.49 | 8.41 | 52.86 | True | 102.38 | 3786.75 |
| `bench/qk-harness-20260612/14b-rerun` | `14B` | `accept` | `explicit` | 22.76 | 39.61 | 74.02 | 60.20 | True | 186.33 | 7551.56 |
| `bench/qk-policy-cap-20260612/32b-1536mb` | `32B` | `accept` | `generic` | 3.44 | 4.16 | 20.98 | 13.52 | True | 1526.25 | n/a |

## Summary

```json
{
  "accepted": 3,
  "accepted_mean_gain": 0.344684828836447,
  "experiments": 3,
  "statuses": {
    "accept": 3
  }
}
```
