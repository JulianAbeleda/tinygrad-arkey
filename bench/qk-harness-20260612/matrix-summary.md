# QK Experiment Matrix

| path | model | status | ref | explicit tok/s | generated tok/s | gain % | % llama | A/B | policy MB | runtime MB |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `bench/qk-harness-20260612/8b` | `8B` | `accept` | `explicit` | 49.35 | 53.49 | 8.41 | 52.86 | True | 102.38 | 3786.75 |
| `bench/qk-harness-20260612/14b` | `14B` | `needs-rerun` | `explicit` | 22.72 | 34.75 | 52.93 | 52.81 | True | 186.33 | 7551.56 |
| `bench/qk-policy-cap-20260612/32b-1536mb` | `32B` | `accept` | `generic` | 3.44 | 4.16 | 20.98 | 13.52 | True | 1526.25 | n/a |

## Summary

```json
{
  "accepted": 2,
  "accepted_mean_gain": 0.14693976589045366,
  "experiments": 3,
  "statuses": {
    "accept": 2,
    "needs-rerun": 1
  }
}
```
