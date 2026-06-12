# QK Experiment Matrix

| path | model | status | ref | explicit tok/s | generated tok/s | gain % | % llama | A/B | policy MB | runtime MB |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `bench/qk-shared-storage-20260612/8b` | `8B` | `accept` | `explicit` | 50.41 | 52.07 | 3.31 | 51.46 | True | 102.38 | 0.00 |
| `bench/qk-shared-storage-20260612/14b` | `14B` | `accept` | `explicit` | 21.77 | 40.55 | 86.29 | 61.63 | True | 186.33 | 0.00 |
| `bench/qk-shared-storage-20260612/32b` | `32B` | `accept` | `explicit` | 11.15 | 17.23 | 54.56 | 55.94 | True | 295.08 | 0.00 |

## Summary

```json
{
  "accepted": 3,
  "accepted_mean_gain": 0.4805189794085383,
  "experiments": 3,
  "statuses": {
    "accept": 3
  }
}
```
