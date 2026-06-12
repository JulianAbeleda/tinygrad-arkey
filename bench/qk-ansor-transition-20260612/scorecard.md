# QK Llama.cpp Scorecard

Objective function for the Ansor-transition loop. This is a read-only
summary over committed QK decisions and optional rollout-comparator
artifacts; it does not run benchmarks.

## Summary

- accepted models: `3/3`
- correctness ok: `True`
- rollout compare ok: `True`
- min % llama.cpp: `51.46`
- mean % llama.cpp: `56.34`
- all models at 70%: `False`
- below 70%: `8B, 14B, 32B`

## Model Rows

| model | generated tok/s | llama ref | % llama | gap to 70 | parity speedup needed | A/B | runtime MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `8B` | 52.07 | 101.20 | 51.46 | 18.54 | 1.94x | `True` | 0.00 |
| `14B` | 40.55 | 65.80 | 61.63 | 8.37 | 1.62x | `True` | 0.00 |
| `32B` | 17.23 | 30.80 | 55.94 | 14.06 | 1.79x | `True` | 0.00 |

## Rollout Comparator

- path: `bench/qwen-rollout-20260612/compare-8b-small/report.json`
- baseline: `8b-generated-small`
- regressions: `0`
- text changes: `0`
- token changes: `0`
