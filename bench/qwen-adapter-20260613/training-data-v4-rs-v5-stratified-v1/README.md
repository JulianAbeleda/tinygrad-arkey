# JSON Rejection-Sampling Data

This artifact samples completions from the current best adapter on V4 train
prompts, keeps strict JSON passes as SFT rows, and carries the original V4
eval rows only for trainer diagnostics. Held-out promotion still uses the
separate V4 rollout gate.

## Summary

- attempts: `2448`
- accepted attempts: `257`
- selected train rows: `217`
- eval rows: `204`
- strict pass: `257/2448`

| category | attempts | accepted attempts | selected train | near miss |
|---|---:|---:|---:|---:|
| `arithmetic` | 272 | 61 | 61 | 7 |
| `categorization` | 272 | 46 | 46 | 22 |
| `code` | 544 | 36 | 20 | 100 |
| `compiler` | 544 | 0 | 0 | 158 |
| `fact` | 272 | 68 | 67 | 1 |
| `string` | 544 | 46 | 23 | 44 |

- sampled categories this run: `code, compiler, string`
- sampled train rows this run: `204`

## Category JSON Axes

| category | parse | schema | type | value | strict |
|---|---:|---:|---:|---:|---:|
| `arithmetic` | 68/272 | 68/272 | 68/272 | 61/272 | 61/272 |
| `categorization` | 68/272 | 68/272 | 68/272 | 46/272 | 46/272 |
| `code` | 136/544 | 136/544 | 136/544 | 36/544 | 36/544 |
| `compiler` | 160/544 | 158/544 | 158/544 | 0/544 | 0/544 |
| `fact` | 69/272 | 69/272 | 69/272 | 68/272 | 68/272 |
| `string` | 228/544 | 90/544 | 90/544 | 46/544 | 46/544 |

## Temperature Summary

| temperature | attempts | accepted | near miss |
|---|---:|---:|---:|
| `0.0` | 408 | 215 | 170 |
| `0.05` | 204 | 41 | 140 |
| `0.1` | 204 | 0 | 0 |
| `0.15` | 204 | 0 | 10 |
| `0.2` | 408 | 0 | 0 |
| `0.25` | 204 | 0 | 1 |
| `0.5` | 408 | 0 | 11 |
| `0.8` | 408 | 1 | 0 |
