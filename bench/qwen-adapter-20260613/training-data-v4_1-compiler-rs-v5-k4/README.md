# JSON Rejection-Sampling Data

This artifact samples completions from the selected adapter on source
train prompts, keeps strict JSON passes as SFT rows, and carries the
source eval rows only for trainer diagnostics. Held-out promotion should
use the separate rollout gate for the matching source dataset.

## Summary

- attempts: `272`
- accepted attempts: `68`
- selected train rows: `68`
- eval rows: `34`
- strict pass: `68/272`

| category | attempts | accepted attempts | selected train | near miss |
|---|---:|---:|---:|---:|
| `compiler` | 272 | 68 | 68 | 0 |

## Category JSON Axes

| category | parse | schema | type | value | strict |
|---|---:|---:|---:|---:|---:|
| `compiler` | 68/272 | 68/272 | 68/272 | 68/272 | 68/272 |

## Temperature Summary

| temperature | attempts | accepted | near miss |
|---|---:|---:|---:|
| `0.0` | 68 | 68 | 0 |
| `0.05` | 68 | 0 | 0 |
| `0.1` | 68 | 0 | 0 |
| `0.2` | 68 | 0 | 0 |
