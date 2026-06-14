# JSON Rejection-Sampling Data

This artifact samples completions from the current best adapter on V4 train
prompts, keeps strict JSON passes as SFT rows, and carries the original V4
eval rows only for trainer diagnostics. Held-out promotion still uses the
separate V4 rollout gate.

## Summary

- attempts: `1632`
- accepted attempts: `216`
- selected train rows: `215`
- eval rows: `204`
- strict pass: `216/1632`

| category | attempts | accepted attempts | selected train | near miss |
|---|---:|---:|---:|---:|
| `arithmetic` | 272 | 61 | 61 | 7 |
| `categorization` | 272 | 46 | 46 | 22 |
| `code` | 272 | 18 | 18 | 50 |
| `compiler` | 272 | 0 | 0 | 79 |
| `fact` | 272 | 68 | 67 | 1 |
| `string` | 272 | 23 | 23 | 22 |
