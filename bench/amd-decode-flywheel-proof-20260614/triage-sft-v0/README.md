# QK Flywheel Triage SFT Dataset

This Phase 3.1 artifact converts the Phase 1 kernel-history triage dataset
into strict JSON SFT rows for adapter training. Holdout rows are included
only as `split=eval` rows for teacher-forced diagnostics and as rollout
prompts; they are not optimized as train rows.

- rows: `83`
- train rows: `45`
- eval/holdout rows: `38`
- oversampled rows: `0`
- schema-support rows: `0`
- holdout ids in train: `0`

## Train Labels

| label | rows |
|---|---:|
| `accept` | 9 |
| `construction_blocked` | 1 |
| `needs_rerun` | 2 |
| `reject` | 20 |
| `tie` | 13 |

## Eval Labels

| label | rows |
|---|---:|
| `construction_blocked` | 19 |
| `diagnostic_only` | 1 |
| `raw_accept_unconfirmed` | 3 |
| `reject` | 9 |
| `tie` | 6 |
