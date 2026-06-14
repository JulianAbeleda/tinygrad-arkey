# AMD Decode Flywheel Triage Baselines

- conclusion: `no_signal`
- examples: `83`
- train rows: `45`
- holdout rows: `38`
- best baseline: `mechanism_prior`

| baseline | accuracy | macro-F1 | false accept | retry precision | retry recall | p@1 | ndcg |
|---|---:|---:|---:|---:|---:|---:|---:|
| `majority_label` | 0.237 | 0.077 | 0.000 | n/a | 0.000 | 0.000 | 0.170 |
| `reject_all` | 0.237 | 0.077 | 0.000 | n/a | 0.000 | 0.000 | 0.170 |
| `random_label` | 0.237 | 0.110 | 0.132 | 0.000 | 0.000 | 0.000 | 0.170 |
| `mechanism_prior` | 0.289 | 0.185 | 0.000 | n/a | 0.000 | 0.000 | 0.218 |
| `simple_family_heuristic` | 0.289 | 0.185 | 0.000 | n/a | 0.000 | 0.000 | 0.218 |

## Model Predictions

| model | accuracy | macro-F1 | false accept | retry precision | retry recall | p@1 | ndcg |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen3_8b_base` | 0.000 | 0.000 | 0.000 | n/a | 0.000 | 0.000 | 0.170 |
