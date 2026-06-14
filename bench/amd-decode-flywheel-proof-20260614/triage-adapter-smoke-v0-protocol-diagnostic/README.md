# QK Flywheel Protocol Diagnostic

This is a Phase 3.0 diagnostic over the existing no-adapter rollout. It
separates strict-output failure from label/reason triage quality. It is not
a promotion artifact and does not replace the strict Phase 2 score.

- conclusion: `protocol_fix_not_enough`
- rows: `38`
- baseline macro-F1: `0.185`
- best diagnostic: `json_extract`

| method | parse | schema | taxonomy | accuracy | macro-F1 | false accept | p@3 | ndcg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `strict_text` | 0/38 | 0/38 | 0/38 | 0.000 | 0.000 | 0.000 | 0.000 | 0.170 |
| `json_extract` | 38/38 | 38/38 | 15/38 | 0.053 | 0.036 | 0.763 | 0.000 | 0.173 |
| `json_extract_taxonomy_repair` | 38/38 | 38/38 | 38/38 | 0.053 | 0.036 | 0.763 | 0.000 | 0.173 |
