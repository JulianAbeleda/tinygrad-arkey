# Adapter JSON Dataset V4

This is the first strict-JSON generation eval large enough to act as a
promotion gate. It keeps the task deterministic: each prompt expects compact
JSON with exactly one key, `answer`, and every answer is scored by the
multi-axis JSON scorer.

- SFT rows: `612`
- train rows: `408`
- held-out eval rows: `204`
- categories: `arithmetic`, `fact`, `code`, `compiler`, `string`, `categorization`
- schema: `{"answer": ...}` with strings and integers
- disjointness: train/eval prompts, answers, and template instances are mechanically checked

The categorization prompts use binary-choice labels rather than raw JSON
booleans so the train/eval answer sets can remain disjoint.

## Category Balance

| category | train | eval |
|---|---:|---:|
| `arithmetic` | 68 | 34 |
| `fact` | 68 | 34 |
| `code` | 68 | 34 |
| `compiler` | 68 | 34 |
| `string` | 68 | 34 |
| `categorization` | 68 | 34 |
