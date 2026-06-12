# Qwen Eval Matrix

This matrix is the shared gate for three compounding goals:
fast local inference, future training/rollout/eval loops, and compiler/search
experiments. Exact token parity is the primitive correctness gate; prompt
quality is reported separately.

| id | model | parity | quality | score | explicit tok/s | generated tok/s | tokens | out |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `8b-shared` | `8B` | `pass` | `pass` | 10/10 | 3.55 | 3.57 | 109 | `bench/qwen-eval-20260612/8b-shared` |
| `14b-shared` | `14B` | `pass` | `pass` | 10/10 | 2.58 | 2.73 | 103 | `bench/qwen-eval-20260612/14b-shared` |

## Summary

```json
{
  "parity_passed": 2,
  "quality_passed_rows": 2,
  "quality_scored_rows": 2,
  "rows": 2
}
```
