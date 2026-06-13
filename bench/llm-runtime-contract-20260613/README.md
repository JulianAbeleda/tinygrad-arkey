# LLM Runtime Contract

This report validates the artifact contract for generated-policy rollout/eval
work. Model files may remain local, but committed datasets, policies, and
evidence artifacts must be present and internally consistent.

| id | type | status | artifact | errors |
|---|---|---:|---|---|
| `qwen3-8b-eval` | `eval` | `pass` | `bench/qwen-eval-20260612/8b-shared` |  |
| `qwen3-14b-eval` | `eval` | `pass` | `bench/qwen-eval-20260612/14b-shared` |  |
| `qwen3-8b-rollout-generated` | `rollout` | `pass` | `bench/qwen-rollout-20260612/8b-generated-small` |  |
| `qwen3-14b-rollout-generated` | `rollout` | `pass` | `bench/qwen-rollout-20260612/14b-generated-small` |  |
| `qwen3-8b-generated-vs-explicit` | `compare` | `pass` | `bench/qwen-rollout-20260612/compare-8b-small` |  |
| `qwen3-14b-generated-vs-explicit` | `compare` | `pass` | `bench/qwen-rollout-20260612/compare-14b-small` |  |
| `qwen3-rollout-training-data-v1` | `training_data` | `pass` | `bench/qwen-rollout-20260612/training-data-v1` |  |
| `qwen3-sft-smoke-v1` | `training_run` | `pass` | `bench/qwen-rollout-20260612/sft-smoke-v1` |  |

## Summary

```json
{
  "failed": 0,
  "missing": 0,
  "passed": 8,
  "rows": 8
}
```
