# LLM Rollout Summary

This is a dataset-style rollout artifact. It is useful for practical eval,
future SFT/RLVR data generation, and compiler/search behavior gates. Timing
is eval-loop sanity data, not the canonical decode benchmark.

## Summary

- mode: `generated`
- model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`
- policy: `bench/qk-shared-storage-20260612/8b/policy.json`
- dataset: `bench/qwen-adapter-20260613/training-data-v2/eval-prompts.jsonl`
- storage: `shared`
- prompt format: `chat`
- prompts: `12`
- generated tokens: `12`
- quality: `fail` (0/12)

| id | tags | quality | generated | tok/s | text |
|---|---|---:|---:|---:|---|
| `sentinel_005` | `sentinel_override,eval` | `fail` | 1 | 0.05 | <think> |
| `sentinel_010` | `sentinel_override,eval` | `fail` | 1 | 2.86 | <think> |
| `sentinel_015` | `sentinel_override,eval` | `fail` | 1 | 3.36 | <think> |
| `sentinel_020` | `sentinel_override,eval` | `fail` | 1 | 3.05 | <think> |
| `sentinel_025` | `sentinel_override,eval` | `fail` | 1 | 2.87 | <think> |
| `sentinel_030` | `sentinel_override,eval` | `fail` | 1 | 2.86 | <think> |
| `sentinel_035` | `sentinel_override,eval` | `fail` | 1 | 3.19 | <think> |
| `sentinel_040` | `sentinel_override,eval` | `fail` | 1 | 3.05 | <think> |
| `sentinel_045` | `sentinel_override,eval` | `fail` | 1 | 2.39 | <think> |
| `sentinel_050` | `sentinel_override,eval` | `fail` | 1 | 3.35 | <think> |
| `sentinel_055` | `sentinel_override,eval` | `fail` | 1 | 3.03 | <think> |
| `sentinel_060` | `sentinel_override,eval` | `fail` | 1 | 3.61 | <think> |

## Quality By Tag

| tag | passed | scored | pass rate |
|---|---:|---:|---:|
| `eval` | 0 | 12 | 0.00 |
| `sentinel_override` | 0 | 12 | 0.00 |
