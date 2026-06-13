# LLM Rollout Summary

This is a dataset-style rollout artifact. It is useful for practical eval,
future SFT/RLVR data generation, and compiler/search behavior gates. Timing
is eval-loop sanity data, not the canonical decode benchmark.

## Summary

- mode: `generated`
- model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`
- policy: `bench/qk-shared-storage-20260612/8b/policy.json`
- adapter: `bench/qwen-adapter-20260613/8b-output-lora-r8-v2`
- dataset: `bench/qwen-adapter-20260613/training-data-v2/eval-prompts.jsonl`
- storage: `shared`
- prompt format: `chat`
- prompts: `12`
- generated tokens: `12`
- quality: `pass` (12/12)

| id | tags | quality | generated | tok/s | text |
|---|---|---:|---:|---:|---|
| `sentinel_005` | `sentinel_override,eval` | `pass` | 1 | 0.05 | OK |
| `sentinel_010` | `sentinel_override,eval` | `pass` | 1 | 2.44 | OK |
| `sentinel_015` | `sentinel_override,eval` | `pass` | 1 | 2.74 | OK |
| `sentinel_020` | `sentinel_override,eval` | `pass` | 1 | 2.54 | OK |
| `sentinel_025` | `sentinel_override,eval` | `pass` | 1 | 2.43 | OK |
| `sentinel_030` | `sentinel_override,eval` | `pass` | 1 | 2.43 | OK |
| `sentinel_035` | `sentinel_override,eval` | `pass` | 1 | 2.62 | OK |
| `sentinel_040` | `sentinel_override,eval` | `pass` | 1 | 2.53 | OK |
| `sentinel_045` | `sentinel_override,eval` | `pass` | 1 | 2.05 | OK |
| `sentinel_050` | `sentinel_override,eval` | `pass` | 1 | 2.77 | OK |
| `sentinel_055` | `sentinel_override,eval` | `pass` | 1 | 2.53 | OK |
| `sentinel_060` | `sentinel_override,eval` | `pass` | 1 | 2.89 | OK |

## Quality By Tag

| tag | passed | scored | pass rate |
|---|---:|---:|---:|
| `eval` | 12 | 12 | 1.00 |
| `sentinel_override` | 12 | 12 | 1.00 |
