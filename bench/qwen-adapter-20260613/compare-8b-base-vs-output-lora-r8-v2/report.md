# LLM Rollout Compare Report

Deterministic offline comparison for committed rollout artifacts. This checks
quality scores, prompt coverage, output equality, and eval-loop timing sanity.
It is not an LLM-as-judge or broad capability benchmark.

## Artifacts

| label | mode | prompts | quality | generated | tok/s | dataset |
|---|---|---:|---:|---:|---:|---|
| `8b-sentinel-base-rollout` | `generated` | 12 | `fail` 0/12 | 12 | 0.52 | `bench/qwen-adapter-20260613/training-data-v2/eval-prompts.jsonl` |
| `8b-output-lora-r8-v2-rollout` | `generated` | 12 | `pass` 12/12 | 12 | 0.50 | `bench/qwen-adapter-20260613/training-data-v2/eval-prompts.jsonl` |

## Comparisons

### `8b-sentinel-base-rollout` vs `8b-output-lora-r8-v2-rollout`

- quality delta: `12` (0 -> 12)
- regressions: `0`
- improvements: `12`
- text changed: `12/12`
- tokens changed: `12/12`
- generated token delta: `0`
- eval-loop tok/s ratio: `0.9685`

| tag | baseline passed | candidate passed | delta | scored |
|---|---:|---:|---:|---:|
| `eval` | 0 | 12 | 12 | 12 |
| `sentinel_override` | 0 | 12 | 12 | 12 |

| id | quality | tokens equal | text |
|---|---|---:|---|
| `sentinel_005` | `fail -> pass` | `False` | OK |
| `sentinel_010` | `fail -> pass` | `False` | OK |
| `sentinel_015` | `fail -> pass` | `False` | OK |
| `sentinel_020` | `fail -> pass` | `False` | OK |
| `sentinel_025` | `fail -> pass` | `False` | OK |
| `sentinel_030` | `fail -> pass` | `False` | OK |
| `sentinel_035` | `fail -> pass` | `False` | OK |
| `sentinel_040` | `fail -> pass` | `False` | OK |
| `sentinel_045` | `fail -> pass` | `False` | OK |
| `sentinel_050` | `fail -> pass` | `False` | OK |
| `sentinel_055` | `fail -> pass` | `False` | OK |
| `sentinel_060` | `fail -> pass` | `False` | OK |
