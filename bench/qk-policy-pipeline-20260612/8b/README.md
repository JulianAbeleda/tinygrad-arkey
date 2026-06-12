# QK Policy Pipeline: Qwen3-8B-Q4_K_M.gguf

Date: 2026-06-12

- commit: `a96534155`
- device: `AMD`
- model size: `8B`
- generated policy: `policy.json`

## Decision

- status: `accept`
- gain: `6.14%`
- explicit mean: `49.61 tok/s`
- explicit decision window: `explicit2, explicit3, explicit4`
- generated mean: `52.65 tok/s`
- generated decision window: `generated3, generated4, generated5`
- generated percent of llama.cpp reference: `52.0%`

Reasons:

- generated beats explicit by 6.14%

## Reproduction

```sh
DEV=AMD JIT=1 QK_GENERATED_POLICY=bench/qk-policy-pipeline-20260612/8b/policy.json PYTHONPATH=. \
  .venv/bin/python -m tinygrad.llm --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --warmup --benchmark 128
```

## Artifacts

- `search.json`, `policy.json`, `semantic-report.md`
- `policy-parity.json`, `policy-parity.md`
- `decode-summary.json`, `decode-summary.md`
- `output-ab.json`, `output-ab.log`

## Decode Summary

# QK Decode Summary

| label | samples | avg tok/s | drop1 | last64 | last32 | last16 | stdev | min | max | Q4 install | Q6 install | policy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `explicit1` | 128 | 49.36 | 49.67 | 47.26 | 44.30 | 41.73 | 6.69 | 9.90 | 54.92 | 162 | 18 | `` |
| `explicit2` | 128 | 49.17 | 49.48 | 48.08 | 45.18 | 49.24 | 7.61 | 9.71 | 54.54 | 162 | 18 | `` |
| `explicit3` | 128 | 50.45 | 50.78 | 50.03 | 49.36 | 49.04 | 4.43 | 9.73 | 54.79 | 162 | 18 | `` |
| `explicit4` | 128 | 49.20 | 49.51 | 47.57 | 49.54 | 49.16 | 7.19 | 9.78 | 54.69 | 162 | 18 | `` |
| `generated1` | 128 | 53.63 | 53.97 | 51.48 | 52.50 | 52.09 | 5.33 | 9.92 | 58.29 | 162 | 18 | `bench/qk-policy-pipeline-20260612/8b/policy.json` |
| `generated2` | 128 | 48.06 | 48.36 | 40.77 | 52.36 | 52.00 | 13.32 | 9.97 | 58.39 | 162 | 18 | `bench/qk-policy-pipeline-20260612/8b/policy.json` |
| `generated3` | 128 | 52.50 | 52.84 | 49.13 | 52.47 | 52.06 | 9.00 | 9.91 | 58.57 | 162 | 18 | `bench/qk-policy-pipeline-20260612/8b/policy.json` |
| `generated4` | 128 | 53.46 | 53.80 | 52.70 | 51.99 | 51.31 | 5.38 | 9.90 | 58.17 | 162 | 18 | `bench/qk-policy-pipeline-20260612/8b/policy.json` |
| `generated5` | 128 | 52.00 | 52.34 | 52.91 | 52.02 | 51.37 | 7.15 | 9.77 | 58.34 | 162 | 18 | `bench/qk-policy-pipeline-20260612/8b/policy.json` |


## Policy Parity Summary

```json
{
  "by_format": {
    "Q4_K": 217,
    "Q6_K": 37
  },
  "effective_mismatches": 18,
  "explicit_installed": 180,
  "explicit_reasons": {
    "policy_fallback": 74,
    "policy_primitive": 180
  },
  "generated_installed": 180,
  "generated_reasons": {
    "policy_fused": 73,
    "policy_missing": 1,
    "policy_primitive": 180
  },
  "generated_unsupported": 0,
  "raw_differences": 254,
  "same_effective": 236,
  "same_raw": 0,
  "total": 254
}
```
