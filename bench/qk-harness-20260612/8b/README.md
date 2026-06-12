# QK Policy Pipeline: Qwen3-8B-Q4_K_M.gguf

Date: 2026-06-12

- commit: `98f51986d`
- device: `AMD`
- model size: `8B`
- reference mode: `explicit`
- generated policy: `policy.json`

## Decision

- status: `accept`
- gain: `8.41%`
- explicit mean: `49.35 tok/s`
- explicit decision window: `explicit1, explicit2, explicit3`
- generated mean: `53.49 tok/s`
- generated decision window: `generated1, generated2, generated3`
- generated percent of llama.cpp reference: `52.9%`

Reasons:

- generated beats explicit by 8.41%

## Reproduction

```sh
DEV=AMD JIT=1 QK_GENERATED_POLICY=bench/qk-harness-20260612/8b/policy.json PYTHONPATH=. \
  .venv/bin/python -m tinygrad.llm --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --warmup --benchmark 128
```

## Artifacts

- `search.json`, `policy.json`, `semantic-report.md`
- `policy-parity.json`, `policy-parity.md`
- `decode-summary.json`, `decode-summary.md`
- `output-ab.json`, `output-ab.log`

## Decode Summary

# QK Decode Summary

| label | samples | avg tok/s | drop1 | last64 | last32 | last16 | stdev | min | max | Q4 install | Q6 install | storage MB | policy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `explicit1` | 128 | 47.39 | 47.69 | 50.10 | 49.39 | 49.19 | 5.71 | 9.40 | 54.41 | 162 | 18 | 3786.75 | `` |
| `explicit2` | 128 | 50.00 | 50.32 | 49.33 | 48.62 | 48.93 | 5.15 | 9.34 | 54.38 | 162 | 18 | 3786.75 | `` |
| `explicit3` | 128 | 50.65 | 50.97 | 50.38 | 49.68 | 49.35 | 5.26 | 9.89 | 54.76 | 162 | 18 | 3786.75 | `` |
| `generated1` | 128 | 54.13 | 54.48 | 52.52 | 51.23 | 50.97 | 4.75 | 9.68 | 58.15 | 162 | 18 | 3786.75 | `bench/qk-harness-20260612/8b/policy.json` |
| `generated2` | 128 | 52.28 | 52.61 | 49.04 | 47.88 | 51.85 | 7.15 | 9.92 | 58.07 | 162 | 18 | 3786.75 | `bench/qk-harness-20260612/8b/policy.json` |
| `generated3` | 128 | 54.07 | 54.42 | 52.84 | 52.28 | 51.92 | 4.54 | 9.86 | 58.27 | 162 | 18 | 3786.75 | `bench/qk-harness-20260612/8b/policy.json` |


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
  "raw_differences": 236,
  "same_effective": 236,
  "same_raw": 18,
  "total": 254
}
```


## Storage Policy

```json
{
  "by_format": {
    "Q4_K": 66060288,
    "Q6_K": 41287680
  },
  "cap_bytes": null,
  "mode": "uncapped_shape",
  "note": "shape-scoped entries multiply at runtime by tensor count",
  "selected_bytes": 107347968,
  "selected_entries": 7,
  "selected_primitive_entries": 4
}
```
