# QK Policy Pipeline: Qwen3-8B-Q4_K_M.gguf

Date: 2026-06-12

- commit: `aa2827350`
- device: `AMD`
- model size: `8B`
- reference mode: `explicit`
- generated policy: `policy.json`

## Decision

- status: `accept`
- gain: `3.31%`
- explicit mean: `50.41 tok/s`
- explicit decision window: `explicit1, explicit2, explicit3`
- generated mean: `52.07 tok/s`
- generated decision window: `generated3, generated4, generated5`
- generated percent of llama.cpp reference: `51.5%`

Reasons:

- generated beats explicit by 3.31%

## Reproduction

```sh
DEV=AMD JIT=1 QK_GENERATED_POLICY=bench/qk-shared-storage-20260612/8b/policy.json PYTHONPATH=. \
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
| `explicit1` | 128 | 49.98 | 50.30 | 47.97 | 46.96 | 48.39 | 5.06 | 9.89 | 54.37 | 162 | 18 | 0.00 | `` |
| `explicit2` | 128 | 50.81 | 51.14 | 49.70 | 49.03 | 48.72 | 4.13 | 8.90 | 54.16 | 162 | 18 | 0.00 | `` |
| `explicit3` | 128 | 50.43 | 50.75 | 49.37 | 48.67 | 48.59 | 4.28 | 9.90 | 54.20 | 162 | 18 | 0.00 | `` |
| `generated1` | 128 | 49.63 | 49.94 | 49.73 | 46.16 | 41.42 | 8.77 | 9.87 | 57.69 | 162 | 18 | 0.00 | `bench/qk-shared-storage-20260612/8b/policy.json` |
| `generated2` | 128 | 48.76 | 49.06 | 42.42 | 31.65 | 27.85 | 11.84 | 9.92 | 57.54 | 162 | 18 | 0.00 | `bench/qk-shared-storage-20260612/8b/policy.json` |
| `generated3` | 128 | 53.74 | 54.08 | 52.67 | 51.85 | 51.52 | 4.35 | 9.93 | 57.98 | 162 | 18 | 0.00 | `bench/qk-shared-storage-20260612/8b/policy.json` |
| `generated4` | 128 | 52.99 | 53.33 | 51.18 | 49.19 | 48.44 | 5.58 | 9.72 | 57.77 | 162 | 18 | 0.00 | `bench/qk-shared-storage-20260612/8b/policy.json` |
| `generated5` | 128 | 49.50 | 49.82 | 50.37 | 51.38 | 50.58 | 9.26 | 8.31 | 57.15 | 162 | 18 | 0.00 | `bench/qk-shared-storage-20260612/8b/policy.json` |


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
  "raw_differences": 164,
  "same_effective": 236,
  "same_raw": 90,
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
