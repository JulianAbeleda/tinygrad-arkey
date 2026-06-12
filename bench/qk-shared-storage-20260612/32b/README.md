# QK Policy Pipeline: Qwen3-32B-Q4_K_M.gguf

Date: 2026-06-12

- commit: `0da79f8ac`
- device: `AMD`
- model size: `32B`
- reference mode: `explicit`
- generated policy: `policy.json`

## Decision

- status: `accept`
- gain: `54.56%`
- explicit mean: `11.15 tok/s`
- explicit decision window: `explicit1, explicit2, explicit3`
- generated mean: `17.23 tok/s`
- generated decision window: `generated1, generated2, generated3`
- generated percent of llama.cpp reference: `55.9%`

Reasons:

- generated beats explicit by 54.56%

## Reproduction

```sh
DEV=AMD JIT=1 QK_GENERATED_POLICY=bench/qk-shared-storage-20260612/32b/policy.json PYTHONPATH=. \
  .venv/bin/python -m tinygrad.llm --model /home/ubuntu/models/Qwen3-32B-Q4_K_M.gguf --warmup --benchmark 128
```

## Artifacts

- `search.json`, `policy.json`, `semantic-report.md`
- `policy-parity.json`, `policy-parity.md`
- `decode-summary.json`, `decode-summary.md`
- `output-ab.json`, `output-ab.log`
- `profile-report.json`, `profile-report.md`

## Decode Summary

# QK Decode Summary

| label | samples | avg tok/s | drop1 | last64 | last32 | last16 | stdev | min | max | Q4 install | Q6 install | storage MB | policy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `explicit1` | 128 | 11.29 | 11.35 | 11.42 | 11.29 | 11.17 | 1.06 | 3.17 | 11.97 | 288 | 32 | 0.00 | `` |
| `explicit2` | 128 | 10.94 | 11.00 | 11.11 | 10.92 | 10.90 | 1.66 | 3.12 | 11.91 | 288 | 32 | 0.00 | `` |
| `explicit3` | 128 | 11.21 | 11.27 | 11.27 | 11.39 | 11.39 | 1.13 | 2.96 | 11.95 | 288 | 32 | 0.00 | `` |
| `generated1` | 128 | 17.19 | 17.30 | 16.89 | 17.37 | 17.30 | 1.71 | 3.13 | 18.54 | 384 | 64 | 0.00 | `bench/qk-shared-storage-20260612/32b/policy.json` |
| `generated2` | 128 | 17.55 | 17.67 | 17.30 | 17.15 | 17.25 | 1.44 | 3.13 | 18.41 | 384 | 64 | 0.00 | `bench/qk-shared-storage-20260612/32b/policy.json` |
| `generated3` | 128 | 16.94 | 17.05 | 17.50 | 17.35 | 17.28 | 2.54 | 3.07 | 18.58 | 384 | 64 | 0.00 | `bench/qk-shared-storage-20260612/32b/policy.json` |


## Policy Parity Summary

```json
{
  "by_format": {
    "Q4_K": 385,
    "Q6_K": 65
  },
  "effective_mismatches": 256,
  "explicit_installed": 320,
  "explicit_reasons": {
    "policy_fallback": 130,
    "policy_primitive": 320
  },
  "generated_installed": 448,
  "generated_reasons": {
    "policy_fused": 1,
    "policy_missing": 1,
    "policy_primitive": 448
  },
  "generated_unsupported": 0,
  "raw_differences": 322,
  "same_effective": 194,
  "same_raw": 128,
  "total": 450
}
```


## Storage Policy

```json
{
  "by_format": {
    "Q4_K": 197591040,
    "Q6_K": 111820800
  },
  "cap_bytes": null,
  "mode": "uncapped_shape",
  "note": "shape-scoped entries multiply at runtime by tensor count",
  "selected_bytes": 309411840,
  "selected_entries": 8,
  "selected_primitive_entries": 7
}
```
