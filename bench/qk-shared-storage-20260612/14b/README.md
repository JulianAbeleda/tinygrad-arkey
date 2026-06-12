# QK Policy Pipeline: Qwen3-14B-Q4_K_M.gguf

Date: 2026-06-12

- commit: `aa2827350`
- device: `AMD`
- model size: `14B`
- reference mode: `explicit`
- generated policy: `policy.json`

## Decision

- status: `accept`
- gain: `86.29%`
- explicit mean: `21.77 tok/s`
- explicit decision window: `explicit3, explicit4, explicit5`
- generated mean: `40.55 tok/s`
- generated decision window: `generated5, generated6, generated7`
- generated percent of llama.cpp reference: `61.6%`

Reasons:

- generated beats explicit by 86.29%

## Reproduction

```sh
DEV=AMD JIT=1 QK_GENERATED_POLICY=bench/qk-shared-storage-20260612/14b/policy.json PYTHONPATH=. \
  .venv/bin/python -m tinygrad.llm --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf --warmup --benchmark 128
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
| `explicit1` | 128 | 23.10 | 23.23 | 22.98 | 22.54 | 21.85 | 1.83 | 6.02 | 24.59 | 180 | 20 | 0.00 | `` |
| `explicit2` | 128 | 20.23 | 20.34 | 16.84 | 17.39 | 21.26 | 4.81 | 5.95 | 24.45 | 180 | 20 | 0.00 | `` |
| `explicit3` | 128 | 22.74 | 22.87 | 22.93 | 23.00 | 23.03 | 2.31 | 5.88 | 24.46 | 180 | 20 | 0.00 | `` |
| `explicit4` | 128 | 22.72 | 22.85 | 22.00 | 23.13 | 23.05 | 2.43 | 6.10 | 24.42 | 180 | 20 | 0.00 | `` |
| `explicit5` | 128 | 19.85 | 19.96 | 18.60 | 19.38 | 19.12 | 4.83 | 6.04 | 24.48 | 180 | 20 | 0.00 | `` |
| `generated1` | 128 | 37.22 | 37.47 | 33.53 | 27.83 | 29.97 | 8.61 | 6.06 | 43.31 | 240 | 40 | 0.00 | `bench/qk-shared-storage-20260612/14b/policy.json` |
| `generated2` | 128 | 40.29 | 40.56 | 39.87 | 39.42 | 39.17 | 3.98 | 6.04 | 43.60 | 240 | 40 | 0.00 | `bench/qk-shared-storage-20260612/14b/policy.json` |
| `generated3` | 128 | 40.70 | 40.98 | 39.95 | 39.44 | 39.23 | 3.49 | 6.07 | 43.44 | 240 | 40 | 0.00 | `bench/qk-shared-storage-20260612/14b/policy.json` |
| `generated4` | 128 | 33.96 | 34.19 | 39.91 | 39.36 | 39.01 | 10.41 | 5.08 | 42.06 | 240 | 40 | 0.00 | `bench/qk-shared-storage-20260612/14b/policy.json` |
| `generated5` | 128 | 40.67 | 40.95 | 39.77 | 39.06 | 38.48 | 3.52 | 6.06 | 43.40 | 240 | 40 | 0.00 | `bench/qk-shared-storage-20260612/14b/policy.json` |
| `generated6` | 128 | 40.21 | 40.48 | 39.53 | 38.86 | 38.43 | 3.50 | 6.02 | 43.25 | 240 | 40 | 0.00 | `bench/qk-shared-storage-20260612/14b/policy.json` |
| `generated7` | 128 | 40.77 | 41.05 | 39.95 | 39.37 | 38.98 | 3.45 | 5.16 | 43.42 | 240 | 40 | 0.00 | `bench/qk-shared-storage-20260612/14b/policy.json` |


## Policy Parity Summary

```json
{
  "by_format": {
    "Q4_K": 241,
    "Q6_K": 41
  },
  "effective_mismatches": 200,
  "explicit_installed": 200,
  "explicit_reasons": {
    "policy_fallback": 82,
    "policy_primitive": 200
  },
  "generated_installed": 280,
  "generated_reasons": {
    "policy_fused": 1,
    "policy_missing": 1,
    "policy_primitive": 280
  },
  "generated_unsupported": 0,
  "raw_differences": 202,
  "same_effective": 82,
  "same_raw": 80,
  "total": 282
}
```


## Storage Policy

```json
{
  "by_format": {
    "Q4_K": 117964800,
    "Q6_K": 77414400
  },
  "cap_bytes": null,
  "mode": "uncapped_shape",
  "note": "shape-scoped entries multiply at runtime by tensor count",
  "selected_bytes": 195379200,
  "selected_entries": 7,
  "selected_primitive_entries": 6
}
```
