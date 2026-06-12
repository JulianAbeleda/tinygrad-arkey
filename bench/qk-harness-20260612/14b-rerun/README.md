# QK Policy Pipeline: Qwen3-14B-Q4_K_M.gguf

Date: 2026-06-12

- commit: `65d2bf5a7`
- device: `AMD`
- model size: `14B`
- reference mode: `explicit`
- generated policy: `policy.json`

## Decision

- status: `accept`
- gain: `74.02%`
- explicit mean: `22.76 tok/s`
- explicit decision window: `explicit1, explicit2, explicit3`
- generated mean: `39.61 tok/s`
- generated decision window: `generated1, generated2, generated3`
- generated percent of llama.cpp reference: `60.2%`

Reasons:

- generated beats explicit by 74.02%

## Reproduction

```sh
DEV=AMD JIT=1 QK_GENERATED_POLICY=bench/qk-harness-20260612/14b-rerun/policy.json PYTHONPATH=. \
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
| `explicit1` | 128 | 23.37 | 23.51 | 23.03 | 22.59 | 22.01 | 1.86 | 6.06 | 24.51 | 180 | 20 | 7300.78 | `` |
| `explicit2` | 128 | 21.62 | 21.75 | 22.53 | 21.67 | 20.36 | 3.53 | 5.44 | 24.31 | 180 | 20 | 7300.78 | `` |
| `explicit3` | 128 | 23.30 | 23.44 | 23.21 | 23.13 | 23.05 | 1.77 | 6.10 | 24.61 | 180 | 20 | 7300.78 | `` |
| `generated1` | 128 | 38.62 | 38.88 | 39.43 | 38.78 | 38.16 | 5.12 | 6.09 | 42.71 | 240 | 40 | 7551.56 | `bench/qk-harness-20260612/14b-rerun/policy.json` |
| `generated2` | 128 | 40.50 | 40.77 | 39.46 | 39.17 | 38.69 | 3.47 | 6.09 | 43.13 | 240 | 40 | 7551.56 | `bench/qk-harness-20260612/14b-rerun/policy.json` |
| `generated3` | 128 | 39.73 | 39.99 | 39.03 | 37.89 | 36.23 | 4.63 | 6.05 | 43.19 | 240 | 40 | 7551.56 | `bench/qk-harness-20260612/14b-rerun/policy.json` |


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
  "raw_differences": 282,
  "same_effective": 82,
  "same_raw": 0,
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
