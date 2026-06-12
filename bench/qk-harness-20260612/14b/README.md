# QK Policy Pipeline: Qwen3-14B-Q4_K_M.gguf

Date: 2026-06-12

- commit: `98f51986d`
- device: `AMD`
- model size: `14B`
- reference mode: `explicit`
- generated policy: `policy.json`

## Decision

- status: `needs-rerun`
- gain: `52.93%`
- explicit mean: `22.72 tok/s`
- explicit decision window: `explicit1, explicit2, explicit3`
- generated mean: `34.75 tok/s`
- generated decision window: `generated3, generated4, generated5`
- generated percent of llama.cpp reference: `52.8%`

Reasons:

- generated unstable: generated3 avg 23.75 is >10% below mean 34.75

## Reproduction

```sh
DEV=AMD JIT=1 QK_GENERATED_POLICY=bench/qk-harness-20260612/14b/policy.json PYTHONPATH=. \
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
| `explicit1` | 128 | 23.50 | 23.64 | 23.32 | 23.15 | 23.05 | 1.62 | 6.08 | 24.60 | 180 | 20 | 7300.78 | `` |
| `explicit2` | 128 | 21.14 | 21.26 | 19.90 | 19.70 | 18.64 | 2.67 | 6.09 | 24.43 | 180 | 20 | 7300.78 | `` |
| `explicit3` | 128 | 23.53 | 23.67 | 23.30 | 23.13 | 23.04 | 1.62 | 6.09 | 24.52 | 180 | 20 | 7300.78 | `` |
| `generated1` | 128 | 39.38 | 39.64 | 39.42 | 38.47 | 37.98 | 3.68 | 6.08 | 43.15 | 240 | 40 | 7551.56 | `bench/qk-harness-20260612/14b/policy.json` |
| `generated2` | 128 | 33.93 | 34.15 | 34.00 | 37.85 | 37.41 | 7.34 | 6.12 | 43.25 | 240 | 40 | 7551.56 | `bench/qk-harness-20260612/14b/policy.json` |
| `generated3` | 128 | 23.75 | 23.89 | 30.13 | 39.12 | 38.69 | 10.31 | 5.93 | 40.10 | 240 | 40 | 7551.56 | `bench/qk-harness-20260612/14b/policy.json` |
| `generated4` | 128 | 39.88 | 40.15 | 38.68 | 38.50 | 38.62 | 4.04 | 6.07 | 43.20 | 240 | 40 | 7551.56 | `bench/qk-harness-20260612/14b/policy.json` |
| `generated5` | 128 | 40.62 | 40.90 | 39.74 | 39.18 | 38.92 | 3.40 | 6.07 | 43.25 | 240 | 40 | 7551.56 | `bench/qk-harness-20260612/14b/policy.json` |


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
