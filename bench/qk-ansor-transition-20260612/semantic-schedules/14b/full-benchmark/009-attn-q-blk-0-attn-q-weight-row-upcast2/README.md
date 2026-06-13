# QK Policy Pipeline: Qwen3-14B-Q4_K_M.gguf

Date: 2026-06-12

- commit: `e0bdf948b`
- device: `AMD`
- model size: `14B`
- reference mode: `policy`
- generated policy: `policy.json`

## Decision

- status: `reject`
- gain: `-5.21%`
- reference policy mean: `38.13 tok/s`
- reference policy decision window: `explicit1, explicit2, explicit3`
- generated mean: `36.14 tok/s`
- generated decision window: `generated1, generated2, generated3`
- generated percent of llama.cpp reference: `54.9%`

Reasons:

- generated slower than explicit by 5.21%

## Reproduction

```sh
DEV=AMD JIT=1 QK_GENERATED_POLICY=bench/qk-ansor-transition-20260612/semantic-schedules/14b/full-benchmark/009-attn-q-blk-0-attn-q-weight-row-upcast2/policy.json PYTHONPATH=. \
  .venv/bin/python -m tinygrad.llm --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf --warmup --benchmark 128
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
| `explicit1` | 128 | 34.38 | 34.60 | 34.62 | 29.63 | 36.84 | 9.67 | 6.10 | 43.46 | 240 | 40 | 0.00 | `bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `explicit2` | 128 | 39.65 | 39.91 | 39.47 | 39.38 | 39.09 | 4.43 | 6.08 | 43.45 | 240 | 40 | 0.00 | `bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `explicit3` | 128 | 40.36 | 40.63 | 39.47 | 38.74 | 38.44 | 3.71 | 6.09 | 43.26 | 240 | 40 | 0.00 | `bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `generated1` | 128 | 36.44 | 36.68 | 36.00 | 35.60 | 35.34 | 3.32 | 6.07 | 38.80 | 240 | 40 | 0.00 | `bench/qk-ansor-transition-20260612/semantic-schedules/14b/full-benchmark/009-attn-q-blk-0-attn-q-weight-row-upcast2/policy.json` |
| `generated2` | 128 | 35.29 | 35.52 | 34.97 | 35.64 | 35.52 | 4.53 | 5.83 | 38.91 | 240 | 40 | 0.00 | `bench/qk-ansor-transition-20260612/semantic-schedules/14b/full-benchmark/009-attn-q-blk-0-attn-q-weight-row-upcast2/policy.json` |
| `generated3` | 128 | 36.70 | 36.94 | 36.02 | 35.60 | 35.44 | 2.93 | 6.06 | 38.81 | 240 | 40 | 0.00 | `bench/qk-ansor-transition-20260612/semantic-schedules/14b/full-benchmark/009-attn-q-blk-0-attn-q-weight-row-upcast2/policy.json` |


## Policy Parity Summary

```json
{
  "by_format": {
    "Q4_K": 241,
    "Q6_K": 41
  },
  "effective_mismatches": 280,
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
  "same_effective": 2,
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
