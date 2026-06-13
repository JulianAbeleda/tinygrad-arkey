# QK Policy Pipeline: Qwen3-8B-Q4_K_M.gguf

Date: 2026-06-12

- commit: `e0bdf948b`
- device: `AMD`
- model size: `8B`
- reference mode: `policy`
- generated policy: `policy.json`

## Decision

- status: `reject`
- gain: `-10.28%`
- reference policy mean: `53.27 tok/s`
- reference policy decision window: `explicit1, explicit2, explicit3`
- generated mean: `47.79 tok/s`
- generated decision window: `generated1, generated2, generated3`
- generated percent of llama.cpp reference: `47.2%`

Reasons:

- generated slower than explicit by 10.28%

## Reproduction

```sh
DEV=AMD JIT=1 QK_GENERATED_POLICY=bench/qk-ansor-transition-20260612/semantic-schedules/8b/full-benchmark/009-attn-q-blk-0-attn-q-weight-row-upcast2/policy.json PYTHONPATH=. \
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
| `explicit1` | 128 | 52.89 | 53.23 | 51.40 | 50.92 | 50.03 | 4.88 | 9.83 | 57.39 | 162 | 18 | 0.00 | `bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `explicit2` | 128 | 53.35 | 53.70 | 52.38 | 51.61 | 51.24 | 4.61 | 9.81 | 57.77 | 162 | 18 | 0.00 | `bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `explicit3` | 128 | 53.56 | 53.91 | 52.40 | 51.64 | 51.25 | 4.45 | 9.80 | 57.84 | 162 | 18 | 0.00 | `bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `generated1` | 128 | 47.92 | 48.22 | 47.38 | 46.61 | 46.53 | 4.36 | 9.87 | 51.90 | 162 | 18 | 0.00 | `bench/qk-ansor-transition-20260612/semantic-schedules/8b/full-benchmark/009-attn-q-blk-0-attn-q-weight-row-upcast2/policy.json` |
| `generated2` | 128 | 48.16 | 48.46 | 47.50 | 46.92 | 46.56 | 3.96 | 9.88 | 51.67 | 162 | 18 | 0.00 | `bench/qk-ansor-transition-20260612/semantic-schedules/8b/full-benchmark/009-attn-q-blk-0-attn-q-weight-row-upcast2/policy.json` |
| `generated3` | 128 | 47.30 | 47.59 | 46.90 | 46.54 | 46.82 | 4.93 | 9.77 | 51.59 | 162 | 18 | 0.00 | `bench/qk-ansor-transition-20260612/semantic-schedules/8b/full-benchmark/009-attn-q-blk-0-attn-q-weight-row-upcast2/policy.json` |


## Policy Parity Summary

```json
{
  "by_format": {
    "Q4_K": 217,
    "Q6_K": 37
  },
  "effective_mismatches": 90,
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
  "same_effective": 164,
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
