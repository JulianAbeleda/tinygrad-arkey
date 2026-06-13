# QK Policy Pipeline: Qwen3-32B-Q4_K_M.gguf

Date: 2026-06-12

- commit: `437f77772`
- device: `AMD`
- model size: `32B`
- reference mode: `policy`
- generated policy: `policy.json`

## Decision

- status: `tie`
- gain: `-2.29%`
- reference policy mean: `17.38 tok/s`
- reference policy decision window: `explicit1, explicit2, explicit3`
- generated mean: `16.98 tok/s`
- generated decision window: `generated1, generated2, generated3`
- generated percent of llama.cpp reference: `55.1%`

Reasons:

- generated within tie band: -2.29%

## Reproduction

```sh
DEV=AMD JIT=1 QK_GENERATED_POLICY=bench/qk-ansor-transition-20260612/benchmarks/32b-confirm/001-ffn-gate-local32-p1/policy.json PYTHONPATH=. \
  .venv/bin/python -m tinygrad.llm --model /home/ubuntu/models/Qwen3-32B-Q4_K_M.gguf --warmup --benchmark 128
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
| `explicit1` | 128 | 17.43 | 17.54 | 16.98 | 16.45 | 16.13 | 1.84 | 3.10 | 18.50 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `explicit2` | 128 | 17.67 | 17.79 | 17.44 | 17.23 | 17.19 | 1.37 | 3.15 | 18.49 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `explicit3` | 128 | 17.03 | 17.14 | 16.25 | 15.08 | 17.27 | 2.36 | 3.07 | 18.50 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `generated1` | 128 | 16.41 | 16.51 | 14.97 | 17.34 | 17.27 | 3.46 | 3.12 | 18.52 | 384 | 64 | 0.00 | `bench/qk-ansor-transition-20260612/benchmarks/32b-confirm/001-ffn-gate-local32-p1/policy.json` |
| `generated2` | 128 | 16.94 | 17.05 | 16.16 | 16.79 | 17.22 | 2.72 | 3.16 | 18.62 | 384 | 64 | 0.00 | `bench/qk-ansor-transition-20260612/benchmarks/32b-confirm/001-ffn-gate-local32-p1/policy.json` |
| `generated3` | 128 | 17.59 | 17.71 | 17.44 | 17.29 | 17.25 | 1.40 | 3.06 | 18.58 | 384 | 64 | 0.00 | `bench/qk-ansor-transition-20260612/benchmarks/32b-confirm/001-ffn-gate-local32-p1/policy.json` |


## Policy Parity Summary

```json
{
  "by_format": {
    "Q4_K": 385,
    "Q6_K": 65
  },
  "effective_mismatches": 384,
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
  "raw_differences": 450,
  "same_effective": 66,
  "same_raw": 0,
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
