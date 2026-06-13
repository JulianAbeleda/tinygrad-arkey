# QK Policy Pipeline: Qwen3-8B-Q4_K_M.gguf

Date: 2026-06-12

- commit: `437f77772`
- device: `AMD`
- model size: `8B`
- reference mode: `policy`
- generated policy: `policy.json`

## Decision

- status: `tie`
- gain: `-0.99%`
- reference policy mean: `51.58 tok/s`
- reference policy decision window: `explicit1, explicit2, explicit3`
- generated mean: `51.07 tok/s`
- generated decision window: `generated1, generated2, generated3`
- generated percent of llama.cpp reference: `50.5%`

Reasons:

- generated within tie band: -0.99%

## Reproduction

```sh
DEV=AMD JIT=1 QK_GENERATED_POLICY=/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/006-ffn-down-blk-4-ffn-down-weight-q4_k_packed_u32-p1-local32/policy.json PYTHONPATH=. \
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
| `explicit1` | 128 | 53.61 | 53.95 | 52.40 | 51.37 | 51.18 | 4.46 | 9.82 | 57.74 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `explicit2` | 128 | 51.42 | 51.75 | 47.61 | 51.83 | 51.42 | 8.19 | 9.48 | 57.88 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `explicit3` | 128 | 49.71 | 50.02 | 47.25 | 41.34 | 51.50 | 10.30 | 9.93 | 57.76 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `generated1` | 128 | 51.69 | 52.01 | 50.56 | 49.84 | 49.49 | 4.13 | 9.87 | 55.58 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/006-ffn-down-blk-4-ffn-down-weight-q4_k_packed_u32-p1-local32/policy.json` |
| `generated2` | 128 | 51.37 | 51.70 | 49.96 | 49.49 | 49.56 | 4.46 | 9.76 | 55.40 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/006-ffn-down-blk-4-ffn-down-weight-q4_k_packed_u32-p1-local32/policy.json` |
| `generated3` | 128 | 50.14 | 50.48 | 48.35 | 49.47 | 49.13 | 6.21 | 7.55 | 55.11 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/006-ffn-down-blk-4-ffn-down-weight-q4_k_packed_u32-p1-local32/policy.json` |


## Policy Parity Summary

```json
{
  "by_format": {
    "Q4_K": 217,
    "Q6_K": 37
  },
  "effective_mismatches": 36,
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
  "raw_differences": 182,
  "same_effective": 218,
  "same_raw": 72,
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
