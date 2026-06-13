# QK Policy Pipeline: Qwen3-8B-Q4_K_M.gguf

Date: 2026-06-12

- commit: `437f77772`
- device: `AMD`
- model size: `8B`
- reference mode: `policy`
- generated policy: `policy.json`

## Decision

- status: `tie`
- gain: `-1.33%`
- reference policy mean: `53.27 tok/s`
- reference policy decision window: `explicit1, explicit2, explicit3`
- generated mean: `52.57 tok/s`
- generated decision window: `generated1, generated2, generated3`
- generated percent of llama.cpp reference: `51.9%`

Reasons:

- generated within tie band: -1.33%

## Reproduction

```sh
DEV=AMD JIT=1 QK_GENERATED_POLICY=/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/001-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p1-local32/policy.json PYTHONPATH=. \
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
| `explicit1` | 128 | 53.39 | 53.73 | 51.96 | 51.30 | 51.24 | 4.64 | 9.84 | 57.42 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `explicit2` | 128 | 53.13 | 53.47 | 51.99 | 51.00 | 50.09 | 4.70 | 9.86 | 57.81 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `explicit3` | 128 | 53.30 | 53.64 | 51.92 | 50.95 | 50.26 | 4.80 | 9.80 | 57.54 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `generated1` | 128 | 52.48 | 52.82 | 51.17 | 50.78 | 50.55 | 4.68 | 9.84 | 57.06 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/001-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p1-local32/policy.json` |
| `generated2` | 128 | 52.71 | 53.05 | 51.27 | 50.72 | 50.05 | 4.56 | 9.80 | 57.14 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/001-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p1-local32/policy.json` |
| `generated3` | 128 | 52.51 | 52.84 | 51.12 | 50.15 | 50.56 | 5.01 | 9.63 | 56.71 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/001-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p1-local32/policy.json` |


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
  "raw_differences": 236,
  "same_effective": 164,
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
