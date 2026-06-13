# QK Policy Pipeline: Qwen3-8B-Q4_K_M.gguf

Date: 2026-06-12

- commit: `437f77772`
- device: `AMD`
- model size: `8B`
- reference mode: `policy`
- generated policy: `policy.json`

## Decision

- status: `reject`
- gain: `-10.77%`
- reference policy mean: `53.22 tok/s`
- reference policy decision window: `explicit1, explicit2, explicit3`
- generated mean: `47.49 tok/s`
- generated decision window: `generated1, generated2, generated3`
- generated percent of llama.cpp reference: `46.9%`

Reasons:

- generated slower than explicit by 10.77%

## Reproduction

```sh
DEV=AMD JIT=1 QK_GENERATED_POLICY=/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/002-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local32/policy.json PYTHONPATH=. \
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
| `explicit1` | 128 | 53.23 | 53.57 | 52.02 | 51.49 | 51.27 | 4.90 | 9.81 | 57.99 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `explicit2` | 128 | 53.38 | 53.73 | 51.89 | 51.08 | 50.94 | 4.61 | 9.87 | 57.51 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `explicit3` | 128 | 53.05 | 53.39 | 52.48 | 51.70 | 51.33 | 5.22 | 9.82 | 57.45 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `generated1` | 128 | 46.85 | 47.15 | 47.97 | 47.30 | 47.14 | 6.98 | 8.43 | 52.76 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/002-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local32/policy.json` |
| `generated2` | 128 | 48.91 | 49.22 | 47.74 | 47.42 | 47.30 | 4.39 | 9.83 | 52.62 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/002-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local32/policy.json` |
| `generated3` | 128 | 46.70 | 46.99 | 47.83 | 46.71 | 45.85 | 8.18 | 9.45 | 52.59 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/002-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local32/policy.json` |


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
