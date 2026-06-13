# QK Policy Pipeline: Qwen3-32B-Q4_K_M.gguf

Date: 2026-06-12

- commit: `437f77772`
- device: `AMD`
- model size: `32B`
- reference mode: `policy`
- generated policy: `policy.json`

## Decision

- status: `reject`
- gain: `-24.65%`
- reference policy mean: `17.13 tok/s`
- reference policy decision window: `explicit1, explicit2, explicit3`
- generated mean: `12.91 tok/s`
- generated decision window: `generated1, generated2, generated3`
- generated percent of llama.cpp reference: `41.9%`

Reasons:

- generated slower than explicit by 24.65%

## Reproduction

```sh
DEV=AMD JIT=1 QK_GENERATED_POLICY=/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/004-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local32/policy.json PYTHONPATH=. \
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
| `explicit1` | 128 | 16.65 | 16.75 | 16.85 | 17.34 | 17.24 | 2.67 | 2.91 | 18.45 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `explicit2` | 128 | 17.25 | 17.36 | 16.62 | 16.23 | 15.28 | 2.08 | 3.10 | 18.49 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `explicit3` | 128 | 17.50 | 17.61 | 17.28 | 16.96 | 17.21 | 1.59 | 3.03 | 18.50 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `generated1` | 128 | 13.10 | 13.18 | 13.11 | 13.04 | 13.00 | 1.29 | 3.04 | 13.98 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/004-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local32/policy.json` |
| `generated2` | 128 | 12.87 | 12.95 | 12.96 | 12.74 | 12.82 | 1.18 | 3.13 | 14.01 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/004-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local32/policy.json` |
| `generated3` | 128 | 12.75 | 12.83 | 12.99 | 13.02 | 12.93 | 1.99 | 3.05 | 14.01 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/004-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local32/policy.json` |


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
