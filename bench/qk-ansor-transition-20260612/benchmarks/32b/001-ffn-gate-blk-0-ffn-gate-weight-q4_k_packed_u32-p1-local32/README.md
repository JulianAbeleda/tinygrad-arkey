# QK Policy Pipeline: Qwen3-32B-Q4_K_M.gguf

Date: 2026-06-12

- commit: `437f77772`
- device: `AMD`
- model size: `32B`
- reference mode: `policy`
- generated policy: `policy.json`

## Decision

- status: `accept`
- gain: `3.24%`
- reference policy mean: `16.70 tok/s`
- reference policy decision window: `explicit3, explicit4, explicit5`
- generated mean: `17.24 tok/s`
- generated decision window: `generated1, generated2, generated3`
- generated percent of llama.cpp reference: `56.0%`

Reasons:

- generated beats explicit by 3.24%

## Reproduction

```sh
DEV=AMD JIT=1 QK_GENERATED_POLICY=/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/001-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p1-local32/policy.json PYTHONPATH=. \
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
| `explicit1` | 128 | 17.47 | 17.58 | 17.15 | 16.75 | 16.07 | 1.69 | 3.13 | 18.48 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `explicit2` | 128 | 16.38 | 16.49 | 15.22 | 13.17 | 8.94 | 3.61 | 3.13 | 18.55 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `explicit3` | 128 | 16.57 | 16.67 | 15.56 | 16.22 | 15.32 | 2.74 | 3.13 | 18.54 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `explicit4` | 128 | 16.88 | 16.99 | 16.45 | 17.08 | 17.20 | 2.46 | 3.15 | 18.43 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `explicit5` | 128 | 16.66 | 16.77 | 17.27 | 16.93 | 16.42 | 2.73 | 3.15 | 18.57 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `generated1` | 128 | 17.01 | 17.12 | 16.14 | 17.36 | 17.26 | 2.42 | 3.16 | 18.53 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/001-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p1-local32/policy.json` |
| `generated2` | 128 | 17.18 | 17.29 | 16.61 | 16.52 | 15.62 | 2.24 | 3.00 | 18.52 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/001-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p1-local32/policy.json` |
| `generated3` | 128 | 17.55 | 17.66 | 17.39 | 17.16 | 16.90 | 1.43 | 3.15 | 18.55 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/001-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p1-local32/policy.json` |


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
