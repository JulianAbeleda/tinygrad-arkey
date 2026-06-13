# QK Policy Pipeline: Qwen3-32B-Q4_K_M.gguf

Date: 2026-06-12

- commit: `437f77772`
- device: `AMD`
- model size: `32B`
- reference mode: `policy`
- generated policy: `policy.json`

## Decision

- status: `reject`
- gain: `-3.65%`
- reference policy mean: `17.38 tok/s`
- reference policy decision window: `explicit1, explicit2, explicit3`
- generated mean: `16.74 tok/s`
- generated decision window: `generated1, generated2, generated3`
- generated percent of llama.cpp reference: `54.4%`

Reasons:

- generated slower than explicit by 3.65%

## Reproduction

```sh
DEV=AMD JIT=1 QK_GENERATED_POLICY=/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/006-ffn-down-blk-8-ffn-down-weight-q4_k_packed_u32-p1-local32/policy.json PYTHONPATH=. \
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
| `explicit1` | 128 | 16.99 | 17.10 | 16.14 | 15.46 | 15.81 | 2.38 | 3.02 | 18.52 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `explicit2` | 128 | 17.67 | 17.78 | 17.43 | 17.33 | 17.23 | 1.39 | 3.15 | 18.56 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `explicit3` | 128 | 17.48 | 17.59 | 17.32 | 17.07 | 16.72 | 1.44 | 3.11 | 18.52 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `generated1` | 128 | 16.67 | 16.78 | 16.34 | 16.49 | 16.74 | 1.93 | 2.92 | 17.96 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/006-ffn-down-blk-8-ffn-down-weight-q4_k_packed_u32-p1-local32/policy.json` |
| `generated2` | 128 | 16.58 | 16.69 | 16.27 | 16.17 | 16.70 | 1.95 | 2.88 | 17.91 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/006-ffn-down-blk-8-ffn-down-weight-q4_k_packed_u32-p1-local32/policy.json` |
| `generated3` | 128 | 16.98 | 17.09 | 16.61 | 16.20 | 15.98 | 1.54 | 3.01 | 18.00 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/006-ffn-down-blk-8-ffn-down-weight-q4_k_packed_u32-p1-local32/policy.json` |


## Policy Parity Summary

```json
{
  "by_format": {
    "Q4_K": 385,
    "Q6_K": 65
  },
  "effective_mismatches": 256,
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
  "raw_differences": 322,
  "same_effective": 194,
  "same_raw": 128,
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
