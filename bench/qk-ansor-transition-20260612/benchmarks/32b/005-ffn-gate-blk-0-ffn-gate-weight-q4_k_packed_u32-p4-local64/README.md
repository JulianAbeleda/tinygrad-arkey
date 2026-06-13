# QK Policy Pipeline: Qwen3-32B-Q4_K_M.gguf

Date: 2026-06-12

- commit: `437f77772`
- device: `AMD`
- model size: `32B`
- reference mode: `policy`
- generated policy: `policy.json`

## Decision

- status: `reject`
- gain: `-25.23%`
- reference policy mean: `17.39 tok/s`
- reference policy decision window: `explicit1, explicit2, explicit3`
- generated mean: `13.01 tok/s`
- generated decision window: `generated1, generated2, generated3`
- generated percent of llama.cpp reference: `42.2%`

Reasons:

- generated slower than explicit by 25.23%

## Reproduction

```sh
DEV=AMD JIT=1 QK_GENERATED_POLICY=/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/005-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local64/policy.json PYTHONPATH=. \
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
| `explicit1` | 128 | 17.69 | 17.80 | 17.47 | 17.31 | 17.16 | 1.36 | 3.08 | 18.52 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `explicit2` | 128 | 17.59 | 17.70 | 17.45 | 17.25 | 17.11 | 1.42 | 3.08 | 18.52 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `explicit3` | 128 | 16.91 | 17.02 | 17.49 | 17.34 | 17.27 | 2.80 | 2.96 | 18.53 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `generated1` | 128 | 12.61 | 12.68 | 12.54 | 12.74 | 12.42 | 1.67 | 3.16 | 13.95 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/005-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local64/policy.json` |
| `generated2` | 128 | 13.20 | 13.28 | 13.00 | 12.78 | 12.46 | 1.10 | 3.13 | 13.97 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/005-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local64/policy.json` |
| `generated3` | 128 | 13.20 | 13.28 | 13.17 | 13.07 | 13.01 | 1.15 | 3.10 | 13.92 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/005-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local64/policy.json` |


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
