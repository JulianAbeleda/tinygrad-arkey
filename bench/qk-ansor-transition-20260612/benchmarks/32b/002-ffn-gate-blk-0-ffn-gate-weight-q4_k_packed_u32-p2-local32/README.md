# QK Policy Pipeline: Qwen3-32B-Q4_K_M.gguf

Date: 2026-06-12

- commit: `437f77772`
- device: `AMD`
- model size: `32B`
- reference mode: `policy`
- generated policy: `policy.json`

## Decision

- status: `needs-rerun`
- gain: `-26.92%`
- reference policy mean: `17.43 tok/s`
- reference policy decision window: `explicit1, explicit2, explicit3`
- generated mean: `12.74 tok/s`
- generated decision window: `generated5, generated6, generated7`
- generated percent of llama.cpp reference: `41.4%`

Reasons:

- generated unstable: generated7 avg 11.29 is >10% below mean 12.74
- generated unstable: generated7 last64 8.82 collapsed below 90% of avg 11.29
- generated unstable: generated7 last16 8.25 collapsed below 85% of avg 11.29

## Reproduction

```sh
DEV=AMD JIT=1 QK_GENERATED_POLICY=/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/002-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local32/policy.json PYTHONPATH=. \
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
| `explicit1` | 128 | 17.04 | 17.15 | 17.10 | 17.34 | 17.27 | 1.80 | 3.14 | 18.51 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `explicit2` | 128 | 17.59 | 17.70 | 17.27 | 16.90 | 16.84 | 1.46 | 3.16 | 18.51 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `explicit3` | 128 | 17.67 | 17.78 | 17.42 | 17.22 | 17.07 | 1.38 | 3.15 | 18.51 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` |
| `generated1` | 128 | 12.42 | 12.49 | 11.00 | 9.53 | 12.29 | 2.89 | 3.16 | 14.39 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/002-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local32/policy.json` |
| `generated2` | 128 | 13.20 | 13.28 | 12.87 | 12.34 | 11.71 | 1.52 | 3.16 | 14.27 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/002-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local32/policy.json` |
| `generated3` | 128 | 13.52 | 13.60 | 13.60 | 13.49 | 13.45 | 1.33 | 3.11 | 14.29 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/002-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local32/policy.json` |
| `generated4` | 128 | 12.43 | 12.51 | 11.24 | 10.18 | 9.72 | 2.43 | 3.16 | 14.34 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/002-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local32/policy.json` |
| `generated5` | 128 | 13.30 | 13.38 | 12.79 | 13.49 | 13.43 | 1.84 | 3.15 | 14.28 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/002-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local32/policy.json` |
| `generated6` | 128 | 13.63 | 13.71 | 13.55 | 13.43 | 13.42 | 0.98 | 3.15 | 14.21 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/002-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local32/policy.json` |
| `generated7` | 128 | 11.29 | 11.35 | 8.82 | 7.28 | 8.25 | 3.46 | 3.03 | 14.27 | 384 | 64 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/32b/002-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local32/policy.json` |


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
