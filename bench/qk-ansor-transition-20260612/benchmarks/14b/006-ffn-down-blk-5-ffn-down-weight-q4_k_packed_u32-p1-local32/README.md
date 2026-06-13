# QK Policy Pipeline: Qwen3-14B-Q4_K_M.gguf

Date: 2026-06-12

- commit: `437f77772`
- device: `AMD`
- model size: `14B`
- reference mode: `policy`
- generated policy: `policy.json`

## Decision

- status: `tie`
- gain: `-2.18%`
- reference policy mean: `39.49 tok/s`
- reference policy decision window: `explicit1, explicit2, explicit3`
- generated mean: `38.64 tok/s`
- generated decision window: `generated1, generated2, generated3`
- generated percent of llama.cpp reference: `58.7%`

Reasons:

- generated within tie band: -2.18%

## Reproduction

```sh
DEV=AMD JIT=1 QK_GENERATED_POLICY=/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/006-ffn-down-blk-5-ffn-down-weight-q4_k_packed_u32-p1-local32/policy.json PYTHONPATH=. \
  .venv/bin/python -m tinygrad.llm --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf --warmup --benchmark 128
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
| `explicit1` | 128 | 39.92 | 40.18 | 38.17 | 38.39 | 38.83 | 4.81 | 6.07 | 43.39 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `explicit2` | 128 | 38.29 | 38.54 | 37.26 | 37.79 | 38.75 | 5.24 | 6.08 | 43.24 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `explicit3` | 128 | 40.28 | 40.55 | 39.19 | 38.65 | 38.11 | 3.89 | 6.08 | 43.28 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `generated1` | 128 | 39.40 | 39.66 | 38.62 | 38.13 | 38.01 | 3.29 | 6.08 | 42.15 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/006-ffn-down-blk-5-ffn-down-weight-q4_k_packed_u32-p1-local32/policy.json` |
| `generated2` | 128 | 37.34 | 37.59 | 38.29 | 38.12 | 37.58 | 6.23 | 5.83 | 42.03 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/006-ffn-down-blk-5-ffn-down-weight-q4_k_packed_u32-p1-local32/policy.json` |
| `generated3` | 128 | 39.17 | 39.43 | 38.68 | 38.08 | 37.59 | 3.55 | 5.92 | 41.95 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/006-ffn-down-blk-5-ffn-down-weight-q4_k_packed_u32-p1-local32/policy.json` |


## Policy Parity Summary

```json
{
  "by_format": {
    "Q4_K": 241,
    "Q6_K": 41
  },
  "effective_mismatches": 200,
  "explicit_installed": 200,
  "explicit_reasons": {
    "policy_fallback": 82,
    "policy_primitive": 200
  },
  "generated_installed": 280,
  "generated_reasons": {
    "policy_fused": 1,
    "policy_missing": 1,
    "policy_primitive": 280
  },
  "generated_unsupported": 0,
  "raw_differences": 202,
  "same_effective": 82,
  "same_raw": 80,
  "total": 282
}
```


## Storage Policy

```json
{
  "by_format": {
    "Q4_K": 117964800,
    "Q6_K": 77414400
  },
  "cap_bytes": null,
  "mode": "uncapped_shape",
  "note": "shape-scoped entries multiply at runtime by tensor count",
  "selected_bytes": 195379200,
  "selected_entries": 7,
  "selected_primitive_entries": 6
}
```
