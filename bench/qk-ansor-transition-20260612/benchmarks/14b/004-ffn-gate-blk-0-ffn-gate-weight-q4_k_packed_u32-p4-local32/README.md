# QK Policy Pipeline: Qwen3-14B-Q4_K_M.gguf

Date: 2026-06-12

- commit: `437f77772`
- device: `AMD`
- model size: `14B`
- reference mode: `policy`
- generated policy: `policy.json`

## Decision

- status: `reject`
- gain: `-25.82%`
- reference policy mean: `38.79 tok/s`
- reference policy decision window: `explicit3, explicit4, explicit5`
- generated mean: `28.78 tok/s`
- generated decision window: `generated3, generated4, generated5`
- generated percent of llama.cpp reference: `43.7%`

Reasons:

- generated slower than explicit by 25.82%

## Reproduction

```sh
DEV=AMD JIT=1 QK_GENERATED_POLICY=/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/004-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local32/policy.json PYTHONPATH=. \
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
| `explicit1` | 128 | 39.94 | 40.21 | 39.24 | 39.39 | 39.09 | 4.27 | 6.09 | 43.20 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `explicit2` | 128 | 37.64 | 37.89 | 33.68 | 32.65 | 36.02 | 6.81 | 5.94 | 43.42 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `explicit3` | 128 | 39.26 | 39.53 | 37.17 | 38.77 | 38.24 | 6.43 | 6.09 | 43.29 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `explicit4` | 128 | 40.45 | 40.72 | 39.49 | 39.37 | 39.22 | 4.28 | 6.07 | 43.59 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `explicit5` | 128 | 36.66 | 36.90 | 38.94 | 37.82 | 36.14 | 8.27 | 6.09 | 43.45 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `generated1` | 128 | 29.11 | 29.29 | 28.67 | 28.32 | 28.41 | 2.26 | 6.08 | 31.08 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/004-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local32/policy.json` |
| `generated2` | 128 | 27.35 | 27.52 | 24.99 | 21.45 | 22.19 | 5.44 | 6.12 | 30.81 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/004-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local32/policy.json` |
| `generated3` | 128 | 29.22 | 29.41 | 28.75 | 28.48 | 28.46 | 2.23 | 6.05 | 31.10 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/004-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local32/policy.json` |
| `generated4` | 128 | 28.31 | 28.49 | 27.81 | 26.91 | 25.93 | 3.67 | 6.12 | 30.75 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/004-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local32/policy.json` |
| `generated5` | 128 | 28.79 | 28.97 | 27.98 | 28.24 | 28.40 | 3.08 | 5.72 | 31.08 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/004-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local32/policy.json` |


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
