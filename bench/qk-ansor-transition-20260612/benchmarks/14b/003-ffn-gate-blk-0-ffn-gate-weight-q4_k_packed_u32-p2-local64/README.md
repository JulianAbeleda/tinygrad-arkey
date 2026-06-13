# QK Policy Pipeline: Qwen3-14B-Q4_K_M.gguf

Date: 2026-06-12

- commit: `437f77772`
- device: `AMD`
- model size: `14B`
- reference mode: `policy`
- generated policy: `policy.json`

## Decision

- status: `reject`
- gain: `-24.35%`
- reference policy mean: `39.46 tok/s`
- reference policy decision window: `explicit4, explicit5, explicit6`
- generated mean: `29.85 tok/s`
- generated decision window: `generated5, generated6, generated7`
- generated percent of llama.cpp reference: `45.4%`

Reasons:

- generated slower than explicit by 24.35%

## Reproduction

```sh
DEV=AMD JIT=1 QK_GENERATED_POLICY=/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/003-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local64/policy.json PYTHONPATH=. \
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
| `explicit1` | 128 | 38.32 | 38.57 | 39.80 | 39.27 | 38.88 | 5.90 | 6.01 | 43.49 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `explicit2` | 128 | 40.40 | 40.67 | 39.72 | 39.06 | 39.14 | 3.48 | 6.04 | 43.44 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `explicit3` | 128 | 38.29 | 38.54 | 35.09 | 31.26 | 24.00 | 7.02 | 6.02 | 43.24 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `explicit4` | 128 | 37.96 | 38.21 | 36.70 | 33.66 | 33.25 | 7.12 | 6.02 | 43.50 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `explicit5` | 128 | 40.34 | 40.61 | 39.72 | 39.51 | 39.23 | 4.06 | 6.07 | 43.52 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `explicit6` | 128 | 40.08 | 40.35 | 39.09 | 39.36 | 39.04 | 4.21 | 5.55 | 43.26 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `generated1` | 128 | 28.46 | 28.64 | 25.64 | 19.60 | 15.51 | 7.09 | 6.03 | 34.14 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/003-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local64/policy.json` |
| `generated2` | 128 | 31.85 | 32.05 | 31.06 | 31.19 | 30.89 | 3.32 | 6.05 | 34.17 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/003-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local64/policy.json` |
| `generated3` | 128 | 30.64 | 30.84 | 30.84 | 31.37 | 31.14 | 4.44 | 6.01 | 34.50 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/003-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local64/policy.json` |
| `generated4` | 128 | 29.90 | 30.09 | 29.04 | 26.53 | 22.42 | 5.15 | 5.97 | 33.97 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/003-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local64/policy.json` |
| `generated5` | 128 | 29.00 | 29.18 | 30.96 | 29.66 | 31.31 | 6.37 | 5.82 | 34.08 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/003-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local64/policy.json` |
| `generated6` | 128 | 29.85 | 30.04 | 28.54 | 29.03 | 31.35 | 4.47 | 5.92 | 34.01 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/003-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local64/policy.json` |
| `generated7` | 128 | 30.70 | 30.89 | 30.49 | 31.50 | 31.39 | 4.04 | 6.07 | 34.15 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/003-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local64/policy.json` |


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
