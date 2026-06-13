# QK Policy Pipeline: Qwen3-14B-Q4_K_M.gguf

Date: 2026-06-12

- commit: `437f77772`
- device: `AMD`
- model size: `14B`
- reference mode: `policy`
- generated policy: `policy.json`

## Decision

- status: `tie`
- gain: `0.41%`
- reference policy mean: `39.60 tok/s`
- reference policy decision window: `explicit1, explicit2, explicit3`
- generated mean: `39.76 tok/s`
- generated decision window: `generated1, generated2, generated3`
- generated percent of llama.cpp reference: `60.4%`

Reasons:

- generated within tie band: 0.41%

## Reproduction

```sh
DEV=AMD JIT=1 QK_GENERATED_POLICY=/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/001-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p1-local64/policy.json PYTHONPATH=. \
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
| `explicit1` | 128 | 39.64 | 39.90 | 37.66 | 35.13 | 37.39 | 4.95 | 6.06 | 43.43 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `explicit2` | 128 | 39.08 | 39.34 | 39.43 | 39.47 | 39.22 | 5.74 | 6.07 | 43.50 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `explicit3` | 128 | 40.08 | 40.35 | 39.05 | 37.61 | 39.19 | 5.21 | 6.03 | 43.15 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `generated1` | 128 | 38.05 | 38.30 | 39.20 | 39.37 | 39.29 | 6.02 | 6.06 | 43.22 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/001-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p1-local64/policy.json` |
| `generated2` | 128 | 41.08 | 41.35 | 40.26 | 39.79 | 39.57 | 3.43 | 6.07 | 43.78 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/001-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p1-local64/policy.json` |
| `generated3` | 128 | 40.17 | 40.44 | 39.89 | 39.74 | 39.39 | 4.60 | 6.03 | 43.91 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/001-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p1-local64/policy.json` |


## Policy Parity Summary

```json
{
  "by_format": {
    "Q4_K": 241,
    "Q6_K": 41
  },
  "effective_mismatches": 120,
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
  "same_effective": 162,
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
