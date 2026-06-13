# QK Policy Pipeline: Qwen3-8B-Q4_K_M.gguf

Date: 2026-06-12

- commit: `437f77772`
- device: `AMD`
- model size: `8B`
- reference mode: `policy`
- generated policy: `policy.json`

## Decision

- status: `reject`
- gain: `-22.40%`
- reference policy mean: `52.31 tok/s`
- reference policy decision window: `explicit1, explicit2, explicit3`
- generated mean: `40.59 tok/s`
- generated decision window: `generated2, generated3, generated4`
- generated percent of llama.cpp reference: `40.1%`

Reasons:

- generated slower than explicit by 22.40%

## Reproduction

```sh
DEV=AMD JIT=1 QK_GENERATED_POLICY=/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/004-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local32/policy.json PYTHONPATH=. \
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
| `explicit1` | 128 | 52.11 | 52.45 | 52.30 | 51.58 | 51.25 | 6.11 | 9.61 | 57.76 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `explicit2` | 128 | 53.38 | 53.72 | 52.59 | 51.76 | 51.35 | 4.49 | 9.85 | 57.81 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `explicit3` | 128 | 51.44 | 51.76 | 52.03 | 51.62 | 51.45 | 7.10 | 9.88 | 57.76 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `generated1` | 128 | 27.12 | 27.26 | 20.50 | 24.65 | 33.16 | 12.59 | 9.83 | 45.28 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/004-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local32/policy.json` |
| `generated2` | 128 | 37.93 | 38.15 | 40.12 | 38.18 | 37.82 | 6.85 | 9.91 | 44.88 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/004-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local32/policy.json` |
| `generated3` | 128 | 41.76 | 42.01 | 40.46 | 39.18 | 39.42 | 3.74 | 9.97 | 45.38 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/004-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local32/policy.json` |
| `generated4` | 128 | 42.09 | 42.34 | 41.46 | 40.96 | 40.78 | 3.18 | 9.84 | 44.90 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/004-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local32/policy.json` |


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
