# QK Policy Pipeline: Qwen3-14B-Q4_K_M.gguf

Date: 2026-06-12

- commit: `437f77772`
- device: `AMD`
- model size: `14B`
- reference mode: `policy`
- generated policy: `policy.json`

## Decision

- status: `reject`
- gain: `-26.50%`
- reference policy mean: `39.12 tok/s`
- reference policy decision window: `explicit1, explicit2, explicit3`
- generated mean: `28.75 tok/s`
- generated decision window: `generated1, generated2, generated3`
- generated percent of llama.cpp reference: `43.7%`

Reasons:

- generated slower than explicit by 26.50%

## Reproduction

```sh
DEV=AMD JIT=1 QK_GENERATED_POLICY=/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/005-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local64/policy.json PYTHONPATH=. \
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
| `explicit1` | 128 | 38.59 | 38.84 | 36.51 | 39.41 | 39.18 | 6.51 | 6.05 | 43.34 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `explicit2` | 128 | 40.20 | 40.47 | 39.80 | 39.22 | 38.89 | 4.42 | 6.06 | 43.28 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `explicit3` | 128 | 38.56 | 38.81 | 39.06 | 38.84 | 39.16 | 6.55 | 5.86 | 43.30 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `generated1` | 128 | 28.88 | 29.06 | 28.19 | 27.38 | 27.97 | 2.85 | 6.02 | 30.80 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/005-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local64/policy.json` |
| `generated2` | 128 | 28.67 | 28.84 | 27.82 | 26.96 | 26.34 | 3.15 | 6.08 | 31.02 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/005-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local64/policy.json` |
| `generated3` | 128 | 28.70 | 28.88 | 28.20 | 27.43 | 27.05 | 2.62 | 5.95 | 30.90 | 240 | 40 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/14b/005-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local64/policy.json` |


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
