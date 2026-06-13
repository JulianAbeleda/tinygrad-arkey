# QK Policy Pipeline: Qwen3-8B-Q4_K_M.gguf

Date: 2026-06-12

- commit: `437f77772`
- device: `AMD`
- model size: `8B`
- reference mode: `policy`
- generated policy: `policy.json`

## Decision

- status: `needs-rerun`
- gain: `-5.25%`
- reference policy mean: `51.48 tok/s`
- reference policy decision window: `explicit5, explicit6, explicit7`
- generated mean: `48.78 tok/s`
- generated decision window: `generated1, generated2, generated3`
- generated percent of llama.cpp reference: `48.2%`

Reasons:

- explicit unstable: explicit5 last16 43.21 collapsed below 85% of avg 52.62
- explicit unstable: explicit7 last16 40.92 collapsed below 85% of avg 50.97

## Reproduction

```sh
DEV=AMD JIT=1 QK_GENERATED_POLICY=/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/003-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local64/policy.json PYTHONPATH=. \
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
| `explicit1` | 128 | 53.60 | 53.94 | 52.26 | 51.30 | 50.62 | 4.53 | 9.94 | 58.00 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `explicit2` | 128 | 50.73 | 51.05 | 46.82 | 40.32 | 35.14 | 9.21 | 9.78 | 57.37 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `explicit3` | 128 | 52.44 | 52.78 | 50.29 | 48.70 | 49.85 | 6.60 | 9.48 | 57.80 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `explicit4` | 128 | 53.59 | 53.93 | 52.09 | 51.20 | 51.36 | 4.64 | 9.88 | 57.75 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `explicit5` | 128 | 52.62 | 52.95 | 50.12 | 46.98 | 43.21 | 7.02 | 9.87 | 57.51 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `explicit6` | 128 | 50.84 | 51.17 | 51.69 | 50.16 | 51.48 | 8.43 | 9.90 | 57.49 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `explicit7` | 128 | 50.97 | 51.30 | 47.56 | 42.04 | 40.92 | 8.83 | 9.90 | 57.87 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` |
| `generated1` | 128 | 48.85 | 49.15 | 48.17 | 47.26 | 47.23 | 4.36 | 9.87 | 52.77 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/003-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local64/policy.json` |
| `generated2` | 128 | 49.08 | 49.39 | 48.04 | 47.40 | 46.95 | 4.09 | 9.88 | 52.58 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/003-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local64/policy.json` |
| `generated3` | 128 | 48.40 | 48.70 | 47.75 | 46.62 | 46.88 | 5.05 | 9.84 | 52.52 | 162 | 18 | 0.00 | `/home/ubuntu/tinygrad-arkey/bench/qk-ansor-transition-20260612/benchmarks/8b/003-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local64/policy.json` |


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
