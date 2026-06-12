# QK Policy Pipeline: Qwen3-32B-Q4_K_M.gguf

Date: 2026-06-12

- commit: `378f8c102`
- device: `AMD`
- model size: `32B`
- reference mode: `generic`
- generated policy: `policy.json`

## Decision

- status: `accept`
- gain: `20.98%`
- explicit mean: `3.44 tok/s`
- explicit decision window: `explicit1, explicit2, explicit3`
- generated mean: `4.16 tok/s`
- generated decision window: `generated1, generated2, generated3`
- generated percent of llama.cpp reference: `13.5%`

Reasons:

- generated beats explicit by 20.98%

## Reproduction

```sh
DEV=AMD JIT=1 QK_GENERATED_POLICY=bench/qk-policy-cap-20260612/32b-1536mb/policy.json PYTHONPATH=. \
  .venv/bin/python -m tinygrad.llm --model /home/ubuntu/models/Qwen3-32B-Q4_K_M.gguf --warmup --benchmark 128
```

## Artifacts

- `search.json`, `policy.json`, `semantic-report.md`
- `policy-parity.json`, `policy-parity.md`
- `decode-summary.json`, `decode-summary.md`
- `output-ab.json`, `output-ab.log`
- `profile-report.json`, `profile-report.md`

## Decode Summary

# QK Decode Summary

| label | samples | avg tok/s | drop1 | last64 | last32 | last16 | stdev | min | max | Q4 install | Q6 install | policy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `explicit1` | 128 | 3.46 | 3.46 | 3.47 | 3.47 | 3.48 | 0.16 | 2.07 | 3.54 |  |  | `` |
| `explicit2` | 128 | 3.43 | 3.43 | 3.40 | 3.32 | 3.30 | 0.21 | 2.33 | 3.56 |  |  | `` |
| `explicit3` | 128 | 3.43 | 3.44 | 3.46 | 3.45 | 3.44 | 0.18 | 2.26 | 3.54 |  |  | `` |
| `generated1` | 128 | 4.13 | 4.14 | 4.02 | 4.01 | 3.95 | 0.35 | 2.77 | 4.34 | 112 | 32 | `bench/qk-policy-cap-20260612/32b-1536mb/policy.json` |
| `generated2` | 128 | 4.21 | 4.21 | 4.23 | 4.22 | 4.20 | 0.24 | 2.89 | 4.34 | 112 | 32 | `bench/qk-policy-cap-20260612/32b-1536mb/policy.json` |
| `generated3` | 128 | 4.15 | 4.16 | 4.20 | 4.19 | 4.25 | 0.31 | 2.83 | 4.35 | 112 | 32 | `bench/qk-policy-cap-20260612/32b-1536mb/policy.json` |


## Policy Parity Summary

```json
{
  "by_format": {
    "Q4_K": 385,
    "Q6_K": 65
  },
  "effective_mismatches": 448,
  "explicit_installed": 320,
  "explicit_reasons": {
    "policy_fallback": 130,
    "policy_primitive": 320
  },
  "generated_installed": 144,
  "generated_reasons": {
    "policy_memory_cap": 305,
    "policy_missing": 1,
    "policy_primitive": 144
  },
  "generated_unsupported": 0,
  "raw_differences": 450,
  "same_effective": 2,
  "same_raw": 0,
  "total": 450
}
```


## Storage Policy

```json
{
  "by_decision": {
    "memory_cap_fused_nonpositive_benefit": 1,
    "memory_cap_fused_over_budget": 304,
    "memory_cap_selected": 144
  },
  "by_format": {
    "Q4_K": 1462763520,
    "Q6_K": 137625600
  },
  "cap_bytes": 1610612736,
  "capped_primitive_entries": 304,
  "mode": "tensor_memory_cap",
  "selected_by_role": {
    "attn_k": 64,
    "attn_v": 64,
    "ffn_down": 16
  },
  "selected_bytes": 1600389120,
  "selected_entries": 449,
  "selected_primitive_entries": 144,
  "unsupported_tensor_infos": 1
}
```
