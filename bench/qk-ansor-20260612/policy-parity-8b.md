# QK Generated Policy Parity

- Model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`
- Policy: `bench/qk-ansor-20260612/8b-level0-policy-full.json`
- Total tensors: `254`
- Effective mismatches: `0`
- Raw differences: `74`
- Generated unsupported winners: `0`
- Explicit installed: `180`
- Generated installed: `180`

## Summary

```json
{
  "by_format": {
    "Q4_K": 217,
    "Q6_K": 37
  },
  "effective_mismatches": 0,
  "explicit_installed": 180,
  "explicit_reasons": {
    "policy_fallback": 74,
    "policy_primitive": 180
  },
  "generated_installed": 180,
  "generated_reasons": {
    "policy_fused": 54,
    "policy_missing": 20,
    "policy_primitive": 180
  },
  "generated_unsupported": 0,
  "raw_differences": 74,
  "same_effective": 254,
  "same_raw": 180,
  "total": 254
}
```

## Differences

| tensor | format | shape | explicit | generated | effective match | raw match |
|---|---|---:|---|---|---:|---:|
| `output.weight` | Q6_K | 151936x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_missing` | True | False |
| `token_embd.weight` | Q4_K | 151936x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_missing` | True | False |
| `blk.0.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.0.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_missing` | True | False |
| `blk.1.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.1.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_missing` | True | False |
| `blk.2.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.2.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_missing` | True | False |
| `blk.3.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.3.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_missing` | True | False |
| `blk.4.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.4.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.5.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.5.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.6.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.6.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_missing` | True | False |
| `blk.7.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.7.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.8.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.8.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.9.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.9.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_missing` | True | False |
| `blk.10.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.10.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.11.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.11.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.12.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.12.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_missing` | True | False |
| `blk.13.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.13.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.14.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.14.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.15.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.15.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_missing` | True | False |
| `blk.16.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.16.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.17.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.17.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.18.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.18.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_missing` | True | False |
| `blk.19.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.19.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.20.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.20.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.21.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.21.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_missing` | True | False |
| `blk.22.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.22.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.23.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.23.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.24.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.24.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_missing` | True | False |
| `blk.25.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.25.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.26.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.26.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.27.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.27.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_missing` | True | False |
| `blk.28.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.28.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.29.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.29.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.30.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.30.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_missing` | True | False |
| `blk.31.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.31.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_missing` | True | False |
| `blk.32.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.32.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_missing` | True | False |
| `blk.33.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.33.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_missing` | True | False |
| `blk.34.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.34.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_missing` | True | False |
| `blk.35.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.35.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_missing` | True | False |
