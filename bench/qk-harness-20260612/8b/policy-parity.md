# QK Generated Policy Parity

- Model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`
- Policy: `bench/qk-harness-20260612/8b/policy.json`
- Total tensors: `254`
- Effective mismatches: `18`
- Raw differences: `236`
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
  "effective_mismatches": 18,
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
  "same_effective": 236,
  "same_raw": 18,
  "total": 254
}
```

## Differences

| tensor | format | shape | explicit | generated | effective match | raw match |
|---|---|---:|---|---|---:|---:|
| `output.weight` | Q6_K | 151936x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `token_embd.weight` | Q4_K | 151936x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_missing` | True | False |
| `blk.0.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.0.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.0.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.0.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.0.ffn_down.weight` | Q6_K | 4096x12288 | `v1_q6_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q6_local64_p2 parts=2 opts=['LOCAL:0:64'] reason=policy_primitive` | False | False |
| `blk.0.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.0.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.1.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.1.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.1.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.1.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.1.ffn_down.weight` | Q6_K | 4096x12288 | `v1_q6_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q6_local64_p2 parts=2 opts=['LOCAL:0:64'] reason=policy_primitive` | False | False |
| `blk.1.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.1.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.2.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.2.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.2.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.2.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.2.ffn_down.weight` | Q6_K | 4096x12288 | `v1_q6_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q6_local64_p2 parts=2 opts=['LOCAL:0:64'] reason=policy_primitive` | False | False |
| `blk.2.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.2.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.3.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.3.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.3.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.3.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.3.ffn_down.weight` | Q6_K | 4096x12288 | `v1_q6_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q6_local64_p2 parts=2 opts=['LOCAL:0:64'] reason=policy_primitive` | False | False |
| `blk.3.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.3.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.4.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.4.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.4.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.4.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.4.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.4.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.5.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.5.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.5.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.5.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.5.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.5.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.6.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.6.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.6.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.6.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.6.ffn_down.weight` | Q6_K | 4096x12288 | `v1_q6_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q6_local64_p2 parts=2 opts=['LOCAL:0:64'] reason=policy_primitive` | False | False |
| `blk.6.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.6.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.7.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.7.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.7.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.7.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.7.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.7.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.8.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.8.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.8.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.8.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.8.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.8.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.9.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.9.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.9.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.9.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.9.ffn_down.weight` | Q6_K | 4096x12288 | `v1_q6_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q6_local64_p2 parts=2 opts=['LOCAL:0:64'] reason=policy_primitive` | False | False |
| `blk.9.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.9.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.10.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.10.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.10.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.10.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.10.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.10.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.11.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.11.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.11.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.11.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.11.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.11.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.12.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.12.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.12.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.12.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.12.ffn_down.weight` | Q6_K | 4096x12288 | `v1_q6_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q6_local64_p2 parts=2 opts=['LOCAL:0:64'] reason=policy_primitive` | False | False |
| `blk.12.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.12.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.13.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.13.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.13.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.13.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.13.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.13.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.14.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.14.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.14.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.14.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.14.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.14.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.15.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.15.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.15.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.15.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.15.ffn_down.weight` | Q6_K | 4096x12288 | `v1_q6_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q6_local64_p2 parts=2 opts=['LOCAL:0:64'] reason=policy_primitive` | False | False |
| `blk.15.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.15.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.16.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.16.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.16.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.16.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.16.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.16.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.17.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.17.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.17.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.17.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.17.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.17.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.18.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.18.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.18.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.18.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.18.ffn_down.weight` | Q6_K | 4096x12288 | `v1_q6_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q6_local64_p2 parts=2 opts=['LOCAL:0:64'] reason=policy_primitive` | False | False |
| `blk.18.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.18.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.19.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.19.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.19.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.19.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.19.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.19.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.20.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.20.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.20.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.20.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.20.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.20.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.21.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.21.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.21.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.21.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.21.ffn_down.weight` | Q6_K | 4096x12288 | `v1_q6_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q6_local64_p2 parts=2 opts=['LOCAL:0:64'] reason=policy_primitive` | False | False |
| `blk.21.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.21.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.22.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.22.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.22.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.22.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.22.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.22.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.23.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.23.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.23.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.23.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.23.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.23.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.24.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.24.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.24.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.24.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.24.ffn_down.weight` | Q6_K | 4096x12288 | `v1_q6_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q6_local64_p2 parts=2 opts=['LOCAL:0:64'] reason=policy_primitive` | False | False |
| `blk.24.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.24.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.25.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.25.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.25.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.25.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.25.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.25.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.26.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.26.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.26.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.26.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.26.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.26.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.27.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.27.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.27.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.27.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.27.ffn_down.weight` | Q6_K | 4096x12288 | `v1_q6_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q6_local64_p2 parts=2 opts=['LOCAL:0:64'] reason=policy_primitive` | False | False |
| `blk.27.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.27.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.28.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.28.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.28.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.28.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.28.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.28.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.29.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.29.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.29.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.29.attn_v.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.29.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.29.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.30.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.30.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.30.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.30.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.30.ffn_down.weight` | Q6_K | 4096x12288 | `v1_q6_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q6_local64_p2 parts=2 opts=['LOCAL:0:64'] reason=policy_primitive` | False | False |
| `blk.30.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.30.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.31.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.31.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.31.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.31.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.31.ffn_down.weight` | Q6_K | 4096x12288 | `v1_q6_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q6_local64_p2 parts=2 opts=['LOCAL:0:64'] reason=policy_primitive` | False | False |
| `blk.31.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.31.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.32.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.32.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.32.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.32.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.32.ffn_down.weight` | Q6_K | 4096x12288 | `v1_q6_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q6_local64_p2 parts=2 opts=['LOCAL:0:64'] reason=policy_primitive` | False | False |
| `blk.32.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.32.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.33.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.33.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.33.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.33.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.33.ffn_down.weight` | Q6_K | 4096x12288 | `v1_q6_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q6_local64_p2 parts=2 opts=['LOCAL:0:64'] reason=policy_primitive` | False | False |
| `blk.33.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.33.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.34.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.34.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.34.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.34.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.34.ffn_down.weight` | Q6_K | 4096x12288 | `v1_q6_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q6_local64_p2 parts=2 opts=['LOCAL:0:64'] reason=policy_primitive` | False | False |
| `blk.34.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.34.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.35.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.35.attn_output.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.35.attn_q.weight` | Q4_K | 4096x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.35.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph parts=0 opts=[] reason=policy_fallback` | `fused_graph parts=0 opts=[] reason=policy_fused` | True | False |
| `blk.35.ffn_down.weight` | Q6_K | 4096x12288 | `v1_q6_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q6_local64_p2 parts=2 opts=['LOCAL:0:64'] reason=policy_primitive` | False | False |
| `blk.35.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
| `blk.35.ffn_up.weight` | Q4_K | 12288x4096 | `v1_q4_packed parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | `q4_local64_p1 parts=1 opts=['LOCAL:0:64'] reason=policy_primitive` | True | False |
