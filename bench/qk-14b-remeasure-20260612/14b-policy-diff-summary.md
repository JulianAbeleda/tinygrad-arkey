# 14B Policy Coverage Diff

## Summary

| format | role | tensors | fused->primitive | primitive option changes |
|---|---|---:|---:|---:|
| Q4_K | `attn_k` | 40 | 40 | 0 |
| Q4_K | `attn_output` | 40 | 0 | 0 |
| Q4_K | `attn_q` | 40 | 0 | 0 |
| Q4_K | `attn_v` | 20 | 20 | 0 |
| Q4_K | `ffn_down` | 20 | 0 | 20 |
| Q4_K | `ffn_gate` | 40 | 0 | 40 |
| Q4_K | `ffn_up` | 40 | 0 | 40 |
| Q4_K | `token_embd` | 1 | 0 | 0 |
| Q6_K | `attn_v` | 20 | 20 | 0 |
| Q6_K | `ffn_down` | 20 | 0 | 20 |
| Q6_K | `output` | 1 | 0 | 0 |

## Effective Transitions

| format | role | explicit effective | generated effective | count |
|---|---|---|---|---:|
| Q4_K | `attn_k` | `fused_graph` | `q4_k_packed_u32` | 40 |
| Q4_K | `attn_output` | `q4_k_packed_u32` | `q4_k_packed_u32` | 40 |
| Q4_K | `attn_q` | `q4_k_packed_u32` | `q4_k_packed_u32` | 40 |
| Q4_K | `attn_v` | `fused_graph` | `q4_k_packed_u32` | 20 |
| Q4_K | `ffn_down` | `q4_k_packed_u32` | `q4_k_packed_u32` | 20 |
| Q4_K | `ffn_gate` | `q4_k_packed_u32` | `q4_k_packed_u32` | 40 |
| Q4_K | `ffn_up` | `q4_k_packed_u32` | `q4_k_packed_u32` | 40 |
| Q4_K | `token_embd` | `fused_graph` | `fused_graph` | 1 |
| Q6_K | `attn_v` | `fused_graph` | `q6_k_packed_u16` | 20 |
| Q6_K | `ffn_down` | `q6_k_packed_u16` | `q6_k_packed_u16` | 20 |
| Q6_K | `output` | `fused_graph` | `fused_graph` | 1 |

## Examples

### New Primitive Coverage

- Q4_K `attn_k`: `blk.0.attn_k.weight`, `blk.1.attn_k.weight`, `blk.2.attn_k.weight`, `blk.3.attn_k.weight`, `blk.4.attn_k.weight`
- Q4_K `attn_v`: `blk.5.attn_v.weight`, `blk.6.attn_v.weight`, `blk.8.attn_v.weight`, `blk.9.attn_v.weight`, `blk.11.attn_v.weight`
- Q6_K `attn_v`: `blk.0.attn_v.weight`, `blk.1.attn_v.weight`, `blk.2.attn_v.weight`, `blk.3.attn_v.weight`, `blk.4.attn_v.weight`

### Primitive Option Changes

- Q4_K `ffn_down`: `blk.5.ffn_down.weight: 4 ['LOCAL:0:32'] -> 2 ['LOCAL:0:32']`; `blk.6.ffn_down.weight: 4 ['LOCAL:0:32'] -> 2 ['LOCAL:0:32']`; `blk.8.ffn_down.weight: 4 ['LOCAL:0:32'] -> 2 ['LOCAL:0:32']`; `blk.9.ffn_down.weight: 4 ['LOCAL:0:32'] -> 2 ['LOCAL:0:32']`; `blk.11.ffn_down.weight: 4 ['LOCAL:0:32'] -> 2 ['LOCAL:0:32']`
- Q4_K `ffn_gate`: `blk.0.ffn_gate.weight: 1 ['LOCAL:0:64'] -> 1 ['LOCAL:0:32']`; `blk.1.ffn_gate.weight: 1 ['LOCAL:0:64'] -> 1 ['LOCAL:0:32']`; `blk.2.ffn_gate.weight: 1 ['LOCAL:0:64'] -> 1 ['LOCAL:0:32']`; `blk.3.ffn_gate.weight: 1 ['LOCAL:0:64'] -> 1 ['LOCAL:0:32']`; `blk.4.ffn_gate.weight: 1 ['LOCAL:0:64'] -> 1 ['LOCAL:0:32']`
- Q4_K `ffn_up`: `blk.0.ffn_up.weight: 1 ['LOCAL:0:64'] -> 1 ['LOCAL:0:32']`; `blk.1.ffn_up.weight: 1 ['LOCAL:0:64'] -> 1 ['LOCAL:0:32']`; `blk.2.ffn_up.weight: 1 ['LOCAL:0:64'] -> 1 ['LOCAL:0:32']`; `blk.3.ffn_up.weight: 1 ['LOCAL:0:64'] -> 1 ['LOCAL:0:32']`; `blk.4.ffn_up.weight: 1 ['LOCAL:0:64'] -> 1 ['LOCAL:0:32']`
- Q6_K `ffn_down`: `blk.0.ffn_down.weight: 1 ['LOCAL:0:64'] -> 2 ['LOCAL:0:64']`; `blk.1.ffn_down.weight: 1 ['LOCAL:0:64'] -> 2 ['LOCAL:0:64']`; `blk.2.ffn_down.weight: 1 ['LOCAL:0:64'] -> 2 ['LOCAL:0:64']`; `blk.3.ffn_down.weight: 1 ['LOCAL:0:64'] -> 2 ['LOCAL:0:64']`; `blk.4.ffn_down.weight: 1 ['LOCAL:0:64'] -> 2 ['LOCAL:0:64']`
