# QK Semantic Codegen Candidates: 14B

This v1 surface promotes one concrete codegen capability into runtime:
`q4_k_packed_u32_direct`, a direct-output Q4_K GEMV that avoids the
separate split-K reduction kernel. Candidates are tensor-scoped so full
decode tests do not change every tensor with the same shape.

## Summary

- candidates: `5`
- single-change candidates: `4`
- current storage bytes: `195379200`

| id | tensor | role | family | scope | parts | opts | full decode |
|---|---|---|---|---|---:|---|---:|
| `current` | n/a | n/a | n/a | n/a | 0 | n/a | `True` |
| `001-ffn-gate-blk-0-ffn-gate-weight-direct-out-tensor` | `blk.0.ffn_gate.weight` | `ffn_gate` | `q4_k_packed_u32_direct` | `tensor` | 1 | `LOCAL:0:32` | `True` |
| `002-ffn-down-blk-5-ffn-down-weight-direct-out-tensor` | `blk.5.ffn_down.weight` | `ffn_down` | `q4_k_packed_u32_direct` | `tensor` | 1 | `LOCAL:0:32` | `True` |
| `003-attn-q-blk-0-attn-q-weight-direct-out-tensor` | `blk.0.attn_q.weight` | `attn_q` | `q4_k_packed_u32_direct` | `tensor` | 1 | `LOCAL:0:64` | `True` |
| `004-attn-k-blk-0-attn-k-weight-direct-out-tensor` | `blk.0.attn_k.weight` | `attn_k` | `q4_k_packed_u32_direct` | `tensor` | 1 | `LOCAL:0:32` | `True` |
