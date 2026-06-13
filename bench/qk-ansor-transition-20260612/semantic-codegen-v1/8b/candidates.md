# QK Semantic Codegen Candidates: 8B

This v1 surface promotes one concrete codegen capability into runtime:
`q4_k_packed_u32_direct`, a direct-output Q4_K GEMV that avoids the
separate split-K reduction kernel. Candidates are tensor-scoped so full
decode tests do not change every tensor with the same shape.

## Summary

- candidates: `4`
- single-change candidates: `3`
- current storage bytes: `107347968`

| id | tensor | role | family | scope | parts | opts | persistent delta | metadata sidecar | full decode |
|---|---|---|---|---|---:|---|---:|---:|---:|
| `current` | n/a | n/a | n/a | n/a | 0 | n/a | 0 | 0 | `True` |
| `001-ffn-gate-blk-0-ffn-gate-weight-direct-out-tensor` | `blk.0.ffn_gate.weight` | `ffn_gate` | `q4_k_packed_u32_direct` | `tensor` | 1 | `LOCAL:0:64` | 0 | 0 | `True` |
| `002-ffn-down-blk-4-ffn-down-weight-direct-out-tensor` | `blk.4.ffn_down.weight` | `ffn_down` | `q4_k_packed_u32_direct` | `tensor` | 1 | `LOCAL:0:32` | 0 | 0 | `True` |
| `003-attn-q-blk-0-attn-q-weight-direct-out-tensor` | `blk.0.attn_q.weight` | `attn_q` | `q4_k_packed_u32_direct` | `tensor` | 1 | `LOCAL:0:64` | 0 | 0 | `True` |
