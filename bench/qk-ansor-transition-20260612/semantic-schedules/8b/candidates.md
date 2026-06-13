# QK Semantic Schedule Candidates: 8B

Second-stage Ansor-transition surface. These candidates carry semantic
schedule/codegen specs instead of only varying `parts` and `LOCAL`.

## Summary

- candidates: `15`
- single-change candidates: `14`
- current storage bytes: `107347968`

| id | tensor | schedule | family | parts | opts | persistent delta | metadata sidecar | full decode |
|---|---|---|---|---:|---|---:|---:|---:|
| `current` | n/a | current | n/a | 0 | n/a | 0 | 0 | `True` |
| `001-ffn-gate-blk-0-ffn-gate-weight-direct-out` | `blk.0.ffn_gate.weight` | `direct_out` | `q4_k_packed_u32_direct` | 1 | `LOCAL:0:64` | 0 | 0 | `False` |
| `002-ffn-gate-blk-0-ffn-gate-weight-row-upcast2` | `blk.0.ffn_gate.weight` | `row_upcast2` | `q4_k_packed_u32` | 1 | `LOCAL:0:64,UPCAST:0:2` | 0 | 0 | `True` |
| `003-ffn-gate-blk-0-ffn-gate-weight-reduce-unroll4` | `blk.0.ffn_gate.weight` | `reduce_unroll4` | `q4_k_packed_u32` | 1 | `LOCAL:0:64,UNROLL:2:4` | 0 | 0 | `True` |
| `004-ffn-gate-blk-0-ffn-gate-weight-two-dim-local4` | `blk.0.ffn_gate.weight` | `two_dim_local4` | `q4_k_packed_u32` | 1 | `LOCAL:0:32,LOCAL:1:4` | 0 | 0 | `True` |
| `005-ffn-down-blk-4-ffn-down-weight-row-upcast2` | `blk.4.ffn_down.weight` | `row_upcast2` | `q4_k_packed_u32` | 4 | `LOCAL:0:32,UPCAST:0:2` | 0 | 0 | `True` |
| `006-ffn-down-blk-4-ffn-down-weight-reduce-unroll4` | `blk.4.ffn_down.weight` | `reduce_unroll4` | `q4_k_packed_u32` | 4 | `LOCAL:0:32,UNROLL:2:4` | 0 | 0 | `True` |
| `007-ffn-down-blk-4-ffn-down-weight-two-dim-local4` | `blk.4.ffn_down.weight` | `two_dim_local4` | `q4_k_packed_u32` | 4 | `LOCAL:0:64,LOCAL:1:4` | 0 | 0 | `True` |
| `008-attn-q-blk-0-attn-q-weight-direct-out` | `blk.0.attn_q.weight` | `direct_out` | `q4_k_packed_u32_direct` | 1 | `LOCAL:0:64` | 0 | 0 | `False` |
| `009-attn-q-blk-0-attn-q-weight-row-upcast2` | `blk.0.attn_q.weight` | `row_upcast2` | `q4_k_packed_u32` | 1 | `LOCAL:0:64,UPCAST:0:2` | 0 | 0 | `True` |
| `010-attn-q-blk-0-attn-q-weight-reduce-unroll4` | `blk.0.attn_q.weight` | `reduce_unroll4` | `q4_k_packed_u32` | 1 | `LOCAL:0:64,UNROLL:2:4` | 0 | 0 | `True` |
| `011-attn-q-blk-0-attn-q-weight-two-dim-local4` | `blk.0.attn_q.weight` | `two_dim_local4` | `q4_k_packed_u32` | 1 | `LOCAL:0:32,LOCAL:1:4` | 0 | 0 | `True` |
| `012-ffn-down-blk-0-ffn-down-weight-row-upcast2` | `blk.0.ffn_down.weight` | `row_upcast2` | `q6_k_packed_u16` | 2 | `LOCAL:0:64,UPCAST:0:2` | 0 | 0 | `True` |
| `013-ffn-down-blk-0-ffn-down-weight-reduce-unroll4` | `blk.0.ffn_down.weight` | `reduce_unroll4` | `q6_k_packed_u16` | 2 | `LOCAL:0:64,UNROLL:2:4` | 0 | 0 | `True` |
| `014-ffn-down-blk-0-ffn-down-weight-two-dim-local4` | `blk.0.ffn_down.weight` | `two_dim_local4` | `q6_k_packed_u16` | 2 | `LOCAL:0:32,LOCAL:1:4` | 0 | 0 | `True` |
