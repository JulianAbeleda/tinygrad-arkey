# QK Semantic Codegen v2 Candidates: 14B

This bounded Family B surface tests row-grouped Q4_K `ffn_down` partial
GEMV. It is microbench-supported first; runtime full-decode installation
is intentionally deferred until a strong raw signal exists.

## Summary

- candidates: `3`
- single-change candidates: `2`
- current storage bytes: `195379200`

| id | tensor | role | family | row group | parts | opts | persistent delta | metadata sidecar | full decode |
|---|---|---|---|---:|---:|---|---:|---:|---:|
| `current` | n/a | n/a | n/a | 0 | 0 | n/a | 0 | 0 | `True` |
| `001-ffn-down-blk-5-ffn-down-weight-row-group2` | `blk.5.ffn_down.weight` | `ffn_down` | `q4_k_packed_u32_grouped` | 2 | 2 | `LOCAL:0:32,UPCAST:1:2` | 0 | 0 | `False` |
| `002-ffn-down-blk-5-ffn-down-weight-row-group4` | `blk.5.ffn_down.weight` | `ffn_down` | `q4_k_packed_u32_grouped` | 4 | 2 | `LOCAL:0:32,UPCAST:1:4` | 0 | 0 | `False` |
