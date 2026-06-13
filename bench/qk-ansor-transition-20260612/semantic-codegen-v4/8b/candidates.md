# QK Semantic Codegen v4 Candidates: 8B

Family C v1 tests an aligned `uint32x4` Q4_K `ffn_gate` partial GEMV. It
is a memory-access probe: it requests one vector load for four adjacent
packed quant words, then unpacks sixteen nibbles from those loaded lanes.

## Summary

- candidates: `2`
- single-change candidates: `1`
- current storage bytes: `107347968`

| id | tensor | role | family | load mode | parts | opts | persistent delta | full decode |
|---|---|---|---|---|---:|---|---:|---:|
| `current` | n/a | n/a | n/a | n/a | 0 | n/a | 0 | `True` |
| `001-ffn-gate-blk-0-ffn-gate-weight-vector-load-u32x4` | `blk.0.ffn_gate.weight` | `ffn_gate` | `q4_k_packed_u32_vector_load` | `aligned_u32x4_global_load` | 1 | `LOCAL:0:64` | 0 | `False` |
