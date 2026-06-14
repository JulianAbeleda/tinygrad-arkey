# QK Semantic Codegen v3 Candidates: 8B

Family C v0 tests a packed-load Q4_K `ffn_gate` partial GEMV. It is a
memory-access probe: it changes the reduce axis from per-position qword
indexing to explicit packed-word lanes that unroll four nibbles from each
loaded uint32.

## Summary

- candidates: `4`
- single-change candidates: `3`
- current storage bytes: `84934656`

| id | tensor | role | family | load mode | parts | opts | persistent delta | full decode |
|---|---|---|---|---|---:|---|---:|---:|
| `current` | n/a | n/a | n/a | n/a | 0 | n/a | 0 | `True` |
| `001-ffn-gate-blk-1-ffn-gate-weight-packed-load-u32x4` | `blk.1.ffn_gate.weight` | `ffn_gate` | `q4_k_packed_u32_packed_load` | `u32_load_once_per_4_nibbles` | 1 | `LOCAL:0:64` | 0 | `False` |
| `002-ffn-gate-blk-2-ffn-gate-weight-packed-load-u32x4` | `blk.2.ffn_gate.weight` | `ffn_gate` | `q4_k_packed_u32_packed_load` | `u32_load_once_per_4_nibbles` | 1 | `LOCAL:0:64` | 0 | `False` |
| `003-ffn-gate-blk-3-ffn-gate-weight-packed-load-u32x4` | `blk.3.ffn_gate.weight` | `ffn_gate` | `q4_k_packed_u32_packed_load` | `u32_load_once_per_4_nibbles` | 1 | `LOCAL:0:64` | 0 | `False` |
