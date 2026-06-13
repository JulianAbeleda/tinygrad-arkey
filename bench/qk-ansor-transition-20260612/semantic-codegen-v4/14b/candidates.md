# QK Semantic Codegen v4 Candidates: 14B

Family C v1 tests an aligned `uint32x4` Q4_K `ffn_gate` partial GEMV. It
is a memory-access probe: it requests one vector load for four adjacent
packed quant words, then unpacks thirty-two 4-bit values from those loaded lanes.

## Summary

- candidates: `2`
- single-change candidates: `1`
- current storage bytes: `195379200`

| id | tensor | role | family | load tile | q/load | parts | opts | persistent delta | full decode |
|---|---|---|---|---|---:|---:|---|---:|---:|
| `current` | n/a | n/a | n/a | n/a | 0 | 0 | n/a | 0 | `True` |
| `001-ffn-gate-blk-0-ffn-gate-weight-vector-load-u32x4` | `blk.0.ffn_gate.weight` | `ffn_gate` | `q4_k_packed_u32_vector_load` | `u32x4_aligned` | 32 | 1 | `LOCAL:0:32` | 0 | `False` |
