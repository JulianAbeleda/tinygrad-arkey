# QK Semantic Codegen v3 Static Gate: 8B

Fail-closed structural validation before microbench. Passing here means
the candidate is a Q4_K ffn_gate packed-load probe, not that it is
eligible for full decode.

## Summary

- candidates: `4`
- passing microbench: `3`
- full-decode supported: `1`
- failing: `0`

| id | status | microbench | full decode | persistent delta | reasons |
|---|---|---:|---:|---:|---|
| `current` | `baseline` | `False` | `True` | 0 | none |
| `001-ffn-gate-blk-1-ffn-gate-weight-packed-load-u32x4` | `pass` | `True` | `False` | 0 | none |
| `002-ffn-gate-blk-2-ffn-gate-weight-packed-load-u32x4` | `pass` | `True` | `False` | 0 | none |
| `003-ffn-gate-blk-3-ffn-gate-weight-packed-load-u32x4` | `pass` | `True` | `False` | 0 | none |
