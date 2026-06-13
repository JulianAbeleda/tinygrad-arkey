# QK Semantic Codegen v4 Static Gate: 8B

Fail-closed structural validation before microbench. Passing here means
the candidate is a Q4_K ffn_gate aligned uint32x4 vector-load probe, not
that it is eligible for full decode.

## Summary

- candidates: `2`
- passing microbench: `1`
- full-decode supported: `1`
- failing: `0`

| id | status | microbench | full decode | persistent delta | reasons |
|---|---|---:|---:|---:|---|
| `current` | `baseline` | `False` | `True` | 0 | none |
| `001-ffn-gate-blk-0-ffn-gate-weight-vector-load-u32x4` | `pass` | `True` | `False` | 0 | none |
