# QK Semantic Codegen v2 Static Gate: 14B

Fail-closed structural validation before microbench. Passing here means
the candidate is a Q4_K ffn_down row-grouped probe, not that it is
eligible for full decode.

## Summary

- candidates: `3`
- passing microbench: `2`
- full-decode supported: `1`
- failing: `0`

| id | status | microbench | full decode | persistent delta | metadata sidecar | reasons |
|---|---|---:|---:|---:|---:|---|
| `current` | `baseline` | `False` | `True` | 0 | 0 | none |
| `001-ffn-down-blk-5-ffn-down-weight-row-group2` | `pass` | `True` | `False` | 0 | 0 | none |
| `002-ffn-down-blk-5-ffn-down-weight-row-group4` | `pass` | `True` | `False` | 0 | 0 | none |
