# QK Semantic Codegen Static Gate: 14B

Fail-closed structural validation before microbench. Passing here means
the candidate is an exact-tensor direct-output Q4_K override that the
runtime can install for full decode.

## Summary

- candidates: `5`
- passing microbench: `4`
- full-decode supported: `5`
- failing: `0`

| id | status | microbench | full decode | reasons |
|---|---|---:|---:|---|
| `current` | `baseline` | `False` | `True` | none |
| `001-ffn-gate-blk-0-ffn-gate-weight-direct-out-tensor` | `pass` | `True` | `True` | none |
| `002-ffn-down-blk-5-ffn-down-weight-direct-out-tensor` | `pass` | `True` | `True` | none |
| `003-attn-q-blk-0-attn-q-weight-direct-out-tensor` | `pass` | `True` | `True` | none |
| `004-attn-k-blk-0-attn-k-weight-direct-out-tensor` | `pass` | `True` | `True` | none |
