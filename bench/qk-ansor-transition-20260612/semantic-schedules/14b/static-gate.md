# QK Semantic Schedule Static Gate: 14B

Fail-closed structural validation before microbench. Passing here does not
mean the schedule compiles or wins; it only means the candidate is shaped
well enough to test in an isolated microbench.

## Summary

- candidates: `15`
- passing microbench: `14`
- full-decode supported: `13`
- failing: `0`

| id | status | microbench | full decode | reasons |
|---|---|---:|---:|---|
| `current` | `baseline` | `False` | `True` | none |
| `001-ffn-gate-blk-0-ffn-gate-weight-direct-out` | `pass` | `True` | `False` | none |
| `002-ffn-gate-blk-0-ffn-gate-weight-row-upcast2` | `pass` | `True` | `True` | none |
| `003-ffn-gate-blk-0-ffn-gate-weight-reduce-unroll4` | `pass` | `True` | `True` | none |
| `004-ffn-gate-blk-0-ffn-gate-weight-two-dim-local4` | `pass` | `True` | `True` | none |
| `005-ffn-down-blk-5-ffn-down-weight-row-upcast2` | `pass` | `True` | `True` | none |
| `006-ffn-down-blk-5-ffn-down-weight-reduce-unroll4` | `pass` | `True` | `True` | none |
| `007-ffn-down-blk-5-ffn-down-weight-two-dim-local4` | `pass` | `True` | `True` | none |
| `008-attn-q-blk-0-attn-q-weight-direct-out` | `pass` | `True` | `False` | none |
| `009-attn-q-blk-0-attn-q-weight-row-upcast2` | `pass` | `True` | `True` | none |
| `010-attn-q-blk-0-attn-q-weight-reduce-unroll4` | `pass` | `True` | `True` | none |
| `011-attn-q-blk-0-attn-q-weight-two-dim-local4` | `pass` | `True` | `True` | none |
| `012-ffn-down-blk-0-ffn-down-weight-row-upcast2` | `pass` | `True` | `True` | none |
| `013-ffn-down-blk-0-ffn-down-weight-reduce-unroll4` | `pass` | `True` | `True` | none |
| `014-ffn-down-blk-0-ffn-down-weight-two-dim-local4` | `pass` | `True` | `True` | none |
