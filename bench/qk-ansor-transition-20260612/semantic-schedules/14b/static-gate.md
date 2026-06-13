# QK Semantic Schedule Static Gate: 14B

Fail-closed structural validation before microbench. Passing here does not
mean the schedule compiles or wins; it only means the candidate is shaped
well enough to test in an isolated microbench.

## Summary

- candidates: `15`
- passing microbench: `14`
- full-decode supported: `13`
- failing: `0`

| id | status | microbench | full decode | persistent delta | metadata sidecar | reasons |
|---|---|---:|---:|---:|---:|---|
| `current` | `baseline` | `False` | `True` | 0 | 0 | none |
| `001-ffn-gate-blk-0-ffn-gate-weight-direct-out` | `pass` | `True` | `False` | 0 | 0 | none |
| `002-ffn-gate-blk-0-ffn-gate-weight-row-upcast2` | `pass` | `True` | `True` | 0 | 0 | none |
| `003-ffn-gate-blk-0-ffn-gate-weight-reduce-unroll4` | `pass` | `True` | `True` | 0 | 0 | none |
| `004-ffn-gate-blk-0-ffn-gate-weight-two-dim-local4` | `pass` | `True` | `True` | 0 | 0 | none |
| `005-ffn-down-blk-5-ffn-down-weight-row-upcast2` | `pass` | `True` | `True` | 0 | 0 | none |
| `006-ffn-down-blk-5-ffn-down-weight-reduce-unroll4` | `pass` | `True` | `True` | 0 | 0 | none |
| `007-ffn-down-blk-5-ffn-down-weight-two-dim-local4` | `pass` | `True` | `True` | 0 | 0 | none |
| `008-attn-q-blk-0-attn-q-weight-direct-out` | `pass` | `True` | `False` | 0 | 0 | none |
| `009-attn-q-blk-0-attn-q-weight-row-upcast2` | `pass` | `True` | `True` | 0 | 0 | none |
| `010-attn-q-blk-0-attn-q-weight-reduce-unroll4` | `pass` | `True` | `True` | 0 | 0 | none |
| `011-attn-q-blk-0-attn-q-weight-two-dim-local4` | `pass` | `True` | `True` | 0 | 0 | none |
| `012-ffn-down-blk-0-ffn-down-weight-row-upcast2` | `pass` | `True` | `True` | 0 | 0 | none |
| `013-ffn-down-blk-0-ffn-down-weight-reduce-unroll4` | `pass` | `True` | `True` | 0 | 0 | none |
| `014-ffn-down-blk-0-ffn-down-weight-two-dim-local4` | `pass` | `True` | `True` | 0 | 0 | none |
