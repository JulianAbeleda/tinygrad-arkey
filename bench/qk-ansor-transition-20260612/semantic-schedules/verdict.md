# QK Semantic Schedule Verdict

This is the 8B/14B gate for the first semantic schedule/codegen surface.
32B is intentionally excluded unless both target models show promise.

## Summary

- overall decision: `semantic_schedule_v0_rejected`
- microbench accepts: `3`
- full-decode candidates: `2`
- full-decode confirmed accepts: `0`
- full-decode raw accepts awaiting confirmation: `0`
- run 32B: `False`

Reasons:

- 8B full decode reject 009-attn-q-blk-0-attn-q-weight-row-upcast2: -10.28%
- 14B full decode reject 009-attn-q-blk-0-attn-q-weight-row-upcast2: -5.21%
- 32B skipped by default because the 8B/14B semantic gate did not accept

## Models

| model | microbench accepts | full-decode ready | full-decode status | gain % | explicit tok/s | generated tok/s |
|---|---:|---|---|---:|---:|---:|
| 8B | 2 | `009-attn-q-blk-0-attn-q-weight-row-upcast2` | `reject` | -10.28 | 53.27 | 47.79 |
| 14B | 1 | `009-attn-q-blk-0-attn-q-weight-row-upcast2` | `reject` | -5.21 | 38.13 | 36.14 |

## Interpretation

The isolated attention `row_upcast2` microbench win did not survive full decode.
This rejects the current semantic schedule v0 surface as a promotion path.
The next research step needs a richer semantic layout/codegen capability, not
another sweep over these same schedule sketches.
