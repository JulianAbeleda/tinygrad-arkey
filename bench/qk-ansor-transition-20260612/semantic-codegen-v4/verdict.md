# QK Semantic Codegen v4 Verdict

This is the 8B/14B gate for Family C v1: exact-tensor Q4_K ffn_gate
aligned uint32x4 vector-load partial GEMV. 32B is intentionally excluded
unless both target models show promise.

## Summary

- overall decision: `semantic_codegen_v4_rejected`
- microbench rows: `2`
- raw microbench accepts: `0`
- strong raw microbench accepts: `0`
- microbench invalid: `2`
- full-decode candidates: `0`
- full-decode confirmed accepts: `0`
- run 32B: `False`

Reasons:

- 8B no raw accepts (1 invalid)
- 14B no raw accepts (1 invalid)
- full decode and 32B skipped because the 8B/14B microbench gate produced no strong raw accepts
- Family C v1 is an aligned uint32x4 vector-load memory-access probe, not a schedule-only knob

## Models

| model | row | status | gain % | current GB/s | candidate GB/s | reasons |
|---|---|---|---:|---:|---:|---|
| 8B | `001-ffn-gate-blk-0-ffn-gate-weight-vector-load-u32x4` | `invalid` | n/a | 188.86 | n/a | candidate status=error |
| 14B | `001-ffn-gate-blk-0-ffn-gate-weight-vector-load-u32x4` | `invalid` | n/a | 352.85 | n/a | candidate status=error |

## Interpretation

Family C v1 is accepted only if the aligned vector-load rewrite produces a
strong raw microbench gain. A weak raw accept is not enough for full decode
because single-tensor gains dilute at model scope. No accept means the next
step is hardware-counter profiling or a deeper memory-layout/codegen
capability, not another schedule-only variant.
