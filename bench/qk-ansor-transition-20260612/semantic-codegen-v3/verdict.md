# QK Semantic Codegen v3 Verdict

This is the 8B/14B gate for Family C v0: exact-tensor Q4_K ffn_gate
packed-load partial GEMV. 32B is intentionally excluded unless both
target models show promise.

## Summary

- overall decision: `semantic_codegen_v3_rejected`
- microbench rows: `2`
- raw microbench accepts: `0`
- strong raw microbench accepts: `0`
- microbench invalid: `0`
- full-decode candidates: `0`
- full-decode confirmed accepts: `0`
- run 32B: `False`

Reasons:

- 8B no raw accepts (1 ties)
- 14B no raw accepts (1 ties)
- full decode and 32B skipped because the 8B/14B microbench gate produced no strong raw accepts
- Family C v0 is a packed-load memory-access probe, not a schedule-only knob

## Models

| model | row | status | gain % | current GB/s | candidate GB/s | reasons |
|---|---|---|---:|---:|---:|---|
| 8B | `001-ffn-gate-blk-0-ffn-gate-weight-packed-load-u32x4` | `tie` | -0.65 | 206.42 | 205.07 | within tie_band=0.030 |
| 14B | `001-ffn-gate-blk-0-ffn-gate-weight-packed-load-u32x4` | `tie` | -0.31 | 367.98 | 366.84 | within tie_band=0.030 |

## Interpretation

Family C v0 is accepted only if the packed-load rewrite produces a strong
raw microbench gain. A weak raw accept is not enough for full decode because
single-tensor gains dilute at model scope. No accept means the next step is
hardware-counter profiling or a deeper renderer memory-layout capability,
not another schedule-only variant.
