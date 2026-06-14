# QK_BLOCK_DOT Microbench

Decision: `qk_block_dot_microbench_rejected`

Repeated dominant-shape microbench for the first `QK_BLOCK_DOT` lowering.
This is still not a runtime integration or full-decode artifact.

## Summary

- metric: `median device_q4_gbs`
- v1 median device Q4 GB/s: `154.10`
- QK_BLOCK_DOT median device Q4 GB/s: `96.32`
- gain: `-37.49%`
- promotion bar: `>=10.00%`
- correctness ok: `True`
- run full decode next: `False`

Device timing is the gate metric. Wall timing is recorded only as secondary
diagnostic data and is noisy when the run is executed with `DEBUG=2` to
collect AMD device times.

## Modes

| mode | runs | median device GB/s | median device ms | median wall GB/s | max_abs max |
|---|---:|---:|---:|---:|---:|
| `v1_partial` | 5 | 154.10 | 0.061240 | 2.71 | 0.00144577 |
| `qk_block_dot` | 5 | 96.32 | 0.097976 | 8.31 | 0.00145054 |

## Interpretation

The compile-shape win did not translate into enough repeated microbench
speedup. Do not integrate `QK_BLOCK_DOT` into runtime or run full decode
from this result. The next research step should inspect why the wider
loads are not paying off at this shape.
