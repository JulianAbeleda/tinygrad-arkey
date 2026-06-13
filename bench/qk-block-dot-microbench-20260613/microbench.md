# QK_BLOCK_DOT Microbench

Decision: `qk_block_dot_microbench_rejected`

Repeated dominant-shape microbench for the first `QK_BLOCK_DOT` lowering.
This is still not a runtime integration or full-decode artifact.

## Summary

- metric: `median device_q4_gbs`
- v1 median device Q4 GB/s: `407.99`
- QK_BLOCK_DOT median device Q4 GB/s: `285.01`
- gain: `-30.14%`
- promotion bar: `>=10.00%`
- correctness ok: `True`
- run full decode next: `False`

Device timing is the gate metric. Wall timing is recorded only as secondary
diagnostic data and is noisy when the run is executed with `DEBUG=2` to
collect AMD device times.

## Modes

| mode | runs | median device GB/s | median device ms | median wall GB/s | max_abs max |
|---|---:|---:|---:|---:|---:|
| `v1_partial` | 5 | 407.99 | 0.069392 | 8.01 | 0.00123835 |
| `qk_block_dot` | 5 | 285.01 | 0.099336 | 24.89 | 0.0012387 |

## Interpretation

The compile-shape win did not translate into enough repeated microbench
speedup. Do not integrate `QK_BLOCK_DOT` into runtime or run full decode
from this result. The next research step should inspect why the wider
loads are not paying off at this shape.
