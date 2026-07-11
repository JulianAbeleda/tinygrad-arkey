# MMQ invocation-v1 generated host factorial

Artifact: `mmq-invocation-v1-20260711.json`

This calibration uses generated, explicitly noncandidate IDs at 0, 64, 128, and 256 false writeback sites. Each point retains the 16x16x256 MMQ argument and launch geometry. Runtime phases reuse preallocated resident AMD buffers; tensor transfer and output allocation are isolated phases.

## Protocol

- 30 samples per point after 5 warmups, randomized and interleaved across 120 trials.
- `perf_counter_ns` empty-pair overhead median: 40 ns, subtracted from host phase medians.
- Direct runtime `wait=True` device duration is reported separately. The enqueue/sync host residual subtracts that duration, and device kernel time is excluded from every host fit.
- Exact generated ID, program key, source/binary/ISA hashes, source size, UOp count, and ISA instruction count are recorded for every point.

## Host relationships

| Phase | Intercept | Per false site | R2 |
| --- | ---: | ---: | ---: |
| UOp construction | 267.5 us | 37.692 us | 1.00000 |
| Schedule creation | 464.2 us | 9.285 us | 0.99993 |
| Warmed compile/cache lookup | 69.9 us | 0.008 us | 0.83113 |
| Enqueue/sync host residual | 302.8 us | 0.232 us | 0.52349 |
| Q4 construct/realize/transfer | 634.5 us | -0.003 us | 0.04217 |
| DS4 construct/realize/transfer | 1805.2 us | -0.003 us | 0.00340 |
| Output allocation | 80.4 us | 0.001 us | 0.87455 |
| Output readback | 92.0 us | 0.011 us | 0.94359 |

UOp count grows from 33 to 1,581 and rendered source grows from 992 to 12,169 bytes. Final ISA remains 153 instructions at all four points, while median device duration remains 7.08-7.52 us. The compiler removes the dynamically impossible stores downstream; the strong slopes therefore measure host graph and scheduling complexity, not v7 device execution cost.

The low-R2 site slopes for transfer, allocation, readback, and enqueue/sync should be treated as invariant overhead/noise observations, not predictive site-cost terms. Invocation modeling should use the UOp-construction and schedule relationships and retain the other phases as separately measured fixed-cost distributions.
