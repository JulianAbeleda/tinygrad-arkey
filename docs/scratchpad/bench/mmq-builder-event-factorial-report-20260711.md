# MMQ builder-event factorial

Artifact: `mmq-builder-event-factorial-20260711.json`

This host-only 2x2x2 generated noncandidate factorial varies quant plus group helper calls 0/8, reduce calls 0/1, and writeback iterations 1/256. Attempted helper results are discarded; every cell canonicalizes to the same admitted 1,246-UOp `ADD1243 / CONST1 / GROUP1 / SINK1` graph and representation hash.

## Protocol

- 30 samples per cell after 5 warmups, randomized and interleaved across 240 trials.
- Channels: group, quant, reduce, equality, store, canonicalization, residual, and total.
- Median overhead from 2,000 empty `perf_counter_ns` pairs is 40 ns and is subtracted once per timed helper channel.
- Untimed `sys.setprofile` passes filter exact helper code objects and match every contracted call count in all eight cells.
- Expected equality/store attempts are recorded per row: `2*writeback_iterations` and `writeback_iterations`.
- No candidate identity, binary, timing, schedule, or device execution is present.

## Main coefficients

The saturated factorial uses binary terms for qg=8, reduce=1, and writeback=256.

| Channel | QG8 main | Reduce1 main | WB256 main |
| --- | ---: | ---: | ---: |
| Group | 21.230 us | 0.015 us | 0.005 us |
| Quant | 231.164 us | -0.001 us | -0.016 us |
| Reduce | 0.230 us | 146.426 us | -0.010 us |
| Equality | -3.377 us | -2.450 us | 1,572.167 us |
| Store | -3.166 us | -3.416 us | 2,678.177 us |
| Canonicalization | -0.091 us | 0.425 us | 1.082 us |
| Residual | 11.477 us | 6.176 us | 217.464 us |
| Total | 259.447 us | 149.261 us | 4,481.569 us |

The qg=8 main effect corresponds to roughly 2.65 us per group call and 28.90 us per quant call. The writeback main effect corresponds to roughly 3.08 us per additional equality attempt and 10.50 us per additional store attempt. Canonical graph construction remains approximately 573-576 us across all cells.

## Interactions

Total two-way coefficients are qg x reduce `-17.117 us`, qg x writeback `+59.182 us`, and reduce x writeback `+31.649 us`; the three-way coefficient is `-35.858 us`. These are saturated eight-cell contrasts with zero residual degrees of freedom, so they describe this bounded run rather than establish uncertainty or extrapolation.

The dominant total relationship is attempted writeback builder work, followed by quant/group and reduce helper work. Since the final sink is fixed exactly, these costs belong to Python construction attempts and orchestration, not the canonical graph size or GPU execution.
