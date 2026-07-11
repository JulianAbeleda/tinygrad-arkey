# MMQ invocation-v3 topology interaction

Artifact: `mmq-invocation-v3-20260711.json`

Invocation-v3 is the exact requested four-cell generated noncandidate contract: base targets 1,024 and 1,280 UOps crossed with 0 and 255 false owner sites. The achieved base sinks are 1,023 and 1,279 UOps. Each false site uses the candidate-shaped per-output `(batch==mi && row==ni)` ownership predicate plus impossible `lane==32`, so all false stores are structurally present and dynamically excluded.

## Protocol

- Exactly 30 samples per cell after 5 warmups, randomized and interleaved across 120 trials.
- Separate UOp construction, schedule creation, and warmed compile/cache lookup phases.
- Resident pre-realized AMD input buffers; no device launch in any measured phase.
- No candidate IDs, candidate binaries, or candidate timings are collected.
- Empty `perf_counter_ns` pair overhead median is 40 ns and is subtracted from host medians.
- Exact sink UOps, rendered source bytes/lines/statements, source hash, and generated program key are recorded per cell.

## Four-cell contrast

The saturated topology model is `intercept + base_uops*B + false_sites*F + base_uops*false_sites*I`.

| Phase | Intercept | B | F | I |
| --- | ---: | ---: | ---: | ---: |
| UOp construction | 96.37 us | 4.236 us/UOp | 54.910 us/site | -0.1186 ns/UOp/site |
| Schedule creation | 312.08 us | 1.642 us/UOp | 7.199 us/site | 0.0938 ns/UOp/site |
| Warmed compile/cache lookup | 65.35 us | 2.25 ns/UOp | 15.17 ns/site | -0.0084 ns/UOp/site |

At the midpoint base complexity, the construction interaction adjusts the false-site slope by about -0.137 us/site and the schedule interaction by about +0.108 us/site. These are small compared with their false-site main effects. Total sink identity grows from 1,023/1,279 UOps at zero sites to 2,110/2,366 UOps at 255 sites; rendered source grows from 11,427/14,244 bytes to 22,560/25,377 bytes.

This is a saturated four-cell contrast with zero residual degrees of freedom. Its algebraic R2 is necessarily 1.0 and supplies no uncertainty estimate. The coefficients apply only to these bounded cells and must not be presented as an independently validated extrapolation.
