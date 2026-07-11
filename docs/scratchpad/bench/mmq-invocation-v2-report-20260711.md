# MMQ invocation-v2 host interaction probe

Artifact: `mmq-invocation-v2-20260711.json`

Invocation-v2 is an independent 3x3 generated noncandidate probe. Requested base graph targets 32, 256, and 768 resolve to 33, 256, and 767 measured base sink UOps. They are crossed with 0, 128, and 256 false sites. The base chain consumes runtime Q4 input and depends on its preceding value, so its host UOps cannot collapse into a constant graph. At every fixed false-site point, exact UOp count and rendered source size increase across all three base levels.

## Protocol

- 30 samples per cell after 5 warmups; all 270 cells are randomized and interleaved.
- UOp construction, schedule creation, and warmed compile/cache lookup are separately timed.
- All phases reuse resident, pre-realized AMD buffers. No kernel launch occurs inside a measured phase, so device time is absent from and excluded from the host fits.
- Empty `perf_counter_ns` pair overhead median is 41 ns and is subtracted from phase medians.
- Every row records its generated noncandidate ID, requested and achieved base UOps, total sink UOps, rendered source counts, program key, and source/binary hashes.

## Interaction fits

The model is `intercept + base_uops*B + false_sites*F + base_uops*false_sites*I`.

| Phase | Intercept | B | F | I | R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| UOp construction | 123.86 us | 4.194 us/UOp | 37.236 us/site | 0.0248 ns/UOp/site | 0.999983 |
| Schedule creation | 374.09 us | 1.601 us/UOp | 9.686 us/site | -0.4235 ns/UOp/site | 0.999895 |
| Warmed compile/cache lookup | 65.80 us | 3.19 ns/UOp | 12.23 ns/site | -0.0074 ns/UOp/site | 0.941314 |

Construction is effectively additive across the two axes. Scheduling has a small negative interaction: at the largest 767-UOp base, it reduces the fitted false-site slope by about 0.325 us/site, roughly 3.4% of the main false-site term. Warm-cache lookup remains nearly flat in absolute terms.

## Candidate coverage

Canonical fixed sink counts are 405 UOps for direct-owner and 1,246 UOps for gated-matrix. The measured base domain is 33-767 UOps. Direct-owner is bracketed; gated-matrix is not. Therefore this probe does not fully cover the fixed candidate baseline, and its interaction model must not be used to extrapolate the gated base-complexity term without a higher-base follow-up point.
