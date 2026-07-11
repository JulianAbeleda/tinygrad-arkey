# MMQ invocation-v5 grouped-predicate closure

Artifact: `mmq-invocation-v5-20260711.json`

Invocation-v5 compares a generated noncandidate 407-UOp baseline, a 255-site topology where one CMPNE predicate is reused across up to four stores, and the invocation-v4 one-AND-per-store topology. No candidate binaries, candidate timings, or device launches are present.

## Admission

| Cell | STORE | INDEX | AND | CMPNE |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0 | 0 | 0 | 0 |
| Grouped, 255 sites | 255 | 255 | 256 | 64 |
| Per-store, 255 sites | 255 | 255 | 255 | 510 |

These exact UOp deltas match the contract. The first grouped formulation used unique `.eq()` ownership predicates and was rejected because tinygrad represented those equalities through 510 additional CMPNE nodes, producing 574 rather than 64 CMPNEs. Replacing the unique ownership side with CMPLT preserved the distinct AND nodes without changing the CMPNE axis. The admitted artifact contains full operation histograms and an empty final failure audit.

## Protocol

- 30 samples per cell after 5 warmups, randomized and interleaved across 90 trials.
- Separate UOp construction, schedule creation, and warmed compile/cache lookup.
- Resident pre-realized AMD inputs; host-only phases and no runtime launch.
- Empty `perf_counter_ns` overhead median is 40 ns and is subtracted from medians.
- Exact sink UOps and rendered source identities are recorded per cell.

## Grouping effect

| Phase | Baseline | Grouped per site | Per-store per site | Grouped saving per site |
| --- | ---: | ---: | ---: | ---: |
| UOp construction | 1.747 ms | 17.974 us | 24.240 us | 6.266 us |
| Schedule creation | 0.995 ms | 8.862 us | 13.444 us | 4.581 us |
| Warmed cache lookup | 65.60 us | 5.56 ns | 8.07 ns | 2.51 ns |

The grouped topology has 1,878 sink UOps versus 2,705 for the per-store topology. Both render to 15,944 bytes and 314 statements, but their distinct source hashes and program keys preserve exact identity. Predicate reuse therefore removes 827 host graph nodes and materially lowers construction and scheduling cost while holding STORE and INDEX counts fixed at the candidate direct-to-gated delta.
