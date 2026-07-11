# MMQ invocation-v4 exact topology probe

Artifact: `mmq-invocation-v4-20260711.json`

Invocation-v4 uses a generated noncandidate 407-UOp base and three cells: baseline, 255 false sites with two CMPNE predicates joined by one AND, and 255 false sites with three CMPNE predicates joined by two ANDs. No candidate binaries, candidate timings, or device launches are present.

## Admission

The first formulation was admitted without iteration. Exact sink-UOp deltas against baseline are:

| Cell | STORE | INDEX | CMPNE | AND |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0 | 0 | 0 | 0 |
| One-AND, 255 sites | 255 | 255 | 510 | 255 |
| Two-AND, 255 sites | 255 | 255 | 765 | 510 |

These exactly match the contract. The artifact includes full operation histograms and an empty bounded failure audit. Any mismatch rejects the run before timing.

## Protocol

- 30 samples per cell after 5 warmups, randomized and interleaved across 90 trials.
- Separate UOp construction, schedule creation, and warmed compile/cache lookup.
- Resident pre-realized AMD inputs; host-only phases with no device execution.
- Empty `perf_counter_ns` overhead median is 40 ns and is subtracted from medians.
- Exact sink-UOp count, rendered source bytes/lines/statements, source hash, and program key are recorded per cell.

## Topology cost

| Phase | Baseline | One-AND per site | Two-AND per site | Added CMPNE+AND per site |
| --- | ---: | ---: | ---: | ---: |
| UOp construction | 1.842 ms | 25.253 us | 36.814 us | 11.561 us |
| Schedule creation | 1.029 ms | 13.641 us | 19.356 us | 5.715 us |
| Warmed cache lookup | 68.22 us | 7.37 ns | 9.41 ns | 2.04 ns |

Sink identity grows from 407 UOps at baseline to 2,705 for one-AND and 3,726 for two-AND. Rendered source is 4,810 bytes at baseline and 15,944 bytes for both false-site cells; their distinct source hashes preserve exact topology identity even though final rendered size is equal. The result attributes the extra host cost to admitted UOp topology, not GPU kernel execution.
