# MMQ invocation-v5 CMPLT scaffolding probe

Artifact: `mmq-invocation-v5-scaffolding-20260711.json`

This three-cell generated noncandidate probe holds the exact core delta at `STORE255 / INDEX255 / AND256 / CMPNE64` and varies CMPLT scaffolding through 0, 128, and 256 nodes. CMPEQ nodes decrease complementarily, so every cell has exactly 1,247 sink UOps against the 1,246 target.

## Admission and iteration

Admission compares the complete opcode histogram, not only selected deltas. All opcode counts match hard-coded expected histograms for all three cells. Exact source hashes and program keys distinguish each cell.

The first formulation used unique constants for comparison identity. Although its tracked deltas were correct, its smallest graph was 1,454 UOps and it was rejected for missing the bounded target. The admitted formulation tags raw comparison identities without constant nodes, then adds 14 common dependent base steps to reach 1,247 UOps. The final failure audit is empty.

## Protocol

- 30 samples per cell after 5 warmups, randomized and interleaved across 90 trials.
- Separate host UOp construction, schedule creation, and warmed compile/cache lookup.
- Resident pre-realized AMD inputs; no device launch or candidate timing.
- Empty `perf_counter_ns` overhead median is 40 ns and is subtracted from medians.

## Result

| Phase | 0 CMPLT | 128 CMPLT | 256 CMPLT | Linear slope | R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| UOp construction | 3.455 ms | 3.430 ms | 3.450 ms | -21.94 ns/CMPLT | 0.046 |
| Schedule creation | 2.435 ms | 2.430 ms | 2.444 ms | 34.05 ns/CMPLT | 0.358 |
| Warmed cache lookup | 68.86 us | 68.75 us | 68.92 us | 0.23 ns/CMPLT | 0.121 |

The response is non-monotonic and the fitted slopes have low explanatory power. At fixed total UOp count and fixed STORE/INDEX/AND/CMPNE core topology, replacing CMPEQ scaffolding with CMPLT has no admitted material host-cost relationship in these samples. The earlier construction and scheduling costs are therefore explained by graph volume and topology, not a standalone CMPLT Python-cost term.
