# MMQ invocation-v7 writeback builder topology

Artifact: `mmq-invocation-v7-20260711.json`

Invocation-v7 crosses writeback iterations 1/256 with two generated noncandidate builder styles. Simple uses per-iteration constants, a 1D index, and constant equality. Candidate-shaped uses shared `SPECIAL` row/batch nodes, repeated mi/ni ownership constants, 2D indices, and one shared 32-deep reduced value.

## Admission and protocol

- All four attempted topology UOp histograms, edge counts, dependency depths, attempt counts, index dimensions, operand build counts, sharing fanout, and representation hashes match exact contracts.
- Every cell then returns the same canonical 1,246-UOp `ADD1243 / CONST1 / GROUP1 / SINK1` graph and hash.
- 30 samples per cell after 5 warmups, randomized across 120 trials.
- Explicit operand, equality, index, store, hash/canonicalization, residual, and total channels.
- Median overhead from 2,000 empty timer pairs is 41 ns.
- Untimed `sys.setprofile` exact-helper crosschecks pass for every cell.
- Host only; no candidate identity, candidate binary, candidate timing, or device launch.

## Medians

| Style / iterations | Operand | Equality | Index | Store | Hash/canonical | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Simple / 1 | 6.54 us | 6.45 us | 22.19 us | 2.43 us | 4.713 ms | 4.755 ms |
| Simple / 256 | 519.49 us | 788.82 us | 1.177 ms | 187.37 us | 4.870 ms | 7.585 ms |
| Candidate-shaped / 1 | 552.74 us | 93.54 us | 40.91 us | 2.43 us | 4.798 ms | 5.492 ms |
| Candidate-shaped / 256 | 554.48 us | 8.769 ms | 1.399 ms | 220.54 us | 4.820 ms | 15.792 ms |

## Factorial contrasts

The saturated model is `intercept + candidate_style + wb256 + candidate_style*wb256`.

| Channel | Candidate main | WB256 main | Interaction |
| --- | ---: | ---: | ---: |
| Operand | 546.202 us | 512.956 us | -511.212 us |
| Equality | 87.089 us | 782.366 us | 7,893.208 us |
| Index | 18.720 us | 1,154.656 us | 203.218 us |
| Store | 0.001 us | 184.938 us | 33.178 us |
| Hash/canonicalization | 84.934 us | 156.133 us | -134.238 us |
| Total | 736.750 us | 2,829.611 us | 7,470.498 us |

The shared deep value behaves as intended: candidate operand cost stays flat from 1 to 256 stores, whereas simple per-iteration values add about 513 us. Candidate-shaped ownership construction dominates at 256 iterations: its equality channel adds 8.676 ms from low to high writeback, versus 0.782 ms for simple equality. Since final canonical graph identity is fixed, this difference is Python attempted-builder topology cost, not final graph size or GPU execution.
