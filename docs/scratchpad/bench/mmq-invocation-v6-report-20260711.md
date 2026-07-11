# MMQ invocation-v6 exact-backbone probe

Artifact: `mmq-invocation-v6-20260711.json`

Invocation-v6 holds the generated ownership core and exact total sink size fixed while varying the filler backbone: comparison filler, candidate-style indexed arithmetic, and arithmetic plus bitwise plus control. Every cell is exactly 1,246 UOps.

## Admission

Admission binds the complete opcode histogram, exact event count and type, flattened builder-operation event counts, and maximum dependency depth:

| Backbone | Events | UOps | Dependency depth |
| --- | ---: | ---: | ---: |
| Comparison filler | 28 | 1,246 | 68 |
| Candidate arithmetic | 14 | 1,246 | 40 |
| Arithmetic + bitwise + control | 9 | 1,246 | 31 |

The first tuning pass produced 1,246 / 1,247 / 1,245 UOps. Reusing one arithmetic multiplier constant and adding one distinct terminal bitmask constant corrected the latter cells exactly without changing the ownership core. Final full-histogram admission passes with an empty failure audit.

## Protocol

- 30 samples per cell after 5 warmups, randomized and interleaved across 90 trials.
- Separate host UOp construction, schedule creation, and warmed compile/cache lookup.
- Resident pre-realized AMD inputs; no device launch or candidate timing.
- Empty `perf_counter_ns` overhead median is 40 ns and is subtracted from medians.
- Every Python construction event is recorded with its event index, type, and intended builder operations.

## Result

| Backbone | UOp construction | Schedule creation | Warmed cache lookup |
| --- | ---: | ---: | ---: |
| Comparison filler | 3.235 ms | 2.474 ms | 68.47 us |
| Candidate arithmetic | 3.485 ms | 2.466 ms | 68.48 us |
| Arithmetic + bitwise + control | 3.455 ms | 2.465 ms | 68.58 us |

At exact equal graph size, scheduling varies by less than 0.4% and warmed lookup by less than 0.2%. Python UOp construction remains sensitive to builder-event mix: candidate-style arithmetic is about 7.7% above comparison filler, while the full backbone is about 6.8% above comparison filler.

## Noncandidate separation

The full backbone is not the candidate implementation. It lives under `generated_noncandidate.mmq_invocation_v6`, the candidate kernel builder is not imported, candidate ID/binary/timing arrays are empty, and all three generated sink hashes are unique. The full cell also has its own source hash and program key. These identities prove separation even though its generated operation families intentionally resemble candidate arithmetic, bitwise, and control structure.
