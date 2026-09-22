# Gate/up Stream-K outer-K unroll sweep — 2026-09-06

All arms use the required llama-compatible compact-Q8 producer, the exact 72
canonical gate/up weights, alternating generated/llama timing, and nine rounds.
All four pass both activation correctness checks and exact candidate/llama
program censuses.

| unroll | generated median ms | llama median ms |
|---:|---:|---:|
| 4 | 21.750267 | 14.287078 |
| 8 | 19.806763 | 14.511715 |
| 16 | 25.458118 | 14.328583 |
| 32 | 25.960151 | 14.258344 |

The retained unroll 8 remains clearly best. This one-variable lane is STOP;
none closes the gate/up service gap. Earlier legacy-producer failures are
invalid performance arms and were not retained as candidate evidence.
