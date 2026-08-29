# NV pp512 final-row prune — STOP (2026-08-29)

The isolated arm is implemented in `extra/llm_research/prefill/nv_final_row_prune_gate.py`;
it is default-off and fail-closed. Numeric comparison of candidate `[0,0,:]`
against control `[0,511,:]` passes: `max_abs=0.0059030056`,
`mean_abs=0.00099487335`, `relL2=0.00030978036`,
`allclose(rtol=.02,atol=.5)=true`; argmax and top-10 are exact. Whole-tensor
hashes differ by design because the schedules have different shapes.

The production captured-graph bracket is complete. The matched unpruned arm
ran at `68.992404 ms` minimum / `69.177272 ms` median with the expected
population of 72 gate/up, 36 K, and 72 Q/O compiler mains. Deep replay, KV
state, activation freshness, and token `198` all passed. The pruned arm ran at
`69.269884 ms` minimum / `69.343492 ms` median with 70 gate/up, 36 K, and 72
Q/O mains. It also passed replay, KV, and token gates.

Pruning therefore regresses the matched production graph by `0.277480 ms` at
the minimum and `0.166220 ms` at the median. The direct uncaptured experiment
is not promoted over this captured authority. No packed-M=1 investment or
default change is justified.
