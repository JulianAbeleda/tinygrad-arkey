# SUPERSEDED / INVALID — Q4-down candidate/control decision

This document must not be used as a performance or correctness result. Its
FP16 control forced host `sum().item()` reads, making the timing apples-to-
oranges, and it did not establish numerical equivalence. It is superseded by
the synchronized same-fixture authority:
`docs/task_workflow/evidence/nv-q4down-matched-ab-20260829/result.json`.
That authority reports correctness failure (`max_abs=2.695646`) and a slower
candidate, so the current Q4-down spelling is STOP.

The synchronized ordinary FP16 control was run over the exact 18 GGUF type-12
roles (blk.4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,28,29), with every
output forced through a host `sum().item()` read in each R9 round.

| route | min ms | median ms | verdict |
|---|---:|---:|---|
| Q8-record + frozen Q4 main candidate | 22.643592 | 23.586751 | PASS |
| installed FP16 overlay control | 356.103492 | 363.353684 | PASS |

The candidate is 333.459900 ms faster at the minimum and 339.766933 ms faster
at the median for the 18-role batch. Both routes are finite and enumerate all
18 roles. The current control used the captured saved-Z input while the earlier
candidate population used its deterministic bounded input; therefore this is
a lifecycle timing comparison, not yet a bitwise output comparison. A same-
fixture rerun is required before claiming numerical equivalence.
