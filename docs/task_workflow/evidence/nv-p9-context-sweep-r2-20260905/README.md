# NV P9 context sweep R2

This normal-route coverage run used one model process at max context 4608 and
explicit `--request-scoped-prewarm`. No promotion flag was changed. Every
completed depth has an atomic R3-by-10-token timing artifact and a captured
PROGRAM census with each distinct actual source text stored once by SHA-256.
GPU state snapshots bracket only the timed decode windows.

Completed results:

| context | route | median ms/token | median tok/s |
| ---: | --- | ---: | ---: |
| 128 | SDPA | 5.0449 | 198.22 |
| 256 | SDPA | 5.9506 | 168.05 |
| 512 | Flash | 4.2287 | 236.48 |

The process then failed while prefilling the ctx1024 census: NV returned
`NV_ERR_NO_MEMORY` for an 8 MiB allocation with the allocator reporting 30.06
GiB used. An external snapshot shortly before the failure showed 31,148 MiB
used and 956 MiB free; immediately after process exit it showed 117 MiB used.
This establishes a same-process accumulated-memory failure after retaining
three depth-specific captures. It does not establish that ctx1024 is
intrinsically too large. The remaining depths must run in separate fresh
processes.

Request-scoped prewarming changes the declared output-horizon metadata and is
recorded in each artifact. It avoided the unrelated five-variant max4608 cold
startup seen in R1. These rows qualify route behavior and within-harness
latency by context; they are not paired llama comparisons.

Some SDPA census metadata contains `symbolic_uop_unresolved` records. These
preserve structural key, op, dtype, expression, argument and any direct BIND
value; they must not be described as resolved launch dimensions.
