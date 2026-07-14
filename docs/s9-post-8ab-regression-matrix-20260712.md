# Post-8ab S9 regression matrix (historical, superseded)

> Superseded 2026-07-13: these runs used a raw pipe route that launched LDS geometry and computed only 1/16 of its
> output tiles. The matrix is useful commit-history evidence, but none of its throughput rows is a valid oracle.

Measured with the exact S9 authority (`DEV=AMD PREFILL_V2=1
PREFILL_GRAPH_GEMM=1`, K=8, warmups=4, rounds=3, ctx512, pinned clocks):

| State | Result | Route relevance |
|---|---:|---|
| `8ab3ee80c` | 116.2 ms / 4405 tok/s | fast S9 baseline |
| `c317132ef` | 116.0 ms / 4413 tok/s | fast S9 baseline |
| `84536d5c6` | 124.8 ms / 4101 tok/s | first slow state |
| `84536d5c6` + direct `load_table()` | 116.4 ms / 4398 tok/s | minimal isolated recovery |

The loader attribution was not reproduced on the corrected stack and is withdrawn. The valid comparison is the
corrected hybrid authority (`244.31 ms / 2095.70 tok/s`) with whole-model parity, not any row above.
