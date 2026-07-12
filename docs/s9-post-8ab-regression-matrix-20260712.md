# Post-8ab S9 regression matrix

Measured with the exact S9 authority (`DEV=AMD PREFILL_V2=1
PREFILL_GRAPH_GEMM=1`, K=8, warmups=4, rounds=3, ctx512, pinned clocks):

| State | Result | Route relevance |
|---|---:|---|
| `8ab3ee80c` | 116.2 ms / 4405 tok/s | fast S9 baseline |
| `c317132ef` | 116.0 ms / 4413 tok/s | fast S9 baseline |
| `84536d5c6` | 124.8 ms / 4101 tok/s | first slow state |
| `84536d5c6` + direct `load_table()` | 116.4 ms / 4398 tok/s | minimal isolated recovery |

The first slow state is the route-ops table-loader adapter introduced at
`84536d5c6`. Later candidate, census, and LM-head commits do not run when the
candidate set is absent and therefore cannot explain the original S9 boundary;
they can affect pure/mixed policies separately. The controlled interaction is
that the adapter must be removed (or its callable resolved/cached before model
capture). Recommended fix: restore the direct loader in the hot model-build
path, preserving route defaults and candidate policy.
