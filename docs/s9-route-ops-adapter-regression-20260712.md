# S9 route-ops adapter regression

The exact authority at `84536d5c6` measured 124.8 ms / 4101 tok/s. In an
isolated worktree, changing only `tinygrad/llm/model.py` from
`qk_ops.prefill_v2_load_table()` back to the direct
`extra.qk.prefill_v2_schedule_search.load_table()` import produced 116.4 ms /
4398 tok/s under the same K8/warmup4/round3/pinned ctx512 protocol.

This proves the route-ops adapter boundary causes approximately 8.4 ms of S9
latency. The underlying table loader is identical; the difference is the
adapter's lazy indirection/import path. No route geometry changed. The
minimal fix is to restore the direct loader call for this hot model-build
path, or make the adapter cache the resolved callable before authority timing.
