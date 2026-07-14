# S9 route-ops adapter regression (superseded)

> Superseded 2026-07-13: current exact A/B runs measured `4108` through the adapter, `4079` through an explicit
> adapter-owned import, and `4096` with the historical direct import in `model.py`. The loader change does not recover
> the fast band on the corrected stack. More importantly, the historical raw pipe used mismatched launch geometry and
> incomplete output, so neither the 4398 nor 4413 result is a valid correctness-qualified performance authority.

The exact authority at `84536d5c6` measured 124.8 ms / 4101 tok/s. In an
isolated worktree, changing only `tinygrad/llm/model.py` from
`qk_ops.prefill_v2_load_table()` back to the direct
`extra.qk.prefill_v2_schedule_search.load_table()` import produced 116.4 ms /
4398 tok/s under the same K8/warmup4/round3/pinned ctx512 protocol.

This historical correlation is retained as provenance, but its causal conclusion is withdrawn. Do not restore the
core-to-research import or optimize against this number. The corrected route geometry is the controlling authority.
