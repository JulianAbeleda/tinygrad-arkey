# QK Semantic Codegen v4

Family C v1: aligned `uint32x4` Q4_K `ffn_gate` vector-load probe.

This artifact follows the memory-access source gate in
`bench/qk-memory-access-20260613/`: raw AMD UOps can now preserve an aligned
`uint32.vec(4)` global load/store. v4 tests whether the real Q4_K GEMV can use
that vector load inside the unpack/dot expression.

Result: rejected at construction. 8B and 14B both fail before candidate timing,
so there are no raw accepts, no full-decode candidates, and no 32B run.

Key files:

- `8b/` and `14b/`: candidate sets, static gates, microbench reports, and logs.
- `load-width/`: DEBUG=4 source-shape logs and parser output.
- `verdict.json` / `verdict.md`: combined 8B/14B decision.

Interpretation: the raw vector load/store lowering exists, but the GEMV path
needs core vector-lane extraction/vector-shape support or a first-class packed
QK load/decode op before Family C can produce a timed vector-load kernel.
