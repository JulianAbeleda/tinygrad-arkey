# Q4-V live model fault (2026-08-29)

## Terminal boundary

The serialized Q4-V asset was accepted structurally from `/tmp/q4v-asset`:

- schema `tinygrad.nv.q4v.asset.v1`
- cubin SHA-256 matches manifest
- launch `(global=(32,8,1), local=(32,2,2))`
- ABI `globals=(0,1,2), outs=(0,), ins=(1,2), vals=()`
- no auxiliary/shared-memory requirement

The first real-model candidate capture was attempted with the exact graph-owned
binding, `--q4-v`, and the established gate/K/QO routes. It did not reach a
single qualified V projection. HCQ stopped at timeline signal `3114` while
waiting for `3118` (30 s), then reported `esr=4` faults on all SMs at
`warp_pc=0x230cd00000`. This is a hard GPU fault, not a timing regression.

Because the standalone producer/main and single-call gates were not isolated
before this attempt, no Q4-V model admission is made. The route remains
default-off and the exact terminal boundary is **before the first real-model
V call**. Do not advance to two-call, 18-call, TinyJit replay, or full graph
tests until a clean isolated single-call test exists.

The failure occurred in `nv_compiler_q4k_gkqo_model_arm.py` during `_capture`,
through `Tensor.realize`/`exec_kernel`; cleanup subsequently repeated the same
stored device error. No established K admission was changed.
