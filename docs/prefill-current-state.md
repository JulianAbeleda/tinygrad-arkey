# 8B Prefill Current State

This is the compact authority for the shipped Qwen3-8B gfx1100 prefill route. Historical scopes and failed benchmark
banks live in Git history, not on the active repository surface.

Last updated: 2026-07-13.

## Shipped route

- Route: `prefill_wmma_lds_dbuf_generated`.
- Status: promoted default for the exact pp512 roles `attn_qo`, `attn_kv`, `ffn_down`, and `ffn_gate_up`.
- Ownership: ordinary tinygrad matmul lowered from a typed `KernelCandidateContext`; there is no hand-emitted pipe/LDS2
  kernel on the runtime path.
- Pipeline: fp16 operands, fp32 accumulation, two LDS slots, 256 threads, and exact role/shape/target admission from the
  canonical candidate set.
- Device: normal `DEV=AMD` / HIP renderer. `DEV=AMD:ISA` is optional compiler-analysis tooling and is not imported or
  selected by production execution.
- Rollback: `PREFILL_GRAPH_GEMM=0` returns to the ordinary tinygrad scheduler. Retired raw-kernel selectors fail loud.

## Durable evidence

The only promotion evidence retained in-tree is under
`bench/prefill-pure-full-kernel/multirole-buffer2-candidate-set-v1/`:

- `candidate-set.json`: four exact admitted payloads and canonical identities.
- `whole-model-quality.json`: PASS, three deterministic greedy cases, baseline/candidate token parity, route bound, and
  healthy GPU after both isolated children.
- `whole-prefill-pinned.json`: clean authority at commit `8045efcef`, pinned pp512 `3561.32 tok/s`, all four candidate
  identities observed, no missing or unexpected bindings.

The current raw-versus-practical placement is owned by BoltBeam in
`BoltBeam/docs/qwen3-8b-current-dual-roofline-20260713.md`. The retired
`bench/qk-prefill-theoretical-ceiling/latest.json` missing-evidence placeholder is intentionally removed.

## Closed branches

- Historical `~4413` and recreated `~4099` pipe results were invalid: leaked LDS geometry launched only 1/16 of the
  pipe-owned output. Correcting geometry and buffer effects produced parity but only `2095.70 tok/s`.
- The raw LDS2 oracle hung the GPU and is not benchmark-eligible.
- The old S9/S10, hybrid, raw pipe/LDS2, single-buffer, and environment-driven local-stage experiments are deleted from
  the active tree. Their conclusions remain in `docs/prefill-lessons-ledger.md` and Git history.

## Change rule

A replacement must supply an exact typed candidate, isolated correctness and GPU-health evidence, whole-model parity,
route-census identity, and pinned timing before it can replace the current default. A faster incomplete or unverified
kernel is not a performance authority.
