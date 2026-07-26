# Retirement record: the Q4_K int8 WMMA-tiled prefill campaign

Route id `prefill_q4k_int8_wmma_tiled_research`, env `PREFILL_Q4K_Q8=wmma_tiled`. Campaign ran 2026-07-05 to
2026-07-14; superseded 2026-07-21. This document exists so the executable machinery can be deleted without losing the
result, the technique, or the one reusable pattern the campaign produced.

**Do not confuse this with packed-WMMA prefill**, which is a different campaign that SHIPPED and is default-on
(`tinygrad/llm/prefill_routes.py:293`, `getenv("TINYGRAD_PREFILL_PACKED_WMMA", 1)`), owned by
`extra/qk/prefill/packed_wmma_prefill_candidates.py`. The two share a name and nothing else. Verified 2026-07-26: no
packed-WMMA file references any file in this campaign.

## What it tried

RDNA3 (gfx1100) exposes `v_wmma_i32_16x16x16_iu8` -- a wave-wide 16x16x16 integer matrix multiply-accumulate, AMD's
tensor-core equivalent. Q4_K weights and Q8_1 activations are both integer, so routing prefill matmuls through int8
WMMA tiles should have delivered a large throughput win over the scalar/dot4 direct-packed route.

## The verdict (measured, quoted from `route_manifest.py` entry `prefill_q4k_int8_wmma_tiled_research`)

> "Generated scheduler-owned tiled WMMA compiles all four exact Qwen3-14B role shapes to iu8 WMMA and passes bounded
> full-K numeric probes; full attn_kv (512,1024,5120) also passed sampled real-GPU correctness. Packed-Q4 decode is
> fused so the 14B model runs without persistent expanded-weight buffers. **A route-bound replay measured 140 tok/s
> versus the recorded direct-packed authority baseline of 364.5 tok/s, so this schedule remains research-only and must
> not become the default.**"

It is a clean negative result, not an abandoned experiment: the approach was correct and reached real WMMA
instructions on every 14B role shape. It lost on speed by ~2.6x. Baseline provenance:
`docs/14b-direct-packed-prefill-authority-baseline-20260710.md`.

Commit `45cfc399c` (2026-07-21) removed the dispatch, stating: "superseded by the shipped scheduler-native
packed-WMMA route (`extra/qk/prefill/packed_wmma_prefill_candidates.py`)."

## Techniques that existed only in this code

**Bounded generated tiled loop.** `emit_q4k_int8_wmma_tiled_scheduler_tensor` in `prefill_int8_wmma_spec.py` used
typed `ScheduleHints` to centralize partial-contiguous ownership and tensor-core selection, so the correction reduction
became a bounded prerequisite and **the full `[groups,M,N]` RAW tensor was never materialized**. The RAW had to stay
tile-local. If a future campaign needs a generated tiled loop over quantized operands, this is the shape that worked.

**No-hand-kernel source scan.** `q4k_wmma_tiled_no_hand_kernel_gate.py` proved a route was machine-generated rather
than hand-authored, by scanning implementation files and the router's dispatch block for hand-kernel tokens (inline
asm, `Ops.WMMA`, `.custom_kernel`, source-string kernels). Nothing else in the repository implements this check
generically. If the machine-search purity contract ever needs enforcing for another route, re-derive it from
`git show 05b67146a -- extra/qk/q4k_wmma_tiled_no_hand_kernel_gate.py` rather than reinventing it.

## Why its gate cannot pass and is not a regression

`q4k_wmma_tiled_no_hand_kernel_gate.py` scans `tinygrad/llm/prefill_routes.py` for `if q8_mode == "wmma_tiled":`. That
branch was deleted with `prefill_research_routes.py` in `45cfc399c`, so the gate returns
`Q4K_WMMA_TILED_NO_HAND_KERNEL_FAIL` (`route.missing_wmma_tiled_branch`) permanently. The strict-xfail added in
`59b95e676` is worded "WIP research; not yet PASS", which is stale -- it cannot become PASS.
`docs/flash-prefill-phase2-plan-20260721.md` and `docs/qwen3-14b-generated-prefill-claude-handoff-20260716.md` both
record this FAIL as historical and not a regression.

## Recovery

- Campaign start: `300a9abcc`, `28d5ab593`, `1d73b7657`, `7fd94a02a`, `6798fd86f` (2026-07-05)
- Milestone / verdict: `05b67146a` "complete scheduler-owned Q4K prefill" (2026-07-14) -- the last substantive change
- Dispatch removal: `45cfc399c` (2026-07-21)
- Final path move (no logic change): `383c71c72` (2026-07-25)

`git show 05b67146a -- <path>` recovers any file at its final substantive state.

## Retirement change set (must land as one change)

Deleting the files alone breaks `test/unit/test_q4k_wmma_tiled_gates.py::test_q4k_wmma_tiled_authority_gate_files_exist`,
which asserts every `.py` path named in the manifest's `authority_gate` string exists on disk. So:

1. Delete `extra/qk/prefill/q4k_wmma_tile_lowering.py`, `q4k_wmma_tiled_lowering_feasibility.py`,
   `q4k_wmma_tiled_microgate.py`, `q4k_wmma_tiled_surface_gate.py`, `q4k_wmma_tiled_role_shape_exec_gate.py`,
   `q4k_wmma_full_role_contract_gate.py`, `q4k_wmma_tiled_no_hand_kernel_gate.py`, `prefill_int8_wmma_spec.py`.
2. In `route_manifest.py`, keep the route row and its `note` (that note is the only record of the measurement), but
   clear `authority_gate` and repoint `promotion_artifacts` at this document. `docs/prefill-lessons-ledger.md` is
   currently named as the promotion artifact and does not mention this campaign at all -- that pointer is stale.
3. Update `test/unit/test_q4k_wmma_tiled_gates.py` (delete it, or reduce it to asserting the route stays non-default).
4. Update the five test files importing `prefill_int8_wmma_spec`: `test_amd_isa_wmma.py`, `test_q4k_wmma_value.py`,
   `test_q4k_wmma_scheduler_decomposition.py`, `test_q4_q4_owner_comparison.py`, and
   `extra/qk/prefill/prefill_mmq_parity_gate.py`.
5. Remove the lazy `_attr` forwarders for `describe_q4k_int8_wmma_*` / `emit_q4k_int8_wmma_*` in
   `tinygrad/llm/route_ops.py` -- they are already never called.
6. `bench/q4k-wmma-tiled-*/latest.json` artifacts report PASS for a route whose dispatch no longer exists; retire them
   in the same change so no future reader treats them as current.

Estimated authored LOC removed: ~1,100 in `extra/qk`, plus test LOC.
