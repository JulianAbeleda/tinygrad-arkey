# Rotating-PV Sequence Primitive — Results (2026-07-23)

Execution of `docs/CLAUDE_FLASHATTN_EXECUTION_PROMPT_20260723.md` (primitive-first).

## What was implemented

A real, typed compiler primitive `Ops.ROTATING_PV_SEQUENCE` that enforces the fixed
per-block emission order for one rotating-PV block:
1. publication/boundary sync → 2. C-window LDS load → 3. PV WMMA → 4. C-window LDS store.

Replaces marker-only sequencing with a native op (no broad `AFTER`-semantics rewrite,
no per-route handcrafted kernels).

| Piece | File | Notes |
|---|---|---|
| Op enum | `tinygrad/uop/__init__.py` | `ROTATING_PV_SEQUENCE = auto()` after `SCOPED_REDUCE`. |
| Typed spec | `tinygrad/uop/ops.py` | `class RotatingPVSequenceSpec(NamedTuple)` (`state`, `block`); builder `.sequence(sync, c_load, pv_wmma, c_store)`; `arg` is the spec instance (not a string tag); `src` fixed order `(sync, c_load, pv_wmma, c_store)`. |
| Verifier | `tinygrad/uop/spec.py` | `validate_rotating_pv_sequence` + dedicated `UPat(Ops.ROTATING_PV_SEQUENCE)` in `spec_shared` (reachable from `spec_full`). GROUP child allow-list extended to accept the new op. `c_store` accepted as `Ops.STORE` **or** the route's typed `rotating_pv_state_write_v1` LDS accumulator write. |
| Intra-block order | `tinygrad/renderer/cstyle.py` | `_hip_expand_rotating_pv_sequence` (registered in `native_repack_matcher`) rewrites the op **pre-linearize** into an explicit `.after()` chain sync→c_load→pv_wmma→c_store, so toposort is structurally forced — not reliant on `AFTER` render-time aliasing. |
| Block contiguity | `tinygrad/codegen/late/linearizer.py` | Each expanded step carries a tag `("rotating_pv_sequence_v1", generation, block, step)`; `linearize()` assigns a disjoint priority band `1000 + generation*100 + block*10 + step` so a block's four steps stay contiguous and a later block's `c_load` (LOAD priority −1) cannot be hoisted into an earlier block's live window. |
| Integration | `tinygrad/schedule/wmma.py` | `amd_gfx1100_rotating_pv_scheduler_probe` builds one `seq` per block; raw `c_store` stays in the `end`/drain chain (drain + publication need the exact `rotating_pv_state_write_v1` node), and `seq` is threaded via the sink so the expander pass reaches it. |

Pass-ordering fact worth keeping: `native_repack_matcher` runs (ctx `count(800)`/`count(700)`)
**before** `pm_add_control_flow`/`do_linearize`, so `ROTATING_PV_SEQUENCE` never survives into
the linearized list — only its four constituent ops (plus the `AFTER` wrappers) do.

## Correctness gate — PASS

- `extra/qk/rotating_pv_abi.py::rotating_pv_kernel_probe()` → `STATUS CONSTRUCTED`
  (`type_verify(sink, spec_full)` passes with the primitive integrated across all 8 blocks).
- Unit sweep (`test_rotating_pv_scheduler`, `test_rotating_pv_state`,
  `test_shared_attention_compiler_capture`, `_synchronization_capture`, `_promotion`):
  **26 passed, 1 failed**. The single failure
  (`test_rotating_pv_drain_reloads_blocks_sequentially_compile_only`) **fails identically at
  baseline HEAD** with our edits stashed — pre-existing, not a regression from this work.
- Production `shared_prefill_attention` still compiles (all four routes via
  `generate_shared_attention_captures`): VGPR 254 / LDS 512 / 0 spills — unchanged path,
  confirming no broad regression.

## Resource gate — NOT REACHABLE (this is the explicit next blocker)

The `VGPR ≤ 192 (measured)` gate cannot be evaluated for the rotating-PV route from this tree:

1. **The rotating-PV route has no backend lowering.** `extra/qk/rotating_pv_abi.py` states
   plainly: "the backend has no verified lowering for the accumulator StateHandle yet."
   `rotating_pv_kernel_probe` only *constructs + type_verifies*; it never compiles to ISA.
   The typed LDS-accumulator ops (`rotating_pv_state_write_v1`, `rotating_pv_loop_read_v1`,
   `rotating_pv_sequential_drain_v1`) have no lowering to real LDS load/store, so the sink
   cannot be turned into registers/VGPR.
2. **The "201 VGPR / 11776 LDS" figures are not measured from this route.** They appear only
   as prose targets / cost-model numbers. The VGPR values in `test_shared_attention_promotion`
   (`192`/`197`) are hardcoded synthetic admission fixtures. `generate_shared_attention_captures`
   measures the *production* `shared_prefill_attention` (254/512), a different construction that
   does **not** route through `amd_gfx1100_rotating_pv_scheduler_probe`.

So the ordering primitive is in place and verified, but the roofline/VGPR gate is gated on a
larger missing piece.

### Next explicit blocker
Implement backend lowering for the rotating-PV **accumulator StateHandle** — i.e. lower
`rotating_pv_state_write_v1` / `rotating_pv_loop_read_v1` (and the sequential drain) to real
AMD LDS store/load — so the probe sink can compile to ISA. Only then can VGPR be measured and
the `ROTATING_PV_SEQUENCE` contiguity win be evaluated against the ≤192 gate. Until that
lowering exists, the resource gate is blocked independent of the ordering primitive.

## Incidental fix
`SharedAttentionCandidateContext.validate()` (`tinygrad/uop/ops.py`) referenced an undeclared
`output_block_base` field (present at HEAD — the capture harness raised `AttributeError` before
compiling anything). Declared `output_block_base: int = 0` (full base-0 accumulator context;
slicing lives in the sibling specs that already carry the field).
