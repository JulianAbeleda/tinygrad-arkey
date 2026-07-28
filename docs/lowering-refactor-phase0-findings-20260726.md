# Lowering refactor — Phase 0 findings

Branch `refactor/lowering-architecture`, baseline `18d2fab52`. Implements LR-000 and LR-001 of
`docs/task_workflow/output/lowering-architecture-refactor-scope-20260726.md`.

Phase 0's rule governs everything after it: **no pass moves until its input/output contract is recorded.**

## LR-000 — fingerprint baseline

`extra/audit/lowering_baseline.py`, artifact `bench/lowering-refactor-baseline/latest.json`.

30 default-path kernels, compiled off-device (`ALLOW_DEVICE_USAGE=0`) from the shipped route specs at real
Qwen3-8B/14B role shapes: direct-packed Q4_K/Q6_K prefill, fused prefill attention, flash decode tile/combine
(8B KV_BOTH, 14B G5 KV_BOTH, 14B G5 K_ONLY), Q4_K/Q6_K GEMV primitives, and the promoted G3 lanemap decode GEMV.
Each entry records source sha256, VGPR/SGPR/LDS/scratch, route_id and shape.

`--check` recompiles and reports **which** thing changed — source hash vs shape vs route — and exits nonzero on any
difference. Every later slice runs it. Verified byte-identical across separate processes.

Coverage limits, deliberate:
- The two packed-load direct-out kernels use `schedule="explicit"`; under `"auto"` the heuristic search picks
  non-compiling options at these shapes. Explicit also removes search nondeterminism from the fingerprint, but **the
  auto-scheduled path is not covered**.
- `lm_head` and legacy non-default variants are outside the sample, to bound runtime.
- The full `AMDResourceArtifact` join is not constructed: it requires per-kernel physical-register role intervals that
  exist only for the hand-annotated MMQ atom.

## LR-001 — pass inventory

`bench/lowering-refactor-baseline/pass_inventory.json`, 93 passes.

| stage | passes | | stage | passes |
|---|---|---|---|---|
| late | 33 | | bufferize | 11 |
| opt | 13 | | renderer | 8 |
| rangeify | 12 | | custom_kernel | 7 |
| indexing | 5 | | dependencies | 4 |

Confidence: 59 high, 28 medium, 6 low. The low/medium entries are the ones to re-derive before moving anything that
touches them.

### Shared mutable state — the real risky boundaries

Eight hazards, each with a named owner problem. These are the boundaries to type first, because a move across any of
them is currently unfalsifiable:

1. **`IndexingContext.range_idx`** — created by `run_rangeify`, then mutated *by reference* by `pm_limit_bufs` after
   `run_rangeify` has returned. Ownership outlives its creator.
2. **`LocalAddBufferContext`** — reused across `to_define_global`, `debuf`, `handle_after`, `renumber_range`,
   `split_store`. Worse: `rangeify_codegen` is called with **two incompatible context types** — `LocalAddBufferContext`
   from `rangeify.py`, but a bare `itertools.count` from `codegen/__init__.py`. That is a latent `AttributeError` if
   the second path is ever reachable.
3. **`LinearScanRegallocContext`** — ~10 mutated fields threaded through two sequential `line_rewrite` passes with
   strict positional-index alignment between them.
4. **`_WARMSTART_OPTS` / `_WARMSTART_CANDIDATE_CONTEXTS`** (`postrange.py`) — process-global, nominally installed
   through a save/restore contextmanager, but mutated **directly** from `extra/llm_research/prefill/prefill_whole_synced.py` and
   `prefill_graph_gemm_route.py`, bypassing that contract entirely.
5. **`to_program_cache` / `schedule_cache`** — process-global, keyed by content hash. A cache-key bug silently reuses
   a stale schedule or lowered program, which would defeat LR-000's fingerprint.
6. **The `("composite_reduce", …)` tuple tag** — produced in two places, resolved in three. Stringly-typed
   provenance; exactly LR-020's target.
7. **`_COMPOSITE_KERNEL_SUBSTITUTIONS["amd_gfx1100"]`** (`postrange.py`) — couples to `tinygrad/llm/fused_attention.py`
   emitters *by string key*. This is the literal seam between the two lowering stacks that §3.1 describes.
8. **`native_*_matcher` renderer hooks** — `native_repack_matcher`, `native_state_lane_matcher`,
   `native_loop_fragment_matcher`, `native_loop_state_matcher`, `native_fragment_opaque_matcher` — reached by
   duck-typed `getattr(ren, ..., None)` and defined only on the AMD `ISARenderer`. A pass **silently no-ops** on any
   renderer lacking them, with no declared interface.

Note on method: a first pass over the inventory's `context_fields` column suggested only three shared contexts and a
much cleaner picture. That was wrong — the hazards above are recorded in the inventory's prose fields, not that
column. Counting a structured field is not the same as reading the evidence.

### Environment variables really are an implicit lowering API

36 of 93 passes are gated by an environment variable. Gates observed include `PCONTIG`, `SPEC`, `IMAGE`,
`WARP_REDUCE_LOWERING`, `V_DOT2_LOWERING`, `SPLIT_REDUCEOP`, `REDUCEOP_SPLIT_THRESHOLD`, `REDUCEOP_SPLIT_SIZE`,
`NOOPT`, `DEBUG`, `VIZ`. This confirms §3.6 with a number: **more than a third of the pipeline's behaviour is
configured by environment, with no typed surface and no single place to read the effective configuration.**

Consequence for the refactor: the trace contract (LR-010) must record the *effective* value of every gate for a given
lowering, or a trace taken on one machine will not explain a result from another.

## Conflicts between the scope and the tree

The scope was written 2026-07-26 13:15. The `DEV=AMD:ISA` retirement landed ~14:00 the same day, so two of its
references are stale:

1. **`extra/qk/kernel_pipeline.py` no longer exists.** The scope lists it in §2 as a current file to preserve, and
   LR-050 proposes promoting "reusable parts" of it into core. It was retired with the ISA support chain because it
   was reachable only under `DEV=AMD:ISA`. It held `DotUpdateRecurrencePlan/Graph/Proof`,
   `HierarchicalKernelPipelinePlan`, `hierarchical_lifecycle_events`, `SchedulerOutputTileLoop`, and
   `build_scheduler_output_tile_loop`. Recover with `git show 348dceeec -- extra/qk/kernel_pipeline.py`.
   **Decision needed:** LR-050 either drops this target or harvests from history. Note the *core*
   `tinygrad/codegen/opt/kernel_pipeline.py` is a different, live module and is unaffected — §5's reference at line
   265 is to that one.
2. The seven `extra/llm_research/decode/*.py` modules named at LR-070 are **proposed** decomposition targets and correctly do
   not exist yet. Not a conflict.

Also cleared: 14 stale `__pycache__/*.pyc` files for modules deleted today. One of these previously caused `sz.py` to
crash the pre-commit hook by walking a path whose source no longer existed.

## Pass order is nowhere declared

The order is the literal statement order of three functions. That is the single most important fact for this refactor,
and it is why LR-010 exists:

- **Kernel-graph construction** — `tinygrad/schedule/rangeify.py:993-1053` (`_get_kernel_graph`): `multi_pm` →
  optional `pm_fold_moved_after` → `pm_native_row_softmax_repack` → `earliest_rewrites` → attention-semantic lowering
  → `run_rangeify` → composite-slot resolution → cost-gated debuf → `limit_bufs` → `add_buffers`/`split_kernels` →
  WAR-dependency fixup.
- **Schedule linearization** — `tinygrad/schedule/__init__.py`: `create_linear_with_vars` → `lower_sink_to_linear` →
  `create_schedule` (a hand-written Kahn topological sort, **not** a `graph_rewrite`) → `pm_resolve_linear_call` →
  `memory_plan_rewrite`. There is no `ScheduleItem` type; a schedule is `UOp(Ops.LINEAR, src=(CALL, ...))`.
- **Per-kernel lowering** — `tinygrad/codegen/__init__.py:74-292` (`full_rewrite_to_sink`), ~35 steps, then
  `do_to_program`'s ISel chain and `pm_to_program` (`do_linearize → do_estimates → do_assemble/do_render → do_compile`).

Order dependencies that are real and undocumented in code:
`pm_native_row_softmax_repack` must precede `pm_mops`'s SHAPED_WMMA rule; `remove_bufferize`'s cost gate depends on
`AxisType.REDUCE` assigned by `run_rangeify`; composite slot resolution must precede const-folding or the slots are
folded away; `WARP_REDUCE_LOWERING` must run before `pm_group_for_reduce` claims the axis; `pm_reduce_acc_upcast_fix`
must precede `pm_add_loads`; `pm_add_gpudims` requires prior scalar-STORE-address lowering.

## Contracts that could not be established

Recorded rather than guessed. Each gates any move that touches it:

- `tinygrad/schedule/flash_fusion.py` and `tinygrad/codegen/late/flash_attn.py`'s top-level `flash_attention` — **no
  callers found.** Likely dead; flagged for LR-081 verification, not asserted dead here.
- `codegen/simplify.py`'s `reduce_simplify_family` — zero external callers found.
- `extra/qk/coalesced_load_lowering.py`, `warp_reduce_lowering.py`, `fdot2_lowering.py`, and
  `codegen_list_scheduler.py` — only their `codegen/experimental.py` forwarding shims were read; internal contracts
  unverified. Recurrence unroll was later promoted to `tinygrad/codegen/late/recurrence.py` with a direct core import.
- `tinygrad/schedule/wmma/{composite,fragments,kernels,softmax,loop_state}.py` internals — only the re-export surface.
- `renderer.isa.X86Renderer` — implements the full ISARenderer hook set but has no confirmed `Device` wiring.

## Two further scope premises contradicted by the tree

Beyond the retired `kernel_pipeline.py` above:

- **`tinygrad/llm/decode_routes.py` contains no `getenv` calls at all**, contradicting both the scope and
  `extra/llm_research/route_manifest.py`'s description of it as an env-gated admission point. The flags live only as
  declarative strings in `route_policy.py`, and `route_manifest.py` is **not imported by any runtime routing code**.
- **`tinygrad/engine/realize.py` does not call into `schedule/__init__.py`.** The import direction is inverted
  relative to the scope's pipeline diagram; `engine/jit.py` reaches into `schedule/memory.py` and
  `schedule/rangeify.py` directly.

Also: the inventory has **zero records** for stages `realize`, `scopes`, and `runtime`. Those exist only in §5's
*target* architecture, not in the current tree. That is itself an LR-001 finding — three of the proposed owners are
new construction, not extraction.

## What Phase 1 needs

- LR-010's trace must capture effective env-gate values, not just pass order (see above).
- The 6 low-confidence and 28 medium-confidence inventory entries are the ones whose contracts are not yet
  established. They gate any move that touches them.
- The `auto`-scheduled path has no fingerprint. Any refactor touching the heuristic optimizer is currently
  unfalsifiable by LR-000 and needs its own gate first.
