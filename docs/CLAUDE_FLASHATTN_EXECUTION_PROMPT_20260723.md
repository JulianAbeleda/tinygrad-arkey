# Claude Execution Prompt — FlashAttention Completion (Primitive-First)

## Objective
Finish the shared flash-attention route to the target roofline gate by closing the final compiler emission gap.

## Current status snapshot
- Core math/pipeline structure is in place (composite reduce + combine + REDUCE_SLOT + rotating PV state markers).
- Current best route is functionally solid but still over the VGPR gate in final emit.
- Current best measured profile on target fused prefill path: `201 VGPR`, `11776 LDS`, `0 spills`, `8 QK + 8 PV`.
- Primary missing gate is strict ordered emission of rotating PV blocks.

## Where to start
1) Read these files first:
- `docs/shared-flash-attention-completion-handoff-20260723.md`
- `docs/shared-flash-scoped-reduce-completion-scope-20260722.md` (latest detailed route context)
- `docs/shared-flash-attention-codex-completion-scope-20260722.md`
- `docs/SHARED_ATTENTION_SEQUENCE_AND_PATTERN_20260723.md`
- `docs/shared-attention-lds-rotating-pv-scope-20260723.md`

2) Capture current branch position:
- `git -C /home/ubuntu/tinygrad-arkey status`
- `git -C /home/ubuntu/tinygrad-arkey log -n 6 --oneline`

3) Confirm current blocker point by checking the rotate-sequence emission path around the renderer and sequence markers.

## Completion principle
- Do not redesign the attention pipeline.
- Do not remove or globally change generic `AFTER`/ordering semantics.
- Add one compiler primitive that enforces this explicit sequence:
  1. publication/boundary sync
  2. C-window LDS load
  3. PV WMMA
  4. C-window LDS store

## Work order (primitive-first)
1) Native op design
- Add a typed, explicit rotating-PV sequence op to UOp/OpSpec.
- Keep operation narrow and local to this route.

2) Verifier / infra
- Add typed spec/invariants for this op.
- Ensure verifier enforces that it appears only in valid rotate-PV contexts.

3) Backend emission (HIP/AMD)
- Implement renderer lowering that emits strict statement order for this op.
- Do not rely on generic reordering to preserve the sequence.

4) Integration point
- Replace current marker-only sequencing for rotating-PV flow with this native op.
- Keep marker metadata only as supporting information if still useful.

5) Evidence run (minimal)
- Compile and inspect emitted ordering in disassembly.
- Confirm C loads are not hoisted outside intended block boundaries.

## Success criteria
1) Resource gate:
- VGPR <= 192 for the target fused prefill geometry where 201 is the current best
- Spills/Scratch = 0
- LDS remains in expected range, no regressions in already-working paths
2) Correctness gate:
- No score/probability materialization in fused path
- Existing numerical checks remain clean
3) Scope gate:
- Applies to fp16 and non-fp16 pathways
- No duplicate per-route kernels
4) Delivery gate:
- Update one handoff doc with results and next explicit blocker (if any)

## Anti-patterns to avoid
- Do not add a broad pass that rewrites all `AFTER` semantics.
- Do not add route-specific ad-hoc kernels.
- Do not switch to per-route handcrafted kernels unless this primitive attempt is blocked.

## If blocked
- Fall back to minimal primitive-only implementation with no speculative global refactors.
- Isolate changes to:
  - op definition,
  - verifier,
  - HIP/AMD rendering.
- Re-attempt emission only after the primitive is in place.

