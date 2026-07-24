> **SUPERSEDED IN PART (2026-07-23).** The rotating-PV probe and its bespoke ABI have been
> RETIRED (production is the accepted design; VGPR-cutting is a measured dead end). The real
> open problem is NOT a rewrite of the prefill path — the kernel already compiles correct at
> 254 VGPR. It is that the production **BoltBeam export path never runs the kernel**: the
> policy never assembles the composite proof that enables the route, and an admission gate
> caps VGPR at ≤192 (production is 254). See `boltbeam-export-triage-8b-14b-20260723.md` for
> the verified diagnosis and the real next steps. The vocabulary-collapse design below is
> retained for reference but is moot now that the duplicate route is removed.

# Shared Flash-Prefill Rewrite Handoff (Gated, non-replacement)

Date: 2026-07-23
Repository: `/home/ubuntu/tinygrad-arkey`

## Purpose

This document records the rewrite plan so we can execute it exhaustively without replacing existing scripts until the rewrite is proven better on model-level evidence.

This is a **gated migration**, not a direct replacement.

- Current legacy prefill paths remain active as rollback.
- The new rewritten path only activates through explicit candidate/policy gating.
- If evidence fails at any gate, keep the existing route unchanged.

## Why this exists

We need one reusable compiler-native attention mechanism that works for:
- 8B prefill (fp16-overlay projection route)
- 14B prefill (bounded packed-weight route)

without duplicating model-specific attention graphs or benchmark scaffolding.

This rewrite must move both to a score-resident fused prefill path that does not materialize full `T x KV` scores/probs.

Current model issue (plain):

- The current rollout is functionally split by path and still relies on one-off, probe-like behavior, so 8B/14B don’t share one proven, compiler-native rewrite pipeline.
- The production path is still not the same ordered rotating-PV contract as the probe/experiment path.
- Because of that, we have not yet met the gated promotion criteria for a safe, default-style replacement.

### Root cause (measured 2026-07-23 — full inventory in the next section)

The pain is not that the *prefill pipeline* is split; it is that the **lowering vocabulary is duplicated**. The attention lowering surface currently carries **~32 distinct dispatch tags, ~30 verifier branches, and ~24 expander functions** — but roughly **7 of those tags are the *same* "typed lane-state write + read (+ sync)" primitive, re-encoded independently** (`rotating_pv_state_*`, `softmax_bridge_*`, `state_loop_*`, `amd_gfx1100_row_state_*`, `amd_gfx1100_attention_loop_state_write_v1`, `amd_register_stage_pair`), each hardcoding its own LDS byte layout / register base and its own arg-tuple convention. Every new attention construct therefore drags a *new* spec class + *new* verifier branch + *new* expander. **That is the "we keep lowering over and over" treadmill.** This refactor's job is to collapse that duplicated vocabulary — not to rewrite the prefill plumbing layer by layer.

**Critical corollary — the simplification and the VGPR-gate blocker are the same problem.** The rotating-PV accumulator (`rotating_pv_state_*`) is lowered only through the **generic LDS-memory degrade** (`INDEX`/`LOAD`/`STORE` on `DEFINE_LOCAL`), whereas its structural siblings loop-state and row-state get a **fixed-VGPR register-pin** via `isel_customi` (renderer/isa/amd.py:859). Give the accumulator the *same* register-resident `StateHandle` lowering the siblings already have, and one change both (a) removes duplicate constructs and (b) unblocks the ≤192 VGPR gate. This is why the accumulator lowering — not prefill plumbing — is the first executable unit.

## Simplification target (the actual design)

Goal restated: **one reusable lowering vocabulary** so new attention math composes from already-lowered pieces instead of adding a new expander each time. Collapse the inventory onto these primitives:

1. **`StateHandle` — one typed lane-state primitive.** read / write / gep-project a typed lane-region, parameterized by `backing = {LDS DEFINE_LOCAL | pinned VGPR span}` and layout (lanes / blocks / fields / dtype). Subsumes all 7 duplicate state encodings (`rotating_pv_state_*`, `softmax_bridge_*`, `state_loop_*`, `amd_gfx1100_row_state_*`, `attention_loop_state_write`, `register_stage_pair`) → one spec class, one verifier, one LDS lowering + one VGPR-pin isel. **This is unification of pieces that already exist**: the generic `StateHandle` + LDS lowering (uop/ops.py:1284, renderer/isa/amd.py:2796) and the fixed-VGPR pin (`isel_customi`, renderer/isa/amd.py:859) are both already in the tree — they are just not shared.
2. **`Ops.WMMA`** — tensor-core math (unchanged). `rotating_pv_wmma_v1` becomes `StateHandle.read → WMMA → StateHandle.write`; the bespoke marker disappears.
3. **Sync = `Ops.WAIT` / `Ops.BARRIER` + publication token.** `rotating_pv_publication_v1` collapses to this.
4. **`Ops.ROTATING_PV_SEQUENCE`** — the single ordering primitive (keep; already landed, commit eecafa037). It already composes (sync, load, WMMA, store); after the collapse those become StateHandle / WMMA / WAIT rather than bespoke markers.
5. **One parameterized output-store** — subsumes `AMD_ATTENTION_OUTPUT_DRAIN` + `AMD_ATTENTION_STATS_DRAIN` + `rotating_pv_sequential_drain_v1` (their bodies are near-identical reciprocal-scale + strided store today).
6. **`AMD_PACKED_FRAGMENT_LOAD`** — genuine distinct primitive (keep).
7. **`ROW_SOFTMAX_REPACK`** — genuine nonlinear compute (keep); its internal state write/read uses `StateHandle`.
8. **Reduction family** (`SCOPED_REDUCE` / `REDUCE_SLOT` / composite) — genuine scheduling machinery, shared with non-attention (keep).

### Countable success metric (this operationalizes "Completion ≠ it compiles")

The refactor is only "simplifying" if these strictly drop. Baseline (measured 2026-07-23) → target:

| measure | baseline | target |
|---|---|---|
| distinct dispatch tags in attention lowering | ~32 | ≤ ~12 |
| dedicated verifier branches | ~30 | ≤ ~12 |
| expander / lowering functions | ~24 | ≤ ~12 |
| independent "typed state" encodings | 7 | 1 (`StateHandle`) |

**If a change does not reduce these counts, it is relocating code, not simplifying — treat it as a one-off and reject it** (per the process rule below). This metric is a hard gate alongside the functional/resource/performance gates.

## What this rewrite is NOT

- Not a script rewrite of existing model/policy files before evidence.
- Not a benchmark-interpretation rewrite.
- Not a fallback removal.
- Not a “better-looking” IR refactor that does not hit gates.

## High-level rule

**No existing script or route gets replaced until all gates are passed.**

Rollback behavior is mandatory at every phase:
- keep old scripts operational,
- keep old route as fallback,
- only gate-to-promotion when the rewritten path clears all checks.

## Process simplification rule (remove one-offs)

Every change in this rewrite MUST follow this fixed sequence and no other local ad-hoc path:

1. **Scope**: update this doc with a single sentence and specific file list for the change.
2. **Reuse first**: attempt to reuse an existing primitive path (`Ops.ATTENTION`, `CompositeReduce`, `RotatingPV` contracts) before adding new logic.
3. **Gate only**: if it doesn’t pass type/spec/rewrite gates, stop and report instead of introducing workaround branches.
4. **Evidence then promote**: only promote if the route-level gates pass; otherwise keep fallback behavior and keep edits minimal.

A “shortcut” implementation that bypasses one of these four steps is treated as a one-off and should not be merged into shared path work.

## Scope map (end-to-end chain)

1. **Model intent capture layer (no behavior change yet)**
   - Keep `prefill_tc_attn` selection and shared semantic entry in place.
   - Verify both 8B/14B can hit the same attention semantic entry (`shared_prefill_attention`).
   - Keep legacy route as fallback if candidate not admissible.

2. **Semantic capture + lowering boundary
   - `Ops.ATTENTION` to rangeify/semantic reducer path remains the single source of truth for rewrite eligibility.
   - Rangeify rewrite remains a strict gate; do not create side-path pattern hacks.
   - If semantic graph shape does not match the contract, do not rewrite.

3. **Unified reduction contract (single source of truth)**
   - Keep one composite reduction contract (`online_softmax_state`) for both scalar and tiled behavior.
   - Preserve slot semantics for `m/l/acc` as logical state vectors.
   - Ensure `REDUCE_SLOT` and `DEFERRED_REDUCE_SLOT` remain provenance-aware and cannot break unrelated paths.

4. **Native attention handoff (existing request path)**
   - Continue using `required_native_attention` as an admission signal.
   - Native handoff should pick the same rewritten path for admissible prefill geometries; no bypass logic per model id.

5. **Production rotating-PV ordering contract
   - Move from “probe-only” sequencing to production path usage.
   - Ensure per-block PV order is preserved semantically and at emission:
     - sync -> C load -> PV WMMA -> C store
   - Preserve typed state/bridge semantics while keeping marker/value contracts untouched for fallback.

6. **Codegen emission order safety**
   - Keep existing fallback renderers unchanged unless proven necessary.
   - Keep sequence ordering local to attention path and enforce it via typed sequence + native lowering, not global reordering policy.

7. **Acceptance evidence**
   - Keep all existing scripts active.
   - Promote only after the rewritten path clears gates on both routes.

## File-level scope and expected edits

### Layer A — Route intent (no replacement)

- `tinygrad/llm/model.py`
  - Keep `prefill_tc_attn` and `shared_prefill_attention` entry behavior.
  - Do not remove legacy SDPA path.
  - Ensure both 8B/14B route through the same semantic entry when gated.

- `tinygrad/llm/prefill_policy.py`
  - Keep policy semantics explicit and conservative.
  - No silent elevation to rewrite path.

### Layer B — Semantic boundary / capture

- `tinygrad/schedule/rangeify.py`
  - Maintain/extend `pm_attention_semantic` matching for all eligible attention shapes.
  - Keep fail-closed semantics for unsupported cases.
  - Ensure rewrite eligibility remains deterministic and auditable.

### Layer C — Reduction/state machinery

- `tinygrad/uop/ops.py`, `tinygrad/uop/spec.py`, `tinygrad/uop/__init__.py`
  - Keep composite/provenance contracts intact.
  - Ensure slot resolution logic remains strict, graph-local, and fail-closed.

- `tinygrad/codegen/late/composite_combines.py`
  - Keep combine registry as single place for combine behavior.
  - No per-path duplicate equations.

- `tinygrad/codegen/late/devectorizer.py`
  - Preserve no-range and range paths without changing fallback behavior.

### Layer D — Native handoff and kernel producer

- `tinygrad/codegen/opt/postrange.py`
  - Keep current native attention handoff logic, but ensure the rewritten composite route participates uniformly.
  - Use one handoff path for both 8B/14B whenever admission is valid.

- `tinygrad/schedule/wmma.py`
  - Keep current probe paths, but make production path own the full ordered sequence for rotating-PV if applicable.
  - Do not fork multiple handoffs by model/route.

### Layer E — Emission ordering guardrail

- `tinygrad/renderer/cstyle.py`
  - Keep native sequence expansion local to the primitive; avoid global ordering rewrites.

- `tinygrad/codegen/late/linearizer.py`
  - Preserve sequence priority hints that keep block-local ordering for the rewritten path.

- `tinygrad/renderer/isa/amd.py`
  - NOTE (corrected 2026-07-23): this is NOT an "as is" layer. The rotating-PV accumulator (`rotating_pv_state_*`) currently has **no register-resident ISA lowering** — it degrades to generic LDS memory ops, unlike loop-state/row-state which pin fixed VGPRs via `isel_customi` (amd.py:859). Step 0 of the execution order adds that register-pin lowering here; it is the critical-path change, not a narrow fix.
  - Collapse the per-construct state lowerings (`lower_rotating_pv_state`, `lower_softmax_bridge`, `lower_state_phase_transfer`, row/loop-state) onto the single `StateHandle` lowering rather than maintaining one function per tag.

## Gates before any promotion (must all pass)

### Functional gates

1. Exactness: prefill attention numerics remain correct at accepted shapes.
2. Correctness parity: non-rewrite fallback output unchanged where rewrite is not selected.
3. Semantic closure: no unsupported attention silently rewritten.

### Memory/resource gates

4. No full score/probability materialization (`T x KV`) in rewrite path.
5. No regression in spill profile on rewrite route (`0` spills target).
6. Register/occupancy profile at or better than current admissible baseline.

### Performance gates

7. Rewritten route must beat current best rollout candidate for measured prefill throughput on target hardware.
8. Improvements must be reproducible under same warm/measurement harness mode.
9. If gated route is enabled and slower, revert and keep fallback.

### Cross-route gates (critical)

10. 8B and 14B must use same shared semantic+rewrite mechanism.
11. No duplicated route-specific flash logic.
12. No route-specific benchmark scripts for rewrite-only behavior.

## Rollback rules (must remain true at all times)

- If any gate fails, keep legacy route enabled and set rewrite to explicit opt-in only.
- Do not edit behavior of non-attention workloads.
- Maintain one shared fallback and one shared admission gate.

## Immediate execution order (de-risk first)

Reordered from "plumbing-first" to "prove the premise, then unify the vocabulary." The highest-risk unknown is measured *first*, before any migration scaffold is built.

0. **Spike (throwaway, ungated) — give the rotating-PV accumulator the register-resident lowering its siblings already have.** Reuse the `isel_customi` fixed-VGPR pin path (renderer/isa/amd.py:859) for `rotating_pv_state_*` instead of the LDS-memory degrade, compile **one** admissible shape, and read real VGPR / spills. This converts the unproven "201" target into a measured number. **If it cannot beat production's ~254 VGPR, stop — the refactor should not happen.** Cheapest possible kill/confirm of the whole premise.
1. **Land the unified `StateHandle` primitive** (`backing = LDS | pinned VGPR`) + its single verifier + LDS lowering + VGPR-pin isel. Migrate **one** duplicate encoding onto it (`rotating_pv_state_*`), keeping the old tags working as fallback. Confirm the count metric dropped.
2. Verify the probe still `type_verify`s **and now compiles to registers**; re-run the spike's shape and confirm the VGPR number holds.
3. Migrate the remaining duplicate encodings (`softmax_bridge_*`, `state_loop_*`, row/loop-state) onto `StateHandle`, one at a time, each behind the gate, each strictly reducing the count metric.
4. Only then wire the **production** path to the ordered rotating-PV contract (`ROTATING_PV_SEQUENCE`) through the shared vocabulary — no per-model fork.
5. Run focused evidence pass on one admissible shape; if gates pass, expand to the full shape grid and both route policies.
6. Promote only with explicit policy flag + benchmark evidence beating fallback.

Baseline references for step 0/1: inventory + measured starting counts are in
`docs/shared-flash-attention-rotating-pv-primitive-results-20260723.md` (primitive + gate state)
and the two sections above (duplication counts, collapse target).

## Outcome definition

Completion is not “it compiles” and not “unit tests pass.”

Completion is when both 8B and 14B can run the rewritten prefill path through the same shared mechanism, under explicit gate, and the route is measurably better and at least as correct as the fallback across accepted conditions.
