# Fused Shared-Attention Enablement — Exhaustive Scope ("Option A", 2026-07-23)

Goal: make the production fused shared flash-attention path actually run end-to-end for 8B
and 14B prefill (currently it silently falls back to SDPA), so a BoltBeam artifact is produced.
Scoped from three parallel investigations. Perf levers (VGPR/occupancy) are settled and out of
scope — see `boltbeam-export-triage-8b-14b-20260723.md`.

## Architecture: two kernel-producing paths (critical to understand)

1. **Harness/capture path** — `generate_shared_attention_captures.py` and the unit tests call
   the hand-built single kernel `amd_gfx1100_q16_grid_hd128_loop_attention` (`schedule/wmma.py:545`)
   **directly**, on a synthetic 61-node single-consumer graph. This is what the 254-VGPR / 0-spill
   proof and the 4× benchmark measured. It works.
2. **Production path** — real model: Tensor ops → `ATTENTION` marker → `lower_attention_semantic`
   (`rangeify.py:19`) → `composite_reduce("online_softmax_state")` → rangeify `_get_kernel_graph`
   → `_lower_composite_no_range_pm` → **CRASH** (`bad reshape: () -> (1,32,1,512,128)`).

The "collapse per-KV-block subgraphs into one kernel" goal (the AST swap at `postrange.py:328-361`
that replaces the AST with the hand-built loop kernel) is **already implemented and correct** — but
it is gated behind the production pipeline surviving rangeify, which it never does. So the crash and
the collapse-goal are **serially dependent, different deliverables**: fixing the crash only lets the
schedule survive far enough to *reach* the already-built native-swap gate.

## The dominant risk (why A is hard to size)

The production composite-reduce lowering has **never once succeeded at full-model graph scale.**
- Capture graph: 1 CALL, ~61 UOps, single consumer, no RoPE/quant/residual/multi-layer.
- Real 8B graph: whole model (32 blocks) scheduled in one `get_kernel_graph` pass, hash-consed
  DAG sharing (e.g. the causal mask built per-block at `model.py:589` becomes one node with 32
  consumers), real quant/RoPE producers feeding Q/K/V.
- The identical rewrite pipeline that compiles the isolated kernel **crashes** on the real graph.

So the true scope is "make composite-reduce/rangeify robust under production-scale multi-consumer
DAG sharing." The first crash is well-understood (below); how many follow-on robustness crashes hide
behind it is **unknown** — this path is dead code that has never run at scale. This is the same
"stacked latent bugs" pattern already seen: VGPR cap → Q-dtype gate → this crash.

## Known crash #1 — root cause + fix (MODERATE, specialist)

`lower_attention_semantic` broadcasts `score` to a fake `hd` axis (`score.expand(b,h,q_len,kv_len,hd)`,
`rangeify.py:86-87`) only to match the accumulator's rank. The "pick one representative lane"
consumption (`primary_repeated`) exists **only for the auxiliary V input**
(`_pack_online_softmax_v_lanes`, `_handle_no_range_generic` aux branch), **never for the primary
score input** — so the score's broadcast EXPAND/RESHAPE survives un-consumed and panics when a later
`symbolic+reduce_collapse+debuf` pass (`rangeify.py:938`) recomputes its shape against an
already-collapsed scalar base.

- **Fix strategy (recommended, surgical):** stop broadcasting `score` to `hd` at Tensor-construction
  time (`rangeify.py:81-115`); build the composite reduce from the natural `(b,h,q_len,kv_len)` score
  shape and let the combine step broadcast to hd lanes — exactly as `_combine_step_online_softmax_state`
  (`composite_combines.py:151-171`, `corr.broadcast(lanes)`) already does for m/l→acc. Reuses correct
  machinery; invents no new primary-repeated mechanism.
- **Risk:** must not regress the hd==16 `owned_map`/WMMA-fragment paths (`rangeify.py:59-71,116-126`)
  that consume `score`/`score_tile` at current shapes. Crosses the construction/lowering boundary.
- **Difficulty:** moderate; needs a rangeify/composite-reduce specialist, not mechanical.

Secondary, independent: `_combine_step_online_softmax` (the legacy non-state combine,
`composite_combines.py:137-149`) has no hd=128 lane-broadcast — likely **moot** because `grid_shape`
forces `online_softmax_state` whenever hd=128 (`rangeify.py:104`); confirm before touching.

## 8B vs 14B

**Same lowering path, geometry-driven not strategy-driven.** `strategy` (FULL_RESIDENT_OVERLAY vs
BOUNDED_PACKED_TILES) is metadata only; `lower_attention_semantic` never reads it. Both funnel through
the same `composite_reduce("online_softmax_state")` and the same parametric `q16_grid` kernel
(group_ratio = q_heads//kv_heads). **14B (Hq=40) has NO extra blocker** — `AMDAttentionGridSpec.validate`
requires `q_tokens%16==0` and `q_heads==kv_heads*group_ratio`, not `q_heads%16`. Both hit the identical
composite-reduce crash; fixing it fixes both.

## Work breakdown (dependency-ordered)

| Phase | Task | Difficulty | Depends | Unknowns / risk |
|---|---|---|---|---|
| **A0** | Q/K/V half-cast at attention boundary (`model.py`) — makes the fused path *eligible* | done (uncommitted) | — | inert in prod until enabled; prerequisite |
| **A1** | Fix crash #1: primary score-broadcast consumption (strategy above) | moderate, specialist | A0 | hd==16 path regression; **may expose crash #2..N** |
| **A2** | Re-run real 8B prefill `schedule_linear` — does it schedule clean or reveal next DAG-sharing crash? | spike/verify | A1 | **the pivotal unknown** — resolves scope size |
| **A3** | Iterate A1-style fixes until real 8B+14B prefill schedules clean | UNKNOWN (1..N fixes) | A2 | count unknown; this is the risk driver |
| **A4** | Confirm the schedule reaches `postrange.py:328` native-swap and emits the 254-VGPR single kernel for the *real* graph | verify | A3 | postrange admission is strict (param slots/sizes/scale); may need follow-up |
| **A5** | End-to-end correctness: real 8B+14B prefill fused vs SDPA (earns `model_*_prefill`) | medium | A4 | — |
| **A6** | Whole-model prefill benchmarks (KV 512..4096) | medium | A5 | handoff doc defers these behind an in-flight multi-wave (G2/G4/G5) experiment |
| **A7** | Decode-nonregression harness (earns `decode_nonregression_*`) — none exists; nearest is `decode_runtime_overhead.py` | medium-high | A5 | must build: run decode with shared-attn on vs off, assert token-parity + no perf regression |
| **A8** | Composite-proof collector: assemble `shared_attention_proof` (target+geometry+v2 artifact+8 flags) and wire adapter activation into model load (currently zero production callers) | medium | A5,A6,A7 | correctness/qk_wmma/pv_wmma/score_resident readable from existing artifact (low); prefill/decode flags gated on A6/A7 |
| **A9** | Raise VGPR cap 192→256 in admission gate | done (commit fa6d29633) | — | — |
| **A10** | BoltBeam prefill-attention route entry (`BoltBeam/boltbeam/policy/route_manifest.py`) | low, mechanical | A6,A7,A8 | no prefill-attn route exists today; template from decode_attention entries |

## Difficulty verdict

- **Compiler (A1-A4) is the uncertain core.** First fix is moderate and well-characterized; total
  count of robustness fixes is **unknown** because the path has never run at production scale. This is
  the single biggest risk to the estimate.
- **Everything else is tractable:** the one-kernel collapse is already built; 8B/14B share one path;
  the enablement tail (A5-A10) is plumbing + two evidence harnesses, no kernel invention.

## SPIKE RESULT (2026-07-23) — the core unknown is now RESOLVED: larger scope

Implemented fix strategy (a) (score built at natural rank; combine step broadcasts). Outcome:
- **Clean and non-regressing** — isolated capture still 254 VGPR / 0 spills; hd==16 unit tests
  unchanged (93 pass / 6 pre-existing fails, identical with and without the fix). The score fix is
  correct and harmless.
- **But the crash PERSISTS (outcome ii).** Instrumentation traced the real failing node: a
  `RESHAPE(1,32,1,512,128)` over `CAST(INDEX(PARAM slot=8, CONST 2097152))` — the **auxiliary V
  (`value_tile`) size-1 broadcast axis** being folded to a single scalar PARAM load by a generic
  reduce-collapse/const-fold pass **without updating the enclosing RESHAPE**. It's the same
  degenerate-axis defect, on V not score.

**Verdict: A is the LARGER scope — a compiler-robustness workstream.** The defect is systemic:
under full-model DAG sharing, generic collapse passes mangle the composite-reduce's degenerate
size-1 broadcast axes, and it recurs per input (score fixed → V surfaces → likely more). The correct
fix is NOT a per-input patch chain but structural: either (1) construct the composite-reduce inputs
without generic-pass-fragile unit broadcast axes at all, or (2) make the collapse passes preserve
RESHAPE targets over collapsed broadcast sources uniformly. This is dedicated specialist compiler
work, sized as a workstream — not a tail-end task. The score fix (rangeify.py) and the model.py
dtype fix are correct partial steps toward it (currently uncommitted).

## (superseded) Recommended next action: a bounded spike (resolves the core unknown)

Implement A1 (strategy above), then run real 8B prefill through `schedule_linear`
(repro: `/home/ubuntu/.claude/jobs/6db6b205/tmp/fused_prefill_force_8b.py`). One of:
- **(i)** it schedules clean and reaches the postrange native-swap → A is mostly-plumbing from here,
  size it as weeks.
- **(ii)** it reveals the next DAG-sharing crash → A is a genuine multi-fix compiler-robustness project,
  size it as a compiler workstream.

This single spike converts the dominant unknown into a number and is the correct first step before
committing to the full estimate.

---

## A2 — DESIGN DECISION + V-side fix (2026-07-23, session resume)

**Decision: Approach (1) — restructure inputs — surgical variant.** Chosen over (2) harden-collapse-passes
because (2) touches generic rewrite passes (`symbolic`/`pm_reduce_simplify`/`pm_const_buffer_folding`)
used by every graph → unacceptable blast radius, and because (1) is the SAME structural move the score-side
A1 fix already proved correct and non-regressing, one input downstream.

**Root-cause confirmation (repro, this session).** Baseline repro crashes exactly as documented:
`bad reshape: () -> (1,32,1,512,128)` in `_get_kernel_graph` → `symbolic+reduce_collapse+debuf` pass
(rangeify.py:951) → `_mop_index` (rangeify.py:272) → shapeless node. The `(1,32,1,512,128)` target is
verbatim `value_tile.reshape(b,h,1,kv_len,hd)` (h=32 grouped-up from 8 KV heads, kv=512, hd=128).

**Why the size-1 axis is removable (not load-bearing).** V has no query dependence. The unit q_len axis on V
existed only to satisfy `scoped_value`'s `len(axis_map)==rank` check and the general "input repeated across
the reduced axes" convention. `_combine_step_online_softmax_state` (composite_combines.py:151-171) already
broadcasts the scalar m/l correction factors to the accumulator's Hd lanes via `corr.broadcast(lanes)`, so
nothing downstream consumes V's q_len axis. Under full-model DAG sharing the generic collapse passes fold
V's degenerate broadcast source to a scalar PARAM load without rebuilding the enclosing RESHAPE → shapeless
node → panic.

**Fix (2 lines, minimal blast radius).** In `lower_attention_semantic` (rangeify.py):
- Build the composite reduce's V input (`logical_v`) at natural rank-4 `(b,h,kv_len,hd)`:
  `logical_v = work_v.cast(qk_dtype).uop.scoped_value((0,1,3,4))` (was rank-5 `reshape(...,1,...).expand(...)`
  with axis_map `(0,1,None,3,4)`). axis_map maps V's 4 source axes to the reduction's logical axes
  b→0,h→1,kv(reduce)→3,hd(lane)→4 — identical to before minus the dropped `None` (q_len) entry.
- `CompositeInputSpec("logical", (0,1,3,4), ...)` (was `(0,1,None,3,4)`); `lane_axis=4` unchanged (Hd's
  logical axis is still 4). `primary_repeated=True` unchanged (that flag governs the SCORE/primary input).

**Decoupling note.** The rank-5 `value_tile` (rangeify.py:101) is intentionally LEFT in place — it is still
consumed raw by the hd==16 owned-fragment carrier path (`construct_hd16_tile_carriers`, guarded
`owned_map_proven and hd==16`, NOT taken for the real hd=128 grid path). For the real 8B/14B path
`value_tile` is now dead (DCE'd) so its fragile RESHAPE never reaches scheduling. This leaves the hd==16
synthetic test geometry and `wmma.py` untouched — the unit-test fail-set must be unchanged.

Outcome classification (converges / next-crash / same-crash) pending the post-fix repro run.

### A2 — post-fix outcome: NEXT-CRASH (same family, DIFFERENT subsystem)

The V-side fix WORKED for its target: the `bad reshape: () -> (1,32,1,512,128)` crash is GONE; scheduling
advances past it. New crash, one pass deeper (same `symbolic+reduce_collapse+debuf` pass, rangeify.py:964):
```
RuntimeError: shape mismatch at Ops.MUL: [(1, 32, 512, 128), ()] [Ops.INDEX, Ops.CAST]  (ops.py:431, DISALLOW_BROADCAST)
```
Instrumented provenance (ATTN_SHAPE_DEBUG dump):
- src[0] = rank-4 activation `(1,32,512,128)`: INDEX(EXPAND(RESHAPE(STAGE bufferized, STACK<7>),
  STACK<7>=[1,512,4096,1,32,512,..]), 0, RANGE(99,LOOP,512), RANGE(101,REDUCE,**4096**)).
- src[1] = scalar `()`: CAST(MEMORY_SEMANTIC(MODEL_PARAMETER) -> ADD(MUL(MUL,CAST), MUL(MUL,CONST -1.0))).

Interpretation: this is NOT a composite-reduce attention input. The REDUCE extent 4096 = the model hidden
dim, and src[1] is a **Q4_K weight dequant** (scale*q + min) collapsed to a scalar. It is a **projection
matmul** (v_proj/o_proj class) UPSTREAM of attention. The same matmul lowers fine on the ordinary SDPA
path; the fused path's added DAG sharing exposes the collapse.

**Design implication.** This instance is structurally OUT OF REACH of design (1) (restructure composite-
reduce inputs) — it is an ordinary quant matmul, not an attention input. Two independent subsystems
(attention-V, now quant-dequant) now fail with the identical `()`-collapse signature on the fused path.
This shifts the evidence toward the crash being rooted in the **generic collapse pass** mishandling
degenerate broadcast axes wherever the fused DAG exposes them — i.e. design (2) territory (harden the
collapse pass to be shape-preserving), which the directive flags as higher blast radius (touches non-
attention graphs). Pending: exact collapse-rule identification (which rewrite folds a broadcast quant
scale to `()` without rebuilding the consumer) before choosing generic-pass fix vs attention-local
avoidance. If the family keeps yielding new subsystem classes, this is the escalation point.

### A2 — CRASH CLASS #2 pinned: upstream Q4_K projection-matmul weight loses its contraction index (ESCALATION)

Applied a shape-preserving guard at the `remove_bufferize` fold tail (design (2), surgical) — it had ZERO
effect; the crash was unchanged. A deep shape-transition walk (printing `_shape` at every ancestor of the
scalar operand) shows WHY: the collapse is NOT a shape-dropping rewrite. The scalar operand's root is:
```
PARAM shape=(65536,144) uchar  ParamArg(12)          # a Q4_K quantized WEIGHT (144 B/superblock)
  INDEX(PARAM, ADD_const, ADD_const) -> shape ()      # indexed by CONSTANT index expressions
  full Q4_K dequant (SHL/BITCAST/WHERE/AND/OR nibble unpack + scale/min) -> shape ()
```
The whole dequant is legitimately evaluated at a SINGLE constant index, so its natural shape IS `()`.
Meanwhile the co-operand (src[0], `(1,32,512,128)`) is indexed by `RANGE(101, AxisType.REDUCE, 4096)` —
a **contraction axis** over the hidden dim. So this MUL is a **projection-matmul body** `A[i,k]·W[k]`
reduced over k, and the weight has LOST its dependence on the contraction range k.

**This is a correctness wall, not just a shape wall.** Broadcasting the scalar weight back to the
activation's shape (the naive "make it schedule" move) would compute `W_const·Σ_k A[k]` instead of
`Σ_k A[k]·W[k]` — a silent miscompile. The shape check is correctly refusing a malformed graph. The
matmul-contraction should never have had its weight index resolved to a constant.

**Why it is beyond design (1) and (2):** it is an ORDINARY Q4_K projection matmul UPSTREAM of attention
(it also exists, correctly, on the SDPA path). Design (1) restructures composite-reduce *attention inputs*
— cannot reach it. Design (2) shape-preserves generic folds — but the value here is genuinely scalar at
its own node; there is no shape to preserve, and forcing one miscompiles. The real defect is that the
fused-attention composite lowering perturbs the whole-model bufferization/range-substitution such that an
upstream matmul weight's contraction RANGE is resolved to a CONST for the weight operand but not the
activation operand (asymmetric). That is a deeper composite-lowering / bufferization-boundary rework.

**Enumerated crash classes so far:**
1. (FIXED, design-1) Attention V-input size-1 q_len axis → `bad reshape: () -> (1,32,1,512,128)`.
   Fix: build `logical_v` at natural rank-4 (rangeify.py), non-regressing (unit tests 6/93 unchanged).
2. (OPEN, deeper) Upstream Q4_K projection-matmul weight loses its contraction-range index under fused
   DAG restructuring → `shape mismatch at Ops.MUL: [(1,32,512,128), ()]`. Broadcast-back = miscompile.
   Root cause = asymmetric RANGE→CONST resolution across a matmul's two operands under the composite
   graph. Requires preventing the erroneous weight-index collapse (keep the weight ranged over k), i.e.
   a composite-lowering/bufferization-boundary fix, NOT a shape tweak.

**Per the directive's "If blocked" clause, this is the escalation point** ("if the crash family keeps
yielding new degenerate-axis classes, STOP and escalate the design question rather than burn GPU time").
The design question for the user: how should the fused-attention composite lowering avoid perturbing
upstream (non-attention) matmul bufferization so a Q4_K weight keeps its contraction index — (a) constrain
the composite lowering's bufferization boundary so upstream matmuls schedule independently (as they do on
the SDPA path), (b) make the range→const substitution contraction-aware (refuse to constant-fold a
reduce-axis index on a matmul weight), or (c) a deeper rework of the composite-reduce lowering. Read-only
mechanism investigation (no GPU) is proceeding to turn this into a concrete candidate before further runs.

### A2 — CLASS #2 mechanism NARROWED by falsification (DECISIVE; escalation stands)

Ran three targeted fixes against class 2 and one decisive causal probe. All GPU runs, 8B forced-fused repro.

Falsified candidate fixes (each left the crash BYTE-IDENTICAL — `shape mismatch at Ops.MUL:
[(1,32,512,128), ()]` at cleanup_dead_axes rangeify.py:481):
1. Shape-preserving reshape+expand guard at `remove_bufferize` tail (rangeify.py:588) — no effect (the
   weight is genuinely rank-0 at its own node; there is no shape to preserve there).
2. `matmul_reduces` guard also bailing when `accessed_buffers` contains a PARAM (rangeify.py:574) — no
   effect (the weight sits behind a STAGE boundary, so `red_gate` never surfaces the PARAM there; the
   collapse does not flow through this guard).
3. Bounding the `composite_owned` walk at GLOBAL-STAGE/AFTER/PARAM kernel boundaries (indexing.py:226) —
   no effect.

DECISIVE causal probe: forced `composite_owned` **completely empty** (DIAG_NO_COMPOSITE_OWNED). The crash
**persists byte-identical.** This DEFINITIVELY FALSIFIES the `composite_consumer`/`composite_owned`
over-scoping mechanism (the leading hypothesis). The upstream Q4_K projection weight loses its
contraction-range index **independent of composite ownership** — i.e. the collapse is intrinsic to how the
composite attention reduce restructures RANGE ASSIGNMENT / bufferization for the whole-model graph, not to
which buffers are flagged composite consumers.

**Remaining locus (not yet fixed):** the asymmetric multi-consumer range-merge in `run_rangeify`
(indexing.py:283-304 — a per-occurrence `all_same(local_rngs)` identity check that can hand one occurrence
of a shared weight an independent `new_range`/realize-axis while the sibling activation keeps the shared
contraction RANGE) and/or `reduce_unparented` (codegen/simplify.py:82-96, which strips a RANGE from a
REDUCE once one operand no longer references it). Confirming which requires deeper interactive tracing of
range assignment across the per-KV-block-unrolled composite subgraph — the composite path's own docstring
(rangeify.py:24-26) notes it "expands into one materialized Tensor subgraph per KV block today," which is
the structural source of the non-identical shared-weight occurrences.

**This is the directive's escalation point.** Class 2 is a genuine deeper composite-reduce-lowering defect
(design (c)), NOT reachable by design (1) [attention-input restructure] or (2) [collapse-pass shape
preserve], and NOT by composite-ownership scoping. Recommended before more GPU time: decide the design
approach — (c1) make the range-merge share the contraction RANGE across all occurrences of a matmul weight
(keep the weight ranged), (c2) prevent `reduce_unparented` from dropping a range still referenced by a
sibling reduce operand, or (c3) rework the composite lowering so it does not unroll per-KV-block into
structurally-distinct weight-index occurrences in the first place. Each is a bounded but real compiler task
in range assignment; the choice affects blast radius on all matmul scheduling.

Class 1 (V-input fix) remains committed, gated off, both gates green (d51bd3e92).
