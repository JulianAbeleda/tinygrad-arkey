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

### A2 — CLASS #2 hypothesis VERIFICATION (per user request "verify first"): real but entangled

Instrumented the suspected merge site (indexing.py:302, the `else: new_range + realize_axis` branch) and
ran the 8B forced-fused repro (DIAG_MERGE). Findings:
- The merge-else branch fires **446 times per schedule; 425 reach a weight PARAM**. It is UBIQUITOUS normal
  scheduling behavior, NOT a smoking gun. Minting a new range for a PARAM-reaching node happens constantly.
- The predicted divergence IS present though: weight PARAMs get DISTINCT-identity ranges on extent-4096
  contraction axes (e.g. `PARAM axis=2 extent=4096 nconsumers=3 distinct_local=3`; `RESHAPE axis=2
  extent=4096 distinct_local=2`). The fused path gives a projection weight EXTRA consumers (3-4) whose
  contraction ranges are not range-identity-equal -> the range-divergence mechanism is real.

Conclusion (evidence-based):
- The range-identity-divergence mechanism is REAL but ENTANGLED with ubiquitous normal scheduling. It is
  not cleanly isolable at the merge without a fused-vs-SDPA differential on the specific collapsing weight.
- **c1 (fix the general merge branch) is now judged too risky**: it runs 446x/schedule for EVERY model;
  altering its range aliasing on an entangled signal risks broad non-attention regressions. Rejected as a
  blind fix.
- The divergence is driven by the EXTRA weight-consumers the fused attention path creates, which points at
  **c3 (remove the duplication at the attention source, lower_attention_semantic)** as the more surgical
  direction. BUT the specific collapsing weight was not isolated (PARAM(65536,144) = 4096x4096 could be
  q_proj / o_proj / an FFN weight), so c3's exact edit point is not yet pinned, and c3 remains a band-aid
  for the general shared-weight/multi-occurrence hazard.

Net: verification correctly steered away from the high-blast-radius c1 and confirmed the mechanism, but did
NOT yield a clean, provable fix. Next required step before ANY implementation: isolate the exact collapsing
weight and its duplication path (fused-vs-SDPA differential trace), then implement c3 at the source and
prove correctness by fused-vs-SDPA next-token numerics. Escalated to user for direction on whether to
invest that isolation or hand class 2 to a dedicated compiler session. Class 1 remains fixed/shipped
(d51bd3e92), gated off, both gates green.

### A2 — CLASS #2: c3 attention-local fix FAILED; four fixes exhausted -> needs interactive expert session

Identified the collapsing weight as **q_proj** by shape (PARAM(65536,144)=4096x4096 rules out k/v_proj at
4096x1024; MUL output (1,32,512,128)=(b,heads,tok,hd) with a separate 4096 reduce is q_proj's signature,
not o_proj's (1,512,4096)). q_proj is pulled into the fused `score` as a lazy subgraph.

Attempted c3 (attention-local): force Q/K/V `.contiguous()` in lower_attention_semantic so the projection
matmuls realize into their own kernels (single weight consumer -> no range divergence), matching the SDPA
path and the proven isolated harness. **NO EFFECT** — crash byte-identical. (Caveat: `.contiguous()`
inserted during a mid-schedule rewrite may not force a real realize boundary in this context, so this is a
weak refutation of the consumer-duplication theory, not a strong one — but it did not work.)

**Status: class 2 has resisted FOUR distinct targeted fixes and one decisive mechanism-falsification:**
1. remove_bufferize shape-preserving reshape/expand (rangeify.py:588) — no effect.
2. matmul_reduces guard bailing on PARAM in accessed_buffers (rangeify.py:574) — no effect.
3. composite_owned walk bounded at kernel boundaries (indexing.py:226) — no effect;
   and DECISIVE: composite_owned forced empty -> crash identical (composite_consumer NOT causal).
4. Q/K/V `.contiguous()` at the attention boundary (rangeify.py:78-81) — no effect.
Verification also showed the range-divergence mechanism is real but ENTANGLED with ubiquitous normal
scheduling (merge-else fires 446x/schedule), so it is not cleanly isolable via one-shot instrumentation.

**Conclusion / recommendation.** Class 2 is a deep, resistant range-assignment/composite-lowering defect
that does not yield to one-shot, hypothesis-driven remote fixes. Every remote-agent mechanism diagnosis
(composite_consumer over-scope; merge range-divergence; q_proj consumer duplication) has been either
falsified or failed to fix when acted on. This class needs INTERACTIVE expert compiler debugging: step
through the actual range assignment for the specific collapsing q_proj node under the fused DAG (pdb /
DEBUG_RANGEIFY at the merge and cleanup_dead_axes), comparing the exact RANGE identities of the weight vs
activation operands of the failing MUL, rather than more remote hypotheses. Alternatively, revisit whether
the composite attention lowering should consume PRE-MATERIALIZED Q/K/V by construction (a real realize
boundary enforced upstream at the model.py call site, att.src[2,3,4], not a mid-rewrite .contiguous()),
which is the structural intent but needs to be done where it actually forces separate kernels.

Class 1 (V-input degenerate axis) remains FIXED, committed (d51bd3e92), gated off, both gates green. That
is the shippable result of this session. Class 2 is fully characterized and handed off with the four
falsified approaches enumerated so no future session repeats them.

### A2 — CLASS #2 ROOT CAUSE PROVEN + approach (a) tested (2026-07-24)

First-principles breakthrough (working from the known-good isolated kernel's input contract):

**PROVEN root cause.** In the real single-layer block, replacing Q/K/V with provenance-free
`Tensor.empty` buffers (the isolated harness's exact contract) makes the fused path SCHEDULE CLEAN;
real Q/K/V crash. So class-2 is entirely about INPUT PROVENANCE, not the composite lowering (which the
harness proves correct). Role-tag ground truth: the collapsing `Ops.MUL [(1,32,512,128), ()]` is the
class-1 `logical_v` (composite V input, tag `ScopedValueSpec(0,1,3,4)`) reading the KV cache (PARAM 8),
spuriously EXPANDED with a 4096 (hidden) axis and multiplied by a Q4_K weight (PARAM 12) collapsed to
`()`. The composite reduce fuses V's projection-matmul + `cache_kv.after(store(stack(k,v)))` provenance
into itself, tangling V's KV reduce with a hidden-4096 projection contraction.

**Ruled out (with evidence):**
- `.realize()` — fundamentally impossible: the prefill-capture path runs under
  `Context(ALLOW_DEVICE_USAGE=0)` by design (schedule-only, must not execute inline).
- Direct pre-cache K/V — still carries v_proj matmul provenance; crashes.
- **`.contiguous()` on K/V in lower_attention_semantic (approach a) — TESTED, INSUFFICIENT.** The
  composite lowering reaches THROUGH the CONTIGUOUS boundary (its V-lane packing,
  `_pack_online_softmax_v_lanes` / devectorizer, walks into V's INDEX/base to rebuild lanes and lands on
  the v_proj reduce). A leaf buffer (empty / a PARAM) has no src to reach through, which is why the
  harness works.

**Loud failure ADDED (permanent).** `tinygrad/uop/ops.py` DISALLOW_BROADCAST site now detects the
class-2 pattern (a `ScopedValueSpec`-tagged composite input meeting a rank-0 `()` sibling) and raises a
descriptive error naming class-2 + the fix requirement + this doc, instead of the cryptic
`shape mismatch [(...),()]`. Runs only on the already-failing path.

**Refined fix direction (next):** give the composite reduce K/V whose reachable base is a clean
BUFFER/PARAM, not a matmul. Most promising: for the fused path, read K/V from the raw `cache_kv` PARAM as
a clean buffer read (not via `assigned_kv = cache_kv.after(store(...))`, which routes the reach-through
into the projection), with the cache store kept as a separate scheduled sink. Alternatively, stop the
composite V-lane packing from reaching through a bufferize boundary. Both are bounded; the cache-PARAM
read matches the harness's leaf-buffer contract most directly.

### A2 — CLASS #2 FIX REQUIREMENT CONFIRMED + exact reach-through localized (2026-07-24)

Isolation experiments on the fast 1-layer repro (env-gated probes in model.py, since reverted):
- K/V as clean raw `cache_kv` PARAM reads, Q left dirty (`q.cast(half)`) -> STILL class-2 (q_proj pulled
  in via `score`). So a PARAM leaf DOES stop the reach-through, but ALL of Q/K/V must be clean.
- K/V clean (cache PARAM) AND Q clean (empty buffer) -> **class-2 GONE** (guard did not fire); only a
  benign `cycle detected while indexing cache_kv` remained (my probe read the cache raw while it was
  also being stored — an ordering cycle, not class-2). CONFIRMS: the fix is to give the composite reduce
  leaf-buffer Q/K/V.

EXACT reach-through mechanism (the fix site): the composite V-lane reconstruction
`_pack_online_softmax_v_lanes` (composite_combines.py:173-192) and the fallback in
`_handle_no_range_generic` (composite_combines.py:249-262) rebuild each V lane as
`base = carrier.src[0]; base.index(*prefix, idx + off)`. In the real path V = `assigned_kv[1,...]` where
`assigned_kv = Tensor(cache_kv.after(store(stack(k,v))))`, so `base = AFTER(cache_kv, store)` and
`base.index(...)` drags the STORE (=> v_proj Q4_K matmul) into every reconstructed lane -> the composite
reduce fuses the projection and collapses its weight. With a clean PARAM/BUFFER base (empty or raw cache)
`base.index(...)` is opaque and correct.

Two candidate fixes (both bounded; ordering/correctness must be preserved):
- (F1) Composite-lowering: in the lane rebuild, peel scheduling-ordering wrappers (Ops.AFTER, and STAGE)
  off `base` so lanes index the underlying buffer, while the composite reduce still depends on V at the
  top level for ordering. Most general (covers Q via the score/primary path too), but must not drop the
  store->read ordering.
- (F2) Model-side: present Q/K/V to the attention op as genuine leaf buffers — K/V via ordered cache
  reads whose base is the cache PARAM (not AFTER), Q via a scratch buffer. Needs the ordering-without-
  cycle and a Q scratch buffer.

Loud class-2 guard (committed 2ebdb2e15) now makes any regression here fail with a clear message.

### A2 — CLASS #2 FIX PLAN (compiler-side; recorded 2026-07-24)

Correction to earlier "model-side F2 is easy": the model marks (`prefill_scratch` etc.) only attach
allocation annotations, not buffer boundaries, and `.contiguous()` is reached-through. Real-valued K/V
require a STORE, and the scheduler's store-to-load forwarding is exactly what inlines the v_proj source
into the read. So model-side buffering hits the same wall — **the fix is compiler-side**, in the
composite-reduce input lowering.

THE FIX (bounded; two sites + one ordering invariant):
1. Make the composite reduce consume Q/K/V as opaque buffer LOADS, not reach-throughs. When the per-input
   index base is an `Ops.AFTER`/`Ops.STAGE` (scheduler-ordering wrapper), load from the underlying buffer
   instead of forwarding the store's source, WHILE keeping the AFTER as an ordering edge (no read-before-
   write race). Two reach-through sites:
   - V-aux: `_pack_online_softmax_v_lanes` / `_handle_no_range_generic` (composite_combines.py 173-192,
     249-262). A naive AFTER-peel here alone was insufficient (left Q dirty).
   - Q/score-primary: the score-primary reconstruction (devectorizer `_select_vector_lane_at` /
     `primary_repeated` path) pulls score's Q (q_proj). MUST pin exact site.
2. Acceptance gate: A4 fused-vs-SDPA next-token numerics (ordering bugs are SILENT; schedule-success is
   not acceptance). Then A3 confirm `amd_gfx1100_q16_grid_hd128_loop_attention` fires; 14B; A5-A8 tail.

The loud class-2 guard (2ebdb2e15) catches any regression with a clear message. The proven root cause and
fix requirement (clean leaf-buffer Q/K/V) are the north star: this is the coupling the isolated 254-VGPR
"lab kernel" always assumed but was never wired to a real model graph.

### A2 — CLASS #2: reach-through is UPSTREAM of rangeify (store-forwarding); attempts tested

Attempted the targeted fix at the rangeify-time V-index site (indexing.py:132-141): peel AFTER wrappers
off the V value before `src.src[0].index(*idxs)` and reattach ordering ends. Instrumented result:
`v_val` at that point already contains the full Q4_K dequant (SHL/MUL/ADD in its toposort) AND one AFTER.
Peeling the AFTER did NOT clear class-2 -> the v_proj recompute is inlined into V's value BEFORE this
site (the store-to-load forwarding already happened upstream), so there is no clean `cache_kv` LOAD left
to preserve at rangeify time. A single-site peel is too late.

Precise state of the fix:
- The composite reduce needs V (and the score/Q side) to reference an opaque BUFFER LOAD. Real V is either
  the cache read (forwarded to the v_proj recompute upstream) or the direct projection -- both reachable
  to v_proj. Only a genuine buffer (empty, or a non-forwarded cache LOAD) is clean (proven).
- The fix must PREVENT the store-to-load forwarding for composite-reduce-consumed cache reads (so V stays
  a buffer LOAD ordered after the store), OR restructure so V/K/Q are genuine buffer LOADs before the
  attention op -- at BOTH the V-aux and score/primary sites. This is upstream of rangeify and multi-site;
  not a single-line patch. Store-forwarding rewrite not yet pinned to one rule.

Realistic remaining work (bounded but real): (1) locate + gate the store-to-load forwarding for composite
V/K so the cache read stays a LOAD; (2) do the equivalent on the score/primary Q path; (3) verify ordering
via A4 fused-vs-SDPA numerics (silent-miscompile risk); (4) A3 kernel-fires; (5) 14B; (6) A5-A8 tail.

Shipped this session: class-1 fix (d51bd3e92), loud class-2 diagnostic (2ebdb2e15), full root-cause proof
+ fix requirement + localization (12fc34305, 1384e0115). The "lab kernel" is correct; the missing
coupling is feeding it buffer-LOAD Q/K/V, and store-forwarding is what defeats every non-buffer attempt.
