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
