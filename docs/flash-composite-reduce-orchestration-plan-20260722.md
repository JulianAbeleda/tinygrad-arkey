# Orchestration plan: composite-reduce flash — Claude drives, low agents execute

**Owner:** Claude (orchestrator + gate). **Executors:** low-effort agents, one concrete phase each. **Verification:** Claude in the main loop after every agent — no result is trusted until I reproduce its artifact.

**Why this supersedes deepseek's "infra ready" framing:** verified 2026-07-22 — deepseek's "full-pipeline proof" was a normal `x.sum()` *injected* at `reduce_to_acc` (late stage), bypassing spec/rangeify. A genuinely-built composite reduce **fails UOp verification** (`at 20 on Ops.REDUCE`). Plus a copy-paste bug (`composite_reduce` duplicated into `UPat`) and no committed end-to-end test. So the primitive is NOT yet a validated first-class construct. Phase A fixes that before the scheduler work.

Repo: `/home/ubuntu/tinygrad-arkey` · Python: `.venv/bin/python` · Env: `DEV=AMD`

## Global rules (every agent)
- **Single GPU lane → agents run ONE AT A TIME.** Claude serializes; never two GPU agents at once.
- **Agents gather verifiable artifacts, Claude decides.** Every task ends with a concrete artifact (a passing test, a per-kernel `__WMMA` dump, a `max_rel_err`, a buffer list) — never a prose conclusion. Claude reproduces it before the next phase.
- **No hand kernels** (no `custom_kernel`, UOp *kernel bodies*, `__builtin_amdgcn`/barriers/LDS, `flash_kernels` imports). This is scheduler/codegen work. (Building a composite-REDUCE UOp *graph* is allowed and required — that is not a hand kernel.)
- **Prove WMMA per-kernel** (dump `__WMMA` call sites + which kernel); never aggregate grep counts.
- **Run the relevant test suite before every commit**; keep it unregressed (baseline: WMMA/packed subset `54 passed/10 skipped/5 xfailed`; `test_tiny`+scheduler `38/1`; pre-existing: 3 `test_wmma_emitted_code_fixtures` subtests, ignore).
- `tm` not wall-clock, warm ≥200. Commit on master, no branches, Co-Authored-By trailer, push. Honest fallback: stop + report the precise blocker rather than fake/force.

## Phase A — Make the composite reduce a real first-class construct (small, do first)
**Goal:** a genuinely-constructed multi-slot composite reduce passes verification and lowers to correct results for ALL slots — the proof deepseek's 1-slot reroute did not give.
1. **Fix the copy-paste bug:** `composite_reduce` is defined twice in `ops.py` (line 594 on UOp, line 1399 inside the `UPat` class with a UOp-creating body). Remove the wrong one from `UPat`.
2. **Diagnose + fix verification:** find the spec check that rejects a REDUCE with a `CompositeReduce` arg (`spec.py` — the `hasattr(x.arg[0],'slots')` relaxation is incomplete; the failure is `at 20 on Ops.REDUCE`). Make a properly-formed composite REDUCE (post-rangeify RANGE-src form, as the scheduler would emit) pass verification.
3. **Real end-to-end test (the Phase A gate):** build a 2-slot `(ADD, MAX)` composite reduce over a small tensor, run it through the full realize pipeline, and assert **BOTH** slots correct (sum AND max) — not just `accs[-1]`. Commit as `test/unit/test_composite_reduce.py`.
**Artifacts:** the passing test (both slots), regression suite counts. **Gate:** Claude reproduces the test.

## Phase B — Scheduler emits the composite reduce for attention (the core, multi-step)
**Goal:** a rangeify graph-rewrite recognizes `softmax(q@kᵀ·scale + mask) @ v` and restructures it into ONE composite REDUCE over KV carrying `(m, l, acc)` — QKᵀ and PV as inner ADD+MUL contractions, softmax correction as the combine — so the `T×KV` score never materializes.
- Anchor: `tinygrad/schedule/rangeify.py` (the graph-rewrite machinery + where the score bufferize is inserted, `remove_bufferize`). The online-softmax combine math (NOT the kernel) ← `extra/qk/flash_kernels.py` for reference.
- Sub-steps (each an agent + Claude gate): (B1) pattern-match the attention subgraph and emit the composite REDUCE structure, correctness first with NO WMMA; (B2) prove the score buffer is gone (per-kernel buffer list) and `max_rel_err ≤ 1e-2` vs fp32.
**Fallback:** if rangeify cannot emit the composite structure without REDUCE→LOOP (kills WMMA) or a hand kernel → stop, report the precise rangeify blocker.

## Phase C — WMMA on both inner contractions
**Goal:** the TC opt WMMA's BOTH the QKᵀ (over Hd) and PV (over KV) contractions inside the composite reduce. Resolve the single-tag limit at `postrange.py:391` (`get_single_element([... tag=="TC"])`) — it currently allows one TC reduce; attention needs two.
**Artifacts:** per-kernel dump with TWO `__WMMA` call sites, score buffer still gone, correctness held.

## Phase D — Gate + wire
Measure the composite-reduce attention vs materialized SDPA at `T=KV=2048`: two-ceiling table (empirical ceilings), absolute `tm` (warm ≥200), per-kernel WMMA, `max_rel_err`. If faster with correctness → wire into `model.py:583-598` + 14B integration test (tok/s vs shipped). Else → honest numbers, do not wire.

## Sequencing
A → B1 → B2 → C → D. Claude gates between each; agents serialized (single GPU). The hard, genuinely-multi-week work is B (scheduler emission) and C (dual-contraction WMMA). A is small but non-negotiable (it makes "the primitive works" actually true). Stop honestly at any real wall.

---

## ⭐ PIVOT (2026-07-22) — Phase A.5 multi-output is a value-model wall; avoid it with two reduces

**Phase A.5 BLOCKED (verified by Claude).** Exposing >1 slot from one composite REDUCE collides with two pervasive, generic assumptions: `rangeify.py:423` (`size = prod(shape)//dtype.count` → 0 for a vec-dtype reduce) and `symbolic.py:212` (GEP-in-dtype-order folds to identity for scalar dtype). A clean fix needs a NEW `Ops` member (e.g. `Ops.REDUCE_SLOT`) audited across every dispatch table — a large refactor. Agent stopped honestly, reverted clean.

**Workaround that stays on existing infra — TWO composite reduces (no multi-output needed):**
- Reduce 1: slots `(m, l)` → surfaces `l` (last slot, via existing `accs[-1]`)
- Reduce 2: slots `(m, acc)` → surfaces `acc` (last slot)
- `out = acc / l`

Both reduces recompute the same deterministic running max `m` → `l` and `acc` stay consistent. Each surfaces only its LAST slot — exactly what Phase A already supports. **Both are score-resident** (score never materialized), so the flash memory win holds; cost is one extra QKᵀ (cheap WMMA). This avoids the value-model wall entirely.

**Revised sequencing:** A ✅ → (A.5 multi-output — SKIP, use two-reduce) → **B: rangeify emits the TWO-reduce structure for `softmax(q@kᵀ)@v`** (score-resident, correct, no WMMA) → C: WMMA both QKᵀ (shared/recomputed) and PV contractions → D: gate vs SDPA (must beat it despite the extra QKᵀ) + wire. If the two-reduce recompute makes it lose to SDPA at D, that's the honest NO-GO.

---

## ✅ Two-reduce formulation VALIDATED + exhaustive next steps (2026-07-22)

**Two-reduce online-softmax = exact attention** (numpy, max_abs_err 7.6e-8, MATCH). Splitting into `(m,l)→l` and `(m,acc)→acc`, sharing the deterministic running max `m`, reproduces standard attention. The formulation is proven; the multi-output wall is genuinely avoided. This is a build, not a wall.

**Concrete steps (each = an agent + Claude gate; no step is blocked, each is a modification of working infra):**
- **S1 ✅ two-reduce math** — validated exact.
- **S2 — 2-slot coupled combines.** Generalize the existing hardcoded 3-slot `"online_softmax"` combine in `reduce_to_acc` into two 2-slot coupled combines: `(m,l)` surfacing `l`, and `(m,acc)` surfacing `acc` (vec-Hd). Each surfaces its LAST slot — existing `accs[-1]` already does this, so NO multi-output needed. Test each combine in isolation (correct `l`, correct `acc`) vs numpy.
- **S3 — two-reduce attention through the pipeline (the workaround test).** Construct `softmax(q@kᵀ)@v` as two composite reduces manually (as Phase A constructed one), realize, assert `max_rel_err ≤ 1e-2` vs fp32 reference. Proves the composite machinery does two-reduce attention correctly (no rangeify emission, no WMMA yet).
- **S4 (Phase B) — rangeify emits the two-reduce structure** from the `softmax(q@kᵀ·scale+mask)@v` subgraph; score-resident (no `T×KV` buffer); correct.
- **S5 (Phase C) — WMMA** on the QKᵀ (over Hd) and PV (over blockKV) contractions inside the composite reduces (dual-contraction; resolve `postrange.py:391` single-tag).
- **S6 (Phase D) — gate** vs SDPA at 2048 (must beat it despite the extra QKᵀ); wire into `model.py:583-598` + 14B if GO.

Mindset: each step is a modification of infrastructure that already works, not a new wall. Prove each with an artifact; only stop if a step is genuinely, verifiably impossible.
