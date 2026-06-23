# Prefill Register-Lifetime / Pool Representation — Exhaustive Scope / Claude Prompt

Date: 2026-06-23

## Mission
Build the **register-lifetime representation** — a liveness-driven VGPR **pool allocator** for `build_gemm_lds2` — so a
**deep** software-pipelined K-loop (full A+B prefetch) fits the **≤ 256-VGPR** envelope without spilling. This is the
one named blocker between the now-emittable schedule template and a real whole-prefill speedup.

The chain so far:
- the prefill ~4–5 % gap to Tensile reduced to **K-loop software pipelining** + **register-pool lifetime** (not tile
  config) — `prefill-schedule-diff-oracle-and-search-reduction-result`;
- the `schedule_template` is **emittable**: `build_gemm_lds2 DBUF=1` is PIPELINED, correct, ~236 VGPR, 0 spill —
  `prefill-kloop-schedule-template-microkernel-result` (`KLOOP_SCHEDULE_TEMPLATE_MICROKERNEL_PASS`);
- but the **Tensile-class depth** (`PLRAB`, full A+B prefetch) **overflows: VGPR 300 > 256** — the falsifiable target
  this scope attacks.

This is **not** a renderer rewrite, **not** a model route, **not** a whole-prefill speed claim until the gate passes.
It is a bounded allocator over the existing hand-asm emitter.

## Core problem (concrete, from `extra/gemm/rdna3_wmma_matmul.py:build_gemm_lds2`)
`build_gemm_lds2` uses **static** VGPR regions stacked linearly:
```
FA=10 → FB=FA+WM*8 → ACCb=FB+WN*8 → CTA=ACCb+WM*WN*8 → CTB=CTA+loadsA*4 → SCR=CTB+loadsB*4 → FB2=SCR+2
assert SCR+2 ≤ 256                       # DBUF/PLRA fit
PLRAB needs FB2 + WM*8 + WN*8 ≤ 256      # full A+B prefetch -> 300 for WM=WN=4 (the wall)
```
The **accumulators** (`ACCb..CTA` = `WM*WN*8` = 128 regs at 4×4) live the **whole** K-loop — an unavoidable floor.
The **coop-load temps** (`CTA`,`CTB`) and **fragment buffers** (`FA`,`FB`,`FB2`) are **short-lived / per-stage**.
Static stacking gives each its own physical regs even when their live windows don't overlap. Tensile uses a **register
pool**: the next-tile prefetch fragments **reuse the physical VGPRs of regions that are already dead** at the prefetch
point (the `PLRA` lever already does a partial version — "prefetch into the DEAD coop-temp regs"). The representation:
make that reuse **systematic and liveness-checked**, so full A+B prefetch fits within the 128 non-accumulator regs.

## Core question
```text
Can a liveness-based VGPR pool fit a deep (A+B) software-pipelined K-loop in <= 256 VGPR,
staying PIPELINED + correct + no-spill?  If yes -> prefill schedule search reopens with a path to ~4-5%.
If no -> the gap is a hardware register-pressure limit (needs a smaller WMMA tile or is Tensile-only).
```

## Required reading
1. `docs/prefill-kloop-schedule-template-microkernel-result-20260623.md`
2. `docs/prefill-schedule-diff-oracle-and-search-reduction-result-20260623.md`
3. `docs/machine-search-representation-expansion-decode-prefill-result-20260623.md`
4. `docs/prefill-amd-gemm-leanaddr-result-20260620.md` (the PLRA dead-reg reuse precedent)
5. `docs/prefill-tensile-schedule-template-extraction-result-20260620.md` (Tensile register-pool evidence)
6. `docs/prefill-primitive-pmc-result-20260619.md`
7. `bench/qk-decode-eval/HARNESS_GUIDE.md`
8. `structure/Development/performance-primitive-research-principles.md`
9. `structure/Development/session-handoff.md`

Inspect: `extra/gemm/rdna3_wmma_matmul.py` (`build_gemm_lds2`, the FA/FB/ACCb/CTA/CTB/SCR/FB2 layout, `PLRA`/`PLRAB`),
`extra/qk_prefill_kloop_template_microkernel.py`, `extra/qk_schedule_interleave_detector.py`,
`extra/qk_amdgpu_isa_primitive_audit.py`, `extra/qk_prefill_whole_synced.py`, `extra/qk_project_search_ledger.py`.

## Non-goals
- No `tinygrad/` source change; no renderer rewrite; no model route; no default flip.
- No whole-prefill speed claim until the register-lifetime + correctness + interleave gates pass.
- No broad tile-config search; no vendored-Tensile promotion; no RL/LoRA/training.
- Do not hand-tune a one-off kernel; build the **allocator/representation** (generalizes across the prefetch depth).
- If it expands into a general SSA register allocator / renderer pass, **stop** and classify `REQUIRES_RENDERER_WORK`.

## Authority
This is a **register-pressure representation proof**. Gate order (stop at first failure):
1. build/compile succeeds;
2. liveness model is sound (max-live ≤ physical, no live-range overlap conflict);
3. **register-lifetime gate**: `max_live_vgpr ≤ envelope (256)`, **0 spill/scratch**;
4. local numeric correctness (rel_rmse ≤ 3e-4) vs numpy;
5. `schedule_interleave_gate`: still **PIPELINED** (the deep A+B prefetch, not collapsed back to phased);
6. **only then** an optional local timing / whole-prefill synced diagnostic.
W==D/whole-prefill is promotion authority and is touched ONLY after 1–5 pass. PROFILE/nosync/local timing = diagnostic.

## Phase 0 — Authority lock
Reproduce: `build_gemm_lds2 DBUF=1` PASS (PIPELINED, ~236 VGPR, correct) **and** `PLRAB` wall (VGPR 300 > 256). Record
the exact per-region VGPR layout (FA/FB/ACCb/CTA/CTB/SCR/FB2) for the target config. Artifact
`bench/qk-prefill-register-lifetime/authority.json`. Verdicts `REGISTER_LIFETIME_AUTHORITY_LOCKED` /
`REGISTER_LIFETIME_BASELINE_DRIFT_STOP`.

## Phase 1 — Liveness model
Build a static liveness table for the K-loop stages: for each register region, record `born_stage`, `last_use_stage`,
`size`, and whether it spans the whole loop (accumulators) or is per-stage (operands/prefetch/coop-temps). Compute the
**max simultaneously-live VGPR** under (a) the current static stacking and (b) an ideal liveness-packed assignment.
Artifact `bench/qk-prefill-register-lifetime/liveness_model.json`. Verdicts `LIVENESS_MODEL_READY` /
`LIVENESS_MODEL_INSUFFICIENT`. **Decision branch:** if even the ideal packed `max_live` for deep A+B prefetch > 256,
record `REGISTER_POOL_INSUFFICIENT_HW_LIMIT` (needs a smaller WMMA tile) and stop before building the allocator.

## Phase 2 — Pool allocator (the representation)
Implement a bounded **pool allocator** that assigns physical VGPRs by liveness window (interval/graph-coloring over the
stage timeline), reusing dead regions for prefetch fragments. Wire it into `build_gemm_lds2` as an **additive mode**
(e.g. `REGPOOL=1`, default off — current static path unchanged). Tool
`extra/qk_prefill_register_pool_allocator.py` (or extend the emitter). Requirements: deterministic; reject if
`max_live > 256`; emit the same WMMA/LDS/prefetch instruction families (only the VGPR indices change). Artifact
`bench/qk-prefill-register-lifetime/pool_allocation.json`. Verdicts `REGISTER_POOL_ALLOCATOR_BUILT` /
`REGISTER_POOL_ALLOCATOR_BLOCKED` / `REQUIRES_RENDERER_WORK`.

## Phase 3 — Deep-pipeline re-emit + gates
Re-emit the **full A+B prefetch** K-loop **with the pool** on the microkernel shape (128×128×256, ≥2 K-tiles). Run, in
order: register-lifetime gate (`max_live ≤ 256`, 0 spill) → correctness (rel_rmse ≤ 3e-4) → interleave detector
(PIPELINED, deep). Artifacts `bench/qk-prefill-register-lifetime/{correctness,interleave,isa_resource}.json`.

## Phase 4 — Classification
- `REGISTER_POOL_FITS_DEEP_PIPELINE` — deep A+B prefetch fits ≤ 256, correct, PIPELINED, 0 spill → the unlock; proceed.
- `REGISTER_POOL_PARTIAL` — a deeper-than-DBUF but not-full prefetch fits (record the achievable depth).
- `REGISTER_POOL_INSUFFICIENT_HW_LIMIT` — even ideal packing > 256 → smaller WMMA tile or Tensile-only.
- `REQUIRES_RENDERER_WORK` — cannot express dynamic pooling in the hand-asm emitter without a general allocator/renderer.

## Phase 5 — (only if FITS) bounded local + whole-prefill diagnostic
If `REGISTER_POOL_FITS_DEEP_PIPELINE`: a local timing sanity (vs the phased/DBUF microkernels), then — and only then —
a **clock-pinned synced whole-prefill** check on the real ffn_down/qo roles via the existing `PREFILL_GEMM_CFG_*`
route override (additive, default off). This is the first legitimate whole-prefill speed measurement for this lane.
Authority = synced whole-prefill; isolated GEMM = diagnostic. Verdict `REGISTER_POOL_DEEP_PIPELINE_WD_PASS` /
`..._NONTRANSFER` / `..._REGRESSION`. **No default flip** — a win is recommend-only.

## Phase 6 — Ledger + result doc
Append a project-ledger entry (lane `prefill`, primitive_class `register_lifetime`, authority local-or-whole-prefill,
the learned rule). Write `docs/prefill-register-lifetime-pool-representation-result-20260623.md`. If the pool fits, add
a `register_lifetime` SearchRow example to the explorer (the representation becomes a real search level).

## Expected final verdicts
Best: `REGISTER_POOL_FITS_DEEP_PIPELINE` + (if whole-prefill transfers) `PREFILL_SCHEDULE_SEARCH_REOPENED`. Honest
likely outcomes given the 128-reg accumulator floor: `REGISTER_POOL_PARTIAL` (some extra depth, sub-Tensile) or
`REGISTER_POOL_INSUFFICIENT_HW_LIMIT` (the 4×4 WMMA tile's 128-reg accumulators leave too little for full A+B prefetch
→ needs a smaller tile, which trades occupancy). Worst-for-search: `REQUIRES_RENDERER_WORK`.

## Claude prompt
You are in `/home/ubuntu/tinygrad-arkey` on branch `qk-prefill-flag-leak-resolution`. Execute the prefill
register-lifetime / pool representation scope. Goal: a liveness-driven VGPR **pool allocator** for `build_gemm_lds2`
so a **deep (A+B) software-pipelined K-loop** fits ≤ 256 VGPR without spilling — the named unlock for the prefill
~4–5 % gap. Read `docs/prefill-register-lifetime-pool-representation-scope-20260623.md` + the kloop-template +
schedule-diff results + `extra/gemm/rdna3_wmma_matmul.py`. Phases: authority lock (reproduce DBUF PASS + PLRAB wall);
liveness model (max-live, accumulator floor — stop with `REGISTER_POOL_INSUFFICIENT_HW_LIMIT` if ideal packing >256);
pool allocator (additive `REGPOOL` mode, default off); deep re-emit + gates (register-lifetime ≤256/0-spill →
correctness → PIPELINED); classify; **only if it fits**, a clock-pinned synced whole-prefill diagnostic via the
additive `PREFILL_GEMM_CFG_*` override. Boundaries: no tinygrad source, no model route, no default flip, no
whole-prefill speed claim until gates pass, no renderer rewrite (stop + classify `REQUIRES_RENDERER_WORK`), no broad
search, no vendored Tensile, no training. Final response: verdict labels; whether the deep pipeline fits ≤256;
correctness; interleave; max-live VGPR; whether prefill schedule search reopens; artifacts; files changed; git status.
