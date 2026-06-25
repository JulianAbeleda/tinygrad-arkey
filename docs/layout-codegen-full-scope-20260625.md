# Full scope: the layout/mapping IR + the renderer-codegen lift (2026-06-25)

Closing the "layout wall" (`docs/layout-mapping-ir-design-20260625.md`) has **two halves**: a searchable
**layout/mapping representation** (so the owned-kernel physical structure is *expressible*) and the
**renderer-codegen lift** behind it (so it's *fast*). This is the full, sequenced, code-grounded scope for both,
with hard gates, kill criteria, and an honest ceiling. Synthesized from 4 parallel capability scopes
(`bench/qk-...`, the workflow run) over the tinygrad-arkey HEAD code + the prior arc docs.

## One wall, two sides
- **SIDE 1 — decode Q4_K GEMV (the live gap):** scheduler packed arm 22 tok/s vs owned ~100. This is where the
  payoff is. Everything below targets it.
- **SIDE 2 — prefill WMMA GEMM (settled):** dependency-free graph-GEMM is already ~96% of Tensile / at-or-above
  llama; the residual is vendored-Tensile-only (+ a beta=work confound) and the deterministic asm-scheduler is
  **measured unable** to close it in-model. Deprioritized — reference only.

## The decisive, repeatedly-confirmed framing
- **Coalescing is necessary but NOT sufficient.** Forced `GROUP(32)` gave no speedup (49.9≈50.9); three scheduler
  GEMV restructures got 50/22/14 vs owned 103. The owned win is the whole *kernel structure*, not a coalesced load.
- **Representation ≠ speed.** The layout IR makes the structure searchable; the codegen lift makes it fast.
- **HEAD-state caveats (verified):** AMD does **not** go through the `ISARenderer`/regalloc path (x86-only,
  `do_linearize:173`) — it hands C-style to LLVM/HIP which owns instruction scheduling; `fdot2` and `OptOps.COALESCE`
  are absent; `tinygrad/renderer/amd/schedule.py` exists but is dead (imported by nothing); the
  `extra/qk_asm_scheduler_inc*_test.py` files are **not on HEAD**.

## Critical correction (vs the naive scope)
The "renderer-codegen lift" is smaller than the prompt framed — two of the three named lifts are non-problems:
- **Load-dedup ("load each word once, dequant 8 nibbles in-register") = ALREADY DONE.** UOps are interned; the single
  `words[...]` INDEX is one node referenced by all 8 nibble extracts → one `LOAD` via `pm_add_loads`. The owned
  "one word/lane, in-register dequant" is the graph's natural CSE form. (So the 22→100 gap is **100% thread-map**,
  not redundant loads — `coalesced-dequant M-A` proves this cheaply.)
- **Prefill asm-scheduler = measured-complete.** Waits already minimal (slack 1); pure reorder perf-neutral; the one
  non-neutral lever (waitcnt-reloc) doesn't transfer. No whole-prefill win exists to capture.

So the real keystone is **ONE** thing: a **LaneMap-aware `add_gpudims`** that can express the owned kernel's
composite per-row-workgroup thread-map (`lane = block_group*8 + word_col`, splitting a REDUCE range across one wave)
— which the scheduler **cannot produce at all** today (`gpudims.py:103` skips REDUCE axes).

## Dependency graph
```
ALREADY BUILT: RANGE/INDEX algebra (apply_movement_op, AxisType, get_idx/split_uop) ·
  M0 coalescing predicate (qk_layout_coalesce_check.py) · M5 cross-lane / WARP_REDUCE_LOWERING ·
  MV_DEQUANT recognition · _sdot4 · float4 uint32 peephole
CRITICAL PATH (decode GEMV):
  layout-ir M1 (LayoutFn + CuTe compose)   ── reuses M0 stride recovery
    → layout-ir M2 (TensorCore.swizzle → first-class LaneMap)
      → coalesced-dequant M-B  [KEYSTONE: LaneMap-aware add_gpudims; REDUCE-split across the wave]
        → M-C (float4 peephole fires free + reuse WARP_REDUCE for cross-lane reduce/store; NO new renderer work)
          → M-E  [W==D close-the-gap = THE PROJECT DECISION GATE]
            → (only if M-E wins) layout-ir M3 (OptOps.COALESCE + static cost + anchored propagation) + M-D (beam-searchable LaneMap)
INDEPENDENT LEVER (does not block the critical path):
  vdot2 M1-M3  ── reuses M5 mechanics; feeds the owned ATTENTION tile (a different kernel); converges with the IR at full tile parity
DEPRIORITIZED (do not fund unless a gate forces it):
  prefill asm-scheduler (measured-complete) · AMD ISARenderer decode-backend (4-8wk; gated behind locally-infeasible ATT attribution; DNR3/4 found every lever sub-material)
```
Keystone insight: **M-B is the convergence node** — upstream of it is representation plumbing; downstream of M-E is
generalization that should only be funded after a W==D win.

## Sequenced roadmap
| Phase | Items | Gate | Effort |
|---|---|---|---|
| **P0 — cheap expose + de-risk** | (1) `vdot2` M1+M2: `pm_fdot2` rule (mirror `pm_warp_reduce`), gated `V_DOT2_LOWERING`, post-devectorize insertion. (2) `coalesced-dequant M-A`: instrument packed arm, prove 1 load/word → gap is 100% thread-map. (3) `schedule-asm M0`: restore the asm tests, confirm Inc0-3, **write the decision** parking prefill-asm + AMD-ISARenderer. | vdot2 structural test + AMD byte-correct + objdump shows `v_dot2`; M-A memo; asm decision written. | ~1.5–2 wk |
| **P1 — build ONLY the plumbing the keystone needs** | `layout-ir M1` (LayoutFn + CuTe **composition** graph_rewrite, with divisibility invariants to reject masked-PAD / mixed-radix). `layout-ir M2` (generalize `TensorCore.swizzle` → first-class validated `LaneMap`; re-express WMMA through it **byte-identically**). Defer `OptOps.COALESCE`/`LAYOUT_TRANSFORM`. | M1 property tests (coeff matches strides; compose byte-equal to manual substitution; masked cases raise). M2: all WMMA shapes byte-identical through LaneMap. | ~7–10 wk |
| **P2 — KEYSTONE codegen lift + the make-or-break gate** | `M-B`: LaneMap-aware `add_gpudims`/`get_grouped_dims` — map a RANGE to `lane%8` (word-col) and split a REDUCE range `bg=lane//8` across one wave; re-express the owned q4k `lane=bg*8+lane4` as a **scheduler** kernel. `M-C`: confirm float4 uint32 vec-load fires for free + wire `WARP_REDUCE_LOWERING` for the cross-lane reduce + single store. `M-E`: in-model clock-pinned W==D, FFN gate/up, ctx 512–4096. | M-B: DEBUG shows 8 adjacent lanes' word offsets consecutive (stride-1). M-C: coalesced uint32 vec-load + warp reduce + single store, tokens match owned. **M-E (DECISION): ≥~90% of owned, or KILL.** | ~7–10 wk |
| **P3 — generalize to searchable (ONLY if M-E wins)** | `layout-ir M3`: `OptOps.COALESCE` + static cost (is_unit_stride on composed layout) pruned in beam **before** timing + Hexcute anchored propagation. `Ops.LAYOUT_TRANSFORM`. `M-D`: beam discovers the LaneMap (no hand flag). optional `vdot2 M3` (attention-tile q.k). | beam reproduces the hand-found coalesced schedule without timing; predicate-vs-measured agree; propagation forces >90% layouts. | ~8–12 wk |
| **P4 — DEPRIORITIZED** | prefill asm-scheduler M1/M2/M5; AMD `ISARenderer` decode-backend + ATT attribution. | Each must beat the ~0.7% whole-prefill noise floor (prior art says they won't); ISARenderer only if P2 forces CUSTOM **and** portability is valued over speed. | prefill ~2–4 wk / decode ~6–12 wk (recommend NOT funding) |

## First 3 moves
1. **`vdot2` M1+M2 (~1 wk):** `extra/qk_fdot2_lowering.py` `pm_fdot2` — detect the post-devectorize half2 fused-MAC idiom and emit one `CUSTOMI` `__builtin_amdgcn_fdot2`, gated `V_DOT2_LOWERING`, inserted alongside the `WARP_REDUCE_LOWERING` block. Cheapest real lever; re-proves the additive CUSTOMI pattern on a different kernel.
2. **`coalesced-dequant M-A` (~0.5 wk):** instrument the packed arm (`Q4K_GEMV_SCHEDULER=2`, DEBUG=4), prove 1 load/word (CSE'd) → write the memo concluding the 22 tok/s loss is 100% strided thread-map (kills load-dedup as scope).
3. **`schedule-asm M0` (~0.5 wk):** restore the asm-scheduler tests, confirm Inc0-3 on gfx1100, write the one-page decision parking the prefill asm-scheduler + the AMD ISARenderer decode-backend. Frees budget for the critical path.

## Honest end-state (three tiers)
- **Tier 1 (achievable — the actual prize):** M-B/M-C make the owned kernel's *structure* (per-row-workgroup
  composite thread-map + coalesced uint32 word load + in-register 8-nibble dequant + cross-lane reduce + single
  store) **expressible and emittable from the scheduler for the first time** — genuinely impossible today. Should move
  the GEMV from 22 to **well past** the 49.9 `MV_DEQUANT` ceiling (49.9 is the output-parallel map *without* the K-split).
- **Tier 2 (plausible — the honest target): ~80–95% of owned** (Tensor-class, searchable), with the last 5–20% lost
  to instruction scheduling the LLVM/HIP lowerer owns — the **same SIA1/PLR/clause wall** the prefill LDS-GEMM arc hit
  (reached edge-of-Tensile-class, not parity).
- **Tier 3 (NOT achievable by pure-search):** byte-for-byte owned/Marlin parity needs the final instruction-schedule
  tuning below the IR, and **on AMD there is no in-tree path** (the only ISARenderer is x86; AMD → LLVM/HIP).

Most likely outcome: **Tier 2** — a large, real, searchable win that doesn't quite touch owned-peak, leaving CUSTOM
as the documented ceiling for the final mile. The truth-teller is **M-E**.

## Kill criteria (the gates that matter)
- **M-E IS THE PROJECT KILL GATE:** if the structurally-isomorphic scheduler kernel plateaus near the 49.9–50
  `MV_DEQUANT` ceiling instead of ≥90 tok/s → **declare CUSTOM fundamentally needed for the GEMV, STOP, do NOT fund
  P3**, re-frame the IR deliverable as searchability/portability only.
- **M-B re-scope trigger:** if splitting a REDUCE range across hardware lanes in `add_gpudims` needs a structural
  rewrite larger than the rest of the roadmap → the owned thread-map may be fundamentally outside the RANGE/AxisType
  model; re-scope before continuing.
- **M2 non-negotiable:** any existing WMMA shape not byte-identical through the LaneMap → stop (it breaks the one
  working thread→fragment path).
- **M3:** kill if the static COALESCE predicate disagrees with measured ordering (it would prune the fast candidate).
- **P4 prefill:** kill any lever that doesn't beat the ~0.7% whole-prefill noise floor (prior art predicts all will).

## Total effort
- **Critical path to the M-E decision (P0+P1+P2): ~16–22 weeks (~4–5.5 months)** of focused single-track work —
  this is the real commitment; everything past it is conditional on M-E.
- If M-E wins, P3 adds ~8–12 wk (total ~24–34 wk). P4 (not recommended): prefill ~2–4 wk, decode ISARenderer ~6–12 wk.
- **Largest estimate risk:** M-B (splitting a REDUCE range across lanes in `add_gpudims` — the code doesn't do this
  today); if it needs a `get_grouped_dims` rewrite, P2 balloons.

---

## Appendix — per-capability scope (condensed)

**A. Layout/mapping IR** (additive on RANGE/INDEX; ShapeTracker/View already deleted). M0 ✓ (coalescing predicate +
reshuffle/manifest). M1 `LayoutFn` (coeff/is_unit_stride/compose via `split_uop`, reuse `heuristic.py:142-145`) + the
missing CuTe **composition** `R(c)=A(B(c))` as a graph_rewrite (`UOp.substitute` + `symbolic`) with admissibility
invariants. M2 `LaneMap` (generalize `tc.py:15` swizzle → first-class `f:(thread,value)→coord`, WMMA byte-identical).
M3 `OptOps.COALESCE` + static cost in beam + anchored propagation; `Ops.LAYOUT_TRANSFORM`. Hook: a layout pass in
`full_rewrite_to_sink` between simplify-ranges and `apply_opts` (where the M5 hook sits). Ceiling: makes structure
*searchable*, not fast.

**B. `v_dot2` renderer lowering** (independent instruction-expose). `pm_fdot2` in `extra/qk_fdot2_lowering.py`
mirroring `pm_warp_reduce`: detect `ADD(MUL(gep(a,0),gep(b,0)), MUL(gep(a,1),gep(b,1)), acc)` over half2 → one
`CUSTOMI` `__builtin_amdgcn_fdot2`. Gated `V_DOT2_LOWERING`, post-devectorize. Feeds the owned attention tile q.k.
Ceiling: instruction only; full tile parity also needs the IR's schedule-expressibility (they converge at the tile).

**C. Coalesced-dequant codegen (the keystone).** Load-dedup already done (CSE). The lift = **M-B** LaneMap-aware
`add_gpudims` expressing `lane=bg*8+lane4` + REDUCE-split across one wave (`gpudims.py:103` skips REDUCE today). M-C
reuses the float4 peephole + `WARP_REDUCE_LOWERING` (no new renderer work). Depends on M1+M2. Ceiling: plausibly
≥90% of owned; the last mile is instruction scheduling AMD hands to LLVM.

**D. Instruction-schedule / asm gap.** PREFILL: a full deterministic scheduler over the hand-asm GEMM is *already
built* (`extra/qk_asm_scheduler.py`, Inc0-3) and **measured unable** to win whole-prefill — park. DECODE: a
deterministic schedule needs AMD to go through the `ISARenderer` (x86-only today) = a 4-8wk compiler-backend project,
gated behind locally-infeasible ATT/SQTT attribution; DNR3/4 found every lever sub-material. Recommend NOT funding
unless M-E proves CUSTOM is needed and portability is explicitly valued over speed.
