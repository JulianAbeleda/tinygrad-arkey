# Phase 2 Plan: WMMA + PCONTIG-fusion coexistence (the flash-prefill build)

**Goal:** make the TC opt WMMA-ify the matmul contractions **inside** a PCONTIG-fused attention kernel, so fused attention is *both* single-kernel (no score spill) *and* on tensor cores — converting the measured 2.45× bracket into a real shipped win.

**Ground truth (all verified, see `flash-prefill-fusion-probe-20260721.md` correction):**
- Rangeify fuses attention today via `PCONTIG` (correct, rel err 0.0). Not the blocker.
- Fused kernel emits **0 WMMA** (vs 2 unfused) → ~2.6× slower. **This is the blocker.**
- `postrange.py:_apply_tc_opt` (305–318) already picks one WMMA-compatible reduce around an epilogue reduce, but tags only one (342). Fused attention has 4 reduces and loses the WMMA-compatible MUL shape.

## Execution rules (hard)
- **Single GPU lane** → agents run **one at a time**, never parallel (parallel GPU = MMU faults, proven). I serialize them.
- **Agents gather, I decide.** Every agent task produces a **verifiable artifact** (a WMMA-call count, kernel count, `tm`, rel-err, or pasted DEBUG output) — never a conclusion. I verify each result in the main loop before the next step. (Two prior agents produced confident-wrong results here; this rule is why.)
- `.venv` python, `DEV=AMD`, `DEBUG=2` `tm` warm ≥200 dispatch, temp in `/home/ubuntu/.claude/jobs/6db6b205/tmp/`. Commit on master, no branches, Co-Authored-By trailer, push.
- Reference config: `T=KV=512` for fast iteration, `T=KV=2048` for the gate. Correctness ref = plain SDPA (`softmax(qkᵀ·scale+causal)@v`), fp16 tol ~1e-2.

## Steps

**P2.1 — Diagnose why the fused kernel emits 0 WMMA.** [agent, low] Instrument `_apply_tc_opt`: on the PCONTIG-fused attention kernel, does it get called? what reduceops does it see? does the compatible-MUL loop (309–317) match? Artifact: pasted logs for PCONTIG=0 vs 8. → I decide the fix shape.

**P2.2 — Get ONE contraction (QKᵀ) WMMA'd inside a fused kernel.** [main-loop-driven edit + agent measurement] Smallest coexistence proof. Success artifact: WMMA-call count ≥1 in a fused (kernel-count-reduced) attention kernel, correctness held, `tm`. If this can't be done, that's the real obstacle → bank + stop.

**P2.3 — Get BOTH contractions (QKᵀ + PV) WMMA'd in the fused kernel.** Extend TC tagging to multiple dot-reduces (the 342 single-tag limit). Artifact: 2 WMMA calls in the fused kernel, correct.

**P2.4 — GATE: measure fused+WMMA attention vs SDPA** at `T=KV=2048`. Artifact: two-ceiling table + absolute `tm` + rel-err. GO if faster than SDPA with correctness held.

**P2.5 — Ship (if GO):** wire via `ScheduleHints.pcontig` on the attention path (`model.py:583–598`), integration-test 14B prefill, geometry-tune via BubbleBeam. If NO-GO: bank the precise obstacle.

## Status log
- P2.1: **DONE.** Instrumented `_apply_tc_opt`. Unfused (PCONTIG=0): QKᵀ matched=True (WMMA); PV matched=False (it's fp32 — `.softmax`+`v.float()`, not fp16); softmax reduces correctly no. Fused (PCONTIG=8): `_apply_tc_opt` **never called**.
- P2.2: **BLOCKED — architectural incompatibility (verified, not a knob).** Root cause: PCONTIG fusion converts REDUCE axes → LOOP axes, so the fused kernel has `n_reduce_axes=0`. Even with TC_OPT=2 (gate passes), `_apply_tc_opt` bails at line 299 (`"no reduce ops for TensorCore"`) because WMMA can only attach to a REDUCE op and fusion removed them. Measured: PCONTIG=8 × TC_OPT∈{0,1,2} → **0 WMMA, ~4928µs** (2.6× slower than unfused 131µs). **WMMA and PCONTIG-fusion are structurally mutually exclusive; no knob bridges it.**

## ⛔ CONCLUSION (2026-07-21) — the cheap scheduler-native path is closed; fused+WMMA needs real new machinery

The three theories all fell:
1. deepseek's "rangeify can't fuse / needs a tuple-accumulator REDUCE" — **false** (PCONTIG fuses today).
2. "fusion just isn't attempting WMMA; flip TC_OPT" — **false** (gate passes, TC opt still bails).
3. The real wall: **WMMA needs a REDUCE op; PCONTIG's fusion turns the matmul contractions into a sequential LOOP with no REDUCE, so there is nothing for the TC opt to grab.**

PCONTIG produces the *wrong structure* for flash: a monolithic reduce-free loop. The **flash** structure is different — an outer KV-block LOOP with the QKᵀ (over Hd) and PV (over block-KV) contractions **preserved as REDUCE ops** (WMMA-able) inside each block, score kept resident. `_apply_tc_opt`'s "epilogue reduction around the dot-product" comment (postrange.py:305-307) shows the TC opt *could* handle an outer-loop + inner-dot shape — but **no existing pass produces that shape for attention.** Making the scheduler emit the flash block-loop-with-preserved-matmul-REDUCEs structure is the genuine multi-week compiler build (either a new rangeify fusion that preserves the contraction REDUCEs, or WMMA-on-loop-accumulation). The 2.45× bracket is the real physics ceiling; reaching it requires that build. **No cheap knob path exists — proven, not assumed.**

## ✅ FEASIBILITY (2026-07-21) — the primitive route (a) is possible and localized

Corrected/extended finding. Two distinct fusions must not be conflated:
- **PCONTIG fusion** destroys the contraction REDUCEs (→ LOOP, `n_reduce=0`) → TC opt bails. Dead end. (This is what P2.2 measured.)
- **REDUCE-preserving fusion** (matmul-REDUCE + epilogue-REDUCE in one kernel, e.g. `(a@b).max(-1)`): instrumented at TC_OPT=2 → `_apply_tc_opt` IS called, finds the dot (`cand=ADD mulop=MUL`), **matched=True**, and `TC()` selects+tags the WMMA axes. **The WMMA selection machinery already works on a fused multi-reduce kernel.** The only gap: the tagged-TC reduce doesn't *emit* WMMA to final code when an epilogue reduce co-resides (applied-but-not-lowered). That is a bounded lowering fix, not new architecture.

**Primitive/centralized route = (a):** one existing WMMA path (TC opt on REDUCE), reused unchanged. The flash build is then two localized pieces on centralized machinery: (i) a rangeify fusion that keeps QKᵀ/PV as REDUCEs with softmax as an epilogue (NOT PCONTIG, which destroys them); (ii) the lowering fix so a tagged-TC reduce survives an epilogue reduce. Option (b) (WMMA-on-loop-accumulation) is rejected: it adds a SECOND WMMA mechanism → decentralized. The scaffolding for (a) already exists (postrange.py:305-317 epilogue-reduce selection, verified firing); reaching the 2.45x ceiling is engineering on the existing path, not invention.
