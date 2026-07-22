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
- P2.1: launched.
