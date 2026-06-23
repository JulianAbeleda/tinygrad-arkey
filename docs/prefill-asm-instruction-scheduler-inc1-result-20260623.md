# Prefill ASM Instruction Scheduler — Inc 1 Result (2026-06-23)

## Verdict: `ASM_SCHED_WAITCOUNT_MODEL_DELIVERED` + `WAIT_CORRECTNESS_NECESSARY_NOT_SUFFICIENT`
Inc 1 builds the **wait-counter (`s_waitcnt`) model** — the async-load counter semantics named as the next capability
in the scope (`prefill-asm-instruction-scheduler-scope-20260623.md`). It is delivered and proven as an audit, an
in-place minimal-count recompute, and a soundness verifier. Two honest findings shape what comes next:
1. **Standalone consumer-only relaxation is ~free** on the hand-tuned `build_gemm_lds2`: the existing full drains are
   already minimal (total relaxable slack = 1, a single perf-irrelevant prologue scalar-load count).
2. **Wait-correctness is necessary but NOT sufficient** to license memory/compute cross-motion: a reorder can be
   register-legal *and* wait-correct yet still compute wrong on hardware. The remaining gap is an RDNA3
   hardware-spacing/scoreboard hazard — Inc 2's hazard recognizer — not the wait model.

## The model (`extra/qk_asm_scheduler.py`)
AMD RDNA3 tracks outstanding async memory ops in per-domain counters: `vmcnt` (VMEM: global/buffer/scratch) and
`lgkmcnt` (LDS + SMEM). A load's destination register is valid only after an `s_waitcnt` drains its counter; same-domain
ops retire in issue order, so to wait for the op at issue-position `s` (0 = oldest) you need `cnt ≤ issued_total − 1 − s`.
- `decode_wait`/`encode_wait` — simm16 ↔ (vmcnt, lgkmcnt), mirroring the in-repo encoder, low nibble preserved.
- `verify_wait_correct(insts) → (ok, reason)` — **soundness gate** for any (reordered) stream: simulate the counters
  with the actual `s_waitcnt`s; fail if any instruction reads/writes a register whose producing load is undrained, if
  LDS is undrained at a barrier, or if memory is undrained at `s_endpgm`.
- `wait_constraints(insts)` — per-`(wait, domain)` `(have, required)` via the issue-score model; `required < have` is
  relaxable slack. Folds in barrier (`lgkm=0`) and endpgm (both `=0`) memory-ordering constraints.
- `recompute_waits_inplace(insts)` — minimal correct counts, byte-layout preserving (only simm16 changes; instruction
  size unchanged → branch offsets stay valid). Conservative: never relaxes a drain that commits stores.
- `build_regions(..., fence_only=True)` / `schedule(..., fence_only=True)` — the Inc 2 substrate where memory ops
  participate in regions and move. **Left OFF by default** (see the necessary-not-sufficient finding).

## Proof (`extra/qk_asm_scheduler_inc1_test.py`, gfx1100, PLRA config, M=N=K=512) — ALL PASS
| check | result |
|---|---|
| Q1 HAND_WAITS_ALREADY_MINIMAL | PASS — 8 (wait,domain) constraints, total relaxable slack = **1** (prologue scalar load) |
| Q2 RECOMPUTE_INPLACE_CORRECT | PASS — minimal-count rewrite (1 wait changed) runs correct, rmse 2.07e-4 |
| Q3 IDENTITY_IS_WAIT_CORRECT | PASS — the gate passes on the unmodified stream |
| Q4 GATE_DISCRIMINATES | PASS — the gate REJECTS a drains-removed stream (not trivially true) |
| Q5 WAIT_MODEL_COMPOSES_WITH_REORDER | PASS — recompute on the Inc-0 (memory-anchored) reorder still correct, rmse 2.07e-4 |
| Q6 WAIT_CORRECTNESS_NECESSARY_NOT_SUFFICIENT | PASS — a fence_only reorder (170 mem ops moved) is wait-correct (gate=True) but cross-motion stays OFF |

## The necessary-not-sufficient investigation (why cross-motion is deferred to Inc 2)
A `fence_only` reorder that lets memory ops move was register-legal and wait-correct, but its prologue (region 1)
computed wrong on hardware (rmse 1.28, not a fault). The diagnosis was exhaustive and rules out the obvious causes:
- **Register DAG is provably complete here**: 0 swapped hazard pairs lack a dependency edge; the produced order has 0
  dependency violations.
- **Per-piece reorders are all correct**: reversing the 128 independent accumulator-init `v_mov`s, moving `loadA0` to
  the region end, and adjacent swaps each stay byte-correct (rmse 2.07e-4). Regions 0,2–7 (including 120 moved memory
  ops in the epilogue) all reorder correctly.
- **Only the prologue full reorder breaks**, despite being register- and wait-legal.
- Conclusion: an **RDNA3 hardware-spacing / scoreboard hazard** (the class LLVM handles with a hazard recognizer +
  `s_nop`/hard-clause insertion) is exposed only by the prologue's tight cross-motion. Modeling it is exactly Inc 2.

So `fence_only` cross-motion is built but **kept off**; the proven-safe reorder remains Inc 0's memory-anchored
compute reorder, now composable with the wait model (Q5).

## Honest standing
- The wait-counter model is the genuine capability Inc 1 set out to build, and it is the correctness backbone Inc 2
  will use as ONE of its gates (alongside the hardware-hazard recognizer).
- It yields **no standalone speedup** on the hand-tuned kernel (waits already minimal) — the lever pays off only with a
  latency-aware reorder, which itself needs the hazard model. This is consistent with the scope's honest ROI (~2–3%,
  realized only by Inc 2/3).
- No `tinygrad/` source, no production path, no default flip, no whole-prefill speed claim.

## Next: Inc 2
A latency-aware list scheduler over `fence_only` regions, gated by BOTH `verify_wait_correct` AND a new RDNA3
hardware-hazard recognizer (VALU→VMEM/WMMA spacing, `s_nop`/hard-clause rules), validated on clock-pinned synced
whole-prefill. Inc 2 is where the wait model + reorder turn into the actual (bounded ~2–3%) prefill win.

## Files
New: `extra/qk_asm_scheduler_inc1_test.py`, this doc. Modified (additive): `extra/qk_asm_scheduler.py`
(wait-counter model + `fence_only` region mode, default off). +1 ledger.
