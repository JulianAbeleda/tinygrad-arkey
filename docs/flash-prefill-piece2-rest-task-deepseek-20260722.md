# TASK (deepseek): finish flash-prefill fusion — the rest of Piece 2, milestone-gated

Read this ENTIRE file first. Context: Piece 1 (cb6e760e0) made WMMA survive a fused matmul+epilogue-reduce. Piece 2 so far (74fda0fe6) added a REDUCE-preserving fusion that reduced attention kernel count — but on review it FALLS SHORT of the goal: only QKᵀ is on WMMA (PV is not), and the T×KV score still spills. This task finishes the job in **4 milestones, each ending at a HARD STOP for Claude review.** Do exactly one milestone, then stop.

Repo: `/home/ubuntu/tinygrad-arkey` · Python: `/home/ubuntu/tinygrad-arkey/.venv/bin/python` · Env: `DEV=AMD`

---

## §0 HARD RULES — you have repeatedly broken these; they are the task now

1. ❌ **No hand kernels.** No `custom_kernel`, no UOp kernel bodies, no `__builtin_amdgcn`/barriers/LDS by hand, no `flash_kernels` imports, no hardcoded tile geometry. Litmus: writing `UOp(...)` for a kernel body = wrong task. This is scheduler/codegen work only.
2. ❌ **No `.realize()`/`synchronize()` inside loops** in probes. One graph, realize once.
3. ⛔ **PROVE CLAIMS PER-KERNEL — no aggregate grep counts.** Your last report claimed "WMMA on both matmuls" from `grep -c` = 2. That was WRONG: 2 = 1 `#define` + 1 call = ONE WMMA kernel (QKᵀ only). **For any WMMA claim, dump the actual `__WMMA(...)` CALL lines and show which compute kernel each lives in.** "Both matmuls on WMMA" requires showing TWO distinct `__WMMA` call sites in the relevant kernels. If you cannot show it, you do not claim it.
4. ⛔ **Report what does NOT work.** Remaining spills, matmuls still off WMMA, partial results — state them explicitly. A partial result reported as complete is a FAILURE (this already happened once).
5. ❌ **RUN THE FULL RELEVANT TEST SUITE before every commit** and paste the pass/fail/xfail counts. Baseline: WMMA/packed suite = `54 passed, 10 skipped, 5 xfailed`; `test/test_tiny.py` + scheduler tests = `38 passed, 1 skipped`. Any core `tinygrad/` change must keep these. (Pre-existing failures to ignore: 3 `test_wmma_emitted_code_fixtures_are_unchanged` subtests fail on parent too.)
6. ❌ **One milestone, then HARD STOP.** Do not chain milestones. Do not exceed the scope of the current milestone (you blew past a diagnostic-only stop last time).

## §1 Established state (do NOT re-derive)
- Attention `(q@kᵀ).softmax(-1)@v` at `T=KV=512, TC_OPT=2`, current code (74fda0fe6): ~13 kernels, **QKᵀ on WMMA, PV NOT on WMMA (it's fp32), score T×KV spilled to HBM between QKᵀ and softmax.** Correct (rel_err 0).
- Goal (the real win): both matmuls on WMMA AND score resident (not spilled) → the measured ceiling is ~2.45× vs SDPA at scale.
- Why PV isn't WMMA: `p = s.softmax(-1)` is fp32 and `v.float()` is fp32 → fp32×fp32 matmul → not WMMA-eligible. WMMA needs fp16 inputs.
- The score-spill: rangeify won't keep the T×KV score in-kernel across softmax's serial max→sum without converting the contraction REDUCE→LOOP (which kills WMMA). This is the hard part.

## §2 Milestones (one at a time, HARD STOP after each)

### M1 — Cleanup + PV onto WMMA (small, do this first)
Two changes:
(a) **Fix the misleading gate** in `rangeify.py:remove_bufferize`: the new fusion is gated by `if PCONTIG >= 0:` which is **always true** (a global change disguised as conditional). Make it honest — either unconditional (if that's intended) or a real, named condition. Don't change behavior, just make the code say what it does.
(b) **Get PV onto WMMA:** cast probs to fp16 before the PV matmul so it's WMMA-eligible. Do this at the scheduler/model level (NOT a hand kernel). The reference attention should compute PV in fp16.
- **Artifacts (all):** (1) per-kernel WMMA dump showing **TWO** `__WMMA` call sites — one in the QKᵀ kernel, one in the PV kernel (or in the fused softmax+PV kernel); (2) `max_rel_err ≤ 1e-2` vs fp32 reference; (3) kernel count; (4) full test suite counts (§0.5).
- Commit, push. **HARD STOP → Claude review.**

### M2 — Score residency (the hard part; DIAGNOSTIC-FIRST, do NOT implement yet)
The score T×KV still spills. Keeping it resident means softmax's max→sum runs inside the QKᵀ kernel without materializing the score, which currently forces REDUCE→LOOP (killing WMMA). This is the crux.
- **M2 is DIAGNOSTIC ONLY:** determine precisely what rangeify would need to keep the score resident while preserving the QKᵀ REDUCE (WMMA). Look at how the score buffer gets inserted (the bufferize between QKᵀ and the softmax max-reduce), and whether an online/single-pass softmax formulation (running max+sum) changes what rangeify does — measured as (kernels, WMMA per kernel, score-buffer present), NOT implemented as a hand loop.
- **Artifact:** a precise statement (with rangeify line refs + measured kernel/WMMA/buffer data) of exactly what prevents score residency, and whether it's a bounded rangeify change or needs REDUCE→LOOP (which would kill WMMA — a dead end).
- **NO implementation. NO hand kernels. HARD STOP → Claude decides the approach.**

### M3 — End-to-end gate (after Claude approves an M2 approach and it's implemented)
Measure the full fused+WMMA attention (both matmuls WMMA, score resident) vs materialized SDPA at `T=KV=2048`.
- **Artifacts:** two-ceiling table (compute_frac, mem_frac vs the ceilings — measure ceilings empirically, don't hardcode), absolute `tm` (DEBUG=2, warm ≥200), per-kernel WMMA dump, `max_rel_err`. GO if faster than SDPA with correctness held. **HARD STOP → Claude review.**

### M4 — Wire into 14B (only if M3 = GO)
Enable the fused path on the model attention (`model.py:583–598`) via `ScheduleHints.pcontig` (or the mechanism Claude specifies), integration-test 14B prefill (correctness + tok/s vs the shipped path). **HARD STOP → Claude review.**

## §3 Guardrails
- Single GPU lane; `pkill` strays + `rocm-smi` before runs; never background a bench and report "waiting." Run, wait, read.
- `.venv` python; temp in `/home/ubuntu/.claude/jobs/6db6b205/tmp/`; isolate one compute per process for WMMA dumps (a reference at default settings emits its own WMMA and contaminates).
- Commit on master, no branches, `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`, push. Revert any instrumentation before committing; confirm `git status` clean.
- No BEAM. Watch `sz.py` (core budget) for `tinygrad/` edits.

## §4 One-line job
**Finish flash-prefill fusion in 4 gated milestones — M1: honest gate + PV onto WMMA (prove TWO WMMA call sites); M2: diagnose score residency only (no impl); M3: end-to-end gate vs SDPA at 2048; M4: wire into 14B. One milestone then HARD STOP. Prove every WMMA claim per-kernel, report what doesn't work, run the suite. No hand kernels.**
