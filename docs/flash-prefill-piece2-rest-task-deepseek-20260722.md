# TASK (deepseek): finish flash-prefill fusion — the rest of Piece 2, milestone-gated

Read this ENTIRE file first. Context: Piece 1 (cb6e760e0) made WMMA survive a fused matmul+epilogue-reduce. Piece 2 so far (74fda0fe6) added a REDUCE-preserving fusion that reduced attention kernel count — but on review it FALLS SHORT of the goal: only QKᵀ is on WMMA (PV is not), and the T×KV score still spills. This task finishes the job in **4 milestones. Run them in sequence, end-to-end — no waiting for Claude between milestones.** Claude reviews ONCE at the very end from the evidence you commit. Because there is no live gating, you MUST self-document every milestone's artifacts (commit them) so the end review can verify each step. STOP early only if you hit a genuine blocker (see §2.5 Fallback) — never fake, force, or paper over.

Repo: `/home/ubuntu/tinygrad-arkey` · Python: `/home/ubuntu/tinygrad-arkey/.venv/bin/python` · Env: `DEV=AMD`

---

## §0 HARD RULES — you have repeatedly broken these; they are the task now

1. ❌ **No hand kernels.** No `custom_kernel`, no UOp kernel bodies, no `__builtin_amdgcn`/barriers/LDS by hand, no `flash_kernels` imports, no hardcoded tile geometry. Litmus: writing `UOp(...)` for a kernel body = wrong task. This is scheduler/codegen work only.
2. ❌ **No `.realize()`/`synchronize()` inside loops** in probes. One graph, realize once.
3. ⛔ **PROVE CLAIMS PER-KERNEL — no aggregate grep counts.** Your last report claimed "WMMA on both matmuls" from `grep -c` = 2. That was WRONG: 2 = 1 `#define` + 1 call = ONE WMMA kernel (QKᵀ only). **For any WMMA claim, dump the actual `__WMMA(...)` CALL lines and show which compute kernel each lives in.** "Both matmuls on WMMA" requires showing TWO distinct `__WMMA` call sites in the relevant kernels. If you cannot show it, you do not claim it.
4. ⛔ **Report what does NOT work.** Remaining spills, matmuls still off WMMA, partial results — state them explicitly. A partial result reported as complete is a FAILURE (this already happened once).
5. ❌ **RUN THE FULL RELEVANT TEST SUITE before every commit** and paste the pass/fail/xfail counts. Baseline: WMMA/packed suite = `54 passed, 10 skipped, 5 xfailed`; `test/test_tiny.py` + scheduler tests = `38 passed, 1 skipped`. Any core `tinygrad/` change must keep these. (Pre-existing failures to ignore: 3 `test_wmma_emitted_code_fixtures_are_unchanged` subtests fail on parent too.)
6. ✅ **Run all milestones end-to-end.** No stopping between them for Claude. Commit each milestone's artifacts as you go so the final review can verify them. STOP early ONLY on a genuine blocker (§2.5) — and then report it precisely, do not force or fake.

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
- Commit, push (with the artifacts in the message/doc). **Continue to M2.**

### M2 — Score residency (the hard multi-step part; diagnose THEN implement)
The score T×KV still spills. Keeping it resident means softmax's max→sum runs inside the QKᵀ kernel without materializing the score, which currently forces REDUCE→LOOP (killing WMMA). This is the crux — the hard work this task exists for.
- **First diagnose:** how the score buffer gets inserted (the bufferize between QKᵀ and the softmax max-reduce), and whether an online/single-pass softmax formulation (running max+sum) lets rangeify keep it resident. Measure (kernels, WMMA per kernel, score-buffer present).
- **Then implement** the scheduler/rangeify change that keeps the score resident **while preserving the QKᵀ and PV REDUCEs (WMMA intact).** This is a scheduler change, expressed in the rangeify/codegen graph rewrites — NOT a hand kernel, NOT a Python block loop.
- **Non-negotiable invariant:** after M2, the per-kernel WMMA dump must STILL show both matmuls on WMMA (M1's gain must not regress) AND the T×KV score buffer must be gone. If the only way to remove the score spill is REDUCE→LOOP that kills WMMA, that is a **dead end** — do NOT ship it; invoke the Fallback (§2.5) and report the precise blocker instead.
- **Artifacts:** per-kernel WMMA dump (both matmuls still WMMA), proof the T×KV score buffer is gone (kernel list / buffer sizes), `max_rel_err`, full test suite counts. Commit. **Continue to M3.**

### M3 — End-to-end gate
Measure the full fused+WMMA attention (both matmuls WMMA, score resident) vs materialized SDPA at `T=KV=2048`.
- **Artifacts:** two-ceiling table (compute_frac, mem_frac vs the ceilings — measure ceilings empirically, don't hardcode), absolute `tm` (DEBUG=2, warm ≥200), per-kernel WMMA dump, `max_rel_err`. If faster than SDPA with correctness held → **continue to M4.** If NOT faster (or score residency was a dead end at M2) → invoke Fallback (§2.5): deliver M1's real gain, report honestly that the full win wasn't reached, and stop. Do NOT wire a non-win into the model.

### M4 — Wire into 14B (only if M3 showed a real speedup)
Enable the fused path on the model attention (`model.py:583–598`) via `ScheduleHints.pcontig` (or the appropriate scheduler mechanism), integration-test 14B prefill (correctness + tok/s vs the shipped path). Commit.

## §2.5 Fallback / stop condition (use it — do NOT force or fake)
Stop early and write the review package (§2.6) if ANY of these hold. Stopping honestly is SUCCESS; forcing a fake result is failure:
- Score residency (M2) is only achievable by killing WMMA (REDUCE→LOOP) or a hand kernel → report the precise rangeify blocker and stop. Deliver M1's real gain.
- A milestone would require a hand kernel to proceed → stop (hand kernels are banned, no exceptions).
- The end-to-end result (M3) is not faster than SDPA → report the honest numbers, do not wire it in.
- A change regresses the test suite and you can't fix it cleanly → revert that change, report it.

## §2.6 Review package (write this at the END for Claude's single review)
Write `docs/flash-prefill-finish-report-<date>.md` with, PER MILESTONE reached: the commit hash, the per-kernel WMMA dump (actual `__WMMA` call sites), `max_rel_err`, kernel count, score-buffer-present yes/no, and the full test-suite counts. Plus a top section: **what works, what does NOT, and where you stopped and why.** No aggregate grep counts anywhere. This doc is what Claude verifies against — if a claim isn't backed by a committed artifact in here, it will be treated as unproven.

## §3 Guardrails
- Single GPU lane; `pkill` strays + `rocm-smi` before runs; never background a bench and report "waiting." Run, wait, read.
- `.venv` python; temp in `/home/ubuntu/.claude/jobs/6db6b205/tmp/`; isolate one compute per process for WMMA dumps (a reference at default settings emits its own WMMA and contaminates).
- Commit on master, no branches, `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`, push. Revert any instrumentation before committing; confirm `git status` clean.
- No BEAM. Watch `sz.py` (core budget) for `tinygrad/` edits.

## §4 One-line job
**Finish flash-prefill fusion end-to-end, no stops — M1: honest gate + PV onto WMMA (prove TWO WMMA call sites); M2: keep the score resident while both matmuls stay WMMA (the hard scheduler work — dead-end if it needs REDUCE→LOOP or a hand kernel); M3: end-to-end gate vs SDPA at 2048; M4: wire into 14B if it's a real win. Prove every WMMA claim per-kernel, report what does NOT work, run the suite each commit, write the §2.6 review package at the end. No hand kernels; stop honestly on a genuine blocker rather than force it. Claude reviews once at the end from your committed artifacts.**
