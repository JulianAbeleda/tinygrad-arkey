# TASK (deepseek): Route (a) — make WMMA survive REDUCE-preserving fusion

Read this ENTIRE file before touching anything. This is scheduler/codegen work with a minimal repro. You will do **Piece 1 only**, in two sub-steps, each ending at a HARD STOP for Claude review. Do not run ahead.

Repo: `/home/ubuntu/tinygrad-arkey` · Python: `/home/ubuntu/tinygrad-arkey/.venv/bin/python` · Env: `DEV=AMD`

---

## §0 HARD BANS — you have gone off-rails 3× before; these are why. Violating any = task failure.

1. ❌ **No hand kernels.** No `custom_kernel()`, no building a compute-kernel body with `UOp(...)`, no importing from `extra/qk/flash_kernels.py`, no `__builtin_amdgcn_*`/`UOp.barrier`/LDS by hand. **Litmus test:** if you are writing `UOp(...)` to build a kernel body, STOP — wrong task. This task edits the *scheduler/TC-opt*, not kernels.
2. ❌ **No `.realize()` / `synchronize()` inside a loop** in any probe. That is exactly what invalidated your last probe (it forced per-block materialization and manufactured every finding). Build ONE graph, realize ONCE.
3. ❌ **No conclusions from experiments — only verifiable artifacts.** Every claim must be a concrete measured value: a WMMA-call count (`grep -cE '__builtin_amdgcn_wmma|__WMMA'` on `DEBUG=4` output), a `max_rel_err` number, or an exact exception `type + file:line + traceback`. **Never** write "this needs N weeks" or "rangeify needs a new primitive" — you did that twice and were wrong both times. Report numbers; Claude concludes.
4. ❌ **PCONTIG is NOT the answer.** PCONTIG fusion converts the contraction REDUCE axes into LOOP axes → the kernel has 0 REDUCE ops → WMMA can't attach (verified). Do **not** use PCONTIG for this task. Route (a) is a *REDUCE-preserving* fusion — a different thing.
5. ❌ **No architectural conclusions.** Your job is a **localized lowering fix**. If you think it needs a rewrite of rangeify, you're wrong — stop and report the exact failing line to Claude.

## §1 What is already established (do NOT re-derive — build on it)

**Route (a)** = reuse the ONE existing WMMA path (the TC opt on a REDUCE op) by (Piece 1) fixing a localized lowering gap, then (Piece 2) adding a REDUCE-preserving fusion for attention. This task is **Piece 1**.

**Minimal repro of the gap:** `(a@b).max(-1)` (a matmul-REDUCE + an epilogue max-REDUCE in one kernel) at `TC_OPT=2`. Verified current state (Claude instrumented it):
- `_apply_tc_opt` (postrange.py:298) **IS called** on the fused 2-reduce kernel (`n_reduceops=2, args=[ADD, MAX]`).
- The candidate loop **finds the dot-reduce** (`cand=ADD mulop=MUL`) → **`matched=True`**.
- `TC(0)` prints (postrange.py:329) → axes selected, the reduce is **tagged TC** (postrange.py:342).
- The WMMA construction block runs (postrange.py:389–427).
- **BUT the final emitted kernel has 0 WMMA calls.** Applied-but-not-lowered.

**Leading hypothesis (CONFIRM, do not assume):** the WMMA construction at postrange.py:412–427 (`candidate_contract.assemble` / the pipeline build) **throws** when an epilogue reduce co-resides, and `hand_coded_optimizations` wraps the whole TC attempt in a `try/except` at heuristic.py:~70 that **silently swallows** the exception → falls back to non-TC → 0 WMMA. Your first job is to prove or disprove this by capturing the exact exception.

## §2 PIECE 1 — make `(a@b).max(-1)` emit WMMA. Two sub-steps, each a HARD STOP.

### Repro script (write to `/home/ubuntu/.claude/jobs/6db6b205/tmp/mm_max.py`):
```python
import os; os.environ['DEV']='AMD'
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
import numpy as np, sys
a=Tensor.randn(512,512,dtype=dtypes.half).realize(); b=Tensor.randn(512,512,dtype=dtypes.half).realize()
ref=(a@b).max(-1).numpy()                       # unfused-ish reference (default TC_OPT)
with Context(TC_OPT=2):
    out=(a@b).max(-1).numpy()
rel=np.abs(out-ref)/(np.abs(ref)+1e-3)
print(f"max_rel_err={rel.max():.5f}")
```
WMMA count: `DEV=AMD PCONTIG=0 TC_OPT=2 DEBUG=4 <py> mm_max.py 2>&1 | grep -cE '__builtin_amdgcn_wmma|__WMMA'` — but note this repro computes a reference too; to count ONLY the fused kernel, make a variant that does just the `with Context(TC_OPT=2)` compute and no reference (see how Claude isolated it: one computation per process). **Current baseline: 0 WMMA.**

### P1.a — FIND THE EXACT FAILURE (instrument only, change no behavior).
- Add a print that captures any exception swallowed by the TC attempt in `hand_coded_optimizations` (heuristic.py, the `try:` around `apply_opt(Opt(OptOps.TC, ...))`, ~line 70): in its `except`, print `type(e).__name__`, `str(e)`, and `traceback.format_exc()`.
- Also add a print at each `except`/`raise KernelOptError` inside the WMMA construction (postrange.py:412–415 and any other in 389–460) capturing what/where it fails.
- Run the repro at `TC_OPT=2`. **Artifact:** the exact exception type + the file:line it originates from + the one-line reason. If it does NOT throw (i.e. WMMA is constructed but removed later), instead find where the WMMA UOp disappears (grep the pre-render UOp graph for `Ops.WMMA` presence vs the final rendered source) and report that boundary precisely.
- **Revert all instrumentation** (`git checkout` the touched files), confirm `git status` clean.
- **HARD STOP.** Report the artifact to Claude. Do NOT attempt a fix yet — the fix depends on what the failure actually is.

### P1.b — MINIMAL FIX (only after Claude confirms the failure site).
- Make the smallest change so `(a@b).max(-1)` at `TC_OPT=2` emits WMMA while staying correct. The fix must be on the existing TC-opt/lowering path (do not add a parallel mechanism).
- **Success artifact (all three):** (1) WMMA-call count ≥1 in the fused `(a@b).max(-1)` kernel, (2) `max_rel_err` ≤ 1e-2 vs the reference, (3) `git diff` of the change (should be small — a localized fix). Also run the existing test suite for the touched file if one exists, and report pass/fail.
- Commit on master, push, Co-Authored-By trailer.
- **HARD STOP.** Report to Claude. Do NOT start Piece 2.

## §3 PIECE 2 — REDUCE-preserving attention fusion (DO NOT START; scoped after Piece 1 lands)

High-level only, so you know where this goes: once WMMA survives a matmul+epilogue-reduce kernel (Piece 1), Piece 2 makes rangeify emit the *attention* shape — QKᵀ and PV kept as REDUCE ops with the softmax as an in-kernel epilogue, score resident, **NOT** PCONTIG (which destroys the REDUCEs). Claude will write the detailed Piece 2 scope after reviewing Piece 1. **Do not touch rangeify fusion for attention until then.**

## §4 Correctness + measurement protocol (hard)
- WMMA presence = `grep -cE '__builtin_amdgcn_wmma|__WMMA'` on `DEBUG=4` output, **on a process that compiles ONLY the kernel under test** (a reference computation at default settings emits its own WMMA and will contaminate the count — isolate, one compute per process).
- Correctness = `max_rel_err` vs a reference computed the plain way, tol ≤ 1e-2 (fp16).
- Timing (only when asked) = `DEBUG=2` `tm`, warm ≥200 dispatches, never wall-clock.

## §5 Guardrails
- **Single GPU lane.** `pkill` strays + `rocm-smi` VRAM check before each run; never background a bench and report "waiting" (MMU faults). Run, wait, read.
- `.venv` python; temp files in `/home/ubuntu/.claude/jobs/6db6b205/tmp/`.
- **Commit on master, no branches**, `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`, push origin/master. **Always revert instrumentation before committing anything else; confirm `git status` clean for `tinygrad/`.**
- **No BEAM** (hangs gfx1100).
- **Core budget:** the fix lives in `tinygrad/` (real core) — keep it minimal, watch `sz.py`.

## §6 One-line job
**Prove exactly why `(a@b).max(-1)` at TC_OPT=2 tags WMMA but doesn't emit it (P1.a, report the exact exception/line to Claude), then land the smallest fix so it emits WMMA and stays correct (P1.b). Two hard stops. No hand kernels, no in-loop realize, no conclusions — only artifacts.**
