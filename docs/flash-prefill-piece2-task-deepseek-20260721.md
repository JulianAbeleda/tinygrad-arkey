# TASK (deepseek): Piece 2 — REDUCE-preserving attention fusion (diagnostic-first)

Read this ENTIRE file first. This is the follow-on to Piece 1 (cb6e760e0, WMMA now survives a fused matmul+epilogue-reduce). Piece 2 makes rangeify fuse *attention* while keeping the matmul contractions as REDUCE ops so the TC opt WMMA's them. **You will do the DIAGNOSTIC step (P2-A) only, then HARD STOP for Claude.** Do not attempt the fusion implementation yet.

Repo: `/home/ubuntu/tinygrad-arkey` · Python: `/home/ubuntu/tinygrad-arkey/.venv/bin/python` · Env: `DEV=AMD`

---

## §0 HARD BANS (same as Piece 1 — you have gone off-rails before)
1. ❌ No hand kernels (`custom_kernel`, UOp kernel bodies, `flash_kernels` imports, `__builtin_amdgcn`/barriers/LDS by hand). Litmus: writing `UOp(...)` for a kernel body = wrong task.
2. ❌ No `.realize()`/`synchronize()` inside a loop in any probe. One graph, realize once.
3. ❌ **Only verifiable artifacts, never conclusions.** Every claim = a measured number (WMMA-call count via `grep -cE '__builtin_amdgcn_wmma|__WMMA'` on `DEBUG=4`, kernel count via `grep -cE GFLOPS`, `max_rel_err`). Do NOT write "needs N weeks" or "rangeify needs X." Report numbers; Claude concludes.
4. ❌ **PCONTIG is NOT the fusion tool here.** Measured: PCONTIG fuses attention but converts the matmul REDUCE axes → LOOP (`n_reduce=0`) → 0 WMMA. Do not use PCONTIG for attention fusion.
5. ❌ **RUN THE TEST SUITE before any commit** (you skipped this in Piece 1). Any change to `tinygrad/` core must show the WMMA/packed-WMMA unit suite is unregressed vs parent.

## §1 Established state (do NOT re-derive)
- Piece 1 fix (cb6e760e0) is landed + verified: WMMA survives a fused matmul+epilogue-reduce (`(a@b).max(-1)` at TC_OPT=2 → WMMA 3, rel_err 0).
- Measured attention baseline (`T=KV=512,H=2`, Piece 1 fix in place):
  - **Unfused (PCONTIG=0): 26 kernels, WMMA=2 (QKᵀ only), the T×KV score is spilled to HBM.**
  - PCONTIG-fused: 17 kernels, WMMA=0 (REDUCEs destroyed — dead end).
- **Key lever:** `(a@b).max(-1)` fuses matmul-REDUCE + epilogue-max-REDUCE into ONE kernel *with the matmul REDUCE preserved*, at **default PCONTIG=0** (no PCONTIG needed) — and Piece 1 makes it WMMA. So **rangeify already does REDUCE-preserving fusion of matmul+reduce natively.** The question Piece 2 answers: why does `softmax(qkᵀ)@v` NOT fuse the same way (it spills the score into 26 kernels instead)?
- PV note: in standard SDPA, `p = s.softmax(-1)` is fp32 and `v.float()` is fp32 → the PV matmul is fp32×fp32 → **not WMMA-eligible**. Flash casts probs to fp16 before PV. Keep this in mind but it is NOT the P2-A focus.

## §2 P2-A — THE DIAGNOSTIC (your whole task): find what forces the score spill

Start from the WORKING REDUCE-preserving fusion (`(a@b).max(-1)`) and **incrementally add attention structure**, measuring at each step, until rangeify breaks into the score-spilling multi-kernel form. This isolates the exact op/pattern that prevents REDUCE-preserving fusion of attention.

Build these expressions (all `a,b` = `Tensor.randn(512,512,half)`, or q/k as needed), each in its OWN process (no reference contamination), under `Context(TC_OPT=2)`, and record **(kernel count, WMMA count)**:
1. `(a@b).max(-1)`                      — baseline: fused, REDUCE preserved, WMMA (Piece 1). Confirm.
2. `(a@b).sum(-1)`                       — matmul + sum-reduce (softmax's 2nd reduce shape).
3. `(a@b - (a@b).max(-1,keepdim=True))`  — matmul + max + broadcast-subtract (softmax step 1).
4. `((a@b) - (a@b).max(-1,keepdim=True)).exp().sum(-1)`   — matmul + max + exp + sum (softmax numerator/denominator).
5. `(a@b).softmax(-1)`                    — full softmax on a matmul (no PV yet).
6. `(a@b).softmax(-1) @ c`               — full attention shape (c = `randn(512,512,half)`), PV in fp16.

For EACH: `(kernel_count, WMMA_count)`, and for the ones that fuse, whether a score buffer of shape ~`(512,512)` appears (a spilled intermediate). **The transition from "fused, few kernels, WMMA present" to "many kernels, score spilled" tells us the exact operation that breaks REDUCE-preserving fusion.** That op is the Piece 2 target.

Also, for the FIRST expression that breaks (many kernels), capture the DEBUG=3 kernel signatures (the `r_...` names) to show where the buffer boundary is inserted.

## §3 Deliverable + HARD STOP
Write to `docs/flash-prefill-piece2-probe-<date>.md`:
1. The 6-row table: expression → (kernels, WMMA, score-buffer present?).
2. The exact expression index where REDUCE-preserving fusion breaks, and the added op that caused it.
3. The DEBUG=3 kernel signatures at that break point.
4. NO conclusions about "how to fix" or "how long" — just the measured breakpoint.
Commit the probe doc + any probe scripts (in `extra/qk/` or tmp). **Then STOP.** Do NOT implement a fusion, do NOT modify rangeify/postrange, do NOT touch PCONTIG. Claude decides the fusion approach from your breakpoint data.

## §4 Guardrails
- Single GPU lane; `pkill` strays + `rocm-smi` before runs; never background a bench and report "waiting." Run, wait, read.
- `.venv` python; temp in `/home/ubuntu/.claude/jobs/6db6b205/tmp/`; isolate one compute per process for WMMA counts (a reference at default settings emits its own WMMA and contaminates the count).
- Commit on master, no branches, Co-Authored-By trailer, push. (Probe doc only — no core edits in P2-A.)
- No BEAM.

## §5 One-line job
**Incrementally grow `(a@b).max(-1)` → `softmax(a@b)@c` under TC_OPT=2, record (kernels, WMMA, score-spilled?) per step, and report the exact expression where REDUCE-preserving fusion breaks. Diagnostic only. Hard stop for Claude. No fusion edits, no PCONTIG, no conclusions — only the measured breakpoint.**
