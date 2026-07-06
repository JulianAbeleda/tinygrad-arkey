# Exhaustive Scope: Track A (schedule-first) & Track B (ISA-renderer pipelining)

Date: 2026-07-06. Follows `docs/prefill-pure-substrate-exhaustive-scope-20260706.md` and the Fable re-scope.

## Why two tracks
Recover the hand kernel's 4413 tok/s 8B pp512 with a pure generated substrate, delete `extra/qk/prefill/wmma.py`.
Verified reframe (see memory `prefill-gemm-hip-renderer-owns-scheduling`): the default `DEV=AMD` prefill GEMM is
emitted by **HIPRenderer** (HIP C → ROCm/LLVM), operands **register-resident** (no LDS tile), and tinygrad has **no
waitcnt/pipelining authority** on that path. So:
- **Track A** = schedule-side wins that live entirely in tinygrad's OptOps/warmstart layer and work on the HIP path.
  Recovers most of the ~1.7× gap with zero renderer work. **Buildable now, low risk.**
- **Track B** = the ideologically-pure "tinygrad owns the pipelining" endgame: move the prefill GEMM onto the opt-in
  **AMDISARenderer** and add counter-targeted `s_waitcnt` so the generated kernel can defer the vmcnt drain past
  compute (the hand kernel's actual latency-hiding). **Large architectural commitment; gated on Track C's measurement.**

Baseline: harness-fixed generated = 2549 tok/s (verified); hand = 4413. Track C (occupancy measurement, run
concurrently) quantifies how much of the residual 40→58 TFLOPS/GEMM is even addressable, i.e. whether B is justified.

---

# TRACK A — schedule-first (buildable now, no renderer work)

## A1. Straggler `.contiguous()` fix — biggest single win
The `attn_output` out-projection runs at 16 TFLOPS (~19% of the forward) because its A-operand is the strided
attention-output view `attn.transpose(1,2).reshape(B,T,4096)` (`model.py:387-389`), so the warmstart `UNROLL=8`
fails `KernelOptError: 8 can't divide CONST arg=2` → silent fallback to the LOCAL-less heuristic
(`postrange.py:533-539`). The existing `.contiguous()` at `model.py:389` is on the matmul **output** (wrong target).
*Fix:* materialize the **input** dense before `_pf16(self.attn_output, out_in)` — `out_in = out_in.contiguous()`.
*Gate:* `WARMSTART_DUMP=1` shows `_warmstart_stats["apply"]` increments for key `(frozenset({512,4096}),4096)` and
no `errs` entry; AMD pp512 rises from ~2549 and the `attn_output` kernel moves 16→~38 TFLOPS.
*Risk:* the diag agent saw a run-order `match=0` caching artifact — MUST verify in-model that materializing the
input doesn't tag the matmul out of the warmstart path. *Files:* `tinygrad/llm/model.py:387-389`.

## A2. Per-role fused re-search
The frozen table was searched on isolated `a@b.transpose()` (`prefill_v2_schedule_search.py:56-88`); several in-model
**fused** instances underperform vs their isolated TFLOPS. Extend the search worker to optionally build/measure the
realistic fused role graph (or at least the contiguous-input variant matching the in-model AST), re-search every role
shape, regenerate the table, and re-validate with `prefill_v2_schedule_table_gate.py`.
*Coverage note:* `DEFAULT_SHAPES` already covers all 9 role shapes (workflow-confirmed) — this is a *fidelity* fix
(search the right AST), not a coverage gap. *Gate:* every role ≥ ~35 TFLOPS in-model; pp512 → projected ~3300+.

## A3. Clean up the dead DBUF scaffold
Per Fable: the `rangeify.py:453-457` DBUF branch is a functional no-op (2× LDS alloc, same `idx`); `_prefill_dbuf_peel`
(`postrange.py:520-531`) is role-unscoped (peels softmax/norm reduces too). Either delete the scaffold or gate
`_prefill_dbuf_peel` to the WMMA/GEMM role only and make the rangeify branch actually offset `idx` by the slot.
Keep everything behind `PREFILL_DBUF` (default 0) so no default-path change. *Files:* `rangeify.py`, `postrange.py`.

## Track A done-criteria
pp512 recovers to ~3300–3400 tok/s (Fable estimate), pure `tinygrad_scheduler_generated`, no `Ops.INS`, all bit-exact,
zero renderer changes. This is "most of the gap for free."

---

# TRACK B — ISA-renderer pipelining (the pure-DBUF endgame; gated on Track C)

Goal: let a *generated* prefill GEMM express the hand kernel's deferred/targeted `waitcnt` so tile k+1's global loads
overlap tile k's WMMA — the ~40→58 TFLOPS lever — with no raw `Ops.INS`.

Known facts (verified): renderer list `[HIPRenderer, AMDLLVMRenderer, HIPCCRenderer, AMDISARenderer]`
(`ops_amd.py:1039`); ISA is opt-in via `DEV=AMD:ISA` (`ops_amd.py:1038`). The ISA renderer's `_insert_waitcnt`
(`tinygrad/renderer/isa/amd.py:747-792`) currently emits **only full-drain `s_waitcnt(simm16=0)`** at RAW/WAR/barrier
boundaries (:750,761,767) — no counter-targeted `vmcnt(n)`/deferred drain.

## B-scope OPEN INVESTIGATION ITEMS (the orchestration must answer these before building)
1. Does `AMDISARenderer` even render a full prefill WMMA GEMM correctly today (TC opts + LDS staging + epilogue),
   or only a subset? Round-trip a representative 4096×4096 GEMM through `DEV=AMD:ISA`, check bit-exactness + that it
   produces WMMA. If it can't render the GEMM, B needs renderer-completeness work FIRST (scope that).
2. Data model of `_insert_waitcnt`: what state does it track (outstanding vmcnt/lgkmcnt per class, per-reg pending)?
   Enough to compute a *targeted* `vmcnt(n)` = "wait until ≤ n VMEM outstanding"? Map the exact insertion points.
3. How would a generated kernel *request* a deferred/targeted wait? Options: (a) a scheduling hint UOp the DBUF
   lowering emits that `_insert_waitcnt` honors; (b) `_insert_waitcnt` inferring the deferral from an unrolled
   load-ahead pattern (issue-early loads + late first-use). Prefer (b) — no new UOp — if the dependency structure
   from the Track-1 unroll-by-2 peel already expresses "load early, use late."
4. LDS staging on the ISA path: the DBUF is an LDS double-buffer, but the mainline GEMM is register-resident. Does B
   require ALSO forcing operand LDS staging (the `_tc_local_stage`/coop-B machinery) onto the ISA GEMM route? Or is
   the register-resident + deferred-vmcnt (no LDS) form sufficient for the overlap? Decide the target shape.
5. Route wiring: how to make ONLY the prefill GEMM use `DEV=AMD:ISA` while decode/other kernels stay on HIPRenderer
   (per-kernel renderer selection, like warmstart is per-kernel). Is that even supported, or is renderer chosen
   per-device? If per-device, B implies rendering the WHOLE model on ISA — scope that blast radius.

## B build sub-steps (after the investigation resolves the above)
- **B1.** Counter-targeted `s_waitcnt` in `isa/amd.py:_insert_waitcnt`: add `vmcnt(n)` emission (not just simm16=0)
  driven by the DBUF load-ahead structure. Gate behind an env flag; default full-drain unchanged.
- **B2.** DBUF lowering (reuse the Track-1 unroll-by-2 peel + double-slot bufferize) emitting the load-ahead pattern
  that B1 recognizes, on the ISA-rendered GEMM.
- **B3.** Route the prefill GEMM (only) onto ISA; verify bit-exact + measure TFLOPS approaches hand ~58.
- **B4.** If B reaches target: delete `extra/qk/prefill/wmma.py` + the `PREFILL_GRAPH_GEMM` route; confirm
  `PURE_MACHINE_SEARCH_ONLY=1` green.

## Track B risk register
- ISA renderer may not render the full GEMM (B could balloon into renderer-completeness work).
- Per-device (not per-kernel) renderer selection → whole-model-on-ISA blast radius.
- Even with counter-targeted waitcnt, matching hand's 58 TFLOPS needs the right unroll/prefetch depth + register
  budget (VGPR≥238 garbage trap, per memory `gfx1100-raw-ins-asm-gotchas`).
- Track C may show the addressable per-GEMM residual is small (e.g. schedule wins already reach ~50 TFLOPS), making
  B low-ROI. **This is why C runs concurrently.**

---

# Orchestration + Track C plan
- **Orchestration workflow (AMD-free, mutations worktree-isolated):** A1 (apply straggler fix), A2 (scope+prototype
  fused re-search), A3 (clean scaffold) as build agents; B investigation items 1–5 as a deep design agent producing
  the exhaustive B implementation plan + go/no-go criteria. Consolidate into a runbook.
- **Track C (parent runs concurrently on main tree, serial AMD, background-no-kill):** measure the per-GEMM lever on
  the HIP path — for a well-scheduled 4096×4096 GEMM, sweep source-unroll / `unr` / `loc` / register pressure and see
  how close comgr gets to ~58 TFLOPS without any DBUF. If comgr already pipelines to ~50+, Track B is low-ROI and A is
  the whole answer; if it plateaus at ~40, B's deferred-waitcnt is the real lever and justified.
- AMD is single-GPU; the workflow stays AMD-free so C never contends. Never force-kill a live `DEV=AMD` run.
</content>
</invoke>
