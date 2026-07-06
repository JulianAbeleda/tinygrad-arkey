# Exhaustive Scope: Pure Generated Prefill Substrate (delete the hand kernel)

Date: 2026-07-06

## North star

Recover the handwritten fp16 graph-GEMM prefill performance with a **pure tinygrad-codegen substrate**
(no raw `Ops.INS`, no `extra/qk/prefill/wmma.py`), then delete the hand kernel. `PURE_MACHINE_SEARCH_ONLY=1`
must stay green throughout.

## Measured ground truth (2026-07-06, AMD gfx1100, 8B Qwen3-Q4_K_M, pp512 len512, warmups=4/rounds=3, clock-pinned)

| Path | tok/s | effective TFLOPS | notes |
| --- | ---: | ---: | --- |
| Hand kernel (`PREFILL_GRAPH_GEMM=1`, `build_gemm_lds2`) | 4413 | 58.6 | `external_handwritten_kernel`, the target |
| Generated, **warmstart ON** (real `__call__` path) | 2688 | ~40 | pure; the honest baseline |
| Generated, warmstart OFF (bench harness bug) | 1741 | 23.1 | phantom — schedule table was inactive |

**Real gap to close: 1.64× (4413 / 2688).** Per-chunk: generated 190.4 ms vs hand 116.0 ms — a 74.4 ms deficit.

Decomposition (device-time trace, warmstart ON, 8B forward = 211 ms device / 729 kernels; elementwise is only 1.3%):
the entire gap is **matmul kernels running below the hand kernel's ~58 TFLOPS**. Two levers:
1. A few GEMM *shapes* still mis-scheduled (e.g. a 4096×4096 attention projection at **16 TFLOPS**).
2. The remaining per-GEMM gap (well-scheduled generated ~40–48 TFLOPS vs hand ~58) = **double-buffering (DBUF)**,
   which current codegen cannot express. This is THE primitive.

Fable-style overhead hypotheses (non-GEMM elementwise/cast/attention) are **refuted** by the trace (1.3%).

---

# STEP 1 — THE PRIMITIVE: generated double-buffered K-loop (scoped first, per request)

**What the hand kernel does** (`extra/qk/prefill/wmma.py::build_gemm_lds2`, `DBUF=1`, lines 250–262): allocates two
LDS tile buffers, unrolls the K-block loop by 2, and issues block N+1's global loads **before** block N's compute,
deferring the VMEM drain (`waitcnt_vm(0)`) until *after* compute — so global-load latency of block N+1 is hidden
behind the WMMA compute of block N. `s_barrier` between compute(this) and the next compute(other) enforces the
LDS visibility. This is the ~40→58 TFLOPS lever. (`PLRA`/`PLRAB` = intra-block register-level fragment prefetch;
second-order, deferred — the shipped kernel runs without them.)

**Why current codegen can't express it** (verified): double-buffering a *rolled* sequential K-range needs two
ordered phase-scopes over the one K-range = **two `END`s over the same range**, which trips the `CFGContext` cycle
assert at `tinygrad/codegen/late/linearizer.py:162` (the documented TG-P12 failure, see
`tinygrad/codegen/late/devectorizer.py:511-513`). `Ops.AFTER` only expresses a *same-instant forward* edge
(`tinygrad/uop/__init__.py:38-39`); `Ops.BARRIER` is all-or-nothing (would serialize away the overlap). So the
loop-carried WAR edge (store slot (k+1)&1 must wait on prior-iter reads of slot k&1, but compute(k) must NOT wait
on store(k+1)) is **inexpressible on a rolled loop**.

**The chosen approach — unroll, don't add a UOp.** Unrolling the K-range by 2 (prologue prefetch → steady-state
compute-slot-k-while-storing-slot-(k+1) → epilogue compute) turns the loop-carried WAR into ordinary **intra-body
forward `AFTER` edges** that `AFTER`/`BARRIER`/`END` already express and `CFGContext` already accepts (one `END`
per range still holds). This mirrors the hand kernel exactly and needs **a new lowering pass, not a new vocabulary**.

## Sub-steps (each is a route-bound, bit-exact, measured gate — no raw `Ops.INS`)

**1a. Single-buffered cooperative LDS staging, route-bound on a real GEMM (Track 2A foundation).**
Prove `bufferize(AddrSpace.LOCAL, removable=False)` + WARP address key + explicit CONTRACT fold emits real
global→LDS→WMMA traffic, bit-exact, on the actual prefill GEMM shape (not just the tiny probe). Reuse the edge
template `_tc_local_stage_coop_b_wmma_post` (`tinygrad/codegen/opt/postrange.py:465-480`:
`group(stores).end(ranges) → barrier → buf.after(barrier) → consume`).
*Files:* `tinygrad/schedule/rangeify.py` (`bufferize_to_store` LOCAL branch, lines 428–432), `postrange.py`.
*Gate:* bit-exact vs fp32 ref; DEBUG=2 shows `__attribute__((shared` + `s_barrier`; TFLOPS ≥ non-staged baseline.
*Status:* substrate probe already passes (design doc milestone-1); needs route-bound proof on the real shape.

**1b. Two-slot LDS allocation + `(k&1)` slot index.**
Extend the LOCAL bufferize to allocate `NBUF=2` and index the store/read by `(k&1)`. Storage only — no ordering
change yet. *File:* `tinygrad/schedule/rangeify.py:428-432`. *Gate:* still bit-exact single-buffer behavior with
2× LDS allocated (LDS ≤ 64 KB assert holds).

**1c. K-range unroll-by-2 peel — the core of the primitive.**
New PatternMatcher in `tinygrad/schedule/rangeify.py` that peels the REDUCE/LOOP K-range into
prologue+steady+epilogue, wiring `store(slot (k+1)&1) → AFTER → prior-iter WMMA reads(slot k&1)` as straight-line
intra-body edges + one hoisted barrier per unrolled step. MUST run **before** `pm_add_control_flow`
(`tinygrad/codegen/__init__.py:174`) so the CFG only ever sees forward edges (never two ENDs over the K-range).
*Gate:* bit-exact; DEBUG=2 shows two LDS buffers + interleaved load/compute; **beats single-buffer TFLOPS**.
*Risk (HIGH):* even with correct structure, tinygrad may insert a conservative full `s_waitcnt` at the load's
first use — killing the overlap. That is sub-step 1d.

**1d. Deferred / targeted `waitcnt` (the actual latency-hiding).**
The hand kernel defers `waitcnt_vm(0)` past compute and uses non-zero counter targets (`waitcnt_vm(LPB)`).
tinygrad emits conservative full waits at dependency boundaries. Scope: does the unrolled structure from 1c let the
existing renderer schedule the vmcnt drain after compute (because the dependency edge lands there), or does the AMD
renderer need a counter-targeted-wait capability? *Files:* AMD renderer waitcnt insertion; `tinygrad/renderer/`.
*Gate:* the DBUF version's measured TFLOPS approaches hand ~58 (the overlap is real), not just ~40.
*This is the residual-risk sub-step; measure 1c first — if 1c already overlaps, 1d may be unnecessary.*

**1e. (DEFERRED) PLRA/PLRAB register-pool fragment prefetch.** Second-order. Do NOT build unless 1a–1d leave a
measured residual vs hand. The hand kernel ships without it.

## Fallback if unrolling is insufficient (keep the loop rolled)
Add a loop-carried-`AFTER` tag (arg flag on `Ops.AFTER` in `tinygrad/uop/__init__.py:41`, or a new `Ops` variant)
meaning "edge holds across the range latch, not within one instant"; exempt tagged edges from the
`backward_slice_with_self` check at `linearizer.py:162` and from `deps`/`nesting` propagation in `CFGContext`.
Strictly more invasive (enum + CFGContext invariant + bufferize). Only if 1c/1d prove unrolling can't hide latency.

## Step-1 done-criteria
Generated (no `Ops.INS`) double-buffered fp16 WMMA GEMM, bit-exact, that measures within ~10% of the hand kernel's
per-GEMM TFLOPS on the representative prefill shapes, route-bound through `prefill_v2_scheduler_matmul_default`.

---

# STEP 0 (prerequisite — tiny, land regardless): fix the bench harness

Not a "feature step," but every Step-1/2 perf number must be measured on the real path. **5-line change**, no risk.
`extra/qk/prefill_whole_synced.py::burst` builds its own `TinyJit(model.forward)` (line 81), bypassing
`model.__call__` where warmstart installs (`model.py:783-786, 797-801`) → measures the phantom 1741.
*Fix (Option 1, no logic duplication):* delete the harness `TinyJit`; warmup/timed loops call
`model(chunk, sp_int, temp, use_flash=True).realize()` (concrete int `sp_int` → the `prefill_v2_jits` per-start_pos
jit + warmstart install, exactly the production path `precompile_concrete_prefill_jits` uses). `use_flash=True` is
required (else `__call__` clobbers `block._use_flash` to False). Preserve all synchronize/K-min-burst timing.
*Gate:* harness now reports ~2688 tok/s for the generated path. *Do NOT* use the manual-`_WARMSTART_OPTS` variant.

---

# STEP 2 — schedule coverage: the mis-scheduled GEMM shapes

After warmstart, the dominant straggler is `r_2_64_32_4_2_2_2_4_4_32_2_8` = a **(M=512, 4096×4096) attention
projection** (`attn_q` or `attn_output`), running at **16 TFLOPS / 19% of the forward** while its sibling 4096²
role runs at 41. It IS covered and HAS a good table entry (loc=4, 38.82 TFLOPS isolated) — but the schedule isn't
landing on this in-model instance. So this is **not "add a missing shape."**

**2a. Diagnose (first).** Run with `WARMSTART_DUMP=1` and inspect `postrange._warmstart_stats["errs"]`: is this key
hitting a `KernelOptError` → silent fallback to the LOCAL-less heuristic (`postrange.py:519-538`), or does it apply
but under-perform because the real fused AST differs from the standalone `a@b.T` the search measured?
*This diagnostic decides 2b vs 2c and definitively distinguishes attn_q vs attn_output.*

**2b. If KernelOptError fallback:** fix the opt so it applies on the fused AST (axis/fusion mismatch), or add a
shape-signature variant the match resolves. *File:* `postrange.py:502-538` (`_warmstart_match`, apply/fallback),
`model.py:675` (re-key).

**2c. If applies-but-underperforms:** re-search this shape *in the fused in-model context* rather than the synthetic
`a@b.transpose()`. *File:* `extra/qk/prefill_v2_schedule_search.py` (worker builds `a@b.T`, lines 56–88) — extend to
optionally search the realistic fused graph.

**2d. Extend search coverage to all role shapes** (attn_kv 1024×4096, ffn shapes) and regenerate the table:
append `(out_f,in_f)` to `DEFAULT_SHAPES` (`prefill_v2_schedule_search.py:22-25`), run the brute-force grid
(`GRID_U0/U1/LOC/UNR`, NO BEAM — beam hangs gfx1100), `load_table()` auto-picks it up. Validate with
`prefill_v2_schedule_table_gate.py` (requires every shape present + uses LOCAL + re-measures OK).
*Gate:* every prefill GEMM role ≥ ~35 TFLOPS in-model; projected pp512 ~3300–3400 tok/s.

---

# STEP 3 — integrate + delete the hand kernel

Wire the Step-1 DBUF lowering into the default `prefill_v2_scheduler_matmul_default` route; measure whole-prefill
pp512 through the fixed harness; confirm generated ≥ hand (or within target). Then delete
`extra/qk/prefill/wmma.py` + the `PREFILL_GRAPH_GEMM` route (`prefill_graph_gemm_route.py`) and the raw-`Ops.INS`
markers, confirm `PURE_MACHINE_SEARCH_ONLY=1` green, and update the census/gates.
*Gate:* no default-selected path executes `extra/qk/prefill/wmma.py`; pp512 ≥ 4000 tok/s pure; all bit-exact.
*Do NOT delete the hand kernel until Step 1 measures within target — it stays as the reference to match.*

---

# Delegation & orchestration plan

Dependency graph (what can run in parallel vs what blocks):
- **Step 0** (harness): independent, 5-line, land immediately — unblocks all measurement.
- **Step 1** (primitive): the long pole. 1a→1b→1c are sequential (each builds on the prior); 1d gated on 1c's
  measurement; 1e deferred. This is one focused codegen workstream, hard to parallelize internally.
- **Step 2** (schedule coverage): 2a (diagnose) independent and immediate; 2b/2c branch on 2a; 2d parallel per-shape
  (each shape is an independent brute-force search — natural fan-out).
- **Step 3**: gated on Step 1 + Step 2.

Proposed orchestration (once approved):
1. **Phase A (parallel):** Step 0 harness fix + Step 2a diagnosis + Step 1a route-bound single-buffer proof — three
   independent agents; all are prerequisites and cheap.
2. **Phase B (primitive, focused):** Step 1b→1c→1d as a single deep codegen agent (worktree-isolated), with a
   bit-exact+TFLOPS gate after each sub-step. Adversarially verify each gate (bit-exactness is easy to fool).
3. **Phase C (parallel):** Step 2d per-shape schedule searches fan out (one agent per role shape, each runs the
   brute-force grid on AMD — serialize the GPU-bound ones; the AMD box is single-GPU so cap concurrency at 1 for
   actual measurement, parallelize only the non-GPU search-setup).
4. **Phase D:** Step 3 integration + deletion, single agent, after B+C green.

Note: AMD is single-GPU and killing a live `DEV=AMD` run wedges the MES ring — GPU-bound gates must run
sequentially with generous timeouts, never force-killed.

## Reading list for any agent picking this up
- This doc + `docs/codegen-wmma-lds-staging-design-20260705.md` (Track 2A/2B).
- Primitive: `tinygrad/schedule/rangeify.py:397-463`, `tinygrad/codegen/late/linearizer.py:135-167`,
  `tinygrad/codegen/late/devectorizer.py:511-518`, `tinygrad/codegen/opt/postrange.py:428-484`, `tinygrad/uop/__init__.py:38-112`.
- Hand reference: `extra/qk/prefill/wmma.py:105-285`, `extra/qk/prefill_schedule_spec.py:60-102`.
- Schedule/straggler/harness: `extra/qk/prefill_v2_schedule_search.py`, `tinygrad/llm/model.py:660-801`,
  `extra/qk/prefill_whole_synced.py:62-106`.
- Diagnostic scripts (this session): `/home/ubuntu/.claude/jobs/2f995982/tmp/{prefill_trace,prefill_trace_ws,parse_trace,authority_ws}.py`.
</content>
</invoke>
