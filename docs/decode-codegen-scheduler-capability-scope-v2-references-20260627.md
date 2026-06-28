# Codegen software-pipelining capability — exhaustive, reference-grounded scope v2 (2026-06-27)

Supersedes `decode-codegen-scheduler-capability-scope.md` (2026-06-26) with: (a) the **route-bound** baseline
(35.0/6.7 tok/s = 33.7%/7.1% of owned — not the old isolated 131–238×), (b) the **literature** that names each
sub-pass, (c) the concrete **RDNA3 mechanism** for each, (d) the **register/occupancy bound** the no-unroll
refutation proved is real. Principle unchanged: build a generic **codegen** capability so the *machine* schedules
competitive kernels (decode attention + prefill GEMM), not more hand-ASM.

## 0. Where this sits (existing foundations)
- **Layer 1 — latency-aware list scheduling** (built): `extra/qk_codegen_list_scheduler.py`, `SCHED_LIST=1`.
  Reorders the linearized UOp list to issue independent work in latency shadows. *Intra-iteration* only.
  Ref: **Gibbons & Muchnick, "Efficient Instruction Scheduling for a Pipelined Architecture," SIGPLAN '86**;
  NP-completeness + the heuristic it approximates: **Hennessy & Gross, TOPLAS 5(3), 1983**.
- **Layer 2 — recurrence unroll** (built): `extra/qk_codegen_recurrence_unroll.py`, `SCHED_UNROLL=8`. Creates
  cross-iteration copies (the ILP raw material). Verified to *help in-model* (no-unroll W==D 22.9/3.8 < 35.0/6.7).
- **Layer 3 — software pipelining / modulo scheduling** (THE MISSING CAPABILITY). Layers 1+2 give copies and an
  intra-iteration order; what's absent is the *cross-iteration steady-state schedule* that overlaps iteration N+1's
  independent work with iteration N's serial merge.

## 1. The bound, in modulo-scheduling terms (the theory that frames the build)
Software pipelining issues loop iterations at a constant **initiation interval II = max(RecMII, ResMII)**
(**Lam, "Software Pipelining: An Effective Scheduling Technique for VLIW Machines," PLDI 1988** — modulo scheduling +
modulo variable expansion; **Rau, "Iterative Modulo Scheduling," MICRO-27 1994 / IJPP 24(1) 1996** — the practical
resource-constrained algorithm; survey: **Allan, Jones, Lee & Allan, "Software Pipelining," ACM Computing Surveys
27(3), 1995** — the MII/RecMII/ResMII taxonomy).

For the decode tile:
- **ResMII** = the K/V LDS load chain. **comgr already pipelines the global loads** (staggered `s_waitcnt vmcnt`),
  so ResMII is largely hidden — *not* our bottleneck.
- **RecMII** = the **loop-carried online-softmax recurrence** `(m, l, acc).after(tt)` fed by the **per-token
  cross-lane reduce** (`ds_bpermute` ladder → `lgkmcnt`). This serial chain sets II, and tinygrad emits it
  **fully exposed** (linearizer is a topo-sort, no modulo scheduler). **This is the lever.**

The win is overlapping iteration N+1's **independent** work — load, `v_dot2`, and the reduce that does *not* depend
on the running `(m,l,acc)` — into the shadow of iteration N's serial merge. That is exactly the owned hand-ASM
steady-state kernel; its **double-buffered LDS tiles are modulo variable expansion** (Lam '88).

## 2. The three sub-passes (each: literature → RDNA3 mechanism → owned target it reproduces)

### P-SWP — software-pipeline the recurrence loop  [PRIMARY lever, Arm A]
- **Literature:** Lam '88 / Rau '94–'96 / Allan '95 (above).
- **RDNA3 mechanism:** there is **no `cp.async`** on gfx11 (**AMD RDNA3 ISA Reference, Feb 2023**); pipelining is
  realized by **`s_waitcnt`-counter scheduling** — issue iteration N+1's `buffer_load`→`ds_store` and its
  `ds_load`/`v_dot2`/reduce early, and **push the matching `vmcnt`/`lgkmcnt` wait *down past* iteration N's merge**.
- **Owned target it reproduces:** waitcnt 21 / shadow_fill 0.2 (vs generated 39–50 / 3.75); the steady-state
  "load next K/V while computing current."
- **Build:** a modulo-scheduling stage in `tinygrad/codegen/late/linearizer.py` (env-gated + in the `to_program`
  cache key at `tinygrad/codegen/__init__.py`), modeled on the existing opt-in lowerings (`qk_fdot2_lowering.py`,
  `qk_warp_reduce_lowering.py`). Partition each loop body into {serial-recurrence, independent} via the
  `.after()` carry chain; schedule independent ops of iter N+1 into iter N's `lgkmcnt` shadow at a target II;
  rename the staged LDS buffer (MVE) for depth-2 double-buffering.

### P-LICM — hoist the invariant cross-lane masks to the prologue
- **Literature:** **Aho, Lam, Sethi, Ullman, "Compilers" (Dragon), 2nd ed. 2006, Ch. 9** (LICM); optimal placement
  that *minimizes hoisted-value register lifetime*: **Knoop, Rüthing & Steffen, "Lazy Code Motion," PLDI 1992**.
- **RDNA3 mechanism:** the 5 XOR-butterfly permute/swizzle lane masks + exec masks are loop-invariant → compute
  once in the preheader, keep resident (owned uses `v8,v26–v29`).
- **Owned target:** "hoisted cross-lane masks" (prologue); inner loop carries data dependency only, no index math.
- **Build:** a LICM/lazy-code-motion pass over the loop UOps for the swizzle-index/mask sub-DAG. **Lazy Code Motion
  framing matters** because naive hoisting inflates live ranges → ties directly into P-REG below.

### P-FUSE — fused K+V LDS double-buffering
- **Literature:** **CUTLASS "Efficient GEMM in CUDA"** (multistage shared-mem double/triple-buffering);
  **FlashAttention (Dao, Fu, Ermon, Rudra, Ré, NeurIPS 2022, arXiv:2205.14135)** — IO-aware pipelined K/V staging
  with online softmax (our exact algorithm template).
- **RDNA3 mechanism:** interleave K and V in LDS so one `ds_load_2addr_stride64_b64` serves both (owned ds_read=1
  vs generated 12); prefetch tile i+1 into a second LDS buffer while computing tile i.
- **Owned target:** the single fused 2-addr K+V load + the 8 KB double-staged tile.
- **Build:** an LDS-layout + load-fusion lowering (interleave `ksh`/`vsh`; depth-2 buffer). Composes with P-SWP
  (the prefetch is the pipeline's producer stage).

### P-REG — bound the pipeline depth by the VGPR/occupancy tradeoff  [governing constraint, not optional]
- **Literature:** the scheduling↔allocation phase-ordering problem: **Goodman & Hsu, "Code Scheduling and Register
  Allocation in Large Basic Blocks," ICS 1988**; **Bradlee, Eggers & Henry, "Integrating Register Allocation and
  Instruction Scheduling for RISCs," ASPLOS-IV 1991**.
- **Why it's load-bearing here:** **measured** — dropping vgpr 80→40 (no-unroll) made W==D *worse* (22.9/3.8), but
  double-buffering (P-FUSE) + deeper pipeline (P-SWP) *raise* VGPRs, and on RDNA3 VGPR count gates wave occupancy.
  So there is a real interior optimum: too shallow = exposed latency; too deep = occupancy collapse. Pipeline depth
  and the MVE rename count must be **searched, not fixed** (Lazy Code Motion's minimal lifetimes + MVE's minimal
  renaming are what keep depth affordable).

## 3. Two architectural arms (which layer the schedule survives at)
- **Arm A — UOp-level modulo pass** (primary, build first): the generic pipelining stage in `linearizer.py` above.
  **Risk (from the 2026-06-26 scope):** comgr may re-schedule and *undo* the UOp-level order. If the hotloop diff
  shows the emitted ISA ignores our ordering, that finding **forces Arm B**.
- **Arm B — tinygrad's own ISA scheduler on `Ops.INS`** (deeper independence): mature the dormant
  `extra/qk_asm_scheduler.py` (already builds a reg def/use DAG over `list[Inst]`) into a real modulo scheduler on
  the `Ops.INS → Ops.LINEAR → assemble_linear` path (`tinygrad/renderer/amd/elf.py`). This emits scheduled ISA
  directly, independent of comgr — the capability the perf-state has **also** named for prefill GEMM, so one
  foundation retires **both** remaining hand-ASM kernels.

## 4. Build order + gates (the loop, re-run on this capability)
1. **Phase 1 — EXPOSE (do first, cheap):** `extra/qk_decode_hotloop_schedule_diff.py` — disassemble both tiles,
   identify the hot-loop body, and report per kernel: in-loop `lgkmcnt(0)`/`vmcnt(0)` counts and the **issue-distance
   from each `ds_bpermute`/load to its first consumer** (latency exposed vs overlapped). Verdict
   `HOTLOOP_SCHEDULE_DIFF__SCHEDULING_BOUND`. This is the baseline the capability must beat.
2. **Phase 2 — build Arm A (P-SWP first, then P-FUSE, with P-LICM + P-REG as supporting passes).**
3. **Validation (in order):** token-match (`qk_decode_attention_block_tile_microgate.py` → PASS) → hotloop diff
   exposed-latency drops toward owned → **route-bound W==D** (`qk_decode_route_attribution_wd.py`: route_bound +
   token_match + tok/s rises from 35.0/6.7) → **generality proof: the same pass moves the prefill-GEMM hot loop**
   (the test that it's a capability, not a kernel hack) → Phase 3: lift pipeline-depth/partition into the search.

## 5. Honest terminal labels
- `SEARCH_PROGRESS__CODEGEN_SCHEDULER` — Arm A/B moves the hotloop diff + W==D. Continue to generality + Phase 3.
- `SEARCH_BLOCKED_BY_CODEGEN__SCHEDULING` (deep) — only if a *properly built* modulo scheduler still cannot approach
  comgr/hand-ASM on the **RecMII** recurrence. Even then the conclusion is "this is the named long-horizon enabler,"
  not "give up" — the capability generalizes to prefill GEMM and is the thing that makes machine search emit fast
  kernels instead of routing among hand-written ones.
- Only abort: correctness/regression. Default-off (env-gated + cache key); shipped route + q4k GEMVs byte-identical.

## 6. One-line synthesis
Build a **modulo scheduler** (Lam '88 / Rau '94–'96) on the existing **latency list scheduler** (Gibbons-Muchnick
'86), use **Lazy Code Motion** (Knoop-Rüthing-Steffen '92) to hoist the invariant cross-lane masks, realize **K/V
LDS double-buffering** (CUTLASS multistage / FlashAttention) via RDNA3 **`s_waitcnt`-counter staging** (no cp.async),
and **search** the pipeline depth under the **VGPR/occupancy tradeoff** (Goodman-Hsu '88) — the measured interior
optimum the no-unroll refutation proved exists.
