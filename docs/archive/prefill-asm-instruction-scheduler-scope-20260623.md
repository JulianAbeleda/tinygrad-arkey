# Prefill ASM Instruction Scheduler — Scope + Inc 0 Result (2026-06-23)

## Why this exists
The adversarial-Tensile-liveness audit (`prefill-adversarial-tensile-liveness-audit-result-20260623.md`) closed off
every *structural* explanation for the ~4% whole-prefill gap to Tensile and attributed the residual to **fine-grained
instruction scheduling below the `build_gemm_lds2` template** — consumer-only `s_waitcnt` counts, `v_wmma` issue
cadence (Tensile SIA1), PGR1/PLR1 prefetch timing, and WGM8 L2-tile traversal. Verdict:
`PREFILL_TENSILE_LIKE_PATH_REQUIRES_ASM_ALLOCATOR` (precisely, an asm **scheduler**). This doc scopes that scheduler
and records the Inc 0 capability that is now built and proven.

## The vendored-Tensile vs asm-scheduler decision (the framing)
- **Vendored Tensile**: route the prefill GEMM to AMD's pre-compiled `.co` kernels. Gets the full ~4% / ~87%-of-llama
  byte-identical *today*, but carries an opaque external binary dependency, gfx1100-specific, un-modifiable — breaks the
  project's dependency-free owned-codegen property.
- **ASM instruction scheduler** (this scope): build, *inside* the project, the pass that produces Tensile's cycle-level
  schedule from our own emitted instructions. Dependency-free, owned, reusable across shapes/kernels; a real
  compiler-backend mini-project.
- **Honest ROI**: the gap is ~4%, and the audit found part of it is a `beta=true` *work confound* (Tensile reads+scales
  C; our kernel skips that work). Genuinely schedulable upside is **meaningfully under 4% (~2–3%), dependency-free**.
  Worth doing as the owned capability; not worth overstating.

## What `build_gemm_lds2` gives us to work with
`extra/gemm/rdna3_wmma_matmul.py:build_gemm_lds2` returns a flat `list[Inst]` — structured AMDGCN instruction objects
(`tinygrad.runtime.autogen.amd.rdna3.ins`), not strings. Each is introspectable:
- `inst.operands` → `{name: (Fmt, bits, OpType)}`; `inst.op_regs` → register span per operand;
- register number decodes exactly from `(inst._raw >> field.lo) & field.mask` using `dict(inst._fields)`;
- `OpType` distinguishes VGPR / SGPR-unified-src / SGPR / inline-const / non-register;
- `inst.size()` byte length, `inst.simm16` mutable (branch offsets back-patched by the builder).

Today there is **no** dependency graph, liveness, scheduling, or register allocation: order == emission order, `s_waitcnt`
is hand-placed as conservative full drains (`simm16=0`), registers are statically hand-assigned, and the only
post-pass is branch back-patching. The instruction stream is exactly the right substrate for a scheduler.

## Design — 4 components
1. **Instruction IR + dependency model** *(keystone — built in Inc 0)*. Per-instruction exact register `defs`/`uses`
   (decoded from the encoding), memory-counter domain (`vm` for VMEM, `lgkm` for LDS+SMEM), and fence/branch class.
   RAW/WAR/WAW dependency edges over physical registers.
2. **Waitcnt-insertion pass** *(highest value-per-effort — Inc 1)*. Replace the conservative full `s_waitcnt(0)` drains
   with minimal **consumer-only** `vmcnt`/`lgkmcnt` counts. This is the literal "Tensile consumer-only s_waitcnt" lever,
   independently shippable, and is **where the async-load model lives** (a load's destination is valid only after its
   domain counter drains — exactly the dependency Inc 0 deliberately does not move across).
3. **Latency-aware list scheduler** *(SIA1 lever — Inc 2)*. Reorder within counter/fence-bounded regions to keep the
   WMMA unit fed and hoist loads early, under a VGPR-liveness budget (never exceed 256).
4. **WGM8 grid remap** *(sibling, Inc 3)*. Workgroup-ID→tile remap for L2 locality — a grid/addressing change, not an
   instruction-schedule change; separate additive flag.

## Increment plan (each additive, default-off; promote only on clock-pinned synced whole-prefill)
- **Inc 0 — capability scaffold + faithfulness proof. ✅ DONE (this commit).**
- **Inc 1 — waitcnt lever**: consumer-only `s_waitcnt` recompute (adds the async-load counter model). Cheapest real win.
  Kill if <0.5% and the detector shows hot-path waits already near-minimal.
- **Inc 2 — list scheduler**: latency-aware reorder of compute+memory within VGPR budget.
- **Inc 3 — WGM8**: grid remap for L2 locality.
- **Rest criterion**: if cumulative gain plateaus well under ~4% and the residual is confirmed `beta`-confound, declare
  the dependency-free frontier and stop chasing noise.

## Validation authority (non-negotiable, per project rules)
- Correctness: `rel_rmse ≤ 3e-4`.
- Speed: clock-pinned (`rocm-smi --setperflevel high`) synced whole-prefill via `extra/qk_prefill_whole_synced.py`,
  ≥3 interleaved rounds. Isolated / W==D timing is **not** promotion authority for prefill.
- No `tinygrad/` source change, no default flip, no vendored-Tensile promotion.

---

# Inc 0 — RESULT: `ASM_SCHED_IR_DAG_FAITHFUL`
Built `extra/qk_asm_scheduler.py` (IR + region/dependency model + scheduler) and `extra/qk_asm_scheduler_inc0_test.py`
(the proof). Inc 0 deliberately does **not** reorder for speed — it proves the IR+DAG is faithful so later increments
can reorder safely.

### What it does
- **`lift(inst, idx)`** — exact register `defs`/`uses` decoded from the encoding (OpType-aware: VGPR / unified-src /
  SGPR / inline; store operands are all reads; load `vdst`/`sdata` is the async def; `v_wmma` `src2`==`vdst` is RMW).
- **`build_regions`** — partitions the stream at **fences (control/sync) AND memory ops**. Inc 0 reorders only
  pure-compute (ALU/wmma) within a region. Rationale (learned the hard way, see below): an async load writes its dest at
  *drain* time, not issue time, so moving memory ops is unsound without the wait-counter model — that is Inc 1's job.
  With memory anchored, no movable instruction ever consumes an un-drained load result inside a region, so register
  RAW/WAR/WAW is sound and the reorder is provably correct.
- **`schedule(insts, mode)`** — `identity` (byte-identical) or `asap` (dependency-respecting greedy; tie-break emits the
  *latest* ready node first, so independent runs come out reversed — a maximal legal permutation).

### Proof (`extra/qk_asm_scheduler_inc0_test.py`, gfx1100, M=N=K=512)
| check | result |
|---|---|
| P1 IDENTITY_BYTE_IDENTICAL | PASS — `schedule(identity)` reproduces the stream bit-for-bit |
| P2 DECODE_COVERAGE_AND_RANGE | PASS — every operand classified; all decoded regs in physical range |
| P3 DAG_LEGALITY_BACKWARD_EDGES | PASS — every dep edge backward in program order (original order is a valid topo sort) |
| P4 LAYOUT_PRESERVED (554 insts moved) | PASS — per-region reorder preserves total byte size; branch offsets stay valid |
| P5 IDENTITY_RUNS_CORRECT | PASS — rmse 2.07e-04 (control) |
| P6 LEGAL_REORDER_RUNS_CORRECT | PASS — rmse 2.07e-04 with 554 instructions permuted (the strong test) |

**Config sweep** (the real route's config space) — all PASS, byte-identical numeric result under heavy reorder:
`default(PLRA)`, `kv_halved(PLRA)`, `plain(DBUF0)`, `DBUF1`, `8-wave(PLRAB)` — 348–652 instructions legally moved,
identity & reorder both rmse 2.07e-04.

### Two real bugs the proof caught (the test earns its keep)
1. **Self-edge on RMW** — `v_wmma` (`src2`==`vdst`) and `v_add v2,_,v2` read+write the same register; the naive WAR pass
   added a self-dependency → false cycle. Fixed by excluding self in the WAR edge.
2. **Unsound async-load hoist** — an early "loads-first" tie-break hoisted an async `ds_load`/`global_load` above its
   address/consumer, producing an **MMU fault** on the GPU. This is exactly the async-completion hazard the synchronous
   register model cannot see — corrected by anchoring memory ops (deferred to Inc 1's wait-counter model).

### Verdict
`ASM_SCHED_IR_DAG_FAITHFUL` — the instruction IR and intra-region dependency DAG are faithful across the prefill route's
config space (proven by byte-identical identity + correct heavy legal reorder on hardware). Ready for **Inc 1 (the
consumer-only `s_waitcnt` lever + async-load counter model)**. No `tinygrad/` source, no default change, no speed claim.

### Files
New: `extra/qk_asm_scheduler.py`, `extra/qk_asm_scheduler_inc0_test.py`, this doc. No production path touched.
