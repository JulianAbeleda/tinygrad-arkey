# Prefill SW-Pipeline Codegen — Project Charter (funded 2026-06-20)

User decision (PTM-4): **fund the software-pipelined K-loop codegen capability** rather than vendor the
external Tensile `.co` or rest dependency-free. This charter turns that funding into a sequenced,
kill-gated project. It supersedes "go build modulo scheduling" with a cheaper-lever-first plan grounded in
the existing codegen map.

## Where we are (measured)

- **PTM-1** (`bb5a10_ptm1_…`): under one interleaved clock, tinygrad authority = **52.97 TFLOPS** (this
  clock), best hand-ASM candidate 39.9, gap to authority **1.33×** (the old 2.34× was a clock artifact).
  The hand-ASM Route-A candidates are **below** tinygrad's own authority.
- **P0 baseline** (`sw_pipeline_p0_baseline_result.json`): the authority kernel is **compute-light** —
  `64 v_wmma` vs **304 int-ALU address-arith** (4.75×) and **544 global_load**, **0 LDS** (global-direct).
  Compute is a small minority; address-recompute + load-handling dominate.

## The wall (who schedules what)

- The authority kernel is **HIP C source compiled by comgr/LLVM** (`HIPRenderer` `cstyle.py:331` →
  `HIPCompiler` `compiler_amd.py:48`). **LLVM** does the final K-loop instruction scheduling; tinygrad emits
  no `s_waitcnt` on this path. `AMDProgram` only loads the finished ELF.
- The linearizer (`codegen/late/linearizer.py:19,34,40-47`) is a **run_count-dominated priority toposort**:
  a K-indexed `LOAD` is pinned inside the RANGE; there is **no modulo-scheduling / loop-rotation** pass, so
  a load cannot be issued one iteration ahead of its WMMA. Proven: a hand-UOp double-buffer compiles
  **byte-identical** to single-buffer (CG-1, `prefill-codegen-software-pipeline-result-20260619.md`).
- No prefetch/async primitive exists: `OptOps` has no PREFETCH (`codegen/opt/__init__.py:6`); `Ops.LOAD` is
  synchronous register-fill (`uop/ops.py:792`); `Ops.WAIT` is dead in codegen. `UNROLL` fully unrolls the
  K-axis but keeps each `load→wmma` synchronous — it is not pipelining.
- `renderer/amd/schedule.py` (prefetch_stage, LDS staging) is **analysis/probe scaffolding, unwired** to the
  live compile path.

## Honest ceiling (state up front)

Route-A's hand-ASM pipeline (`build_gemm_pipe`) **does** express the double-buffer and gets **+32%**, but
caps at **24-32 TFLOPS** for a *single-wave / VGPR* reason — and even a renderer emitting single-wave
register double-buffering inherits that cap. Pure-tinygrad WMMA plateaus at **~42** (POWN); LLVM warmstart
~48; **Tensile ~66**. LDS-multiwave is **refuted on RDNA3** (net-negative, IC-served global reads). So:
**realistic ceiling of this whole effort ≈ LLVM (~42-48), not Tensile's 66.** The only proven ~66/87%-llama
path is the vendored `.co`, which was declined. Funding buys *dependency-free* prefill gains up to ~LLVM,
not Tensile parity. This is the central risk; every phase gates against it.

## Plan — cheap lever first, expensive capability gated behind it

### Phase 0 — baseline + premise (DONE)
`extra/qk_sw_pipeline_p0_baseline.py` → `PASS`. Instruction mix + authority baseline recorded above.

### Phase 1 — Lever A: addressing-mode lowering (cheap, local, high-confidence) — **NEXT**
The 304:64 address-ALU:WMMA ratio is the biggest *static* overhead and is **not** the pipelining problem.
tinygrad's AMD renderer emits no `base + immediate offset:` loads and does not strength-reduce the base
pointer across K, so it recomputes full 64-bit addresses per strided load.
- **Change:** AMD renderer index→address lowering (`cstyle.py` AMD path / LLVM-IR or asm) to emit
  immediate-offset loads + hoist/strength-reduce the base pointer across the K-loop. **No new IR op, no new
  pass.**
- **Measure:** in-harness via `qk_amd_bb5a10_ptm1_same_harness_bridge` (interleaved, one clock, vs the
  authority row in the *same* run).
- **KILL-GATE:** ≥ **1.2×** isolated/in-harness improvement, byte-identical correctness (expected ~42→~57).
  If < 1.2×, **kill Lever A and do not proceed to Lever B on addressing grounds.**

### Phase 2 — Lever B: the pipelining capability (net-new, multi-day) — **gated behind Phase 1**
Only if Phase 1 lands and prefill is still meaningfully below the authority/Tensile target. Net-new:
1. a prefetch/async-load IR op (new `Ops.PREFETCH` or revive `Ops.WAIT` with deferred-consume semantics);
2. a modulo-scheduling / loop-rotation pass (prologue + steady-state + epilogue; hoist iter k+1's load
   ahead of k's WMMA) — the linearizer structurally cannot do this today;
3. register double-buffer lowering (Route-A proved *register*, not LDS, is the RDNA3 form);
4. relaxed `s_waitcnt` — either emit C/index LLVM's pipeliner can rotate, **or** move the matmul onto the
   `assemble_linear` native-ISA path where tinygrad owns `s_waitcnt` (as Route-A does).
- **KILL-GATE:** must **exceed the authority row** in the PTM-1 harness (catching up to it is not a win).
  Expected ceiling ~LLVM single-wave (24-48). To exceed ~48 also needs multi-wave occupancy (global-direct,
  since LDS is refuted) = the deepest, IC-capped sub-problem — a separate gate if Phase 2 even reaches it.

### Phase 3 — ship / rest
Whatever passes its gate ships behind a flag, measured in-model (prefill pp512) under the PTM-1 clock
discipline. If both levers fail their gates, the honest outcome is: dependency-free prefill rests at
authority (~47% llama), and ~87% requires reopening the declined `.co` policy.

## Stop / discipline rules
- Report every TFLOPS with its sclk provenance; only the PTM-1 interleaved harness is timing authority.
- No mixed-harness comparison; no standalone LDS; one lever at a time behind its kill-gate.
- Do not start Phase 2 before Phase 1's gate resolves.

## Next concrete action
Phase 1: implement the AMD renderer addressing-mode lowering, then measure under the PTM-1 harness against
the 52.97 authority baseline. **This is the first real renderer change — the start of the multi-day risk —
so confirm before proceeding.**
