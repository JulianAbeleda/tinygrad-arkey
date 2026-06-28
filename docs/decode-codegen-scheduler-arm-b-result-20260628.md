# Arm-B codegen scheduler — result (2026-06-28)

Scoped + ran the Arm-B preflight (tinygrad emits scheduled AMD ISA itself, bypassing LLVM). **Terminal: blocked on a
missing capability — there is no native AMD ISA backend.** This closes the scheduler investigation: both arms
characterized, the precise missing capability named.

## Verdict
`SEARCH_BLOCKED_BY_CODEGEN__SCHEDULER_NOT_WIRABLE` — Arm B's premise (schedule the instruction stream *after*
bypassing LLVM) cannot be wired to the **generated** decode block tile, because tinygrad has **no UOp→`Ops.INS`
instruction-selection backend for AMD**. Every AMD renderer routes through LLVM; the direct-assembly substrate only
serves hand-emitted assembly. Building Arm B = building a complete native AMD/rdna3 ISA backend — a major capability,
not a scheduling pass.

## The preflight, measured
The Arm-B idea was: route the block tile through `Ops.INS → assemble_linear` (`tinygrad/renderer/amd/elf.py:14`,
which encodes `Inst`s to ELF directly, no LLVM), where the order is tinygrad's and a scheduler (`schedule.py` /
`extra/qk_asm_scheduler.py`) could control it. The preflight asked: can the block tile reach that path? It cannot:

1. **Every AMD renderer routes through LLVM.** Device renderers are `[HIPRenderer, AMDLLVMRenderer, HIPCCRenderer]`
   (`tinygrad/runtime/ops_amd.py:1026`); the active default is **HIPRenderer** (HIP-C → comgr → LLVM). `DEV=AMD:LLVM`
   selects `AMDLLVMRenderer`, which renders **LLVM IR** (verified: a trivial matmul emits `define amdgpu_kernel void
   @...` and goes through LLVM's MachineScheduler — *not* the bypass path). `HIPCCRenderer` is C → hipcc → LLVM.
2. **The block tile doesn't even compile on the one non-C AMD path.** `DEV=AMD:LLVM` on the block tile →
   `RuntimeError: failed to render Ops.CUSTOMI ... [float, half.vec(2), half.vec(2)]` — the `fdot2` (`v_dot2`)
   builtin is a HIP-C intrinsic the LLVM-IR renderer can't express.
3. **`Ops.INS` is produced only by `tinygrad/renderer/isa/x86.py`** — a native ISA backend (UOp→`Ops.INS`→assemble,
   bypassing LLVM) exists **for x86 only**. `ls tinygrad/renderer/isa/` = `{x86.py}`. There is no `isa/amd.py`.
4. **The direct-assembly substrate serves hand-emitted assembly, not generated UOps.** `assemble_linear`,
   `tinygrad/renderer/amd/schedule.py` (latency-class / wait-group / LDS-stage *metadata*), and the dormant
   `extra/qk_asm_scheduler.py` (reg def/use DAG over `list[Inst]`) all operate on the prefill GEMM's hand-emitted
   `build_gemm_lds2` `Inst` stream — there is no path from the generated block tile's UOps to `Ops.INS`.

## Combined scheduler conclusion (Arm A + Arm B)
- **Arm A** (UOp reorder in `linearize`, `docs/decode-codegen-scheduler-arm-a-result-20260628.md`): wirable but
  **LLVM owns the schedule** — every reorder stays in LLVM's 42–52 waitcnt envelope, never owned's 21.
- **Arm B** (bypass LLVM via tinygrad's own assembly): **no native AMD ISA backend exists** to bypass LLVM for
  generated kernels. The substrate (assemble_linear + schedule metadata + reg-DAG scheduler + the `isa/x86.py`
  template) is the start, but the AMD instruction-selection backend itself is the major missing piece.

**The precisely-characterized abstraction limit:** closing decode attention to owned-quality via pure machine search
requires a **native AMD/rdna3 ISA backend in tinygrad** (UOp→`Ops.INS` instruction selection + register allocation
for the tile's ops — `fdot2`, `ds_bpermute`, LDS staging, online-softmax — then a latency scheduler on that stream).
That is the *only* lever that owns the instruction schedule instead of handing source to LLVM. It is a major,
multi-increment build (the x86 backend is the template; the prefill `qk_asm_scheduler` increments are the scheduler
precedent), out of scope for a single pass. Until it exists, LLVM's scheduling envelope is the ceiling for the
generated decode tile (35.0/6.7 = 33.7%/7.1% of owned), and the owned hand-ASM tile remains the shipped default.

## Path forward (scoped, not built)
Build a native AMD ISA backend incrementally, templated on `tinygrad/renderer/isa/x86.py`:
1. UOp→`Ops.INS` instruction selection for the block tile's op set (start: the hot-loop ops — `fdot2`,
   `ds_bpermute`, LDS load/store, the online-softmax ALU), with register allocation.
2. Prove it byte-faithful (identity-schedule reproduces a correct kernel) — the `qk_asm_scheduler` inc-0 pattern.
3. A latency/modulo scheduler on the `Inst` stream (consumer-only `s_waitcnt`, load/reduce interleave) — mature
   `extra/qk_asm_scheduler.py` + `tinygrad/renderer/amd/schedule.py` metadata onto it.
4. Gate: `assemble_linear` produces a correct block tile (microgate) → waitcnt drops toward owned's 21 → route-bound
   W==D rises from 35.0/6.7 → generality on prefill GEMM (already partly served by the same substrate).

This is the same foundation the perf-state names for prefill GEMM's residual — one backend, both kernels.
