# Codex brief — native AMD ISA backend, with LLVM's AMDGPU backend as the solved map

You (Codex) turn this into the executable, staged prompt; I (Claude Code) build + GPU-verify it increment by
increment. **Core directive: do NOT design an AMD backend from scratch. LLVM's AMDGPU (AMDGCN) backend already solves
isel + regalloc + spill + LDS mapping + waitcnt + scheduling — and we have that exact LLVM installed
(`/opt/rocm-7.2.4/llvm/bin/llc`, AMD LLVM 22.0.0git, target `amdgcn`/gfx1100, the one comgr uses).** Treat LLVM as
the reference model: audit it, extract a working model per dimension, then translate that model into tinygrad's
`ISARenderer` framework. The only place we intend to *beat* LLVM is the scheduler (owned hand-ASM hits
`s_waitcnt`=21 where LLVM's MachineScheduler plateaus at 42–52 — measured, `decode-codegen-scheduler-arm-a-result-20260628.md`).

## 1. Goal + why (one paragraph)
Pure machine search for AMD is capped because the machine hands the final mile (isel/regalloc/**scheduling**) to LLVM,
whose scheduling envelope limits the generated decode tile to 33.7%/7.1% of owned. A native UOp→`Ops.INS`→
`assemble_linear` backend gives tinygrad the whole pipeline (no LLVM), making the schedule a machine-owned, searchable
decision → generated kernels can reach hand-ASM quality → hand kernels retire. See
`docs/amd-isa-backend-scope-and-enablement-20260628.md`.

## 2. PHASE 0 — audit LLVM's AMDGPU backend (DO THIS FIRST; we have no working model yet)
We have observed LLVM's *output* (disasm) but have NOT extracted its *model*. Build that model first — it is the map.
Method (we own the exact LLVM): compile a representative kernel (a trivial elementwise, a GEMV, and the block tile's
HIP-C) and dump LLVM's per-pass decisions:
- `clang -cc1 ... -triple amdgcn-amd-amdhsa -target-cpu gfx1100 -S -emit-llvm` → LLVM IR; then
  `llc -mtriple=amdgcn -mcpu=gfx1100 -print-after-all -O3 in.ll` to dump every MachineFunction pass.
- Targeted pass debug (release+asserts permitting; else `-print-after=<pass>`): `-mllvm -debug-only=si-insert-waitcnts`,
  `-print-after=si-insert-waitcnts`, `-print-after=machine-scheduler`, `-print-after=greedy`,
  `-print-after=si-lower-control-flow`, `-print-after=si-memory-legalizer`.
- `llc -print-after-isel` for the selected DAG; `llvm-mc -arch=amdgcn -mcpu=gfx1100 -show-encoding` to confirm
  encodings match `tinygrad.runtime.autogen.amd.rdna3.ins`.

Produce a **working-model artifact** (`bench/amd-llvm-backend-model/latest.json` + a doc) covering EACH dimension —
this is the map every later increment is built from. **If a dimension below is already audited elsewhere in the repo,
cite it and skip; otherwise audit it here.**

| dimension | what LLVM does (the solved model to extract) | how to audit it (this LLVM) | LLVM source (the map) |
|---|---|---|---|
| **isel coverage for real UOps** | SelectionDAG/GlobalISel patterns: each IR op → AMDGCN inst (`v_add_f32`, `v_fmac`, `v_cvt`, `v_mul`, etc.) | `llc -print-after-isel`; enumerate the UOp→inst map tinygrad needs | `lib/Target/AMDGPU/SIInstructions.td`, `VOP{1,2,3}Instructions.td`, `AMDGPUInstructionSelector.cpp`, `AMDGPUISelDAGToDAG.cpp` |
| **register allocation quality** | Greedy RA over VGPR/SGPR classes; occupancy-aware (VGPR count gates waves) | `-print-after=greedy`; record vgpr/sgpr counts vs owned 64 | `SIRegisterInfo.cpp`, `GCNRegPressure.cpp`, `SIMachineFunctionInfo` |
| **spill/fill/copy correctness** | spill to scratch (`scratch_store/load`), SGPR→VGPR spills, cross-class copies | force high pressure; capture spill code + frame layout | `SIRegisterInfo::spillSGPR/eliminateFrameIndex`, `PrologEpilogInserter` |
| **LDS / threadgroup / shared-memory mapping** | `__shared__`→LDS: `ds_load/ds_store` + group_segment_size in the kernel descriptor; address calc | a `__shared__` kernel; capture ds ops + descriptor lds size | `lib/Target/AMDGPU/` LDS lowering, `AMDGPUMachineFunction::allocateLDSGlobal` |
| **v_dot2 / ds_bpermute / barriers / waitcnt semantics** | intrinsics→`v_dot2acc_f32_f16`, `ds_bpermute_b32`, `s_barrier`; **`s_waitcnt` counter algorithm** (vmcnt/lgkmcnt/expcnt tracking) | `__builtin_amdgcn_fdot2`/`ds_bpermute` kernels; **`-debug-only=si-insert-waitcnts`** to extract the waitcnt insertion algorithm | **`SIInsertWaitcnts.cpp`** (THE waitcnt model), `SIMemoryLegalizer.cpp` (barriers/fences), AMD RDNA3 ISA Ref (counter semantics) |
| **scheduler maturity** | GCN MachineScheduler (region list-sched + occupancy targeting + hazard recognizer) — the part we intend to BEAT | `-print-after=machine-scheduler`; diff its waitcnt schedule vs owned (42–52 vs 21) to localize where it loses | `GCNSchedStrategy.cpp`, `GCNIterativeScheduler.cpp`, `GCNHazardRecognizer.cpp` |

Comparative anchor: we already have owned hand-ASM disasm
(`bench/qk-decode-attention-isa-diff/disasm_owned_flash_tile_gqa_whole.txt`) and LLVM's block-tile disasm — diff
LLVM-model vs owned to mark exactly where LLVM is *correct-but-suboptimal* (scheduling) vs *correct* (isel/regalloc).

## 3. What we already have (verified — do NOT rebuild; build on it)
- **Framework**: `tinygrad/renderer/isa/__init__.py` (`ISARenderer` + `IselContext` + regalloc plumbing in
  `codegen/__init__.py:210`). **Template**: `tinygrad/renderer/isa/x86.py` (905 lines, the worked example).
- **AMD substrate**: `renderer/amd/dsl.py` (`Reg` model), `runtime.autogen.amd.rdna3.ins` (1357 encodable insts),
  `renderer/amd/elf.py:assemble_linear` (Ops.INS→runnable ELF), `renderer/amd/schedule.py` (latency metadata),
  `extra/qk_asm_scheduler.py` (reg def/use DAG + reorder).
- **Foundation VERIFIED on gfx1100** (`qk_asm_scheduler_inc0_test` all-pass): assemble is byte-faithful, the
  assembled kernel **runs correct (P5)**, and a dependency-respecting `Inst`-reorder **still runs correct (P6)**. So
  the back half (assemble + reorder + scheduler substrate) is proven; PHASE 0 + the build supply the front half.

## 4. Staged build (each gated; grounded in the PHASE-0 map)
- **Inc 0** — new `tinygrad/renderer/isa/amd.py` `AMDISARenderer(ISARenderer)`: isel for the minimal op set
  (DEFINE_GLOBAL/index/LOAD/STORE, ADD/MUL, SPECIAL workitem) translated from the LLVM isel map; `copy/spill/fill/
  asm_str` from the LLVM spill/copy model; wired `DEV=AMD:ISA`. Gate: trivial kernel runs numerically correct via
  `assemble_linear`.
- **Inc 1** — op coverage (casts, sub/div/max/exp2/fma, cmp/select, RANGE/END reduce, gated load/store) from the LLVM
  isel map. Gate: small GEMV correct.
- **Inc 2** — block-tile ops: `v_dot2`, `ds_bpermute`, LDS staging (DEFINE_LOCAL→ds), barriers, **waitcnt insertion
  ported from `SIInsertWaitcnts` model**. Gate: block tile compiles via the backend + `BLOCK_TILE_MICROGATE_PASS`.
- **Inc 3** — scheduler: mature `qk_asm_scheduler` + `renderer/amd/schedule.py` into a latency/modulo scheduler that
  **beats LLVM's MachineScheduler on the recurrence** (the localized weakness from PHASE-0). Gate: block-tile
  `s_waitcnt` drops toward 21; route-bound W==D rises from 35.0/6.7.
- **Inc 4 — BubbleBeam/search binding** (scope it now): lift the searchable decisions — schedule (pipeline depth,
  list-sched priority), regalloc (occupancy vs spill tradeoff), waitcnt placement — into BubbleBeam/FutureSight as a
  candidate space, with the route-bound W==D + token-match gates as the evaluator. Generality proof: the same backend
  + scheduler moves the prefill GEMM. This is where "the machine schedules competitive kernels" actually lands.

## 5. Gates / constraints
Default-off opt-in renderer (`DEV=AMD:ISA`); shipped default (HIPRenderer) + owned route byte-identical until a gate
promotes; correctness-first per increment (microgate token-match, then route-bound W==D — never promote on isolated
timing); do not edit `tinygrad/runtime/autogen/**`; bracketed-prefix commits; if an increment proves an LLVM model
component is not portable to the `Inst`/assemble path, document it as a precise blocker rather than forcing it.

## 6. Sources (verify live; do not fabricate)
- **LLVM AMDGPU User Guide** — https://llvm.org/docs/AMDGPUUsage.html (ABI, kernel descriptor, calling conv, target features).
- **LLVM AMDGPU backend source** (`llvm-project/llvm/lib/Target/AMDGPU/`): `SIInsertWaitcnts.cpp` (waitcnt counter
  model — the key one), `GCNSchedStrategy.cpp` / `GCNIterativeScheduler.cpp` / `GCNHazardRecognizer.cpp` (scheduling),
  `AMDGPUInstructionSelector.cpp` / `AMDGPUISelDAGToDAG.cpp` + `SIInstructions.td` / `VOP*Instructions.td` /
  `DSInstructions.td` / `FLATInstructions.td` (isel), `SIRegisterInfo.cpp` (regalloc/spill), `SIMemoryLegalizer.cpp`
  (barriers/fences). The installed `/opt/rocm-7.2.4/llvm` is AMD's fork of this.
- **AMD RDNA3 Instruction Set Architecture Reference Guide**, Feb 2023 — https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf (encodings + s_waitcnt vmcnt/lgkmcnt/expcnt semantics).
- **LLVM MachineScheduler** — https://llvm.org/docs/CodeGenerator.html#the-machine-instruction-scheduler (the list-scheduling framework GCNSchedStrategy plugs into).
- **tinygrad precedent** (in-repo): `tinygrad/renderer/isa/x86.py`, `tinygrad/renderer/isa/__init__.py`,
  `tinygrad/renderer/amd/{dsl,elf,schedule}.py`, `extra/qk_asm_scheduler.py` (+ inc0–3 tests, foundation verified).
- **Scheduling theory anchor** (from the v2 scope, for inc-3): Lam PLDI'88 + Rau MICRO-27'94 (modulo scheduling),
  Gibbons-Muchnick '86 (list scheduling) — see `docs/decode-codegen-scheduler-capability-scope-v2-references-20260627.md`.
