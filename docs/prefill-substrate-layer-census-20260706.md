# Layer Census: how low we can go, and what's covered at each layer

Date: 2026-07-06. The full lowering stack for the pure-generated prefill WMMA GEMM (gfx1100/RDNA3), with
per-layer coverage (file:line) and the agnostic compiler/GPU-arch reference that grounds each layer.

## Framing (agnostic)
The stack is **progressive lowering across abstraction levels** — the multi-level-IR design ([MLIR](https://en.wikipedia.org/wiki/MLIR_(software)),
[progressive raising](https://grosser.science/static/7d02fb58ecc49e4d2097d11bc9e8152a/chelini-2021-abstraction-raising.pdf)):
each step "translates constructs from one level into equivalent, more detailed constructs at the next level down."
A compiler back-end's canonical phases are **instruction selection → scheduling → register allocation → code emission**
([Writing an LLVM Backend](https://llvm.org/docs/WritingAnLLVMBackend.html)). Our stack realizes exactly this.

## The 9 layers, coverage, and grounding reference

| # | Layer | Our realization | Coverage | Owns sched/waitcnt? | Agnostic ref |
|---|---|---|---|---|---|
| 1 | Tensor / high-level IR | `tensor.py`, `uop/` (34 `Ops`) | complete | — | [MLIR](https://en.wikipedia.org/wiki/MLIR_(software)) |
| 2 | Schedule / tiling | `schedule/` (rangeify, bufferize, wmma), `codegen/opt` | complete for tiling; DBUF *shape* scaffolded | — | [CUTLASS efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html) |
| 3 | Instruction selection | HIP/LLVM renderers (complete); `AMDISARenderer` isel (44 ops) | ISA path **partial — no WMMA** | — | [LLVM backend](https://llvm.org/docs/WritingAnLLVMBackend.html) |
| 4 | Instruction scheduling / **software pipelining** | HIP→**LLVM** (`SI Machine Scheduler`); ISA→`_schedule` list sched | HIP: not ours. ISA: list-sched, no modulo/pipeline | **decisive** | [modulo scheduling (Rau)](https://dl.acm.org/doi/10.1145/192724.192731), [instr. scheduling](https://en.wikipedia.org/wiki/Instruction_scheduling) |
| 5 | Register allocation | HIP→LLVM; ISA→`codegen/late/regalloc.py` + isa pools | ISA: single-VGPR linear-scan, **no consecutive-VGPR fragments, no spill** | — | [LLVM backend](https://llvm.org/docs/WritingAnLLVMBackend.html) |
| 6 | Async-memory sync (**waitcnt**) | HIP→**LLVM `SIInsertWaitcnts`**; ISA→`_insert_waitcnt` | HIP: not ours. ISA: **full-drain only, no `vmcnt(n)`** | **decisive** | [SIInsertWaitcnts](https://llvm.org/doxygen/SIInsertWaitcnts_8cpp.html), [RDNA3 ISA](https://gpuopen.com/news/rdna3-isa-guide-now-available/) |
| 7 | Tensor-core instruction | `codegen/opt/tc.py`, `schedule/wmma.py`; HIP/LLVM emit WMMA | codegen+HIP/LLVM: complete. **ISA renderer: absent** | — | [WMMA on RDNA3](https://gpuopen.com/learn/wmma_on_rdna3/), [RDNA3 ISA §7.9](https://www.techpowerup.com/gpu-specs/docs/amd-rdna3-isa.pdf) |
| 8 | Assembly / object (ELF) emission | `renderer/amd/elf.py::assemble_linear`, `dsl.py`; autogen `rdna3/ins.py` | **COMPLETE** (encoding + ELF pack) | — | [LLVM backend (code emission)](https://llvm.org/docs/WritingAnLLVMBackend.html) |
| 9 | Hardware | gfx1100 RDNA3 (WMMA units, VGPR/LDS, vmcnt/lgkmcnt) | — | — | [RDNA3 ISA Reference](https://www.techpowerup.com/gpu-specs/docs/amd-rdna3-isa.pdf) |

## Layer detail (condensed from exhaustive census)

### Layer 8-9 — the FLOOR is COMPLETE (proven by the hand kernel end-to-end)
- `assemble_linear` (`elf.py:15`) takes a `LINEAR` of already-final `Ops.INS`, does **no** sched/regalloc/waitcnt —
  only scans registers for the descriptor (`elf.py:18-35`), sizes LDS from the sink (`elf.py:60-76`), packs ELF
  bytes (`elf.py:77,119-145`). Encoding lives in `dsl.py:416` (`Inst.to_bytes`).
- Autogen `rdna3/ins.py` has EVERY instruction the hand kernel uses: `v_wmma_f32_16x16x16_f16` (+5 variants, `:1764-1769`),
  `ds_load/store_b128` (`:570-572`), `global_load_b128` (`:634`), `global_store_b16` (`:636`), `s_waitcnt` full simm16
  (`:1138`), `s_delay_alu` (`:1136`), `s_barrier` (`:1168`). Encoding gap = NONE.
- Runtime `ops_amd.py::AMDProgram` (`:576`) launches any ELF regardless of provenance (LLVM/ISA/hand identical).
- Proof: `extra/qk/prefill/wmma.py` builds `Inst` lists → `assemble_linear` → ELF → launches, bit-exact.

### Layer 3/5/7 — the ISA renderer (`renderer/isa/amd.py`) is PARTIAL (this is where Track B lives)
44 `AMDOps` (`:57-83`) cover: scalar/uniform int (`S_IMUL/IADD/ISHL/WGID`), VALU (`V_ADD/MUL/SUB/cvt/cmp/where/exp/dot2`),
LDS `DS_LOAD/STORE` (b16/b32 only), `GLOBAL_LOAD/STORE` (b32 only, vec scalarized), index math, `WG_ID/WI_ID`,
`BARRIER`, `GATED_STORE`, `ACCUM_READ/WRITE` (pinned), `DS_BPERMUTE`. Regalloc: single-VGPR linear-scan + physical
pools (`:33-42`), **no spill** (`:625-629`), VGPR≥238 trap only avoided-by-construction for pinned accumulators (no
active guard). Scheduling: `_schedule` list scheduler (`:676`, default-on). Waitcnt: `_insert_waitcnt` (`:747-792`)
**full-drain `s_waitcnt(0)` only** — no targeted `vmcnt(n)`.
**MISSING vs hand kernel: (1) WMMA entirely (no op, no isel, no fragment regalloc); (2) wide mem b128/b64;
(3) targeted waitcnt; (4) `s_delay_alu`; (5) VGPR≥238 guard.** Verified foundation: `qk_asm_scheduler_inc0_test`.

### Layer 2/7 — codegen owns the WMMA MATH + tiling, delegates the rest
- 11 `OptOps` (`opt/__init__.py:6-9`): TC, UPCAST, UNROLL, LOCAL, THREAD, GROUP, GROUPTOP, NOLOCALS, PADTO, SWAP,
  COALESCE(stub). TC→WMMA lowering `_apply_tc_opt` (`postrange.py:228-328`) covers gfx1100 16×16×16 (half/bf16/iu8),
  fragment CONTRACT/UPCAST/UNROLL reshaping + warp-lane swizzle (`tc.py:140-147`, LaneMap `tc.py:7-109`), ragged
  PADTO. **GROUP+TC forbidden** (`postrange.py:177`) → stock OptOps can't lower segmented MMQ.
- DBUF scaffold (default-off, `PREFILL_DBUF`): storage NBUF=2 (`rangeify.py:447-457`), paired `(k&1)` offset
  (`postrange.py:466-481`), K-peel via extra UNROLL(2) avoiding the two-END cycle (`postrange.py:521-534`).
  **1d waitcnt is explicitly the renderer residual** (`rangeify.py:25`).
- Boundary: codegen DECIDES tiling/WMMA/ordering (priority toposort `linearizer.py:8-48`); DELEGATES regalloc +
  instruction scheduling + waitcnt to the renderer/LLVM. The `SCHED_MODULO` probes (`linearizer.py:54-128`) conclude
  LLVM re-schedules from the dep DAG and ignores UOp order on the HIP path ("SCHEDULER_NOT_WIRABLE").

## Synthesis: how low we can go, and where the gap is

**We can descend all the way to the ELF floor — and the floor + the top are already done.** The blockers to a
generated equivalent of the hand kernel are ALL concentrated in ONE file, `tinygrad/renderer/isa/amd.py`, and map
exactly to three canonical back-end phases ([LLVM backend](https://llvm.org/docs/WritingAnLLVMBackend.html)):

| Canonical phase | Gap in `AMDISARenderer` | Difficulty |
|---|---|---|
| Instruction selection (layer 3) + tensor-core (7) | add `AMDOps.V_WMMA` + isel rule emitting `v_wmma_f32_16x16x16_f16` | medium |
| Register allocation (layer 5) | **consecutive-VGPR fragment allocation** (A=8, B=8, C/D=8 contiguous VGPRs) — allocator is single-VGPR today | **HARD (largest gap)** |
| (memory ops) | `ds_load/store_b128`, `global_load_b128`, `global_store_b16` (currently b32/b16-LDS) | medium |
| Instruction scheduling / software pipelining (layer 4) + waitcnt (6) | **targeted `vmcnt(n)`** partial waits driven by the DBUF load-ahead — the actual latency-hiding | **HARD (the perf lever)** |
| (hazard) | `s_delay_alu(1)` interlock for cvt/FP→VALU; VGPR≥238 guard | low |

Everything **below** (encoding/assembly/runtime, layer 8-9) is complete and proven; everything **above** (WMMA math,
tiling, DBUF shape, layers 1-2/7-codegen) is complete. **Track B == completing `AMDISARenderer` for these 5 items**,
routing only the prefill GEMM to `DEV=AMD:ISA`. This is the standard back-end descent, not a from-scratch backend.

Independent proof the target is reachable on this exact hardware via these exact techniques: [seb-v reaches 50 TFLOPS,
60% over rocBLAS, via ISA-level double-buffering + prefetch on gfx11](https://seb-v.github.io/optimization/update/2025/01/20/Fast-GPU-Matrix-multiplication.html).

## Sources
Progressive lowering: [MLIR (Wikipedia)](https://en.wikipedia.org/wiki/MLIR_(software)), [MLIR Tutorial](https://llvm.org/devmtg/2020-09/slides/MLIR_Tutorial.pdf), [Progressive Raising](https://grosser.science/static/7d02fb58ecc49e4d2097d11bc9e8152a/chelini-2021-abstraction-raising.pdf).
Backend phases: [Writing an LLVM Backend](https://llvm.org/docs/WritingAnLLVMBackend.html).
Scheduling/pipelining: [Instruction scheduling](https://en.wikipedia.org/wiki/Instruction_scheduling), [Iterative Modulo Scheduling (Rau)](https://dl.acm.org/doi/10.1145/192724.192731), [CUTLASS efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html), [Colfax pipelining](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/).
Waitcnt/LLVM AMDGPU: [SIInsertWaitcnts](https://llvm.org/doxygen/SIInsertWaitcnts_8cpp.html), [AMDGPU User Guide](https://llvm.org/docs/AMDGPUUsage.html), [SI Machine Scheduler](https://www.phoronix.com/news/SI-Machine-Scheduler-LLVM).
RDNA3 ISA/WMMA: [RDNA3 ISA Guide](https://gpuopen.com/news/rdna3-isa-guide-now-available/), [RDNA3 ISA PDF](https://www.techpowerup.com/gpu-specs/docs/amd-rdna3-isa.pdf), [WMMA on RDNA3](https://gpuopen.com/learn/wmma_on_rdna3/).
Concrete perf: [seb-v Fast RDNA3 Matmul](https://seb-v.github.io/optimization/update/2025/01/20/Fast-GPU-Matrix-multiplication.html).
</content>
