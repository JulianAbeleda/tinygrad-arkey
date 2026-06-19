# Scope - prefill Tensile primitive extraction and codegen transfer

This scopes the post-EBT-1 path: extract the mature backend primitive, including its launch contract, then decide
whether the recovered schedule should remain an external HSACO artifact or become a tinygrad codegen target.

EBT-1 closed the easy route. The HIP runtime cannot coexist in-process with tinygrad `DEV=AMD` HCQ/KFD, so
rocBLAS/hipBLASLt cannot be called directly on tinygrad buffers. The only remaining way to use the mature backend
without a separate HIP process is to extract the compiled Tensile primitive and launch it through tinygrad HCQ.

## What "extract the primitive" means

The primitive is not just a `.hsaco` file. The full contract is:

- selected solution for a specific GEMM shape/layout;
- code object path and kernel symbol;
- kernel descriptor (`.kd`) and resource metadata;
- exact kernarg byte layout, including hidden dispatch fields;
- global/local launch geometry;
- alpha/beta, strides, transposes, leading dimensions, and pointer ordering;
- workspace and any post-GSU/fixup kernels;
- artifact version and fallback policy;
- correctness and in-model transfer gates.

If any one of those is missing, this is not a runnable primitive. It is only a fast library measurement.

## Current fixed facts

| fact | value |
|---|---|
| Lane A | **KILL**: HIP runtime and tinygrad HCQ/KFD are mutually exclusive in one process (`prefill-external-bridge-ebt1-result-20260619.md`) |
| Lane B through TPE-4 | **PASS for ffn_gate/up fixed shape**: extracted rocBLAS Tensile primitive launches through HCQ at 66.91 TFLOPS median, correct/no-copy/no-HIP (`prefill-tensile-tpe4-perf-result-20260619.md`) |
| target | 8B PREFILL_V2 fp16 matmul bucket, especially pp512/pp1024 |
| tinygrad ceiling | ~40.8-42.0 TFLOPS on ffn_gate/up |
| external ceiling | hipBLASLt 69.8 TFLOPS ffn_gate/up; rocBLAS 70.9 ffn_down; rocBLAS 76.7 attn_q/o |
| Amdahl upper bound | ~1.4-1.45x full pp before extraction/routing overhead |
| route posture | research-only, no default, no decode changes |
| artifact policy | must be explicitly accepted: generated/installed ROCm Tensile artifacts are an external dependency |

Local ROCm evidence:

- `/opt/rocm-7.2.4/lib/rocblas/library/Kernels.so-000-gfx1100.hsaco` is a loadable AMDGPU ELF with many `Cijk_*`
  symbols and `.kd` descriptors.
- `/opt/rocm-7.2.4/lib/hipblaslt/library/Kernels.so-000-gfx1100.hsaco` is also a loadable AMDGPU ELF with many
  `Cijk_*` symbols and `.kd` descriptors.
- rocBLAS has `TensileLibrary_Type_HH_HPA_Contraction_l_*_gfx1100.{dat,co}` files for fp16 input/fp32 accumulate
  GEMM variants.
- hipBLASLt has `TensileLibrary_HH_HH_HA_*_Type_HH_HPA_Contraction_l_*_gfx1100.{dat,co}` files for the same family,
  including bias/aux-capable variants.
- The `.dat` files expose solution names, predicates, macro-tile/workgroup fields, and kernel-name fields for at
  least hipBLASLt. The `.co` files report as `data` locally; the loadable code object is the `Kernels.so-000-gfx1100.hsaco`.
- HSA metadata in the HSACO includes `.args`, kernarg segment sizes, hidden block counts, hidden group sizes,
  hidden remainders, and hidden global offsets.

tinygrad runtime constraint:

- `AMDProgram` already parses HSACO and dispatches through HCQ.
- But current `AMDProgram` uses the first `.rodata` kernel descriptor; it does not resolve the descriptor by the
  requested `name`. Loading a multi-kernel Tensile `Kernels.so` through `AMDProgram` is therefore not enough.
- Lane B needs either a single-kernel extracted HSACO or a tinygrad-side named-symbol/named-descriptor loader.

## Arc 1 - Lane B: Tensile HSACO extraction through HCQ

### TPE-0 - authority lock

Before build work, decide:

- allowed artifacts: installed ROCm files only, copied/pinned HSACO+metadata, generated artifacts, or none;
- backend family priority: hipBLASLt first for ffn_gate/up, rocBLAS first for ffn_down and attn_q/o, or one unified
  source;
- route posture: standalone probe only, one-block transfer, or full `PREFILL_TENSILE_GEMM=1` research flag;
- fallback: unsupported artifact/shape/device returns to PREFILL_V2 silently;
- acceptance gate: research >=1.25x full pp512, strong >=1.35x pp512/pp1024, no default.

Gate: explicit artifact/dependency policy. Kill: if external HSACO artifacts are unacceptable, Lane B stops here.

### TPE-1 - solution selection and trace discovery

Goal: identify the exact solution and kernel symbol the HIP-only oracle selected for each target shape.

Work:

- extend or wrap `extra/qk_prefill_blas_ceiling.cpp` to emit selection metadata;
- run HIP-only, not tinygrad, because HIP and HCQ cannot coexist;
- collect rocBLAS logs/traces where available;
- use `rocblas_gemm_ex_get_solutions` / `rocblas_gemm_algo_solution_index` when possible to enumerate and force
  candidate solution indices;
- collect hipBLASLt heuristic records for hipBLASLt-selected shapes;
- use code-object tracing/profiling if logs do not expose symbol names;
- record code object path, solution index/algo, kernel symbol, achieved ms/TFLOPS, and any auxiliary kernels.

Targets:

| role | shape | expected best source |
|---|---|---|
| ffn_gate/up | 512 x 4096 -> 12288 | hipBLASLt from PXB-1 |
| ffn_down | 512 x 12288 -> 4096 | rocBLAS from PXB-1 |
| attn_q/o | 512 x 4096 -> 4096 | rocBLAS from PXB-1 |
| attn_k/v | 512 x 4096 -> 1024 | likely low EV; include only to close matrix |

Artifact: `bench/qk-tensile-extraction/selection.json`.

Gate:

- ffn_gate/up selected solution is known: library family, solution index/algo if available, code object, kernel symbol;
- standalone timing remains within 5% of PXB-1;
- auxiliary kernels are identified if present.

Kill:

- selected solution cannot be observed;
- the fast path is not a stable kernel symbol;
- performance depends on multiple opaque runtime-selected kernels that cannot be mapped.

### TPE-2 - launch-contract extraction

Goal: produce a machine-readable contract for one selected ffn_gate/up kernel.

Work:

- parse the relevant `.dat` solution entry for macro-tile, depthU, workgroup, GSU/StreamK, predicate constraints,
  and kernel-name mapping;
- parse HSACO ELF symbols to find the function and `.kd` descriptor;
- parse AMDGPU metadata to recover kernarg offsets, sizes, pointer/value kinds, required hidden args, group/private
  segment sizes, wavefront size, and kernarg segment alignment;
- determine exact pointer order and value order by matching metadata, `KernelArguments` conventions, and/or traced
  runtime kernargs;
- determine global/local launch geometry for the fixed shape;
- determine whether workspace is unused, caller-provided, or required for GSU/post-processing.

Important implementation note:

- Current `AMDProgram` must not be trusted to select the desired Tensile kernel out of a multi-kernel HSACO. TPE-2
  must produce either:
  - a single-kernel HSACO/code-object slice, or
  - a tinygrad helper that resolves `kernel_symbol.kd` and `kernel_symbol` by name and sets `aql_prog_addr`,
    `prog_addr`, segment sizes, and resource registers from that descriptor.

Artifact: `bench/qk-tensile-extraction/ffn_gate_up_contract.json`.

Gate:

- contract includes kernel symbol, descriptor symbol, code object hash/path, kernarg layout, launch geometry,
  workspace contract, and all static shape/layout assumptions;
- contract is reproducible from installed files and does not require HIP runtime in the tinygrad process.

Kill:

- kernarg layout is not recoverable;
- hidden args are runtime-generated in an opaque way;
- selected kernel requires private runtime state rather than only caller buffers/scalars/workspace.

### TPE-3 - minimal HCQ launch proof

Goal: launch the selected ffn_gate/up Tensile kernel from tinygrad HCQ on tinygrad-owned buffers.

Work:

- build `extra/qk_tensile_hcq_launch.py` as a standalone probe;
- allocate A/B/C as tinygrad AMD tensors/buffers;
- load the selected HSACO using a named-descriptor loader or single-kernel extraction;
- fill raw kernargs exactly from the TPE-2 contract;
- use conservative HCQ synchronization and `wait=True` timing;
- compare output to tinygrad fp16 oracle.

Likely helper surface:

- `extra/qk_tensile_contract.py` for `.dat` + HSACO metadata parsing;
- `extra/qk_tensile_hcq_launch.py` for the first launch;
- possibly `extra/qk_tensile_named_program.py` or a tightly scoped runtime helper if `AMDProgram` needs named
  descriptor support.

Gate:

- kernel runs under `DEV=AMD` without HIP runtime loaded;
- output is correct within fp16/fp32 tolerance;
- no crash/hang/corruption after repeated runs;
- launch uses tinygrad-owned buffers with no copies.

Kill:

- named descriptor cannot be loaded;
- kernel requires HIP runtime services;
- raw kernargs fail correctness after a bounded debug pass;
- workspace/post-GSU flow requires opaque runtime orchestration.

### TPE-4 - isolated performance parity

Goal: prove the extracted primitive keeps most of the mature backend speed when launched through HCQ.

Measure:

- ffn_gate/up device time, TFLOPS, and percent of PXB-1 standalone;
- warmup sensitivity;
- repeated-run stability;
- PMU/profile sanity if available;
- compare to tinygrad POWN-1 and PXB-1.

Gate:

- ffn_gate/up reaches >=90% of the PXB-1 HIP-only time, or at minimum >=62 TFLOPS;
- if below 62 TFLOPS, explain whether the loss is launch geometry, wrong kernel, workspace/GSU, or metadata mismatch.

Kill:

- extracted launch falls near the 42 TFLOPS tinygrad plateau;
- performance requires a host-side multi-kernel selection path that cannot be represented in HCQ.

### TPE-5 - shape matrix  — STATUS: DONE, PASS (2026-06-19, `prefill-tensile-tpe5-shape-matrix-result-20260619.md`)

Result: all 3 high-share roles launch correct/stable/no-workspace through HCQ from one `Ailk_Bljk` code object with
one pointer convention — ffn_gate/up 66.8, ffn_down 68.9 (StreamK, no workspace), attn_q/o 58.9 TFLOPS. Weighted
model **~1.40× full pp512** (~95% llama). Gates met (all correct/stable, no layout copies, no workspace, ≥1.25×); no
kill condition triggered. → proceed to TPE-6.

Goal: decide whether one extracted primitive is enough or every role needs its own contract.

Work:

- repeat TPE-1 through TPE-4 for ffn_down and attn_q/o only if ffn_gate/up passes;
- include attn_k/v only as a low-EV closure row unless the weighted model says it matters;
- compute weighted matmul-bucket model after actual extracted timings.

Artifact: `bench/qk-tensile-extraction/shape_matrix.json`.

Gate:

- weighted extracted matmul model predicts >=1.25x full warm pp512;
- no role needs layout copies or separate HIP preprocessing;
- contracts are stable across repeated runs.

Kill:

- only ffn_gate/up works and total pp upside is below ~1.15x;
- each role requires a substantially different opaque contract that is not maintainable.

### TPE-6 - one-block transfer

Goal: prove the primitive survives a real prefill block before touching full model routing.

Work:

- route only one block or one-layer harness behind a research flag;
- use PREFILL_V2 realized fp16 weights;
- keep decode untouched;
- compare one-block outputs and timing to PREFILL_V2;
- verify no compile/recompile storm and no extra realizes/copies.

Gate:

- selected block improves by >=1.20x after all routing overhead;
- fp16 oracle tolerance passes;
- fallback is clean.

Kill:

- isolated speed does not transfer to block timing;
- model layout forces transposes/copies;
- TinyJit/HCQGraph cannot represent the call without material overhead.

### TPE-7 - full in-model research route

Goal: optional, only after TPE-6 passes.

Work:

- route eligible prefill matmuls behind `PREFILL_TENSILE_GEMM=1`;
- measure warm pp512/pp1024;
- run dNLL <=0.01 gate;
- verify decode ctx sweep unchanged;
- verify fallback when artifacts are missing or device is not gfx1100/compatible.

Gates:

- research pass: >=1.25x warm pp512;
- strong pass: >=1.35x warm pp512 and pp1024;
- no default without separate policy review.

Kill:

- full pp gain <1.15x;
- fallback or quality fails;
- artifact coupling is too brittle for the measured gain.

## Arc 2 - tinygrad codegen transfer from the extracted contract

This is option 2 from the research synthesis: teach tinygrad to express the same Tensile-class schedule. It should
not start as a build until Arc 1 recovers a working contract and reaches >=62 TFLOPS under HCQ. Otherwise there is
no proven local target to imitate.

### TCG-0 - schedule anatomy from the selected primitive

Inputs:

- TPE-2 contract;
- selected `.dat` solution fields;
- disassembly of the selected `Cijk_*` kernel;
- POWN-1 failed configs and timings.

Work:

- extract macro tile, workgroup, MFMA shape, depthU, GSU/StreamK, vector widths, global-read pattern, LDS layout,
  local read/write schedule, prefetch, number of independent accumulators, VGPR/SGPR counts, and epilogue stores;
- compare each item against tinygrad's POWN-1 kernel shape;
- label the delta as frontend schedule, UOp expressibility, renderer instruction scheduling, register allocation,
  or runtime launch contract.

Gate:

- produces a table of "Tensile does X, tinygrad currently does Y, missing capability is Z" for the selected shape.

Kill:

- selected primitive cannot be disassembled or mapped to schedule concepts.

### TCG-1 - minimal codegen capability list

Expected missing capabilities to audit:

- enough independent WMMA/MFMA accumulators to hide latency;
- software-pipelined global/LDS load overlap with MFMA issue;
- stable register allocation for large accumulator tiles;
- descriptor-aware/raw-kernarg launch when the kernel is not generated by tinygrad;
- vectorized epilogue stores without layout copies;
- optional GSU/post-GSU support if selected solutions use it.

Gate:

- identify the smallest tinygrad codegen change that could plausibly move 42 TFLOPS toward >=62 TFLOPS;
- if the smallest change is "rewrite the AMD renderer/scheduler," mark as project-level, not a prefill arc.

### TCG-2 - proof kernel, not model route

Only if TCG-1 names a bounded capability:

- build a single-shape tinygrad-generated or assembly-assisted GEMM for ffn_gate/up;
- require >=62 TFLOPS isolated before any model work;
- compare instruction schedule and occupancy to the extracted Tensile primitive.

Kill:

- proof kernel stays near 42 TFLOPS;
- it requires hand-maintained per-shape assembly equivalent to Lane C;
- it duplicates the extracted HSACO route with worse maintainability.

## Decision tree

| result | next action |
|---|---|
| TPE-0 rejects artifacts | rest at PREFILL_V2; do not start codegen transfer |
| TPE-1 cannot observe selected solution | Lane B blocked; only project-level codegen remains |
| TPE-2 cannot recover kernarg/launch contract | Lane B blocked; codegen transfer may still use disassembly if available |
| TPE-3 launches correctly but slow | debug contract; if still <62 TFLOPS, close Lane B as non-transferable |
| TPE-4 reaches >=62 TFLOPS | continue to shape matrix |
| TPE-5 predicts <1.15x full pp | bank as isolated success, no model route |
| TPE-6 transfers to one block | consider full research route |
| TPE succeeds but artifact policy is unacceptable | use TCG scope to decide if tinygrad codegen can absorb the schedule |

## What not to do

- Do not reintroduce HIP runtime in the tinygrad process.
- Do not use a separate HIP process plus copies and call it a pass.
- Do not route anything by default.
- Do not touch decode.
- Do not reopen LDS tiling, bigger tiles, more waves, BK32/BK64, or noLDS as standalone knobs.
- Do not treat the full `Kernels.so` as usable until named-descriptor selection is proven.
- Do not begin a tinygrad codegen rewrite until an extracted primitive proves the target schedule transfers under HCQ.

## Recommended next step

Proceed with TPE-1 only if the project accepts a research-only Tensile artifact dependency. The first deliverable is
`selection.json`: for ffn_gate/up, identify the exact hipBLASLt/rocBLAS solution, code object, kernel symbol, and
standalone timing. If that cannot be made explicit, Lane B should be closed before any HCQ launcher work.
