# AMD GPU Holistic Primitive Model — From HIP to LLVM AMDGPU to AMDGCN ISA (2026-06-23)

## Purpose

This document consolidates what this project has learned about AMD GPU performance primitives, and ties those
learnings to the public ROCm / LLVM / AMDGCN documentation.

The goal is not to document one kernel. The goal is to understand the GPU holistically:

```text
model primitive -> tensor graph / runtime lifecycle -> HIP or tinygrad lowering -> LLVM AMDGPU -> AMDGCN ISA
-> resources / occupancy / memory movement -> whole-decode W==D transfer
```

The project repeatedly learned that isolated kernel speed, source-level intent, and graph identity are not enough.
The real primitive is the full lifecycle above.

## External Reference Map

| Layer | Reference | Why it matters |
|---|---|---|
| LLVM AMDGPU backend | `https://llvm.org/docs/AMDGPUUsage.html`, `https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html` | AMD's compiler target path: LLVM IR -> AMDGPU code object / ISA. This is the AMD analog to NVIDIA's PTX/SASS toolchain boundary. |
| ROCm compiler reference | `https://rocm.docs.amd.com/projects/llvm-project/en/latest/reference/rocmcc.html` | ROCm clang/LLVM is the production compiler stack for HIP/OpenMP/OpenCL to AMDGPU. |
| HIP programming model | `https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html` | Defines blocks/threads, shared memory, device memory, kernel launch model. |
| HIP C++ language extensions | `https://rocm.docs.amd.com/projects/HIP/en/develop/reference/cpp_language_extensions.html` | `__shared__` maps to LDS-like shared memory; launch bounds and device functions shape lowering. |
| Reading AMDGCN ISA | `https://rocm.blogs.amd.com/software-tools-optimization/amdgcn-isa/README.html` | Practical guide to reading AMDGCN disassembly. |
| Machine-readable ISA | `https://gpuopen.com/machine-readable-isa/` | Official XML ISA descriptions for decoding and tooling. |
| Cross-lane operations | `https://gpuopen.com/learn/amd-gcn-assembly-cross-lane-operations/` | Explains cross-lane data exchange, relevant to wave reductions (`ds_bpermute`, swizzle-like idioms). |
| Occupancy | `https://gpuopen.com/learn/occupancy-explained/` | Clarifies waves, SIMD scheduling, and resource-limited occupancy. |
| RDNA scheduling | `https://gpuopen.com/amd-gpu-architecture-programming-documentation/` | RDNA3 scheduling/MES context; not a kernel guide, but useful for queue/runtime mental model. |
| GCN ISA docs | `https://docs.amd.com/v/u/en-US/gcn3-instruction-set-architecture` | Older but still useful for ISA categories and concepts. |

## Naming: PTX / SASS Equivalents On AMD

The NVIDIA analogy is useful but imperfect:

| NVIDIA | AMD / ROCm analog | Project meaning |
|---|---|---|
| CUDA C++ | HIP C++ | High-level GPU C++ source. |
| PTX | LLVM AMDGPU IR / sometimes AMDGCN assembly in practice | PTX is virtual ISA; AMD has LLVM IR plus AMDGPU backend, while AMDGCN asm is closer to real ISA. |
| SASS | AMDGCN machine ISA | Actual instruction layer visible in disassembly. |
| `ptxas` | ROCm clang / LLVM AMDGPU backend / lld / comgr | Compiler + assembler + linker path. |
| `.cubin` | `.hsaco` / `.co` AMDGPU code object | GPU binary/code object loaded by HSA runtime. |
| `nvdisasm` | `llvm-objdump`, `roc-objdump`, `amdllvm-objdump` | Disassembly/metadata inspection. |

In this repo, "owned AMDGCN tile" usually means:

```text
HIP/C++ or AMDGCN-controlled source -> ROCm clang/LLVM -> AMDGPU code object -> tinygrad custom_kernel / HCQ graph node
```

## The Primitive Stack

### 1. Algorithmic primitive

Examples:

- Q4_K GEMV for FFN gate/up/down.
- GQA decode attention.
- KV-cache append/read.
- Split-KV flash decode.
- Online softmax + PV accumulation.

The algorithmic primitive says what math is done. It does **not** determine performance by itself.

### 2. Work decomposition primitive

This is often the decisive layer.

Examples from this project:

| Primitive | Bad decomposition | Winning decomposition |
|---|---|---|
| FFN Q4_K GEMV | one thread/row serial K reduce | one wave/row, K-block-parallel, warp reduce |
| batch-1 attention | too few workgroups / SDPA underoccupancy | split-KV across many workgroups |
| owned attention short ctx | route guard assumed empty splits unsafe | empty-split-safe tile, route allowed at ctx512+ |
| split combine | standalone combine looked expensive | W==D showed combine overlaps and is not critical |

Project principle:

```text
Same math + different work decomposition can be the whole win.
```

### 3. Memory movement primitive

This includes:

- global memory load width;
- coalescing;
- LDS staging;
- register reuse;
- cache dtype;
- materialization/cast copies;
- part/meta round-trips;
- KV-cache lifecycle.

Measured examples:

| Finding | Lesson |
|---|---|
| owned tile initially read fp32 cache as fp16 | dtype contract is a memory primitive, not just a type annotation. |
| FO2 fp16 cache removed fp32->fp16 copy and added ~7% over cast route | cache dtype/materialization can be W==D-critical. |
| runtime-KV copy tax was real but later incremental | lifecycle tax ranking changes after a bigger route lands. |
| B5 combine 2.4x local improvement did not move W==D | not every memory movement is on the wall critical path. |

Project principle:

```text
Bytes moved at the wrong lifecycle boundary can dominate; bytes moved off the critical path may not matter.
```

### 4. ISA primitive

Important AMDGCN-level primitives for this project:

| ISA / resource primitive | Why it matters |
|---|---|
| `v_dot2_f32_f16` / fdot2 lowering | dense fp16 dot throughput for attention QK/PV-like work. |
| `global_load_dwordxN` / vector loads | coalesced global memory movement. |
| LDS `ds_read*` / `ds_write*` | software-managed staging for K/V or reductions. |
| `ds_bpermute` / cross-lane ops | wave reductions, lane exchange, warp-style reductions. |
| `s_waitcnt` placement | latency hiding and memory dependency control. |
| VGPR count | occupancy and spill risk. |
| SGPR count | occupancy/control overhead. |
| LDS bytes | workgroup occupancy and tile feasibility. |
| spills / scratch | often catastrophic. |
| branch/mask structure | empty-split and bounds behavior. |

ISA inspection should answer:

```text
Did the compiler actually emit the primitive we intended?
```

Example: saying "LDS-staged tile" is not enough. Disassembly should show LDS operations. Saying "fp16 dot" is not
enough. Disassembly should show the expected packed dot op or explain why not.

### 5. Runtime / graph primitive

On tinygrad, this includes:

- TinyJit capture/replay;
- GraphRunner var patching;
- HCQ graph nodes;
- custom kernels as `Ops.PROGRAM`;
- buffer lifetime;
- `.after()` dependency semantics;
- fallback paths;
- route guards.

Measured examples:

| Finding | Lesson |
|---|---|
| external `.co` can be injected as `Ops.PROGRAM` via `Tensor.custom_kernel` + binary source | escape-hatch graph node is viable. |
| GraphRunner args for runtime-KV were correct | not every replay failure is scalar patching. |
| owned tile dtype bug looked like runtime-KV failure | validate real data path before blaming graph mechanics. |
| `@function` full-cache copy provided persistence | functional graph semantics can hide lifecycle taxes. |

Project principle:

```text
A kernel is not a primitive until it is correct inside the replayed model lifecycle.
```

### 6. W==D transfer primitive

The final authority is whole-decode W==D:

- wall-clock token/s;
- `.item()` inside timing window;
- byte-identical tokens;
- route actually fires;
- repeated spread;
- context sweep.

Local A/B is diagnostic only.

## Project Learnings By Primitive

### Q4K GEMV warp

What won:

- work decomposition, not new math;
- wave/row K-parallel schedule;
- warp reduction;
- lossless FP reassociation;
- W==D transfer.

Primitive extraction:

```text
row-parallel quantized GEMV schedule + wave reduction + correct dtype/shape guard
```

ISA/implementation signals to audit:

- wave-level reductions;
- memory coalescing over quant blocks;
- VGPR occupancy;
- no spills;
- no q8 lifecycle tax.

### Owned AMDGCN decode attention

What won:

- real-cache dtype contract fixed;
- fp16 route cache / FO2;
- owned tile graph-node route;
- ctx guard lowered after empty-split safety confirmed;
- W==D all-context pass.

Primitive extraction:

```text
owned external AMDGCN/HIP attention tile as a graph node + native fp16 cache contract + split-KV route policy
```

ISA/implementation signals:

- `v_dot2` or fdot2 lowering;
- LDS staging;
- finite neutral empty-split meta;
- combine correctness;
- no fp32-as-fp16 read;
- no degenerate-cache-only validation.

### Split-KV combine

What did not transfer:

- standalone combine speedups;
- 2.4x cheaper combine;
- Amdahl projection treating standalone combine as serial.

Primitive extraction:

```text
split-KV combine must be judged by W==D overlap, not standalone us
```

### Runtime-KV

What remains:

- runtime-managed KV is architecturally valid;
- but after owned attention + fp16 cache, it is incremental;
- previous GraphRunner/cache theories were refuted;
- opaque append NaN remains open if resumed.

Primitive extraction:

```text
runtime-owned cache lifecycle only matters if residual materialization remains large after promoted routes
```

### tinygrad-native codegen

What is still not native:

- llama-class LDS + vector-dot fused attention codegen;
- coupled tiled reductions with online softmax/PV;
- some runtime cache lifecycle semantics.

Primitive extraction:

```text
first prove escape-hatch primitive W==D; only then decide if native codegen should learn it
```

## HIP Pattern -> Expected ISA

| HIP / source pattern | Expected AMDGPU/ISA signal | Failure mode |
|---|---|---|
| `__builtin_amdgcn_fdot2` | `v_dot2_f32_f16`-class instruction | scalar fp16 ops or no packed dot. |
| `__shared__` arrays | LDS group segment + `ds_*` instructions | global-memory-only "tile". |
| wave shuffle / builtin permute | `ds_bpermute` / cross-lane ops | scalar LDS/global reductions. |
| vectorized loads | `global_load_dwordx2/x4` or wide loads | scalar global loads. |
| `restrict` pointer args | fewer alias barriers, vectorization possible | conservative reloads. |
| compile-time tile sizes | static LDS, unrolled loops | dynamic indexing / poor codegen. |
| route-level fp16 cache | no fp32->fp16 materialization copy | hidden cast/copy kernels. |

## ISA Audit Checklist

For any candidate kernel:

1. **Correctness first**
   - real data, not degenerate zeros;
   - multi-step token correctness;
   - dtype/layout contract verified.

2. **Metadata**
   - code object hash;
   - kernel symbol;
   - gfx target;
   - group segment / LDS bytes;
   - private segment / spills;
   - VGPR/SGPR;
   - kernarg size/layout.

3. **Instruction mix**
   - VALU;
   - SALU;
   - LDS;
   - global loads/stores;
   - vector dot ops;
   - cross-lane ops;
   - branches;
   - waits.

4. **Memory behavior**
   - global load width;
   - coalescing pattern;
   - LDS bank risk;
   - HBM round trips;
   - materialization/cast kernels.

5. **Occupancy**
   - workgroups;
   - waves/workgroup;
   - VGPR-limited waves;
   - LDS-limited waves;
   - CU coverage.

6. **Graph lifecycle**
   - route fires;
   - graph nodes present;
   - fallback tested;
   - runtime vars patch;
   - buffer identity/dtype correct.

7. **W==D**
   - token/s;
   - spread;
   - byte-identical;
   - context sweep.

## Proposed Tooling Artifact

Future tool:

```text
extra/qk_amdgpu_isa_primitive_audit.py
```

Inputs:

- code object `.co` / `.hsaco`;
- optional HIP source;
- tinygrad captured graph metadata;
- W==D artifact.

Outputs:

```text
bench/qk-amdgpu-isa-primitive-audit/<candidate>.json
```

Fields:

- kernel symbol;
- gfx target;
- instruction counts;
- key instruction flags:
  - has_v_dot2;
  - has_lds;
  - has_cross_lane;
  - has_spill;
  - has_vector_global_load;
- resources:
  - VGPR;
  - SGPR;
  - LDS bytes;
  - scratch bytes;
- ABI:
  - kernarg layout;
  - pointer/scalar order;
  - dtype contract;
- graph:
  - node count;
  - route identity;
  - runtime vars;
- W==D:
  - local A/B;
  - whole-decode delta;
  - verdict.

## What This Means For Project Principles

### 1. A primitive is lifecycle-complete or it is not a primitive

Local kernels, graph nodes, dtype casts, route guards, and token correctness are part of the same primitive.

### 2. ISA validates intent

Do not trust source-level claims like "uses LDS" or "uses dot2" without disassembly or code object metadata.

### 3. Dtype is an ABI

The fp32-cache/fp16-tile bug is now a first-class lesson: dtype mismatch can silently produce plausible local
success and catastrophic in-model failure.

### 4. Measure transfer, not elegance

B5 combine was elegant and faster locally; it did not transfer. FO2 was "just dtype/cache", and it transferred.

### 5. Escape hatches are learning instruments

Owned HIP/AMDGCN kernels reveal what tinygrad-native codegen should eventually learn. They are not merely hacks if
they are default-gated, tested, documented, and W==D-proven.

## Current Primitive Ranking After Owned Attention + FO2

This ranking must be refreshed by the post-default audit, but the current high-level state is:

| Lane | State | Next use |
|---|---|---|
| Q4K GEMV warp | W==D pass, default-eligible | owner default decision / combined default. |
| owned AMDGCN attention + fp16 cache | W==D pass, default-eligible | owner default decision. |
| runtime-KV | deferred incremental | reopen only if post-default residual copy tax is large. |
| native tinygrad attention codegen | long-term | learn from owned tile ISA once default path is stable. |
| ISA primitive audit | missing | build to systematize future wins. |

## Sources

- LLVM AMDGPU backend user guide: https://llvm.org/docs/AMDGPUUsage.html
- ROCm LLVM AMDGPU guide: https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html
- ROCm compiler reference: https://rocm.docs.amd.com/projects/llvm-project/en/latest/reference/rocmcc.html
- HIP programming model: https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html
- HIP C++ language extensions: https://rocm.docs.amd.com/projects/HIP/en/develop/reference/cpp_language_extensions.html
- Reading AMDGCN ISA: https://rocm.blogs.amd.com/software-tools-optimization/amdgcn-isa/README.html
- AMD machine-readable ISA: https://gpuopen.com/machine-readable-isa/
- AMD GCN cross-lane operations: https://gpuopen.com/learn/amd-gcn-assembly-cross-lane-operations/
- AMD GPUOpen occupancy explained: https://gpuopen.com/learn/occupancy-explained/
- AMD GPU architecture programming documentation: https://gpuopen.com/amd-gpu-architecture-programming-documentation/
- AMD GCN3 ISA documentation: https://docs.amd.com/v/u/en-US/gcn3-instruction-set-architecture
