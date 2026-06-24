# GEMM / Tensor Contraction First-Principles Open-Source Audit

Date: 2026-06-20

## Purpose

Consolidate the first principles behind fast GEMM/tensor contractions from open-source implementations and map them
back to the prefill question:

- why Tensile-class kernels win;
- why blind search is not enough yet;
- what primitives a tinygrad-native search space must expose before BEAM/search can be useful.

This is not a new benchmark result. It is a source-grounded design audit.

## Sources Read

| Source | URL | Why it matters |
|---|---|---|
| ROCm Tensile legacy repo | `https://github.com/ROCm/Tensile` | Describes Tensile as the benchmark-driven GEMM/tensor-contraction backend, now retired into ROCm Libraries. |
| ROCm Libraries super-repo | `https://github.com/ROCm/rocm-libraries` | Current home for Tensile and ROCm library source. |
| Tensile docs | `https://rocm.docs.amd.com/projects/Tensile/en/latest/` | Public documentation for Tensile concepts and kernel parameters. |
| NVIDIA CUTLASS | `https://github.com/NVIDIA/cutlass` | Clean reference for hierarchical GEMM decomposition and reusable data-movement/MMA abstractions. |
| CUTLASS Efficient GEMM | `https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md` | Describes threadblock tiling, global memory movement, K partitioning, and tuning tradeoffs. |
| ROCm Composable Kernel | `https://github.com/ROCm/composable_kernel` | AMD tile-based programming model, tensor coordinate transformation, kernel/Invoker structure. |
| CK Tile README | `https://github.com/ROCm/composable_kernel/blob/develop/include/ck_tile/README.md` | Explicitly names tile programming and layout/index transformation concepts. |
| Triton matmul tutorial | `https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html` | High-level programmable blocked matmul: block-level matrix multiply, pointer arithmetic, L2 ordering, autotune. |
| BLIS | `https://github.com/flame/blis` | CPU-side GEMM first principles: packing, microkernels, and loop hierarchy. |

## First Principles

GEMM is a tensor contraction with two free output indices and one reduction index:

```text
C[M,N] = A[M,K] @ B[K,N]
C[i,j] = sum_k A[i,k] * B[k,j]
```

General tensor contractions extend this:

```text
Output[free_indices] = sum_reduction_indices A[...] * B[...]
```

The winning implementation problem is therefore not "how do we multiply?" It is:

1. how often can each loaded operand be reused before going back to memory;
2. how close to the tensor/matrix unit can we keep the operands;
3. how much memory latency can be hidden behind compute;
4. how many independent tiles/waves/warps can stay resident without spilling;
5. how efficiently the final epilogue writes the result.

## The Common GEMM Stack

All serious GEMM systems converge on the same hierarchy.

| Level | CPU / BLIS language | GPU / CUTLASS language | AMD / Tensile-CK language | Purpose |
|---|---|---|---|---|
| Problem | GEMM/contraction | GEMM/convolution-as-GEMM | GEMM/tensor contraction | define M/N/K/free/reduction axes |
| Outer tile | cache block | threadblock tile | workgroup / macro tile | split work across cores/CUs/SMs |
| Packed/shared tile | packed panels | shared-memory tile | LDS tile | reusable operand layout |
| Inner tile | microkernel | warp/wave tile | wave/tile operator | feed vector/tensor instructions |
| Atom | FMA register block | MMA/tensor op | WMMA/MFMA/tile op | actual multiply-accumulate primitive |
| Pipeline | packed-panel loop | multistage copy/MMA loop | PGR/PLR/K-loop scheduling | overlap movement and compute |
| Epilogue | store/update C | epilogue visitor/store | store/fusion path | write result without losing the win |

This is the first-principles answer: fast GEMM is hierarchical data reuse plus a hardware-specific inner compute atom.

## What Each Open-Source System Teaches

### Tensile

Tensile is a domain-specific autotuning/codegen system for AMD GEMM-like contractions. The legacy repo states that it
creates benchmark-driven backend libraries for GEMMs, batched GEMMs, and N-dimensional tensor contractions, mainly as
the backend for rocBLAS. Current source lives under `ROCm/rocm-libraries`.

What matters for us:

- Tensile is not one kernel. It is a generated solution space plus benchmark selection.
- Its parameters encode a rich GEMM design space: macro tile, workgroup, matrix instruction, depth/unroll, vector
  widths, prefetch behavior, LDS usage, and resource policy.
- The precompiled `.co` we launch is one selected solution from that space.

Implication:

```text
Tensile result = generated template + selected parameters + tuned binary
```

So reverse engineering must recover the selected solution's primitive rows, not just disassemble instructions.

### CUTLASS

CUTLASS is the clearest open-source statement of the modern GPU GEMM hierarchy. It explicitly decomposes GEMM into
hierarchical abstractions and data movement. Its efficient GEMM documentation describes threadblock tiles loading
input tiles from global memory and computing accumulated matrix products, with tile sizes tuned against memory reuse,
occupancy, and problem shape.

What matters for us:

- It names the design decomposition we need: device -> kernel -> collective/copy -> tiled MMA -> atom.
- Shared memory is not the goal; it is the staging layer that lets the atom run efficiently.
- Tile size is a resource tradeoff: larger tiles improve reuse but can reduce occupancy or waste work on edges.
- Split-K/parallel reductions exist because shape can make the obvious tile decomposition underfill the GPU.

Implication:

```text
Our future search space should look more like a GEMM template hierarchy than flat OptOps.
```

### Composable Kernel / CK Tile

CK is AMD's open-source tile-based programming model for performance-critical ML kernels. It emphasizes:

- tile-based programming;
- tensor coordinate transformation;
- templated tile operators;
- templated kernel/invoker layers;
- instantiated kernels and client API.

CK Tile's README explicitly calls tensor coordinate transformation the layout/index transform abstraction, and tile API
/ distributed tensor the programming model.

What matters for us:

- CK exposes the abstraction layer tinygrad currently lacks for this problem: explicit tile layout and coordinate
  transforms.
- CK is closer to the representation we would want for native AMD GEMM search than current BEAM over global-direct
  UOps.
- CK changelogs also show modern GEMM features such as async/ping-pong/rotating-buffer style pipelines and block-scale
  quantized GEMM support.

Implication:

```text
The missing substrate is not "more search"; it is tile/layout/pipeline representation.
```

### Triton

Triton presents a programmable blocked matmul in a compact form: block-level matrix multiplication, multidimensional
pointer arithmetic, program ordering for cache locality, and autotuning.

What matters for us:

- It is a useful model for parameterized search once the kernel template exists.
- It shows that a small exposed template space can be productive: block sizes, group ordering, K block, num warps,
  num stages.
- But Triton-level abstraction is not automatically enough for RDNA3 WMMA/LDS details unless the backend lowers those
  primitives well.

Implication:

```text
Triton-style search is useful after the template exposes the right memory hierarchy.
```

### BLIS

BLIS/GotoBLAS are the CPU-side first-principles reference: packing, loop hierarchy, microkernel. The hardware differs,
but the core invariant is the same:

```text
move data into a layout where the inner compute kernel can reuse it cheaply
```

What matters for us:

- Packing on CPU is analogous to shared/LDS staging on GPU.
- The microkernel is analogous to the tensor-core atom.
- Most of the system is about feeding the microkernel, not inventing the multiply instruction.

Implication:

```text
The prefill gap is about feeding WMMA, not about whether WMMA exists.
```

## Consolidated Primitive Matrix

| Primitive | First-principles role | Tensile | CUTLASS | CK | Triton | tinygrad today |
|---|---|---|---|---|---|---|
| Problem/contraction description | express free/reduction axes | yes | yes | yes | yes | yes |
| Macro tiling | split C tile across workgroups | yes | yes | yes | yes | partial |
| Memory layout transform | convert strided operands to compute-friendly tiles | yes | yes | yes | manual pointer math | weak/implicit |
| Shared/LDS staging | operand reuse near compute | yes | yes | yes | backend-dependent | missing/weak for authority path |
| Tensor op atom | hardware matrix instruction | WMMA/MFMA class | MMA/WGMMA | tile ops | `dot` lowering | yes: WMMA exists |
| K-loop pipeline | overlap next tile movement with current compute | yes | yes | yes | `num_stages` style | missing/weak |
| Wait/barrier semantics | make pipeline correct without over-waiting | yes | yes | yes | backend-lowered | weak |
| Resource policy | balance VGPR/LDS/waves/scratch | yes | yes | yes | autotune params | partial |
| Epilogue | store/fuse result | yes | yes | yes | explicit | partial |
| Autotune/search | select good parameters | benchmark library | profiler/templates | instance factory/profiler | autotune | BEAM, but wrong space |

## What This Means For Our Prefill Gap

The current clean matrix is:

| engine | pp512 result | clock state |
|---|---:|---|
| tinygrad WMMA | ~1436 tok/s | high clock |
| tinygrad + Tensile | ~2664 tok/s | high clock |
| llama.cpp | ~3136 tok/s avg | high clock |

The first-principles interpretation:

1. tinygrad has the tensor op.
2. Tensile has a full GEMM dataflow around the tensor op.
3. llama prefill has a different high-share path: quantized MMQ/matmul plus rocBLAS GEMM.
4. Clock is not the gap.
5. Search is not ready until the missing dataflow primitives exist.

So the failure mode is precise:

```text
current search space:
  tune global-direct WMMA and existing UOp schedules

needed search space:
  tune GEMM template with explicit tile layout, LDS/shared staging,
  K-loop pipeline, waits/barriers, tensor op atom, resource constraints
```

## Why "Just Add LDS" Failed

LDS is a storage primitive, not a performance primitive by itself.

The useful pattern is:

```text
global coalesced load
  -> LDS store in tensor-op-friendly layout
  -> LDS vector load / fragment construction
  -> WMMA/MFMA/MMA
  -> prefetch next K tile while current tile computes
  -> wait/barrier only where needed
```

Standalone LDS can be slower because it adds extra instructions and barriers without enough reuse or overlap. This
matches our local P8 attempts: correct LDS-staged candidates did not beat the LLVM authority.

## Why BEAM Was Not The Lever

BEAM is useful when the winning point is inside tinygrad's existing schedule space.

For this problem, the winning point is outside that space:

- no explicit Tensile-class LDS layout row;
- no first-class K-loop pipeline stage;
- no semantic wait/barrier search;
- no resource-aware pruning tied to LDS/waves/scratch;
- no layout-correctness oracle for the macro tile.

Therefore:

```text
BEAM over current global-direct WMMA = low value
BEAM/search over a Tensile-class template = potentially high value
```

## Machine Search Readiness

Machine search becomes meaningful only when these are first-class:

| Requirement | Why |
|---|---|
| tile/layout representation | search needs something equivalent to CUTLASS/CK coordinate transforms |
| LDS/shared-memory staging | search must move operands through reusable tiles |
| tensor-op fragment mapping | correctness depends on layout -> fragment relationship |
| K-loop pipeline stages | performance depends on load/compute overlap |
| wait/barrier model | pipeline must be correct and not over-synchronized |
| resource model | reject scratch/spill and bad occupancy before timing |
| correctness ladder | small tile -> macro tile -> full authority shape |
| same-harness timing | avoid clock/kernel-identity artifacts |

Without those, search can only tune the wrong abstraction.

## Audit Consequences

The strong prefill audit should read the open-source systems through this lens:

1. **Tensile audit:** recover selected solution parameters and dataflow rows.
2. **llama prefill audit:** split the pp512 MMQ/matmul bucket into source-visible primitive rows.
3. **CK/CUTLASS audit:** use their abstractions to name the portable representation tinygrad lacks.
4. **Search design:** define a smaller Tensile-class template space, not an unbounded search.

## Practical Next Rows

| Row | Action | Pass Condition |
|---|---|---|
| Tensile solution parameter extraction | map selected kernel to MT/DepthU/WorkGroup/vector/prefetch/LDS params | parameters match disasm/resource facts |
| Tensile dataflow diagram | global -> LDS -> fragment -> WMMA mapping | enough to write correctness-only microkernel |
| CK/CUTLASS abstraction map | name equivalent concepts across AMD/NVIDIA/template systems | transfer terms are stable across GPUs |
| llama prefill source map | map high-share MMQ/matmul kernels to source/functions | no high-share pp512 kernel remains opaque |
| tinygrad substrate gap | list required IR/renderer/search objects | machine search row says ready/not-ready |

## Bottom Line

The open-source ecosystem agrees on the same first principles:

```text
fast GEMM = hierarchical tiling
          + data layout/packing
          + near-compute staging
          + tensor/matrix instruction atom
          + overlapped K-loop pipeline
          + wait/resource/epilogue policy
          + autotuned parameters
```

Tensile is AMD's mature autogenerated/autotuned version of that stack. CUTLASS is the clearest NVIDIA template model.
CK is the AMD tile/layout abstraction model. Triton is the compact programmable/autotuned model. BLIS is the CPU
packing/microkernel model.

For tinygrad prefill, the conclusion is direct:

```text
we should not search harder yet;
we should first expose the GEMM dataflow primitives that successful open-source systems already make explicit.
```
