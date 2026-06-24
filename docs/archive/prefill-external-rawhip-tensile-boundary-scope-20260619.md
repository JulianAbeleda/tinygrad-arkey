# Scope - prefill external/raw-HIP/Tensile boundary after bounded kernels are exhausted

This is the full scope for the only material prefill route left after:

- `prefill-wmma-lds-tiling-result-20260619.md`: hand-LDS WMMA is refuted (1.02x).
- `prefill-own-wmma-kernel-result-20260619.md`: bounded pure-tinygrad WMMA configs are refuted (best 42.0 TFLOPS).
- `prefill-external-blas-result-20260619.md`: external BLAS ceiling is real (hipBLASLt 69.8 TFLOPS on ffn_gate/up).

No route/default exists. This scope is about whether the measured external/Tensile-class ceiling can transfer into
the tinygrad `DEV=AMD` model path without violating the project's authority, fallback, and portability rules.

## The decision before any build

The old ">=1.5x full pp" shipping gate is likely too high for the measured ceiling. If the ~74% matmul bucket moves
from ~41 TFLOPS to ~70 TFLOPS, the Amdahl upper bound is about 1.4-1.45x before bridge/layout overhead. Therefore:

| decision | consequence |
|---|---|
| keep strict >=1.5x full warm pp as a route gate | stop here; the measured ceiling probably cannot clear it |
| allow a research/opt-in gate around >=1.25x pp512 and >=1.35x strong | proceed with bridge probes, still no default |
| disallow external dependencies at runtime | skip HIP BLAS integration; only Tensile-HSACO extraction or raw-HIP/assembly HCQ lanes remain |
| require pure tinygrad only | stop here; POWN-1 closed the bounded no-deps route |

This is a policy boundary, not a kernel question. The scope below is useful only if a non-default research flag is
allowed to cross either an external-library boundary or a raw-kernel/HSACO boundary.

## Fixed facts

| fact | value |
|---|---|
| target phase | 8B prefill, especially pp512/pp1024 |
| current authority baseline | `PREFILL_V2` fp16 realized weights + WMMA + warmstart-TC |
| dominant share | ~74% fp16 WMMA matmul in PREFILL_V2 |
| tinygrad matmul plateau | ~40.8-42.0 TFLOPS on ffn_gate/up |
| external ceiling | hipBLASLt 69.8 TFLOPS ffn_gate/up; rocBLAS 70.9 ffn_down; rocBLAS 76.7 attn_q/o |
| weak external shape | attn_k/v only 51.8 TFLOPS, 1.27x tinygrad reference |
| correctness class | fp16 path, no lossy quantization; dNLL should remain <=0.01 |
| runtime mismatch | tinygrad AMD uses HCQ/HSA/KFD queues; rocBLAS/hipBLASLt use HIP runtime streams |

## Boundary lanes

### Lane A - HIP-runtime BLAS call with tinygrad VA pointers

Call rocBLAS/hipBLASLt from a tiny C/C++ shim, passing tinygrad `HCQBuffer.va_addr` pointers directly.

Why try it first:
- fastest way to test transfer into real tinygrad allocations;
- uses the measured library ceiling directly;
- no need to reverse-engineer Tensile kernel ABI.

Main risks:
- HIP may not accept KFD/HCQ-allocated VRAM pointers as HIP device pointers;
- if copies are required, the route probably dies on overhead and memory traffic;
- ordering between HCQ queues and HIP streams is not automatic;
- TinyJit/HCQGraph cannot capture opaque HIP runtime calls.

Initial synchronization can be conservative: `dev.synchronize()` before the BLAS call, `hipStreamSynchronize()` after
it, then resume HCQ. That is not the final design, but it isolates pointer validity and correctness before optimizing
sync.

### Lane B - extract/load the Tensile HSACO through HCQ

Use rocBLAS/hipBLASLt only as the oracle for selecting a solution, then load the chosen Tensile HSACO through
tinygrad's `AMDProgram`/HCQ path.

Why this is attractive:
- avoids running the HIP runtime in the model hot path;
- fits tinygrad's existing HCQ execution model (`AMDProgram` already loads HSACO and fills C-like kernargs);
- may be TinyJit/HCQGraph-compatible if the kernel args and launch dimensions are stable.

Main risks:
- solution selection, HSACO path, kernel name, launch geometry, workspace, and arg ABI may be hard to extract;
- rocBLAS/Tensile kernels may depend on packed metadata or workspace conventions that are not public/stable;
- version/packaging policy is still an external-artifact boundary.

### Lane C - raw HIP / assembly / Tensile-like kernel launched through HCQ

Write or generate one shape-specialized fp16 GEMM kernel, compile to HSACO, and launch it through `AMDProgram`.

Why it exists:
- avoids HIP runtime at execution;
- gives control over the software-pipelined WMMA loop that tinygrad `custom_kernel` could not express.

Main risks:
- this is effectively hand-reimplementing a small slice of Tensile;
- high chance of landing near the 42 TFLOPS plateau unless instruction scheduling/software pipeline is genuinely
  better than the UOp kernel;
- maintenance burden is high for every shape/layout.

### Lane D - deep tinygrad renderer/codegen rewrite

Teach tinygrad's codegen to express the software-pipelined WMMA/Tensile pattern directly. This is not scoped as a
near-term build. It is a project-level codegen effort, not a prefill primitive edit.

## Phases and gates

### Phase EBT-0 - authority/policy lock

Decide and document:

- allowed lanes: HIP runtime, extracted HSACO, raw-HIP/assembly, or none;
- allowed dependency surface: `/opt/rocm-7.2.4` shared libs, generated HSACO artifacts, or source-only;
- route posture: research flag only (`PREFILL_EXTERNAL_GEMM=1`), no default;
- acceptable full-pp gate: strict >=1.5x (likely stop), research >=1.25x, strong >=1.35x;
- fallback contract: unsupported lib/shape/device silently falls back to PREFILL_V2.

Gate: policy is explicit before touching model routing. Kill: if pure tinygrad/no deps and strict >=1.5x are both
required, stop and rest at PREFILL_V2.

### Phase EBT-1 - tinygrad-buffer pointer interop spike (Lane A)

Build a standalone bridge probe:

- allocate A/B/C as tinygrad AMD tensors or buffers, not HIP-owned buffers;
- expose their `HCQBuffer.va_addr` to a C/C++ shim;
- call `hipPointerGetAttributes` on those pointers;
- run one hipBLASLt/rocBLAS GEMM into a tinygrad-owned output buffer;
- verify output against a tinygrad fp16 oracle after returning to HCQ.

Artifact: `bench/qk-prefill-external-bridge/interop.json`.

Gate:
- HIP accepts tinygrad VRAM pointers without copies;
- single-shape GEMM is correct;
- synchronous bridge time is within 10% of standalone HIP timing for ffn_gate/up.

Kill:
- HIP rejects HCQ/KFD pointers;
- pointer path requires host/device copies;
- correctness depends on undefined ordering or corrupts tinygrad buffers.

### Phase EBT-2 - bridge overhead and shape matrix

If EBT-1 passes, run all four prefill shapes through the same bridge using tinygrad-owned buffers:

- ffn_gate/up: 512 x 4096 -> 12288;
- ffn_down: 512 x 12288 -> 4096;
- attn_q/o: 512 x 4096 -> 4096;
- attn_k/v: 512 x 4096 -> 1024.

Measure:
- kernel time vs PXB-1 standalone;
- bridge overhead per call;
- handle/heuristic/workspace reuse;
- sync cost with conservative HCQ/HIP fencing;
- zero-copy guarantee.

Gate:
- ffn_gate/up keeps >=1.55x over tinygrad after bridge overhead;
- weighted matmul bucket model predicts >=1.25x full pp512;
- no shape needs a layout copy.

Kill:
- bridge overhead erases the ffn_gate/up win below ~1.3x;
- layout conversion or transposes add material memory traffic;
- per-call setup cannot be cached.

### Phase EBT-3 - one-block / one-layer transfer

Route only one prefill block or a narrow one-layer harness through the bridge. Do not edit default model behavior.

Requirements:
- inputs are the same realized fp16 weights used by PREFILL_V2;
- route is behind a single flag and easy fallback;
- measure warm one-layer/block time with and without external GEMM;
- verify fp16 output tolerance against PREFILL_V2.

Gate:
- selected block's matmul-heavy share moves by >=1.25x after all routing overhead;
- no compile/recompile storm;
- no decode path touched.

Kill:
- isolated GEMM speed does not transfer to block timing;
- TinyJit/lazy graph boundaries force extra realizes/copies.

### Phase EBT-4 - full in-model warm pp

Route all eligible prefill matmuls behind `PREFILL_EXTERNAL_GEMM=1`.

Measure:
- warm pp512 and pp1024;
- dNLL <=0.01;
- decode ctx sweep unchanged;
- fallback on missing lib/unsupported shape;
- memory footprint and workspace reuse;
- repeated-run stability.

Gates:
- research pass: >=1.25x full warm pp512;
- strong pass: >=1.35x full warm pp512 and pp1024;
- default candidate would require a separate policy decision, broader soak, and likely a higher bar than this scope.

Kill:
- full pp gain <1.15x;
- quality or fallback failure;
- host/runtime overhead dominates because calls are outside HCQGraph/TinyJit capture.

### Phase EBT-5 - reduce synchronization/runtime cost

Only if EBT-4 passes but overhead is the limiting layer:

- cache handles, streams, heuristics, and workspaces once per device/shape;
- minimize HCQ/HIP fencing;
- test whether HIP events can be used as a narrow cross-stack fence;
- if HIP runtime remains too costly, pivot to Lane B (Tensile HSACO through HCQ).

Gate:
- overhead reduction improves full pp by >=5%;
- no race/hang under repeated runs.

Kill:
- ordering cannot be made safe without full synchronize per call;
- HIP runtime global state conflicts with HCQ device ownership.

### Phase EBT-6 - Tensile HSACO extraction (Lane B)

Only if Lane A is policy-blocked or runtime-overhead-blocked.

Work:
- use rocBLAS/hipBLASLt logging/profiling to identify selected solutions for the passing shapes;
- locate the HSACO and kernel symbol;
- recover launch dimensions, workspace requirements, and arg ABI;
- load the HSACO with `AMDProgram`;
- call it on tinygrad `HCQBuffer` args;
- validate one shape, then all shapes.

Gate:
- one dominant shape reaches within 10% of PXB-1 standalone without HIP runtime calls in the hot path;
- launch can be represented as an HCQ program with stable args;
- artifact/version policy is explicit.

Kill:
- kernel ABI is opaque/unstable;
- workspace or metadata conventions are not reconstructable;
- performance depends on rocBLAS host-side orchestration that cannot be replicated.

### Phase EBT-7 - raw-HIP/assembly kernel (Lane C)

Only if both Lane A and Lane B are blocked and the project accepts a raw-kernel maintenance burden.

Work:
- start with ffn_gate/up only;
- compile a shape-specialized HIP/assembly HSACO;
- launch through `AMDProgram`;
- target >=62 TFLOPS isolated, stretch >=70 TFLOPS;
- no model routing until the isolated gate passes.

Kill:
- isolated kernel stays near 42 TFLOPS;
- correctness or maintenance burden exceeds the external dependency route.

## Implementation surfaces

Likely files if pursued:

- `extra/qk_prefill_external_interop_probe.py`
- `extra/qk_prefill_external_bridge.cpp`
- `bench/qk-prefill-external-bridge/`
- `tinygrad/runtime/external_blas_amd.py` or one contained adapter module if the bridge passes
- `tinygrad/llm/model.py` route behind `PREFILL_EXTERNAL_GEMM=1` only after EBT-3 passes
- `docs/prefill-external-rawhip-tensile-boundary-result-20260619.md`

Relevant existing code:

- `tinygrad/runtime/ops_amd.py`: `AMDProgram`, `AMDAllocator`, `KFDIface.alloc/map`, `HCQBuffer.va_addr`.
- `tinygrad/runtime/support/hcq.py`: `HCQProgram.__call__`, `HCQBuffer`, timeline synchronization.
- `extra/qk_prefill_blas_ceiling.cpp`: standalone rocBLAS/hipBLASLt ceiling timer.
- `extra/qk_prefill_wmma_sweep.py`: no-deps WMMA sweep that closed the bounded tinygrad route.

## What not to do

- Do not route external BLAS by default.
- Do not copy tensors to HIP-owned buffers and call that a pass.
- Do not mix decode changes into this arc.
- Do not reopen LDS tiling, bigger WMMA tiles, more waves, BK32/BK64, or noLDS as standalone ideas.
- Do not treat an isolated library win as model authority.
- Do not accept a route without fallback when the library, shape, or backend is unsupported.

## Expected outcomes

| outcome | meaning |
|---|---|
| Lane A passes and pp >=1.25x | viable opt-in research route; still needs policy before any default |
| Lane A pointer interop fails | external HIP runtime route closed; try Tensile HSACO extraction only if dependency policy allows artifacts |
| Lane A works but pp gain <1.15x | ceiling does not transfer; bank as bridge/runtime overhead failure |
| Lane B works | best technical fit: external-generated kernel, HCQ execution, less runtime mismatch |
| Lane B fails | external/Tensile route is blocked without a raw-kernel project |
| Lane C fails | rest at PREFILL_V2; remaining gap is outside scoped project work |

## Current recommendation

Proceed only if the user explicitly accepts a research-only external/raw-HIP boundary. If accepted, start with
EBT-1 (tinygrad-buffer pointer interop), because it cheaply separates "HIP can use our buffers" from the harder
model-routing problem. If EBT-1 fails, the remaining credible path is Tensile HSACO extraction through HCQ; if that
also fails, the project should rest at PREFILL_V2 rather than start a from-scratch Tensile rewrite.
