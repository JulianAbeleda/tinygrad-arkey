# DType orthogonality migration — EXP (2026-07-29)

## Goal

Align EXP with upstream's separation of scalar numeric dtype, value lanes/shape, address space, target
spelling, and model storage format. This is a compiler-ownership cleanup, not a new kernel or an LLM
precision policy.

The intended ownership is:

| Concern | Owner |
| --- | --- |
| Scalar numeric semantics (`uint32`, `float16`) | `DType` |
| Value lanes and logical extent | UOp shape / the render site |
| Pointer and address-space semantics | UOp/buffer metadata |
| Metal, HIP, CUDA, or C source spelling | target renderer |
| GGUF quant/storage format | model loader and quant lowering |
| Native AMD fragment ABI | explicit fragment/layout contract |

## Upstream trace

The old upstream `dtype_shape` branch was an experiment that made vector lane count visible through UOp
shape and forced shape validation. EXP already carries much of that shape machinery, including custom
AMD op shape rules. That branch is stale and is not safe to cherry-pick.

Current upstream completed the direction as a sequence of migrations rather than one dtype patch:

- `2fc7e5341` — final removal of `PtrDType` (2026-07-07)
- `63cb1369c` — removal of `ImageDType` (2026-07-08)
- `fdffc6c0c` — removal of dtype `.base` ownership (2026-07-08)
- `03ecad948` — full removal of `DType.vec`, `count`, and `_scalar` (2026-07-12)

EXP has substantial AMD fragment, late-codegen, and machine-search work after its upstream merge base.
The architecture should therefore be ported by ownership boundary, not by replaying those commits.

## Stage 1 — renderer boundary (complete)

`CStyleLanguage` now exposes three distinct operations:

- `render_dtype(dtype)` renders one scalar target type.
- `render_vector_dtype(dtype, lanes)` renders an explicitly sized native vector.
- `render_type(uop)` derives lanes from `uop.max_numel()` and renders the complete value type.

Clang, HIP, CUDA, and Metal call sites that previously smuggled lane width through `DType.count` now pass
the lane count explicitly. Metal and CUDA bitcasts use `render_type(uop)`, matching upstream's ownership.
Metal owns its compact MSL scalar spellings (`uint`, `ushort`, and related forms), so a two-lane uint is
composed as `uint` + `2` -> `uint2`; the core dtype name never becomes `unsigned_int2`.

This stage intentionally leaves vector `DType` objects available behind the boundary. AMD instruction
selection and fragment validation still use them as an internal ABI, but they no longer dictate how a
C-style backend discovers or spells value lanes.

## Audit at the Stage 1 checkpoint

The remaining compatibility surface is large and performance-sensitive:

| Legacy representation | Lines | Python files | Largest owners |
| --- | ---: | ---: | --- |
| `dtype.count` | 226 | 28 | x86 ISA, UOp shape, expander, AMD ISA, register-store lowering |
| `.vec(...)` | 423 | 59 | WMMA kernels, softmax, LLVM IR, UOp spec, AMD tests |
| `.vcount` | 27 | 9 | devectorizer, UOp spec/ops, rangeify |
| `PtrDType` / `.ptrdtype` | 173 | 37 | WMMA kernels, AMD ISA, devectorizer, LDS optimizer |
| `ImageDType` | 39 | 10 | devectorizer, dtype/UOp core, C-style and NIR renderers |

Counts include `tinygrad`, `test`, `extra`, and `examples`; they are migration sizing, not claims that
each line must be changed.

## Validation at the Stage 1 checkpoint

- Metal/Objective-C/MetalGraph: 11 passed.
- CPU vector, composite-reduce, pipeline, Q4 decode, and renderer-focused tests: 49 passed.
- Renderer plus selected AMD/HIP/Q4 tests: 31 passed; failures were existing host/tool constraints or
  unrelated attention assertions (missing AMD hardware, missing LLVM inspection tools, and existing
  attention scheduling/barrier failures).
- Qwen3 0.6B Q8 on `DEV=METAL JIT=1`: model loaded and generated successfully.
- Three-token diagnostic: prefill 1.84 tok/s; decode 0.97, 1.53, then 70.68 tok/s after JIT capture.
- A DEBUG=4 model run emitted 380,983 bytes of diagnostic output with 112 compact native vector tokens
  and zero invalid descriptive integer-vector spellings.

The token rates are diagnostic evidence for path continuity, not a published performance benchmark.

### Metal 8B reactivation checkpoint

Restoring the backend files was not sufficient to make Metal active: `ALL_DEVICES` still listed only
AMD and CPU, so an unconfigured macOS process selected CPU. EXP now registers Metal ahead of AMD and
CPU, matching upstream's discovery order for the backends retained by this fork. A platform-independent
registration test prevents the backend from becoming present-but-unreachable again.

On the local 16 GB Apple M4 / macOS 26.5 host, the registered backend passed the focused Metal compiler,
allocator, graph, Objective-C bridge, and dtype tests (12/12), plus the ordinary tiny tensor suite
(15 passed, one expected CL-image skip). Float32/float16/int32 elementwise reductions and a 64x64 fp16
matmul also matched their CPU references.

The downloaded Qwen3-8B Q4_K_M model then completed a real Metal load and decode at explicit context 128:

```bash
DEV=METAL PYTHONPATH=. python3 -m tinygrad.llm.cli \
  --model /Users/julianabeleda/models/Qwen3-8B-Q4_K_M.gguf \
  --max_context 128 --stream off --warmup --benchmark 3
```

The 5,027,783,488-byte model admitted successfully and produced 10.07, 11.68, and 11.39 tok/s after JIT
warmup (mean 11.05 tok/s). A same-machine, same-model, same-context upstream-control run at
`6ea7d366f` produced 5.01, 4.86, and 4.83 tok/s (mean 4.90 tok/s). This short diagnostic establishes
correct Metal execution and a useful EXP baseline; it is not a statistically sufficient release
benchmark or evidence that the AMD-only generated candidates apply to Metal.

The next Metal work is therefore profiling and machine-searching the generic generated graph. It must
not enable the gfx1100-only packed-WMMA or flash-attention candidates on Apple hardware, and it must not
introduce a hand-authored Metal hot-path kernel.

## Remaining stages

### Stage 2 — generic value-lane API

Make UOp shape the only generic source of value lane count. Replace generic codegen reads of
`dtype.count`/`vcount` with UOp lane/shape helpers, starting with symbolic transforms, decomposition,
expansion, devectorization, and reduction lowering. Preserve explicit native-instruction widths.

Gate: CPU and backend codegen suites produce equivalent programs; no AMD fragment code changes yet.

### Stage 3 — explicit AMD fragment/layout contracts (custom bridge complete; physical gate pending)

Move packed WMMA and attention carrier widths from vector `DType` identity into fragment/layout metadata.
The scalar accumulator/input dtype and the hardware lane/register layout must remain separately
inspectable by machine search and final ISA validation.

The custom AMD attention bridge now follows that model: `AMD_PACKED_FRAGMENT_LOAD`,
`AMD_ROW_SOFTMAX_REPACK`, and `AMD_ROW_SOFTMAX_SLOT` carry scalar dtypes, while their descriptors own
the 16-lane or 8-lane shape. SPEC=3 verifies these scalar-plus-shape contracts. The final generic WMMA
carrier still uses the fork's legacy vector dtype until the physical AMD gate can compare generated ISA.

Gate: gfx1100 compile evidence, instruction counts, VGPR/spill contracts, and known numerical fixtures are
unchanged on an AMD validation host.

The exact pending checklist is [AMD dtype-orthogonality validation](dtype-orthogonality-amd-validation-20260729.md).

### Stage 4 — pointer and image ownership

Move pointer size, pointee type, address space, and image shape to UOp/buffer metadata following the
upstream sequence. Do this after value lanes so pointer vectors are not migrated twice.

Gate: CPU, HIP, AMD ISA, Metal, Python, NIR, graph, and image tests pass.

### Stage 5 — remove vector state from `DType`

Delete `.vec`, `.count`, `.vcount`, `_scalar`, `PtrDType`, and `ImageDType` only after all consumers use
the new owners. At that point `DType` is scalar-only throughout the compiler, not merely at renderer
boundaries.

Gate: full applicable test suite, Metal small-model regression, AMD 8B hot-path correctness and compile
evidence, and no generated-kernel diffs outside intentional source spelling changes.

## Promotion rule

Stage 1 may remain on EXP until AMD validation is available. Do not promote the later stages based only
on macOS/Metal results: the custom AMD fragment ABI is the largest remaining divergence from upstream
and is part of the machine-search hot path.

## Non-goals

- No eGPU reset or AMD hardware work in this stage.
- No hand-authored kernel, new hot-path route, or search-policy change.
- No change to GGUF quantization or attention's fp16 boundary.
- No claim that legacy vector/pointer dtype state has already been removed.
