# Cross-Vendor ISA Primitive Audit + Search Readiness Scope (2026-06-23)

## Mission

Clarify whether the project's ISA audit principle is AMD-only or general, and scope how to use it going forward.

Answer:

- The **principle is general**: every GPU backend has a final code-object / machine-instruction layer that can and
  should be audited before trusting a performance primitive.
- The **current implementation is AMD-specific**: `extra/qk_amdgpu_isa_primitive_audit.py` inspects AMDGPU code
  objects (`.co` / `.hsaco`) using ROCm/LLVM tooling and AMDGCN instruction/resource signals.
- A future cross-vendor tool should have vendor backends:
  - AMD: AMDGCN / ROCm / LLVM tools;
  - NVIDIA: cubin / SASS / CUDA binary utilities;
  - Intel: VISA/GEN/Xe tooling where available.

The audit is not a replacement for W==D. It is an early reject and explanation layer in the primitive lifecycle:

```text
source/schedule -> compiler/backend -> code object -> ISA/resources -> graph lifecycle -> W==D transfer
```

## Why This Matters Now

The 8B decode project is at a clean exhaustion checkpoint:

- FFN Q4K GEMV is closed at/near llama parity.
- Owned AMDGCN attention is closed/near parity and default-on.
- The largest remaining lever is KV materialization, but it is core-runtime-blocked.
- The fallback lane is small-ops / activation fusion, but likely overlapped and must prove W==D.
- Machine search is not justified until a bounded searchable knob is identified.

The ISA audit tool should therefore be used in two ways:

1. **Guard future candidates**: reject kernels/schedules that do not emit the intended low-level primitive.
2. **Explain wins/misses**: when local speed does not transfer, distinguish ISA/resource failure from runtime/graph
   lifecycle failure.

## External Reference Map

### AMD

- LLVM AMDGPU backend guide:
  - `https://llvm.org/docs/AMDGPUUsage.html`
  - `https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html`
- ROCm compiler reference:
  - `https://rocm.docs.amd.com/projects/llvm-project/en/latest/reference/rocmcc.html`
- HIP programming model:
  - `https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html`
- HIP C++ language extensions:
  - `https://rocm.docs.amd.com/projects/HIP/en/develop/reference/cpp_language_extensions.html`
- Reading AMDGCN ISA:
  - `https://rocm.blogs.amd.com/software-tools-optimization/amdgcn-isa/README.html`
- AMD machine-readable ISA:
  - `https://gpuopen.com/machine-readable-isa/`
- AMD cross-lane operations:
  - `https://gpuopen.com/learn/amd-gcn-assembly-cross-lane-operations/`
- AMD occupancy:
  - `https://gpuopen.com/learn/occupancy-explained/`

### NVIDIA

- CUDA binary utilities:
  - `https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html`
- CUDA binary utilities PDF:
  - `https://docs.nvidia.com/cuda/pdf/CUDA_Binary_Utilities.pdf`
- PTX ISA documentation:
  - `https://docs.nvidia.com/cuda/parallel-thread-execution/`

NVIDIA exposes tooling such as:

- `cuobjdump`;
- `nvdisasm`;
- `cu++filt`;
- `nvprune`.

Useful analogy:

| NVIDIA layer | Audit target |
|---|---|
| CUDA C++ | source intent |
| PTX | virtual ISA / compiler IR boundary |
| SASS | final machine instruction layer |
| cubin | code object |
| `cuobjdump`, `nvdisasm` | disassembly/resource evidence |

### Intel

- oneAPI Level Zero:
  - `https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/INTRO.html`
  - `https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2023-0/intel-oneapi-level-zero.html`
- Intel Graphics Compiler VISA documentation:
  - `https://github.com/intel/intel-graphics-compiler/blob/master/documentation/visa/1_introduction.md`

Intel's public low-level story is less directly analogous to AMDGCN/SASS for this project, but the principle remains:
inspect the compiler/backend output and resource usage before declaring a primitive.

## Cross-Vendor Mapping

| Concept | AMD | NVIDIA | Intel |
|---|---|---|---|
| high-level GPU language | HIP C++ | CUDA C++ | SYCL / DPC++ / OpenCL |
| virtual/intermediate GPU layer | LLVM AMDGPU IR | PTX | VISA / LLVM/SPIR-V depending path |
| final machine ISA | AMDGCN | SASS | GEN/Xe ISA |
| code object | `.co` / `.hsaco` | `.cubin` / fatbin section | device binary/module |
| disassembler / binary tools | `llvm-objdump`, `roc-objdump`, `amdllvm-objdump`, `readelf` | `cuobjdump`, `nvdisasm` | IGC/ocloc/Level Zero-related tooling |
| occupancy/resource evidence | VGPR, SGPR, LDS, scratch | registers, shared memory, local memory, spills | GRF/thread payload/shared local memory equivalents |
| cross-lane primitive | `ds_bpermute`, `ds_swizzle` | shuffle / warp-level ops | subgroup shuffles |
| shared memory | LDS | shared memory | SLM |
| dot/tensor primitive | `v_dot2`, MFMA/WMMA-like paths | tensor cores, IMMA/HMMA, DP4A | DPAS / Xe matrix paths |

## General ISA Audit Contract

Every backend should emit a normalized record:

```json
{
  "candidate": "...",
  "vendor": "amd|nvidia|intel",
  "arch": "...",
  "code_object": "...",
  "symbols": ["..."],
  "resources": {
    "registers": null,
    "shared_memory_bytes": null,
    "scratch_or_local_bytes": null,
    "spills": null
  },
  "instruction_flags": {
    "has_vector_dot": false,
    "has_matrix_or_tensor_op": false,
    "has_shared_memory": false,
    "has_cross_lane": false,
    "has_vector_global_load": false,
    "has_spill": false
  },
  "graph_lifecycle": {
    "route_fires": false,
    "runtime_vars_patch": null,
    "fallback": null
  },
  "wd": {
    "tokens_match": null,
    "delta_pct": null,
    "contexts": []
  },
  "verdict": "..."
}
```

The vendor-specific parser may differ, but the normalized questions are the same:

1. Did the final binary emit the intended instruction family?
2. Did it use the intended memory hierarchy?
3. Did it avoid spills/scratch?
4. Is occupancy plausibly sufficient?
5. Is the dtype/layout ABI correct?
6. Does the kernel actually run in the model graph?
7. Does it transfer to W==D?

## AMD Backend: Current State

Current tool:

```text
extra/qk_amdgpu_isa_primitive_audit.py
```

Current audit target:

- AMDGPU code object from owned decode attention route.

Known confirmed signals:

- `v_dot2_f32_f16` present;
- LDS present (`ds_store` / `ds_load`-class evidence);
- cross-lane reduction present (`ds_bpermute`);
- 56 VGPR;
- 0 scratch/spill;
- owned tile is real code-object evidence, not source-only.

Near-term AMD extensions:

- make the tool accept candidate metadata JSON;
- support multiple code objects in one run;
- emit normalized JSON under `bench/qk-isa-primitive-audit/`;
- inspect tinygrad-generated residual kernels when code objects are discoverable;
- attach W==D artifact links.

## NVIDIA Backend: Future Shape

This is not needed for the current AMD project, but it proves the principle is not AMD-exclusive.

Potential tool:

```text
extra/qk_nvidia_isa_primitive_audit.py
```

Inputs:

- `.cubin`;
- executable/fatbin;
- CUDA source metadata;
- optional W==D artifact.

Tools:

- `cuobjdump`;
- `nvdisasm`.

Candidate normalized signals:

- register count;
- shared memory bytes;
- local memory/spill evidence;
- tensor core / matrix op mnemonics;
- vector/global load patterns;
- warp shuffle instructions;
- barrier/wait structure;
- occupancy-relevant resource usage.

Useful verdicts:

- `NVIDIA_SASS_PRIMITIVE_CONFIRMED`
- `NVIDIA_PTX_ONLY_INSUFFICIENT`
- `NVIDIA_SPILL_OR_RESOURCE_GAP`
- `NVIDIA_TOOLING_UNAVAILABLE`

Important note:

PTX alone is not final proof. It is closer to a virtual ISA. For performance claims, the audit should prefer SASS from
`nvdisasm` / `cuobjdump` when available.

## Intel Backend: Future Shape

Potential tool:

```text
extra/qk_intel_isa_primitive_audit.py
```

Inputs:

- device binary/module;
- IGC/ocloc output;
- Level Zero module metadata if available.

Candidate normalized signals:

- GRF/register pressure;
- SLM usage;
- DPAS/matrix op evidence;
- subgroup/cross-lane evidence;
- spills/scratch;
- memory send patterns if available.

Useful verdicts:

- `INTEL_ISA_PRIMITIVE_CONFIRMED`
- `INTEL_VISA_ONLY_PARTIAL`
- `INTEL_TOOLING_UNAVAILABLE`

Intel should be treated as a future backend. Do not block current AMD progress on it.

## How This Connects To Runtime-KV

Runtime-KV is **not** primarily an ISA problem.

The latest diagnostic says:

- MAXC shrink transfers strongly;
- `E_49152` is on the W==D critical path;
- opaque append passes standalone;
- model-local opaque append is core-runtime blocked;
- the blocker is TinyJit / `@function` persistence without materialization.

ISA audit still helps by proving:

- the append kernel itself is not the bottleneck if it emits expected stores and has no obvious resource failure;
- owned tile read path is real-cache-correct and ISA-confirmed;
- remaining failure is runtime lifecycle, not codegen.

So the correct classification is:

```text
RUNTIME_GRAPH_LIFECYCLE_GAP
```

not:

```text
ISA_CODEGEN_GAP
```

## How This Connects To Small-Ops Fusion

Small-ops/activation fusion is the bounded fallback, but it is not yet machine-search-ready.

Before any search:

1. identify a real kernel group by rendered source/AST;
2. verify it is not mislabeled KV/cache work;
3. prove it is on the W==D path or at least not fully overlapped;
4. define a single fusion candidate;
5. use ISA audit to verify the fused candidate actually removed launches/loads/stores;
6. require >=1-2% W==D before expanding.

ISA audit can reject false wins:

- source says fused but binary still has separate loads/stores;
- fusion increases spills;
- fusion lowers occupancy enough to lose;
- final graph still emits the old kernel group.

## Machine Search Readiness

Machine search is only justified after one of these is true:

1. Runtime-KV core capability exists and exposes tunable implementation knobs.
2. Small-op fusion proves one bounded transferable fusion.
3. A residual kernel has a verified ISA/codegen gap and a local correctness harness.

Until then:

```text
do not run broad kernel search
```

The current state is:

| Lane | Search readiness | Reason |
|---|---|---|
| attention | low | closed near parity; more variants risk non-transfer |
| GEMV | low | closed at parity |
| Runtime-KV | not yet | core-runtime blocked, not a kernel search surface |
| small ops / activation | maybe later | needs one fusion gate first |
| ISA audit | ready | tooling can become general infrastructure |

## Proposed Tooling Architecture

Create a vendor-neutral wrapper later:

```text
extra/qk_isa_primitive_audit.py
```

Backend modules:

```text
extra/qk_amdgpu_isa_primitive_audit.py
extra/qk_nvidia_isa_primitive_audit.py
extra/qk_intel_isa_primitive_audit.py
```

For now, only AMD must work.

CLI shape:

```bash
PYTHONPATH=. .venv/bin/python extra/qk_isa_primitive_audit.py \
  --vendor amd \
  --candidate owned_decode_attention \
  --code-object /tmp/b4_tile.co \
  --wd-artifact bench/qk-post-owned-attention-default-audit/wd.json \
  --out bench/qk-isa-primitive-audit/owned_decode_attention.json
```

Normalized output should allow future CI/search filters:

- reject if `has_spill=true`;
- reject if required `has_vector_dot=false`;
- reject if required `has_shared_memory=false`;
- reject if route does not fire;
- reject if W==D token correctness fails.

## Verdicts

Allowed cross-vendor verdicts:

- `ISA_AUDIT_GENERAL_PRINCIPLE_CONFIRMED`
- `AMD_ISA_AUDIT_READY`
- `NVIDIA_ISA_AUDIT_BACKEND_SCOPED`
- `INTEL_ISA_AUDIT_BACKEND_SCOPED`
- `RUNTIME_KV_NOT_ISA_BLOCKED`
- `SMALL_OPS_NEEDS_FUSION_GATE_BEFORE_SEARCH`
- `MACHINE_SEARCH_NOT_READY`
- `MACHINE_SEARCH_READY_FOR_LANE`

Current expected verdict:

```text
ISA_AUDIT_GENERAL_PRINCIPLE_CONFIRMED
AMD_ISA_AUDIT_READY
RUNTIME_KV_NOT_ISA_BLOCKED
SMALL_OPS_NEEDS_FUSION_GATE_BEFORE_SEARCH
MACHINE_SEARCH_NOT_READY
```

## Claude Prompt

You are in `/home/ubuntu/tinygrad-arkey` on branch `qk-prefill-flag-leak-resolution`.

Read:

```text
docs/cross-vendor-isa-primitive-audit-and-search-scope-20260623.md
docs/amd-gpu-holistic-primitive-model-20260623.md
docs/post-default-runtime-kv-diagnostic-result-20260623.md
docs/small-ops-activation-fusion-scope-20260623.md
```

Answer and document whether the ISA audit principle is AMD-specific or general.

Expected conclusion:

- The principle is general across GPU vendors.
- The current implementation is AMD-specific.
- AMD backend is ready now.
- NVIDIA backend would use cubin/SASS via `cuobjdump`/`nvdisasm`.
- Intel backend would use IGC/VISA/Level Zero-era tooling where available.
- Runtime-KV is not ISA-blocked; it is runtime graph lifecycle blocked.
- Small-ops fusion needs one bounded W==D gate before machine search.
- Machine search is not yet justified until a bounded lane is proven.

If useful, update or create:

```text
docs/cross-vendor-isa-primitive-audit-and-search-result-20260623.md
```

Do not implement NVIDIA/Intel tooling unless explicitly asked. Do not start machine search. Do not reopen attention/GEMV.

Final response must include:

- general vs AMD-specific answer;
- vendor mapping table;
- how this changes Runtime-KV;
- how this changes small-ops fusion;
- machine-search readiness verdict;
- files changed;
- git status.
