# Cross-Vendor ISA Primitive Audit + Search Readiness — Result (2026-06-23)

## 1. Final verdicts
- `ISA_AUDIT_GENERAL_PRINCIPLE_CONFIRMED`
- `AMD_ISA_AUDIT_READY`
- `NVIDIA_ISA_AUDIT_BACKEND_SCOPED` (tooling absent on this host)
- `INTEL_ISA_AUDIT_BACKEND_SCOPED` (tooling absent on this host)
- `RUNTIME_KV_NOT_ISA_BLOCKED`
- `SMALL_OPS_NEEDS_FUSION_GATE_BEFORE_SEARCH`
- `MACHINE_SEARCH_NOT_READY`

## 2. General vs AMD-specific
**The principle is general; the implementation is AMD-specific.**
Every GPU backend has a final code-object → machine-ISA → resource/occupancy layer that can and should be audited
*before* trusting a performance primitive — source intent and graph identity are not proof. The project's tool
`extra/qk_amdgpu_isa_primitive_audit.py` is an **AMD-specific instance** of that general contract (it parses AMDGPU
code objects with ROCm/LLVM tooling). The normalized questions (intended instruction family? memory hierarchy? no
spills? occupancy? dtype ABI? runs in the graph? transfers to W==D?) are **vendor-independent**; only the parser
and mnemonics differ.

Verified on this host: AMD tooling present and working (`llvm-objdump`/`llvm-readelf`/`clang-offload-bundler` →
owned tile 56 VGPR / 0 spill / `v_dot2` / LDS / `ds_bpermute`). **NVIDIA** (`cuobjdump`/`nvdisasm`/`nvcc`) and
**Intel** (`ocloc`/`iga`/`igc`) tooling are **absent** (this is a gfx1100 box) → those backends are *scoped only*,
not buildable here.

## 3. Vendor mapping
| Concept | AMD (ready) | NVIDIA (scoped) | Intel (scoped) |
|---|---|---|---|
| high-level language | HIP C++ | CUDA C++ | SYCL / DPC++ / OpenCL |
| virtual / IR layer | LLVM AMDGPU IR | **PTX** (virtual ISA — *not* final proof) | VISA / SPIR-V |
| **final machine ISA** | **AMDGCN** | **SASS** | GEN / Xe ISA |
| code object | `.co` / `.hsaco` | `.cubin` / fatbin | device binary/module |
| disassembler / metadata | `llvm-objdump`, `llvm-readelf`, `roc-objdump` | `cuobjdump`, `nvdisasm` | IGC / `ocloc` / Level-Zero tooling |
| occupancy/resources | VGPR, SGPR, LDS, scratch | registers, shared mem, local mem, spills | GRF, SLM, spills |
| cross-lane | `ds_bpermute`, `ds_swizzle` | warp `shfl` | subgroup shuffle |
| shared memory | LDS | shared memory | SLM |
| dot/tensor | `v_dot2`, WMMA/MFMA | DP4A, HMMA/IMMA (tensor cores) | DPAS (Xe matrix) |

**Key caveat (NVIDIA): PTX is not final proof.** PTX is a virtual ISA / compiler-IR boundary (the AMD analog of
LLVM-AMDGPU-IR, *not* AMDGCN). A performance claim must read **SASS** via `nvdisasm`/`cuobjdump`, exactly as the
AMD audit reads AMDGCN, not LLVM IR. (`NVIDIA_PTX_ONLY_INSUFFICIENT`.)

## 4. Normalized ISA-audit contract
A vendor-neutral record (realizable for AMD **today** — see `bench/qk-isa-primitive-audit/owned_decode_attention.json`):
`{candidate, vendor, arch, code_object, symbols, resources{registers, shared_memory_bytes, scratch_or_local_bytes,
spills}, instruction_flags{has_vector_dot, has_matrix_or_tensor_op, has_shared_memory, has_cross_lane,
has_vector_global_load, has_spill}, graph_lifecycle{route_fires, runtime_vars_patch, fallback}, wd{tokens_match,
delta_pct, contexts}, verdict}`. The example record for the owned tile is fully populated from real code-object
evidence + the W==D artifact → the contract is **not hypothetical for AMD**. NVIDIA/Intel backends would fill the
*same* record via their parsers.

CI/search filter shape (future): reject if `has_spill`, or required `has_vector_dot=false`, or required
`has_shared_memory=false`, or route doesn't fire, or W==D token-correctness fails.

## 5. How this changes Runtime-KV classification
**It does not move it — and that is the point.** The ISA audit proves the runtime-KV failure is **not** codegen:
the opaque append kernel is byte-correct standalone (microbench rel_rmse e-7) and the owned tile read path is
ISA-confirmed and real-cache-correct. The model-local failure is therefore **`RUNTIME_GRAPH_LIFECYCLE_GAP`**
(TinyJit/`@function` persistence-without-materialization), **not `ISA_CODEGEN_GAP`**. The ISA audit's role here is
*disambiguation*: it lets us say "runtime-KV is `RUNTIME_KV_NOT_ISA_BLOCKED`" with code-object evidence rather than
guesswork. Runtime-KV stays core-runtime-deferred.

## 6. How this changes small-ops fusion
The ISA audit becomes a **false-win guard** for the fusion lane: a fusion that "should" merge kernels must show, in
the code object, that the separate loads/stores/launches actually disappeared (and that it did not add spills or
crater occupancy) — *before* trusting any local speed. Combined with the existing gate (rendered-source evidence →
one fusion → ≥1–2% W==D), the small-ops lane is `SMALL_OPS_NEEDS_FUSION_GATE_BEFORE_SEARCH`: not yet
search-ready; it needs one bounded, ISA-verified, W==D-positive fusion first.

## 7. Machine-search readiness: `MACHINE_SEARCH_NOT_READY`
| Lane | Search readiness | Reason |
|---|---|---|
| attention | low | closed near parity; variants risk non-transfer |
| GEMV | low | closed at parity |
| runtime-KV | not yet | core-runtime blocked — not a kernel-search surface |
| small-ops/activation | maybe later | needs one ISA-verified W==D fusion gate |
| ISA audit | **ready** | tooling proven; becomes general guard/infrastructure |

Broad kernel/machine search remains **unjustified** until one of: (a) a runtime-KV core capability exposes tunable
knobs, (b) small-ops proves one bounded transferable fusion, or (c) a residual kernel has a verified ISA/codegen gap
+ a local correctness harness. None hold yet.

## 8. Files changed
- New: this result doc; `bench/qk-isa-primitive-audit/owned_decode_attention.json` (normalized AMD contract record).
- No NVIDIA/Intel tooling implemented (scoped only). No machine search. No source/default changes. No
  attention/GEMV/runtime-KV work. The AMD ISA tool `extra/qk_amdgpu_isa_primitive_audit.py` is unchanged (already
  shipped).

## 9. Git status
Clean before this task (HEAD `5da39fc0e`). This task adds the result doc + one normalized contract artifact only;
no `tinygrad/` source or default changes.

## Recommendation
Hold machine search. The ISA-audit principle is general and the AMD backend is production-ready as a **guard +
disambiguation** layer — use it to gate every future candidate. The next *bounded* action is the single small-ops
fusion gate (`docs/small-ops-activation-fusion-scope-20260623.md`); runtime-KV is a separate, owner-authorized
**core-runtime** task. A vendor-neutral `extra/qk_isa_primitive_audit.py` wrapper (AMD backend only for now) is a
cheap, optional infrastructure follow-on.
