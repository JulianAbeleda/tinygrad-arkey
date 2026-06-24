# TPE-2 RESULT — ffn_gate/up rocBLAS launch contract fully recovered (PASS)

Executed Phase TPE-2 of `prefill-tensile-primitive-extraction-and-codegen-scope-20260619.md`: produce a
machine-readable launch contract for the selected ffn_gate/up Tensile kernel so it can be launched through tinygrad
HCQ (no HIP runtime). **Verdict: PASS** — kernarg layout, descriptor, code object, launch geometry, and workspace
contract are all recovered from installed files, with **no hidden args and no private runtime state**. Probe:
`extra/qk_tensile_contract.py`; artifact: `bench/qk-tensile-extraction/ffn_gate_up_contract.json`. No tinygrad route,
no defaults, decode untouched.

## How (no HIP runtime, no deps) [M]
1. The Tensile `.co` is a **compressed clang offload bundle** (`CCOB` magic), not a bare ELF. Unbundled the
   gfx1100 code object: `clang-offload-bundler --type=o --unbundle --targets=hipv4-amdgcn-amd-amdhsa--gfx1100`.
2. The GEMM kernel is in the **`Ailk_Bljk`** layout `.co` (`TensileLibrary_Type_HH_HPA_Contraction_l_Ailk_Bljk_…gfx1100.co`),
   **not** the `Kernels.so-000-gfx1100.hsaco` (which holds only PostGSU/helper kernels).
3. Tensile emits **one `NT_AMDGPU_METADATA` note per kernel** (290 in this object); hand-rolled an ELF-note +
   msgpack parser (no pyelftools/msgpack deps) to scan all notes and find the selected `.name`.

## The recovered contract [M] (`Cijk_…MT128x128x16…AMAS0…GLVWA8…SU0…WGM8`, 512×4096→12288)

| field | value |
|---|---|
| descriptor symbol | `<kernel>.kd` (present; resolvable by name) |
| code object | unbundled gfx1100 ELF from the `Ailk_Bljk` `.co` (CCOB), sha pinned |
| **kernarg_segment_size** | **128 bytes**, align 8 |
| group (LDS) / private | 25088 B / 0 |
| sgpr / vgpr / wavefront | 58 / 256 / 32 |
| launch geometry | grid (512, 96, 1), workgroup (128,1,1) — fixed shape, from TPE-1 trace |
| workspace | **none** (SU0_SUM0_SUS0 = no StreamK/GSU; scratch 0; single kernel) |
| **hidden args** | **NONE** (0) — no hidden block/grid/remainder/global-offset fields |

**kernarg layout — 19 `by_value` args (the pointers are by-value int64 VAs, not HIP buffer descriptors):**
```
off  0  Tensor2dSizeA(8)   off  8 Tensor2dSizeB(8)
off 16  AddressD(8)        off 24 AddressC(8)   off 32 AddressA(8)   off 40 AddressB(8)
off 48  Alpha(4 fp32)      off 52 Beta(4 fp32)
off 56  StridesD(8)        off 64 StridesC(8)   off 72 StridesA(8)   off 80 StridesB(8)
off 88  SizesFree(12=3×i32: M,N,batch)          off 100 SizesSum(4=K)
off 104 NumWorkGroups0(4)  off 108 NumWorkGroups1(4)
off 112 NumFullBlocks(4)   off 116 WgmRemainder1(4)  off 120 MagicNumberWgmRemainder1(4)
```

## Why this is a strong PASS
- **No hidden args / no private state** — the TPE-2 kill gates ("hidden args runtime-generated opaquely", "requires
  private runtime state") do **not** trigger. The kernel needs only caller buffers (A/B/C/D VAs) + scalars in a
  128-byte kernarg buffer + the fixed launch geometry. This is exactly what tinygrad HCQ can fill.
- **Reproducible from installed files**, no HIP runtime in-process (unbundling + ELF/msgpack parsing are host-side).

## The one TPE-3 prerequisite (bounded, not opaque)
The last 5 kernarg fields are Tensile's **WGM8 workgroup-remapping** values for the fixed shape:
`NumWorkGroups0/1`, `NumFullBlocks`, `WgmRemainder1`, `MagicNumberWgmRemainder1` (a magic-number division constant).
These are **static for the 512×12288 shape** and must be computed via Tensile's WGM formula **or** captured once from
the kernarg buffer rocBLAS builds (a one-time HIP-side trace, separate process — Lane A bars in-process HIP, not a
separate trace). This is a known bounded computation, not opaque runtime state — so it does **not** kill TPE-2; it is
the first task of TPE-3.

Also confirmed (runtime constraint): `AMDProgram` loads the **first** `.kd` of a multi-kernel object, so TPE-3 needs
a **named-descriptor loader** (resolve `kernel_symbol.kd` by name; set `aql_prog_addr`/segment sizes/resource regs
from it) or a single-kernel slice.

## Verdict + next step
**TPE-2 PASS → proceed to TPE-3 (minimal HCQ launch proof)**: build a named-descriptor loader + fill the 128-byte
kernarg from this contract (compute the WGM fields), launch the kernel from tinygrad HCQ on tinygrad-owned A/B/C
buffers, and verify vs the fp16 oracle. Target: correct output, no copies, no HIP runtime. If the WGM fields or the
named-descriptor load block it, that is where Lane B kills.

## Files
`extra/qk_tensile_contract.py`, `bench/qk-tensile-extraction/ffn_gate_up_contract.json`, this doc. Provenance:
`prefill-tensile-tpe1-selection-result-20260619.md`, `prefill-tensile-primitive-extraction-and-codegen-scope-20260619.md`.
No kernel/model/default changes.
