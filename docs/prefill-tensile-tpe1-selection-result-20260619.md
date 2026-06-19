# TPE-1 RESULT — ffn_gate/up Tensile solution identified (PASS); a gate-vs-extractability tension for TPE-2

Executed Phase TPE-1 of `prefill-tensile-primitive-extraction-and-codegen-scope-20260619.md` (Lane B: extract the
selected Tensile kernel to run through tinygrad HCQ). TPE-0 artifact authority: research-only Tensile dependency
accepted. **Verdict: PASS** — the ffn_gate/up selected solutions are observable, stable, named `Cijk_*` symbols with
identified code objects and single-kernel (no GSU/aux) flow. Method is HIP-only (Lane A proved the HIP runtime can't
coexist with tinygrad `DEV=AMD`). Probe: `extra/qk_tensile_selection.py`; artifact:
`bench/qk-tensile-extraction/selection.json`. No tinygrad route, no defaults, decode untouched.

## Method [M]
Built the standalone ceiling binary (`extra/qk_prefill_blas_ceiling.cpp`, host C++ + rocBLAS + hipBLASLt) and ran it
under `rocprofv3 --kernel-trace --output-format csv`. Parsed the kernel trace (`extra/qk_tensile_selection.py`):
identified the ffn_gate/up dispatch per library (rocBLAS kernels are shape-specific → `grid_y = N/128 = 96`;
hipBLASLt uses one generic `UserArgs` kernel for all shapes). Exact commands in `selection.json.method` + the probe
docstring.

## Selected solutions for ffn_gate/up (512×4096→12288, fp16→fp32) [M]

| | rocBLAS | hipBLASLt |
|---|---|---|
| kernel symbol | `Cijk_Ailk_Bljk_HHS_BH_MT128x128x16_MI16x16x16x1_…AMAS0…GLVWA4_GLVWB4…SU0…WGM8` | `Cijk_Ailk_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT96x96x32_MI16x16x1…` |
| macro-tile | 128×128×16 | 96×96×32 |
| grid / workgroup | (512, 96, 1) / (128,1,1) | flattened (33024,1,1) / (128,1,1) |
| VGPR / LDS / scratch | 256 / 25088 B / 0 | 256 / 30720 B / 0 |
| TFLOPS (trace) | **62.9** | **64.8** (PXB-1 isolated 69.8) |
| UserArgs? | **no** | **yes** |
| code object | `…/rocblas/library/Kernels.so-000-gfx1100.hsaco` (881 KB) | `…/hipblaslt/library/Kernels.so-000-gfx1100.hsaco` (22 MB) |
| auxiliary kernels | **none** (only `__amd_rocclr_fillBufferAligned`); single-kernel GEMM, no GSU/fixup | none |

## Gates → PASS
- selected solution known (kernel symbol + code object) ✓ for both libs;
- stable kernel symbol (named `Cijk_*`, not runtime-opaque) ✓;
- single-kernel fast path (no opaque multi-kernel GSU/fixup for ffn_gate/up) ✓ — avoids the TPE-1 kill;
- standalone timing consistent with PXB-1 (trace TFLOPS within range; trace includes warmup) ✓.

## The key finding for TPE-2 — extractability vs the gate
There is a real tension the next phase must weigh:
- **rocBLAS kernel** = `MT128x128…AMAS0`, **no UserArgs** → the *simplest* launch contract (direct kernarg layout),
  in the *smaller* 881 KB code object → **most extractable.** But it runs **~61–63 TFLOPS — at/just under the 62
  TFLOPS gate.**
- **hipBLASLt kernel** = `MT96x96…UserArgs` → **clears the gate (~69.8 TFLOPS)**, but uses the **UserArgs** indirect
  kernarg/bias/aux convention in a 22 MB code object → a **harder launch contract** to recover.

So TPE-2 (launch-contract extraction) should **start with the rocBLAS shape-specific kernel** (the extractable
contract) and accept ~61–63 TFLOPS as the first HCQ-launch target — that already proves the route and is ~1.5× the
tinygrad 42 TFLOPS plateau, just shy of the 62 gate. Only if a strict ≥62 floor is mandatory does TPE-2 need the
hipBLASLt UserArgs path (and its harder contract). Either way, the **runtime constraint stands**: `AMDProgram` loads
the *first* descriptor from a multi-kernel `Kernels.so`, so TPE-2/TPE-3 need a **named-symbol/named-descriptor
loader** (resolve `kernel_symbol.kd` by name) or a single-kernel HSACO slice.

## Verdict + next step
**TPE-1 PASS → proceed to TPE-2 (launch-contract extraction) for the rocBLAS `MT128x128…AMAS0` ffn_gate/up kernel**:
parse the HSACO ELF for the function + `.kd` descriptor, recover the kernarg byte layout (pointer/value order,
hidden dispatch fields) from the AMDGPU metadata, and the fixed-shape launch geometry — producing
`bench/qk-tensile-extraction/ffn_gate_up_contract.json`. If the kernarg/hidden-arg layout proves opaque or
runtime-generated, TPE-2 kills Lane B and only the codegen-transfer arc (TCG) remains.

## Files
`extra/qk_tensile_selection.py`, `bench/qk-tensile-extraction/selection.json`, this doc. Reuses
`extra/qk_prefill_blas_ceiling.cpp`. Provenance: `prefill-tensile-primitive-extraction-and-codegen-scope-20260619.md`,
`prefill-external-bridge-ebt1-result-20260619.md`. No kernel/model/default changes.
