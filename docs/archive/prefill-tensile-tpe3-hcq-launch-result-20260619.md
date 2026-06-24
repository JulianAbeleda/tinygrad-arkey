# TPE-3 RESULT — rocBLAS Tensile ffn_gate/up kernel launched from tinygrad HCQ (PASS) — Lane B is real

Executed Phase TPE-3 of `prefill-tensile-primitive-extraction-and-codegen-scope-20260619.md`: launch the selected
rocBLAS Tensile ffn_gate/up kernel from tinygrad HCQ on tinygrad-owned buffers, with **no HIP runtime and no copies**.
**Verdict: PASS.** This is the point where Lane B becomes real instead of dying: the mature backend's compiled kernel
runs inside tinygrad's command-queue path, correct and stable. rocBLAS contract only (no hipBLASLt/UserArgs). No model
route, no defaults, decode untouched. Probe: `extra/qk_tensile_{kernarg_capture.cpp,hcq_launch.py}`; artifacts:
`bench/qk-tensile-extraction/{hcq_launch.json,kernarg_capture.json}`.

## Result [M]
6 consecutive launches, byte-identical: **rel_err 0.00035** vs the tinygrad fp16 oracle (tol 2e-2), max_abs 0.1287,
**no hang/corruption, fully stable.**

| TPE-3 PASS gate | result |
|---|---|
| no HIP runtime in tinygrad process | ✓ (named-descriptor loader + raw kernarg + HCQ dispatch; HIP only in the separate capture) |
| named descriptor loads correctly | ✓ (resolved `<kernel>.kd` by name from the 305-kernel object) |
| kernel launches through HCQ | ✓ |
| output correct vs oracle | ✓ rel 0.00035 |
| no copies | ✓ (tinygrad buffer VAs written straight into the kernarg) |
| repeated launches stable | ✓ 6/6 identical |

## How (three pieces)
1. **Named-descriptor loader** (`NamedAMDProgram`, probe-local subclass — *no runtime change*): unbundle the `.co` →
   gfx1100 ELF; parse the symtab for `<kernel>.kd` `st_value`; use that as `rodata_entry` (AMDProgram normally takes
   the *first* `.rodata` descriptor). Everything else (rsrc1/2/3, `prog_addr = base + rodata_entry +
   kernel_code_entry_byte_offset`, segment sizes) derives from that descriptor.
2. **Exact kernarg by capture** (scope-blessed): an `LD_PRELOAD` shim (`qk_tensile_kernarg_capture.cpp`) intercepts
   `hipExtModuleLaunchKernel` in a **separate HIP-only** rocBLAS run and dumps the 128-byte kernarg + launch geometry
   for the ffn_gate/up GEMM (identified by the embedded sizes M=512,N=12288,K=4096). This **removed all WGM
   guesswork** — the 5 WGM8 fields came out concrete: `NumWG0=4, NumWG1=96, NumFullBlocks=12, WgmRemainder1=8,
   MagicNumberWgmRemainder1=0x10000001`. Lane A's in-process-HIP ban does not bar a separate-process capture.
3. **Raw-kernarg launch**: override `fill_kernargs` to write the captured 128 bytes verbatim, substituting only the 4
   `Address` VAs (D/C at off16/24, A at off32, B at off40) with the tinygrad buffer pointers. Launch
   `global=(4,96,1)` workgroups × `local=(128,1,1)` (→ grid 512×96×1) through the HCQ compute queue, `wait=True`.

GEMM verified (from captured strides): col-major `C[512,12288] = A[512,4096]·B[4096,12288]`, α=1/β=0 — mapped to
row-major tinygrad tensors (A_t[K,M], B_t[N,K], C_t[N,M]) with oracle `C_t = B_t @ A_t`.

## One robustness note (hardened)
The kernel's `.kd` descriptor `kernarg_size` field reads **0** (the true 128 lives only in the metadata). The probe
now allocates `max(desc.kernarg_size, len(raw)) = 128` so it never writes past the kernarg allocation — PASS holds
identically after the fix.

## Verdict + next step
**TPE-3 PASS → proceed to TPE-4 (isolated performance parity).** Lane B is a runnable primitive: the rocBLAS Tensile
ffn_gate/up kernel executes through tinygrad HCQ on tinygrad-owned buffers, correct and stable, with no HIP runtime
and no copies. TPE-4 measures device time / TFLOPS through this HCQ launch and compares to the **≥62 TFLOPS** target
(≥90% of the PXB-1 HIP-only time). If it lands near the 42 TFLOPS tinygrad plateau instead, the gain is a launch/setup
issue to diagnose; if it reaches ~62, continue to the shape matrix (TPE-5).

## Files
`extra/qk_tensile_kernarg_capture.cpp`, `extra/qk_tensile_hcq_launch.py`,
`bench/qk-tensile-extraction/{hcq_launch.json,kernarg_capture.json}`, this doc. Provenance:
`prefill-tensile-tpe2-contract-result-20260619.md`. No kernel/model/default changes; no runtime files modified.
