# Scope — TPE-7c/d UOp injection (build it) + codegen-transfer oracle (TCG). Do both.

Two tracks, both research-only (no ship/bank/default; external artifact research-only):
- **Track 1 (build the injection):** get the precompiled Tensile kernel to run as a tinygrad realize/JIT graph node
  driven by `TensileRunner` (TPE-7b, conformant). Then route a one-block harness and, if it lands, in-model behind
  `PREFILL_TENSILE_GEMM=1`; measure.
- **Track 2 (codegen-transfer oracle):** disassemble the extracted Tensile kernel, recover its schedule anatomy, and
  produce the concrete "Tensile does X / tinygrad does Y / missing capability Z" table — the oracle for a future
  pure-tinygrad GEMM. Sidesteps the external dependency.

## Track 1 — UOp injection (the concrete plan)
Confirmed mechanism: HCQGraph (graph/hcq.py:175) does `q.exec(runtime, ji_args[j], ast.arg.global_size,
ast.arg.local_size)` — `runtime` = the object from `get_runtime`, dims from the PROGRAM UOp's `ProgramInfo`. So
injection = a PROGRAM UOp whose `ProgramInfo` carries **Tensile dims** (global=(4,96,1), local=(128,1,1) for
ffn_gate/up) + `TensileRunner` returned by `get_runtime`.

Approach (least-fragile, empirical):
1. **TPE-7c-0 probe:** build a minimal `custom_kernel` (trivial store) for the matmul out-shape and inspect the
   realized `Ops.PROGRAM` UOp — its `ProgramInfo` (global_size/local_size/outs/ins/vars), `.key`, and how
   `runtime_cache` is keyed. This tells us the exact structure to produce/patch.
2. **TPE-7c-1 runtime bind:** put a `TensileRunner` into `runtime_cache[(program.key, 'AMD')]` so realize/HCQGraph
   uses it instead of the codegen'd AMDProgram.
3. **TPE-7c-2 dims:** ensure the PROGRAM UOp's `ProgramInfo.global_size/local_size` = Tensile's — either by shaping
   the `custom_kernel` ranges to those dims, or by `program_uop.replace(arg=replace(arg, global_size=…))` + re-keying
   the cache. The Tensile kernel reads NumWorkGroups from its kernarg, but the HW still needs the right grid.
4. **TPE-7c-3 buffers/correctness:** confirm the CALL UOp's bufs map to Tensile's (out,A,B) so `TensileRunner.
   fill_kernargs(bufs)` binds the right VAs; verify output vs fp16 oracle through realize, then under TinyJit.
5. **TPE-7c-4 one-block harness:** route the FFN block ([feature,T] space) through injected nodes; verify + time.
6. **TPE-7d:** in-model behind `PREFILL_TENSILE_GEMM=1` (no default), warm pp512/pp1024 + dNLL ≤0.01, decode
   untouched, clean fallback. Gate ≥1.25× research / ≥1.35× strong.

KILL: PROGRAM UOp can't carry a precompiled lib + Tensile dims through realize without per-call recompile; bufs/dims
can't be made to match; JIT capture drops or mis-schedules the node; correctness/quality fails.

## Track 2 — codegen-transfer oracle (TCG-0/1 from the parent scope)
1. **TCG-0 schedule anatomy:** disassemble the selected `Cijk_…MT128x128…` kernel (llvm-objdump on the unbundled
   ELF); extract macro-tile, MFMA/WMMA shape, depthU, global-read vector widths, LDS layout, prefetch
   (PGR/PLR), independent accumulators, VGPR/SGPR. Cross-read the `.dat`/kernel-name fields.
2. **TCG-1 capability delta:** table of "Tensile does X, tinygrad POWN-1 does Y, missing capability Z" labeled as
   frontend-schedule / UOp-expressibility / renderer-scheduling / register-alloc / runtime-contract. Identify the
   smallest tinygrad codegen change that could plausibly move 42→≥62 TFLOPS; if it's "rewrite the AMD
   renderer/scheduler", mark project-level.
3. Output: `docs/prefill-tensile-codegen-oracle-tcg-result-20260619.md` + `bench/qk-tensile-extraction/codegen_oracle.json`.

## Constraints
No model default; decode untouched; research flag only; reuse `TensileRunner` + committed captures; keep probes
local; fallback to PREFILL_V2 on any unsupported shape/device. External-artifact dependency is research-only.

## Deliverables
Track 1: `extra/qk_tensile_inject.py` (+ one-block / in-model harness if it lands), `bench/qk-tensile-extraction/inject.json`,
`docs/prefill-tensile-tpe7cd-injection-result-20260619.md`. Track 2: `extra/qk_tensile_disasm.py`,
`bench/qk-tensile-extraction/codegen_oracle.json`, `docs/prefill-tensile-codegen-oracle-tcg-result-20260619.md`.
