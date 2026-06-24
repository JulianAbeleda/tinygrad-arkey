# Scope - pure-tinygrad prefill speedup via software-pipelined WMMA (no dependency). Redo CG-1 correctly, then integrate.

User decision: **no external dependency.** Drop the Tensile-artifact route (Option A, measured 1.41× llama but ships
a rocBLAS HSACO). Pursue Option B: reproduce what the Tensile kernel does **in tinygrad's own codegen** and improve
the matmul there, so the win is dependency-free and general.

## What Tensile does (the oracle, from CG-0/TCG-0)
tinygrad already MATCHES Tensile on the 128×128×16 macro-tile and the RDNA3 WMMA 16×16×16 fragment. The whole 42→66
TFLOPS gap is **one schedule shape**: a **software-pipelined K-loop** — double-buffered global→LDS→register prefetch
of tile k+1 overlapped with the WMMA of tile k (Tensile: PGR1+PLR1, 1LDSB0 double LDS buffer; ISA: next-tile
`global_load` issued *during* the current WMMA, `vmcnt` deferred). tinygrad's current kernel puts each tile's
`global_load` on the critical path (after the previous barrier), so global-load latency is exposed every iteration.

## The loose end (why CG-1's "fork B / not expressible" is UNPROVEN)
CG-1 (`qk_wmma_pipeline_kernel.py`) added a register prefetch `a_pf=a[k+1]` but **the LDS store still stored
`a[k_tile]` (current), never `a_pf`** → the prefetch was **dead code, DCE'd** → byte-identical ISA. So CG-1 did NOT
actually test a double-buffer; it tested a no-op. The "not UOp-expressible" verdict must be re-run with a correctly
wired pipeline before concluding renderer-level work is required.

## Phases
- **CG-R1 — correct double-buffer (the real test):** rebuild the pipelined WMMA kernel so it genuinely double-buffers:
  two LDS buffers `A[2],B[2]`; prologue loads tile 0 → buf0; loop k: store tile k+1's global → buf[(k+1)%2] (the
  prefetch, wired), barrier, WMMA tile k from buf[k%2] (loaded last iter). The next-tile load and the current WMMA use
  different buffers ⇒ can overlap. Measure TFLOPS + disasm (does `global_load` now issue during the WMMA?).
  - **fork A** (overlap appears, TFLOPS climbs toward 62): the pipeline IS UOp-expressible → it was just the dead-code
    bug. Proceed to CG-R2.
  - **fork B** (linearizer/renderer still serializes a *correctly-wired* double buffer): genuinely a renderer
    capability. Document the exact failure (cross-iteration LDS state in the REDUCE idiom; or clang re-serializing the
    HIP source) and scope the renderer change (CG-R3).
- **CG-R2 (fork A only) — proof kernel:** tune the pipeline (depth, LDS layout, vector widths) to **≥62 TFLOPS
  isolated** on ffn_gate/up; then on ffn_down/attn shapes. Gate before any model wiring.
- **CG-R3 (fork B only) — renderer change spec/build:** the minimal tinygrad codegen addition — `OptOps.PREFETCH` /
  a software-pipelining loop transform + double-LDS-buffer lowering — emitting HIP C++ that clang schedules as a
  pipeline (tinygrad's AMD path emits HIP source; clang does the final waitcnt/regalloc, so the lever is the *source
  structure* tinygrad emits). Scope size honestly (bounded pass vs renderer rewrite).
- **CG-R4 — integrate (either fork):** make the pipelined schedule reachable for the prefill matmul shapes — as a new
  default schedule for these shapes, a warmstart-able opt, or an OptOp — so PREFILL_V2 (and decode where applicable)
  picks it up with NO external artifact. Re-measure in-model warm pp512 + dNLL vs the current PREFILL_V2 and vs llama
  (3394 tok/s re-measured baseline).

## Gates
- CG-R1 decides fork A/B (the corrected test).
- proof kernel must hit **≥62 TFLOPS isolated** before model work; KILL a fork if it stays ≤46 (plateau) or needs
  hand-maintained per-shape assembly.
- in-model (CG-R4): warm pp512 ≥1.25× PREFILL_V2 research / ≥1.35× strong, dNLL ≤0.01, decode unchanged, NO dependency.
- no BEAM on gfx1100 (hangs); the schedule must come from UOp structure or a new pass.

## Non-goals / constraints
No external artifact (the whole point). No model default until a proof kernel clears ≥62 TFLOPS. Reuse
`extra/gemm/amd_copy_matmul.py` as the WMMA base; the Tensile kernel/disasm is the oracle only. Decode untouched until
prefill lands.

## Deliverables
`extra/qk_wmma_pipeline_kernel.py` rewritten with a CORRECT double buffer (CG-R1), `bench/qk-codegen-pipeline/*.json`,
result doc `prefill-codegen-pipeline-redo-result-20260619.md` with the fork A/B verdict (corrected), and — if fork A —
the proof-kernel TFLOPS + the integration plan.
