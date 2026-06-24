# Scope — pure-tinygrad software-pipelined WMMA GEMM (close prefill 42→≥62 TFLOPS), the codegen-transfer target

The no-external-dependency path to llama-class prefill. The codegen oracle (`prefill-tensile-codegen-oracle-tcg-result-20260619.md`)
proved tinygrad already MATCHES the Tensile kernel on the two hardest things — the **128×128×16 macro-tile** and the
**RDNA3 WMMA fragment** — and that the entire 42→~66 TFLOPS gap is two schedule capabilities tinygrad lacks:
1. **software-pipelined K-loop**: double-buffered global→LDS→register prefetch, overlapped with WMMA issue
   (Tensile: PGR1+PLR1, 1LDSB0 double LDS buffer, ds_load_b128 prefetch);
2. **spill-free large-accumulator allocation**: hold the full thread-tile accumulator in VGPRs (Tensile: TT4×64,
   vgpr256, no spill; tinygrad POWN-1 spills → 11 TFLOPS when accumulators grow).

This scope decides whether either is **expressible/buildable in tinygrad today** (a proof kernel) or requires a
**new renderer/optimizer capability** (project-level), and — if buildable — hits ≥62 TFLOPS isolated.

## Grounding facts (established)
- **tinygrad has NO software-pipelining opt.** `OptOps = {TC, UPCAST, UNROLL, LOCAL, THREAD, GROUP, GROUPTOP,
  NOLOCALS, PADTO, SWAP}` (codegen/opt/__init__.py). There is no PREFETCH / DOUBLE_BUFFER / PIPELINE op → the
  capability is **absent**, not merely BEAM-gated. BEAM cannot "find" a pipeline it has no action for.
- **POWN-1 exhausted the bounded config sweep** (waves/tiles/BK/noLDS) → 42 TFLOPS plateau; no-LDS was within 10%
  ⇒ tinygrad's LDS staging buys ~0 *without* the prefetch overlap. Every lever regressed.
- **`extra/gemm/amd_copy_matmul.py`** is the existing hand-UOp WMMA kernel (single-buffered, the POWN base).
- The Tensile schedule fingerprint (oracle): v_wmma ×13810, ds_load_b128 ×9324, ds_store_b128 ×2144, double LDS
  buffer, ~55k vmcnt waits = overlapped load/compute.

## The central fork (CG-1 decides it)
Either:
- **(A) UOp-expressible:** a hand-written `custom_kernel` can stage the double buffers and issue the next tile's loads
  before the current tile's WMMA, and tinygrad's linearizer/renderer *preserves* that ordering (doesn't serialize or
  hoist a premature `s_waitcnt`). Then it's a **buildable proof kernel** → ≥62 TFLOPS is reachable in pure tinygrad,
  and model integration is ordinary tinygrad (no injection, no external artifact).
- **(B) renderer/optimizer capability:** the linearizer collapses/serializes the manual pipeline (it schedules loads
  and WMMA in program order with conservative `s_waitcnt`), so overlap needs a real codegen change — a new
  software-pipelining pass or `OptOps.PREFETCH` + renderer support for double-buffered LDS and `s_waitcnt` placement.
  Then it's **project-level** (extend the AMD renderer scheduler), the deep-capability wall.

## Phases
- **CG-0 — ISA diagnosis (cheap):** disassemble the POWN-1/amd_copy WMMA kernel; quantify the gap vs Tensile —
  global-load↔WMMA overlap (are next-tile loads issued before the current WMMA?), single vs double LDS buffer,
  `s_waitcnt` density, accumulator VGPR live-range. Confirms the exact missing instruction-ordering.
- **CG-1 — UOp-expressibility test (the decision):** hand-build a 2-stage software-pipelined K-loop in a
  `custom_kernel` on `amd_copy_matmul`'s base: explicit LDS buffer A/B, prefetch tile k+1 global→LDS while WMMA
  consumes tile k from the other buffer, swap per iteration. Disassemble + measure TFLOPS.
  - overlap preserved & TFLOPS climbs toward 62 → **fork A** → CG-2.
  - linearizer serializes it (ISA shows no overlap, TFLOPS ~42) → **fork B** → CG-3.
- **CG-2 — proof kernel (fork A only):** full software-pipelined WMMA GEMM for ffn_gate/up; tune pipeline depth
  (deeper PGR), LDS layout, vector widths. **Gate: ≥62 TFLOPS isolated** before any model work.
- **CG-3 — capability spec (fork B only):** specify the minimal renderer/optimizer change — an `OptOps.PREFETCH` (or a
  software-pipelining rewrite pass) + double-buffered-LDS lowering + `s_waitcnt` scheduling that overlaps load with
  WMMA. Estimate scope (bounded pass vs "rewrite the AMD scheduler"); per the parent gate, if it's the latter, mark
  **project-level**.
- **CG-4 — register-allocation sub-problem (both forks):** diagnose POWN-1's accumulator spill (more-acc→11 TF). Is
  it allocation order, too many live LDS-staging temps, or WMMA-acc placeholder handling? Determine if fixable at the
  UOp level (fewer live temps / explicit REG tiling) or needs renderer regalloc work.

## Gates / kill
- CG-1 is the go/no-go: UOp-expressible (proof-kernel path) vs renderer-capability (project-level).
- CG-2 proof kernel must reach **≥62 TFLOPS isolated** before model integration; KILL if it stays ≤46 (plateau) or
  needs hand-maintained per-shape assembly (that's the rejected Lane C).
- No BEAM on gfx1100 (hangs); the pipeline must come from UOp structure or a new pass, not BEAM search.

## Why this matters / relationship to the injection track
The injection track (TPE-7c, eager-proven) reaches ~95% llama prefill via an **external** rocBLAS artifact. This
codegen track reaches the same bar with **no dependency** — if fork A holds. The two are complementary: the extracted
Tensile kernel is the **oracle** (the exact schedule to imitate, 66 TFLOPS the bar); this scope tries to reproduce its
schedule in tinygrad's own codegen. A fork-A proof kernel would also generalize beyond this model (any fp16 GEMM) and
beyond gfx1100, unlike the per-shape Tensile extraction.

## Constraints
No model route; no defaults; decode untouched; reuse `amd_copy_matmul` as the WMMA base; no BEAM. Probe-local until a
proof kernel clears ≥62 TFLOPS isolated. This is a renderer/codegen research arc, not a model change.

## Deliverables
`extra/qk_wmma_pipeline_diag.py` (CG-0 ISA diagnosis), `extra/qk_wmma_pipeline_kernel.py` (CG-1 hand-pipelined kernel),
`bench/qk-codegen-pipeline/*.json`, result doc `prefill-codegen-software-pipeline-result-20260619.md` with the fork
A/B verdict and (fork A) the proof-kernel TFLOPS or (fork B) the renderer-capability spec.
