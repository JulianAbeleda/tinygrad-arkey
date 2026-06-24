# CG-R1 RESULT — software pipeline IS expressible in tinygrad, but is NOT the gfx1100 lever (Infinity-Cache served)

Executed CG-R1 of `prefill-codegen-pipeline-redo-scope-20260619.md`: rebuild the pipelined WMMA kernel with a
**correctly-wired** double buffer (CG-1's prefetch was dead code). **Two findings that overturn the prior verdict:**
(1) the software pipeline **is** UOp-expressible — the overlap appears in the ISA — so CG-1's "fork B / not
expressible" was an artifact of the dead-code bug; (2) but the pipeline **does not speed up gfx1100** — the global
loads are already Infinity-Cache-served, so overlapping them buys nothing. Pure tinygrad, no dependency. Probe:
`extra/qk_wmma_pipeline_kernel.py`; ISA `/tmp/cgr1.txt`.

## Result [M]
| kernel | TFLOPS | correctness |
|---|---:|---|
| single-buffer (CG-0 base, amd_copy) | 48.5 | — |
| **correct double-buffer (CG-R1)** | **46.8** | mse 6.66e-7 (exact) |
| Tensile (oracle) | ~66 | — |

The double-buffer ISA **does** interleave the next tile's `global_load` with the current tile's `v_wmma` (32 wmma + 54
global_load, interleaved in the steady state) — the overlap the oracle showed Tensile uses. **Yet it is 46.8 vs 48.5
TFLOPS — no gain (marginally slower from the extra LDS/barriers).**

## Interpretation — the pipeline is a red herring on gfx1100 [M]
- **CG-1 was a flawed test.** Its prefetch stored `a[k_tile]` (current), never the prefetched `a[k+1]` → dead code →
  DCE'd → byte-identical ISA → the "not UOp-expressible" verdict was unfounded. The pipeline **is** expressible
  (CG-R1 proves it: real double buffer, real overlap, correct output).
- **But overlapping the global load gives ~0 on gfx1100.** The RX 7900 XTX's 96 MB Infinity Cache already serves the
  operand reuse, so the global-load latency was never on the critical path — the same IC effect that refuted
  LDS-tiling for decode-attention (`amd-decode-next-step`) and for prefill (PWLT-A2). Tensile's software pipeline is
  the right lever for *HBM-latency-bound* GPUs; gfx1100 isn't one for this tile.
- **So the 48→66 TFLOPS gap is NOT memory pipelining.** It is **WMMA-issue rate / occupancy / register allocation** —
  Tensile's thread-tile `TT4_64` + scheduling packs the `v_wmma` issue more densely than tinygrad's `WAVES 2×2,
  TM32×TN4` emitted HIP (clang schedules the WMMA issue from tinygrad's source). POWN-1 already swept the bounded
  config space (waves/tiles/BK/noLDS) → 42–48 plateau, every lever regresses. The residual is per-thread
  codegen/issue quality — the same internals wall hit by decode (`per-thread codegen`) and POWN-1.

## Verdict for the pure-tinygrad path
- **Software pipelining: REFUTED as the lever** (expressible, but IC-served → no gain). This corrects the codegen
  oracle (CG-0/TCG-0), which inferred the gap was the pipeline from Tensile's ISA — the pipeline is present in Tensile
  but is not *why* it's faster here.
- **The real gap (WMMA issue/occupancy) is the deep codegen-internals wall** — not a bounded kernel edit and not a
  single named pass like prefetch. Closing it means changing how tinygrad emits/schedules the WMMA issue (clang's
  hands, driven by tinygrad's source structure), which POWN-1's exhausted sweep + this result place in the
  project-level / BEAM-hang class.
- **Net for "no dependency":** there is no cheap pure-tinygrad win for prefill matmul on gfx1100. The bounded levers
  (config sweep, LDS-tiling, software-pipeline) are all refuted; the remainder is deep WMMA-issue codegen. The
  dependency-free prefill stays at PREFILL_V2 (~80% of llama, re-measured); matching llama needs either the external
  Tensile route (1.41× llama, but the rocBLAS HSACO dependency the user declined) or deep AMD-renderer WMMA-issue
  work.

## Files
`extra/qk_wmma_pipeline_kernel.py` (correct double buffer), base `extra/gemm/amd_copy_matmul.py`, ISA `/tmp/cgr1.txt`,
scope `prefill-codegen-pipeline-redo-scope-20260619.md`. Oracle: `prefill-tensile-codegen-oracle-tcg-result-20260619.md`
(now corrected: pipeline is present in Tensile but not the gfx1100 lever). No kernel/model/default changes.
