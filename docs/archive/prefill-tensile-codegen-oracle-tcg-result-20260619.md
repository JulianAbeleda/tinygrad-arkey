# TCG-0/1 RESULT — codegen-transfer oracle: the exact tinygrad capability gap to the Tensile prefill kernel

Executed the codegen-transfer-oracle track (TCG-0/1) of `prefill-tensile-tpe7cd-injection-and-codegen-oracle-scope-20260619.md`:
recover the selected Tensile GEMM kernel's schedule anatomy and produce the concrete "Tensile does X / tinygrad does
Y / missing capability Z" table — the oracle for a future pure-tinygrad GEMM (sidesteps the external dependency).
Probe: `extra/qk_tensile_disasm.py`; artifact: `bench/qk-tensile-extraction/codegen_oracle.json`.

## Tensile schedule anatomy [M] (`Cijk_…MT128x128x16_MI16x16x16x1…`, from the exhaustive name + disasm of the unbundled ELF)
| field | value | meaning |
|---|---|---|
| macro-tile M×N×K | 128×128×16 | per-workgroup output tile + DepthU 16 |
| WMMA fragment | 16×16×16 (`v_wmma` ×13810 in-object) | RDNA3 WMMA, same primitive tinygrad uses |
| **PGR / PLR** | **1 / 1** | **prefetch global-read + prefetch local-read** (software pipeline) |
| **1LDSB** | **1LDSB0 = DOUBLE LDS buffer** | next K-tile streamed into buffer B while WMMA consumes buffer A |
| LDS reads | `ds_load_b128` ×9324 (LRVW16) | wide 128-bit vectorized LDS loads |
| thread-tile | TT 4×64, vgpr256, **no spill** | enough independent accumulators to hide WMMA latency |
| global load vec | GLVWA/B 4, GRVW 4 | 64-bit coalesced global reads |

The instruction mix (13810 `v_wmma`, 9324 `ds_load_b128`, 2144 `ds_store_b128`, 55224 `vmcnt` waits) is the fingerprint
of a **double-buffered, software-pipelined K-loop**: load/store LDS for the next tile overlapped with WMMA on the
current one.

## Capability delta — Tensile vs tinygrad POWN-1 [M]
tinygrad POWN-1 best (`docs/prefill-own-wmma-kernel-result-20260619.md`): 128×128×16, WMMA 16×16×16, waves 2×2, **42
TFLOPS** — *every* lever regressed (waves→28 TF, more-acc→11 TF on spill, no-LDS→38 TF).

| aspect | Tensile (~66 TF) | tinygrad (42 TF) | missing capability | class |
|---|---|---|---|---|
| macro-tile | 128×128×16 | 128×128×16 | **none** (identical) | — |
| WMMA fragment | 16×16×16 | 16×16×16 | **none** (both RDNA3 WMMA) | — |
| **K-loop prefetch** | PGR1+PLR1, double LDS buffer → next tile prefetched during WMMA | single-buffered, no pipeline (no-LDS within 10% ⇒ tinygrad's LDS staging buys ~0 without the overlap) | **software-pipelined K-loop** (double-buffer global→LDS→reg prefetch overlapped with WMMA issue) | **renderer instruction-scheduling** — NOT UOp-expressibility |
| accumulators | TT4×64, vgpr256, no spill | more-acc → spill (11 TF) | spill-free large-accumulator allocation | **register allocation** |
| LDS read width | LRVW16 / `ds_load_b128` | narrower | wide vectorized LDS reads (minor) | renderer codegen (minor) |

## The oracle conclusion [M]
tinygrad already **matches Tensile on the two things that look hardest** — the 128×128 macro-tile and the RDNA3 WMMA
fragment. The entire 42→~66 TFLOPS gap is **one schedule shape it cannot emit**: a *software-pipelined K-loop* that
double-buffers the global→LDS→register prefetch and overlaps it with WMMA issue, plus the *spill-free
large-accumulator allocation* that pipeline needs. Both are **AMD-renderer instruction-scheduling / register-allocation
capabilities, not frontend UOp expressibility**, so the smallest closing change is **project-level** (extend the AMD
renderer's scheduler), the same BEAM-hang-class wall POWN-1 already hit.

This is the value of the oracle: it converts "llama/rocBLAS is faster" into a single, named, located codegen target —
*software-pipelined double-buffered K-loop + spill-free accumulators in the AMD renderer* — that a future pure-tinygrad
effort can aim at, with the extracted kernel as the exact schedule to imitate and the 66 TFLOPS as the bar. It needs
no external artifact.

## Files
`extra/qk_tensile_disasm.py`, `bench/qk-tensile-extraction/codegen_oracle.json`, this doc. Provenance:
`prefill-own-wmma-kernel-result-20260619.md` (POWN-1), `prefill-tensile-tpe5-shape-matrix-result-20260619.md`. No
kernel/model/default changes.
