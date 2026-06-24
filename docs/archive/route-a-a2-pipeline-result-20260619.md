# Route A / A2 result — software pipelining: IMPLEMENTED, CORRECT, +32% — but plateaus below LLVM (KILL)

## What was built (the named A2 lever)
`build_gemm_pipe()` in `extra/gemm/rdna3_wmma_matmul.py`: a **double-buffered, software-pipelined** RDNA3 WMMA
GEMM. Unroll-by-2 (F0 = even-k fragments, F1 = odd-k), prefetch next-k global loads while WMMAs on the current
buffer run, replacing the naive per-iteration `s_waitcnt(0)` full barrier with **targeted `s_waitcnt(vmcnt=LPB)`**
so current WMMAs proceed while next-k loads are in flight. Last prefetch is guarded (loop runs NK/2−1 prefetched
iterations + a no-prefetch tail) to avoid the out-of-bounds over-read that would hang the GPU.

Two encoding/structure facts nailed this session (both were blockers):
- **vmcnt encoding** for this assembler is `expcnt[2:0] | lgkmcnt[9:4] | vmcnt[15:10]` (per the proven in-repo
  `extra/gemm/amd_asm_matmul.py` encoder) — **NOT** the LLVM GFX11 `vmcnt[3:0]|[15:14]` layout. My first attempt
  used the LLVM layout → the targeted wait became a no-op → WMMAs read uninitialized VGPRs → RMSE=nan. Fixed.
- The pipeline **structure** was validated independently with a `FULLWAIT=1` toggle (full barrier → RMSE 0.0002,
  no speedup), isolating the nan to the encoding before fixing it.

## The honest number (standalone, same harness as A1, best-of-3, RMSE 0.0002 CORRECT throughout)
| shape | unpipelined champion | **pipelined (best)** | pipeline gain | LLVM ref |
|---|---:|---:|---:|---:|
| N=2048 square | 18.5 (TM4/TN4) | **24.5** (TM2/TN4) | **+32%** | ~48 warmstart |
| prefill ffn 512×4096×12288 | 29–31 (TM4/TN4) | **32.4** (TM4/TN2) | **+9–11%** | ~42 tinygrad-WMMA-peak / 66 Tensile |

The pipeline **works and helps** — it removed the load-latency stall, which is why the square-shape gain is large
(+32%). At the real prefill shape the big N already amortizes load latency, so the marginal gain shrinks to ~+10%.
**It does not clear the bar (beat LLVM's 48).** Best result ≈ 32 TFLOPS at the prefill shape = ~26% of the 122
TFLOPS hardware peak, ~48% of Tensile's 66, and roughly at parity with tinygrad's *own* LLVM-WMMA peak (~42, POWN).

## Binding reason (now evidenced, not just argued)
The pipeline removed load-latency serialization (the lever the A1 doc named). The **remaining** gap to LLVM/Tensile
is structural and not addressable by single-wave global-load pipelining:
- **One wave32 per workgroup** → no inter-wave latency hiding for the WMMA units themselves. After load latency is
  hidden, throughput is bound by single-wave WMMA issue/occupancy.
- **VGPR cap forces small tiles for the double buffer**: at TM4/TN4 the 128 accumulator VGPRs + 2×64 fragment
  buffers + addresses exceed 256, so the pipeline only fits at TM4/TN2 / TM2/TN4 (acc=64). The amortization vs
  pipelining trade-off is real but both sides land ~24–32 TFLOPS.
- Closing the 32→48→66 gap needs **LDS-staged, multi-wave tiling** (the `extra/gemm/rdna4_asm_matmul.py`-class
  structure: global→LDS→registers, 4 waves/workgroup, `ds_load` interleaved with WMMAs). That is a separate
  multi-day rewrite, and its dependency-free ceiling is **capped by the Infinity-Cache-served caveat (CG-R1)** —
  realistic expected outcome "maybe match LLVM, not Tensile."

## Verdict — KILL (productive)
Per the A2 continuation criterion: *correct but plateaus clearly below 48 with the binding reason identified →
STOP and report.* The named A2 lever (single-wave software pipelining) is **done, correct, and a real +32%/+10%
improvement** over A1 — banked. It does **not** beat LLVM's 48. The only surviving path to ≥48 is the LDS
multi-wave staged rewrite, which is a distinct multi-day project with an IC-served-capped, uncertain ceiling — a
fresh funding decision, not a continuation of this lever.

Fallbacks unchanged: PREFILL_V2 (~80% llama, shipped) or the external Tensile `.co` (1.41× llama, dependency).

## Files / provenance
`extra/gemm/rdna3_wmma_matmul.py` — `build_gemm_pipe()` + `waitcnt_vm()` (this session); `GEMM=1 USEPIPE=1` runs
the pipeline through the A1 standalone harness (the trustworthy measurement); `PIPE=1` runs an interleaved
base-vs-pipe ratio harness (ratio fair, absolutes contaminated by cross-kernel cache thrash — use USEPIPE for
absolutes). A1/binding-constraint: `route-a-rdna3-wmma-result-20260619.md`. Continuation:
`route-a-a2-continuation-prompt-20260619.md`. IC-served caveat: CG-R1. LDS multi-wave reference:
`extra/gemm/rdna4_asm_matmul.py` (RDNA4 encodings, idea-only for RDNA3).
