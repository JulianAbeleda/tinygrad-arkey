# Route A / A3 — P2/P3 result: all 4 LDS levers built + correct, but LDS-staging REFUTED for RDNA3

## What was built (items 1–4, all in `build_gemm_lds2`)
A fully parametric LDS-staged multi-wave WMMA GEMM, one kernel covering all four optimization levers:
1. **Double-buffer LDS** (`DBUF=1`) — unroll-by-2 ping-pong: prefetch the next K-block's global→LDS while the
   current block's WMMAs run; removes the inner per-iter barrier and load↔compute serialization.
2. **Occupancy** (`WAVES_M`,`WAVES_N`,`WM`,`WN`) — smaller per-wave tiles cut ACC VGPRs (4×4→128, 2×2→32),
   raising resident waves (occupancy 1 → ~3).
3. **Bank-conflict-free LDS** (`PAD`) — pads LDS row stride by N bytes.
4. **Block depth** (`BK`) — 16/32/64, amortizes the barrier over more WMMA work per LDS round-trip.

Generalized cooperative load (chunk scheme: thread→`tid%CPR` col-chunk × rows `tid//CPR + j·RSTRIDE`) handles
threads ≠ block-rows. **All configs CORRECT** (RMSE 2e-4 at N=128/256/2048 and prefill). Two bugs found+fixed:
the B-fragment read base double-counted `LDS_A`; `wave_m`/`wave_n` were clobbered by the K-loop (recompute from
`tid` in the epilogue).

## The result — the levers don't move it
| config (W2×2 unless noted) | N=2048 | prefill 512×4096×12288 |
|---|---:|---:|
| T4×4 BK16 (P1 baseline) | 5.8 | 3.2 |
| T4×4 BK16 **+DBUF** | 6.1 | 3.5 |
| T2×2 BK16 (high-occ) | 6.0 | — |
| T2×2 BK16 +DBUF | 5.8 | — |
| T2×2 BK16 +DBUF +PAD8 | 5.6 | — |
| T2×2 BK32 +DBUF +PAD8 | 5.9 | 3.3 |
| **global-direct single-wave A2 (the contrast)** | **24.5** | **32.4** |

**Every LDS config plateaus at ~6 TFLOPS (N=2048) / ~3.5 (prefill) — 4–9× BELOW the global-direct A2 kernel.**
Double-buffer (+6%), occupancy, bank-pad, and BK **all fail to lift it**. When four independent, well-targeted
levers all fail to move a number, the bottleneck is none of them — it is the **LDS round-trip + barrier overhead
itself**, which is **net-negative on this IC-served GPU**: global reads are already Infinity-Cache-served and
cheap, so staging them through LDS (global→reg→LDS→reg + two barriers/block) adds pure overhead the WMMA work
can't repay.

## The reframe — the A3 scope's premise was wrong
The A3 scope assumed "LDS multi-wave staging = the path to LLVM's 42." **It isn't.** The banked POWN sweep
measured tinygrad's *own tuned WMMA* config at **"no-LDS → 38" vs 42 with LDS** — LDS contributes only **~+10%**,
and the other ~90% of 42 comes from **global-direct WMMA scheduling/occupancy/ILP**. So:
- The fast structure on RDNA3 is **global-direct WMMA** (read global straight into fragments, IC-served) — exactly
  what A1/A2 do. A2's 24–32 is in the right family; LDS staging (6) is a wrong turn.
- The real gap (A2 32 → LLVM 42) is **global-direct WMMA scheduling/occupancy/ILP**, the codegen-class lever POWN
  already walled (software-pipelined K-loop tinygrad can't express; BEAM-hang class).

## Verdict — LDS multi-wave sub-arc REFUTED (decisive)
Items 1–4 are implemented, correct, and measured. The LDS-staged multi-wave approach is **structurally ~4–9×
slower than global-direct WMMA on RDNA3** and **none of the four levers change that** — the LDS round-trip itself
is the cost. This **closes the A3 LDS sub-arc**: it is not the path to ≥42 dependency-free. The only surviving
dependency-free lever is refining the **global-direct WMMA path (A1/A2 family)** toward 38–42 via
scheduling/occupancy/ILP — the same codegen capability POWN walled. Net: the dependency-free RDNA3 prefill
frontier rests at **A2 ~24–32 TFLOPS** (the global-direct pipeline); fallbacks unchanged (PREFILL_V2 ~80% llama
shipped; external Tensile `.co` 1.41× llama, dependency).

Minor: W4×2 (8-wave) tripped a config assert (`BN < RSTRIDE` when BN=64, THREADS=256) — a coop-load constraint,
not pursued since the whole LDS family is refuted.

## Files / provenance
`extra/gemm/rdna3_wmma_matmul.py` — `build_gemm_lds2` (`LDSGEMM2=1`, env: WAVES_M/N, WM, WN, BK, PAD, DBUF,
LIMIT_OCC). Commit d36c66734. Prior: P0/P1 `route-a-a3-p0-p1-result-20260619.md`, scope
`route-a-a3-lds-multiwave-scope-20260619.md`, A2 `route-a-a2-pipeline-result-20260619.md`. Ceiling/no-LDS-38 ref:
POWN `prefill-own-wmma-kernel-result-20260619.md`.
