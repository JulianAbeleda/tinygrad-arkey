# PROMOTED: Dependency-Free RDNA3 Prefill GEMM — ~96% of Tensile, +11% over LLVM authority

Date: 2026-06-20

## The banked result

A correct, **dependency-free** hand-asm RDNA3 GEMM for the ffn_gate/up prefill shape (M=512, N=12288, K=4096),
consistently at **~96% of the vendored Tensile `.co`** and **+11% over tinygrad's own LLVM-WMMA authority**,
measured cleanly and reproduced across 3 independent sessions.

**Canonical config:** `build_gemm_lds2(M, N, K, WAVES_M=2, WAVES_N=2, WM=4, WN=4, BK=32, PAD=16, DBUF=0,
PLRA=1)` launched at **wg2 LDS (32768 B)**, square 128×128 tile. (`extra/gemm/rdna3_wmma_matmul.py`.)

## Benchmark (each kernel measured ALONE, pinned clock, best-of-120, 3 sessions × 5 rounds)

| | median TFLOPS |
|---|---:|
| Tensile `.co` (vendored dep) | ~64.2 |
| **ours (dependency-free)** | **~61.6** |
| LLVM authority (tinygrad's own WMMA) | ~55.5 |

- **ours / Tensile = 0.96** (median 0.955–0.973 across sessions; run-to-run 0.93–0.98).
- **ours / authority = +11%** (consistent).
- Correctness re-checked after timing: **rel RMSE 2.08e-4**.

**Method matters:** each kernel is measured *alone* in its own tight loop (NOT interleaved — interleaving a
foreign-`.co` launch perturbs our kernel, which caused the earlier spurious "57 / 0.92×"). Clock pinned via
`rocm-smi --setperflevel high`. `extra/qk_amd_gemm_promotion_benchmark.py`.

## Honest framing

- **~96%, not parity.** The fact-check's "63 vs 62 (ours ahead)" was optimistic — the clean each-alone Tensile
  number is ~64, so ours (~61.6) is a stable ~4% behind. The truth sits between the artifact-low "92%"
  (interleave perturbation) and the optimistic "parity": **consistently ~96%.**
- This is on a shape **Tensile never tuned** (M=512 and N=12288 both off its tuning grid); the `.co` is a
  nearest-neighbor fallback, and the realistic tuned ceiling for this shape is ~65 (the representative cluster,
  not the M=384=79 outlier). So ~96% of the fallback `.co` ≈ ~95% of the realistic ceiling.
- Cross-library absolute ratios mix clock (`.dat` offline vs ours pinned); the robust claims are the
  **same-session each-alone-pinned ratio (~0.96, reproduced 3×)** and the **+11% over the LLVM authority**.

## How it was built (the levers that mattered, all measured)

| lever | effect |
|---|---|
| wg2 occupancy (LDS-pad to 32768) | avoids the wg4 L2-contention dip |
| **PAD16 bank-conflict-free LDS** | **+13%** (PMC bankcf 28.6→2.7/cyc) — the biggest single win |
| A-prefetch PLR (into dead coop-temp regs) | +9% (PMC-confirmed latency hiding) |
| BK32 depth | the compute-density sweet spot (deeper overflows VGPR) |
| square 128×128 tile | optimal (non-square refuted for this shape) |
| LEANADDR (scalar-base addressing) | VALU −18% to Tensile's count — *neutral* on throughput (kept off) |

## Levers measured and ruled out (no further gain)

prefetch depth (full A+B PLR — dominated), L2 locality (ours already better), VALU/address overhead
(matched-neutral), ds_load 2× (+3% ceiling, hidden), shape-specific non-square tiles (refuted). **No
identified lever closes the remaining ~4%** — it is hidden micro-costs + clock + the fact that this shape is
awkward for everyone (Tensile included).

## Provenance / reproduce

```bash
DEV=AMD PYTHONPATH=. python3 extra/qk_amd_gemm_promotion_benchmark.py    # ours vs .co vs authority, pinned
```

Full arc (every gate, every probe) in the dated `docs/prefill-amd-gemm-*-20260620.md` series; correctness,
emission, and timing all gated and committed.

## Verdict

**PROMOTED.** Dependency-free RDNA3 prefill GEMM at **~96% of vendored Tensile** and **+11% over tinygrad's
LLVM authority**, correct (2e-4), reproduced across 3 sessions, zero dependencies. The win is banked: the
config above is the canonical dependency-free kernel for this shape; every lever is measured; the residual ~4%
is small, characterized, and not closeable by any identified kernel change.
