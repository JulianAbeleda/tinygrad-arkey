# Prefill AMD GEMM Runnable Timing — Scope (timing gate)

Date: 2026-06-20

## The one question this answers

Now that the candidate provably computes A@B, **how fast is it under a fair one-clock harness, and does it
beat the threshold?**

**Answer: PASS.** Verdict `PASS_GEMM_RUNNABLE_TIMING_GATE`. The correctness-passing candidate
(`build_gemm_lds`, single-buffer LDS) runs at the authority shape M=512, N=12288, K=4096 at **~38–39 TFLOPS
best / ~33 median**, **~1.3× the same-run global-direct hand-asm baseline** (~29 best / ~25 median), with
correctness intact after timing (relative RMSE 2.1e-4) and the GPU witnessed active by power draw.

This is a **timing gate only**: no BEAM/search, no routing/default change, no new kernel family. It does
**not** claim Tensile-class (that needs ≥60 TFLOPS) and does **not** treat the single-buffer candidate as the
double-buffer A0/B0/A1/B1 schedule.

## Deliverables

| artifact | role |
|---|---|
| `extra/qk_amd_gemm_runnable_timing_probe.py` | interleaved one-clock timing + telemetry + post-timing correctness |
| `bench/amd-broad-backend-roadmap/amd_gemm_runnable_timing_result.json` | timing result (`bench/**` gitignored, reproducible) |

Run:

```bash
PYTHONPATH=. python3 extra/qk_amd_gemm_runnable_timing_probe.py     # CNT=300 RAMP=200 by default
```

The probe **refuses to run** unless the prior correctness result is `PASS_GEMM_RUNNABLE_CORRECTNESS`, the
candidate is `build_gemm_lds`, and the authority shape `512×12288×4096` passed correctness.

## Harness (PTM-1 discipline)

Single process, interleaved round-robin, per-launch `Device['AMD'].synchronize()` + `perf_counter`, warm
cache, an untimed compile+clock-ramp burst excluded, best-of-N (CNT=300). **Interleaving is the primary
clock control** — both rows are sampled across the same clock trajectory, so the *ratio* is clock-fair even
though absolute TFLOPS is clock-volatile. No cross-session numbers are used for the verdict.

## Results

| row | best TFLOPS | median TFLOPS | post-timing rel RMSE |
|---|---:|---:|---:|
| **candidate_lds_single_buffer** (`build_gemm_lds`, LDS 8192 B) | **39.3** | 32.8 | **0.000208** |
| baseline_global_direct_hand_asm (`build_gemm_pipe` T4×2) | 29.2 | 25.5 | — |

- **Ratio candidate / global-direct: 1.35× (best), 1.29× (median).**
- Beats the prior slow native LDS family (~18–21 TFLOPS): **yes**.
- Reaches Tensile-class (≥60 TFLOPS): **no** — not claimed.
- Candidate resources: LDS 8192 B, scratch/private 0, grid `[96, 4, 1]`, workgroup `[128, 1, 1]`.

**Harness calibration / trust basis.** The global-direct baseline reads ~29 best, squarely in its
independently-known ~24–32 range — this calibrates the harness, so the candidate's same-run 39 is trustworthy
*relative* to it. The robust claim is the **interleaved ratio (~1.3×)**; absolute TFLOPS is clock-volatile
provenance (the p10–p90 spread 1.4–2.4 ms reflects sclk drift; best-of-N catches the high-clock moment).

## Clock / activity provenance

`rocm-smi` **sclk is unreliable on this RX 7900 XTX** (reads low DPM levels — 3–30 MHz — even under load; a
known misread for this card). **Power draw is the honest activity witness**: median **~50 W** during the run
vs ~5 W idle — you cannot sustain tens of TFLOPS at idle clock. The gate therefore validates activity on
power (≥25 W, 20 samples) and reports sclk only as best-effort provenance.

| metric | value |
|---|---|
| telemetry samples | 20 |
| power median / idle ref | ~50 W / ~5 W |
| sclk (unreliable) | median ~5 MHz, max ~30 MHz (misread) |
| gpu_use (unreliable) | ~0% (instantaneous, catches gaps) |

## Important correction to the prior record

This **supersedes** the earlier conclusion that `build_gemm_lds` was ~3.2 TFLOPS at the prefill shape and
that LDS-multiwave was "refuted (net-negative)". That measurement used **65536 B LDS**
(`LIMIT_OCC=1`) which **artificially crippled occupancy to 1 workgroup/CU**. At the candidate's **true 8192 B
footprint**, occupancy is unconstrained and the single-buffer LDS path runs **~1.3× the global-direct
baseline** — i.e., LDS staging is *not* net-negative for this prefill shape when measured at its real
footprint. The "refuted" verdict was an occupancy-confounded artifact, not a property of the kernel.

(This does not reopen the *double-buffer* A0/B0/A1/B1 PGR1 path — that remains out of scope here; its
correctness was never proven and its overlap is a separate question.)

## Verdict policy (as applied)

Precedence: post-timing correctness regression → clock-invalid → below-threshold → pass.

- `FAIL_GEMM_RUNNABLE_TIMING_CORRECTNESS_REGRESSION` — rel RMSE stayed 2.1e-4 < 0.02 → not triggered.
- `BLOCKED_GEMM_RUNNABLE_TIMING_CLOCK_INVALID` — power-witnessed active (50 W, 20 samples) → not triggered.
- `BLOCKED_GEMM_RUNNABLE_TIMING_GATE_NOT_MET` — candidate 39 ≥ 18 floor and beats same-run global-direct → not triggered.
- **`PASS_GEMM_RUNNABLE_TIMING_GATE`** — met.

Precondition/launch refusals (`..._PRECONDITION`, `..._LAUNCH`) are wired for missing/wrong correctness
artifact, wrong candidate identity, and launch failure.

## What is and isn't claimed

- **Claimed**: correct (A@B, 2.1e-4) and ~1.3× the same-run global-direct hand-asm baseline at the authority
  shape, under a fair interleaved harness with power-witnessed active compute.
- **Not claimed**: Tensile-class (≥60 TFLOPS — candidate ~39 best); absolute TFLOPS as a stable number
  (clock-volatile); the double-buffer A0/B0/A1/B1 schedule; any routing/default change.

## Next

The candidate computes A@B correctly **and** clears the first-pass timing threshold. Natural follow-ups
(each its own gate, none authorized here):

1. **Occupancy/parameter sweep** of the LDS-staged family (`build_gemm_lds2`: WAVES/WM/WN/BK/PAD) toward the
   ~42–48 LLVM-authority band — but only if it remains correctness-gated and interleaved; **still no BEAM**.
2. Pin or telemetry-bin the clock so absolute TFLOPS is reportable, not just the ratio.
3. Compare against the tinygrad LLVM authority row in the same interleaved process (not just global-direct).

Order completed: **contract → K-loop → lowering plan → emission → runnable+correctness → timing ✓.**
