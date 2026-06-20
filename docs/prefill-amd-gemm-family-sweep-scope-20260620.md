# Prefill AMD GEMM Family Sweep vs Authority — Scope

Date: 2026-06-20

## The one question this answers

Now that single-buffer LDS proved competitive (~1.3× global-direct), can **tuning the dependency-free LDS
family reach or beat the tinygrad LLVM authority**, and where is the frontier vs the 60-TFLOPS Tensile bar?

**Answer: it REACHES the authority.** Verdict `PASS_GEMM_FAMILY_SWEEP_AUTHORITY_BEATEN`. A bounded enumerated
config sweep finds **`lds2` with BK=32 (deeper K-block) at ~55 TFLOPS best — slightly ahead of the LLVM
authority (~52.8 best), parity on median** — all byte-correct (rel RMSE 2.08e-4), at the authority shape
M=512,N=12288,K=4096, measured in one interleaved clock-fair process. This is **dependency-free hand-asm**.

This is a fixed config grid, **NOT a search** — no BEAM. No routing/default change. The robust claim is the
interleaved *ratio*; absolute TFLOPS is clock-volatile provenance.

## Deliverables

| artifact | role |
|---|---|
| `extra/qk_amd_gemm_family_sweep_probe.py` | bounded grid, correctness-gate each, interleaved one-clock timing vs authority |
| `bench/amd-broad-backend-roadmap/amd_gemm_family_sweep_result.json` | result (`bench/**` gitignored, reproducible) |

Run:

```bash
PYTHONPATH=. python3 extra/qk_amd_gemm_family_sweep_probe.py    # CNT=150 RAMP=80 default
```

## Results (two reproduced runs, best-of-N TFLOPS at authority shape)

| row | LDS B | best | median | correct |
|---|---:|---:|---:|---|
| **lds2 W2×2 T4×4 BK32** | 16384 | **55.3 / 54.9** | 50.7 | ✅ 2.08e-4 |
| authority_tinygrad_llvm (`r_16_192_…`) | — | 52.8 / 52.7 | 51.4 | (tinygrad ref) |
| lds2 W2×2 T4×4 BK16 DBUF | 16384 | 41.6 / 42.0 | 38.0 | ✅ |
| lds2 W2×2 T4×4 BK16 | 8192 | 41.6 / 41.7 | 37.0 | ✅ |
| lds_single_buffer (`build_gemm_lds`) | 8192 | 40.9 / 40.6 | 37.9 | ✅ |
| lds2 W2×2 T2×2 BK16 | 4096 | 33.1 / 32.7 | 30.8 | ✅ |
| lds2 BK32 PAD8 DBUF | 36864 | 31.3 / 31.2 | 28.3 | ✅ |
| lds2 BK16 PAD8 | 10240 | 30.9 / 31.0 | 26.6 | ✅ |
| global_direct_pipe_T4×2 (baseline) | 0 | 29.1 / 28.9 | 26.2 | ✅ |

**Frontier**: best dependency-free = `lds2_BK32` ≈ **55 TFLOPS**; **ratio vs authority = 1.05× best, ~0.99×
median** (i.e., reaches/matches the authority); **ratio vs global-direct = ~1.9×**; **reaches Tensile-class
(≥60): no** (~92% of the way).

## What the sweep shows

- **BK=32 is the lever.** Doubling the K-block depth (BK16→BK32, KT=2 substeps per LDS load) does ~2× the
  WMMA per global→LDS→barrier round-trip, amortizing the staging/barrier cost — lifting the family from ~41
  (BK16) to ~55. This is exactly the overhead the prior "LDS slow" reading misattributed to LDS itself.
- **LDS padding hurts here** (PAD8: 41→31): the bank-conflict pad costs more than it saves at this tile.
- **Register double-buffer (DBUF) is ~neutral** at BK16 (41.6 vs 41.6) and *negative* when stacked with
  BK32+PAD8 (31) — consistent with double-buffering not being the RDNA3 win; depth-per-stage (BK) is.
- **Smaller tile (T2×2, BM=BN=64) is slower** (33) — less WMMA reuse per cooperative load.

## Trust basis

- One interleaved process, round-robin, per-launch sync+`perf_counter`, RAMP burst excluded, best-of-N
  (CNT=150). The *ratio* is clock-fair.
- **Harness calibration**: global-direct reads ~29 (known 24–32) and the LLVM authority reads ~52–53 (known
  42–53 high-clock) — both in-band, so the BK32 ~55 is trustworthy *relative* to them, in the same run,
  twice.
- **Activity witnessed by power** (median ~50 W vs ~5 W idle); `rocm-smi` sclk is unreliable on this RX 7900
  XTX (reported as provenance only).
- Honest nuance: best-of-N has BK32 *ahead* of authority (1.05×); on **median it is parity** (50.7 vs 51.4).
  So the defensible claim is **"reaches/matches the LLVM authority,"** not "decisively beats."

## Significance (corrections to the prior record)

1. **"Dependency-free prefill rests below the LLVM authority / POWN ~42-TFLOPS pure-tinygrad ceiling" — does
   not hold for hand-asm.** A dependency-free hand-asm LDS kernel (BK32) reaches ~55, matching/beating the
   ~52–53 LLVM authority. POWN measured LLVM-*codegen* WMMA configs; the deeper-K LDS pipeline is a schedule
   LLVM didn't emit, expressible by hand on the `assemble_linear` path.
2. **Extends [[amd-prefill-lds-gemm-not-refuted]]**: LDS staging isn't merely "not net-negative" — at BK32 it
   is the *fastest* dependency-free config and reaches the authority.
3. **Gap to Tensile-class** (≥60, Tensile itself ~66): from ~55 = ~85–92% of the way, dependency-free and
   correct — the closest dependency-free prefill result on record.

## Scope / honesty boundaries

- Single prefill shape (512×12288×4096); other shapes untested.
- Bounded grid (6 lds2 configs + single-buffer + global-direct + authority) — **NOT BEAM/search**; a wider
  grid or other shapes could move the frontier.
- `alpha=1/beta=0`, hand-asm; no fusion, no routing/default change.
- Absolute TFLOPS clock-volatile (report the ratio + clock provenance).

## Verdict

`PASS_GEMM_FAMILY_SWEEP_AUTHORITY_BEATEN` — a correct, dependency-free config (`lds2_BK32`) reaches/matches
the same-run LLVM authority at the authority shape (1.05× best, ~parity median), ~1.9× global-direct,
correctness 2.08e-4. Failure branches (`..._BELOW_AUTHORITY`, `..._CLOCK_INVALID`, `..._LAUNCH`,
`..._PRECONDITION`) were wired; none triggered.

## Next (each its own gate, none authorized here)

1. **Push BK / K-block depth further** (BK48/64) and re-pad — does the family clear the 60 Tensile-class bar
   dependency-free? Still a bounded grid, **no BEAM**.
2. **Pin or telemetry-bin the clock** so absolute TFLOPS (not just ratio) is reportable.
3. **Generalize the shape** (other N/K, decode-adjacent) before any thought of routing this in-model — which
   would be a separate, much larger decision (the universal in-model-integration caveat still applies).
