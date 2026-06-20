# Prefill AMD GEMM Runnable + Correctness — Scope

Date: 2026-06-20

## The one question this answers

Does a runnable candidate with **real addressing** actually compute **A @ B correctly** on the GPU?

**Answer: YES, at the authority shape.** Verdict `PASS_GEMM_RUNNABLE_CORRECTNESS`. The LDS-staged, 4-wave
cooperative, 128×128 macro-tile WMMA GEMM launches on the AMD GPU and matches `A @ B` to **relative RMSE
≈ 2.1e-4** at the selected ffn_gate/up shape **M=512, N=12288, K=4096** (and a bring-up shape). This replaces
the prior structural-emission placeholder addressing (`STRUCTURAL_EMISSION_ONLY`, not runnable) with a real
fixed-shape address model that is now numerically verified.

This does **not** time, makes **no performance claim**, changes **no routing/defaults**, and runs **no**
BEAM/search — by rule, timing comes only now that correctness has passed.

## Deliverables

| artifact | role |
|---|---|
| `extra/qk_amd_gemm_runnable_correctness_probe.py` | builds the runnable candidate, launches on GPU, validates vs `A@B` |
| `bench/amd-broad-backend-roadmap/amd_gemm_runnable_correctness_result.json` | numeric result (`bench/**` gitignored, reproducible) |

Run:

```bash
PYTHONPATH=. python3 extra/qk_amd_gemm_runnable_correctness_probe.py
```

Inputs: `amd_gemm_emission_result.json`, `amd_gemm_lowering_plan_result.json`,
`ffn_gate_up_contract.json`, and the proven RDNA3 builder `extra/gemm/rdna3_wmma_matmul.py:build_gemm_lds`
(+ its 128-thread launcher `_run_insts_lds`).

## Numeric correctness (the gate)

Reference `ref = A.astype(f32) @ B.astype(f32)` with `B = Bt.T` (the kernel takes `Bt` = B transposed,
`N×K` row-major). Inputs fp16 (scaled 0.1), accumulation fp32. Pass threshold: relative RMSE `< 0.02`.

| shape | M | N | K | max abs err | RMSE | relative RMSE | result |
|---|---:|---:|---:|---:|---:|---:|---|
| bring-up | 128 | 128 | 256 | 0.0002 | — | **0.000205** | ✅ PASS |
| **authority** | 512 | 12288 | 4096 | 0.0010 | — | **0.000208** | ✅ PASS |

Both are ~100× under threshold, consistent with fp16-input/fp32-accumulate at these K depths.

## 1. Real fixed-shape address model

The candidate replaces placeholder addresses with the derived tiled model (`build_gemm_lds`):

- **Macro tile** BM=BN=128, BK=16; **grid** `(N//128, M//128, 1)`; **workgroup** 128 threads = 4 wave32 (2×2).
- **Cooperative load**: thread `tid` stages global row `(gy*128 + tid)` of the 128×16 A-slice and `(gx*128 +
  tid)` of the 128×16 Bt-slice, then `ds_store` to LDS at `tid*32` (A region) / `tid*32 + LDS_B` (B region).
- **A global address** `vA = (gy*128 + tid)*K*2 + k_block*32` (A is M×K row-major, fp16).
- **B global address** `vB = (gx*128 + tid)*K*2 + k_block*32` (Bt is N×K row-major = B transposed).
- **C global address** `row = gy*128 + wave_m*64 + mi*16 + (i*2+parity)`, `col = gx*128 + wave_n*64 + ni*16 +
  (lane&15)`, store fp16 at `(row*N + col)*2`.

These addresses were validated end-to-end at M=512, N=12288, K=4096 by the numeric pass — the address model
is correct at the authority dims.

## 2. Fragment mapping (documented + verified)

- **Wave layout**: `wave = tid>>5` (0..3); `wave_m = wave>>1`, `wave_n = wave&1` → 2×2 wave grid; each wave
  owns a 64×64 sub-tile = WM×WN = 4×4 = **16 WMMA tiles**.
- **LDS regions**: `LDS_A=0` (128·16·2 = 4096 B), `LDS_B=4096` (4096 B); single-buffer, total **8192 B**.
- **`ds_store` offsets**: A-slice rows at `tid*32` within `LDS_A`; Bt-slice rows at `tid*32` within `LDS_B`.
- **`ds_load_b128` fragments**: `vAfrag = wave_m*2048 + (tid&15)*32`, tile `mi` at `+mi*512`;
  `vBfrag = LDS_B + wave_n*2048 + (tid&15)*32`, tile `ni` at `+ni*512` (each fragment = 2× `ds_load_b128`).
- **WMMA feed**: `v_wmma(acc[mi*WN+ni], src0=FA[mi], src1=FB[ni], src2=acc[mi*WN+ni])` for `mi,ni ∈ 0..3`.
- **Accumulator → output**: the 16 accumulator fragments `ACCb+(mi*WN+ni)*8` map to the 64×64 wave sub-tile
  via the C address formula. The correct numeric result confirms this mapping end-to-end.

## 3. Smallest runnable correctness candidate

`build_gemm_lds` — alpha=1, beta=0, no fusion, no timing. It is the **same global_load → LDS store → barrier
→ LDS read → WMMA structure** the emission probe emitted, now with runnable addressing. The bring-up shape
exercises the identical 128×128 macro-tile mapping at reduced N/K; the authority shape proves it at the real
dims. (The bring-up is a sanity gate, not the final shape — the authority shape is the binding result.)

## Honest scope boundary

- **Single-buffer LDS.** This candidate uses one barrier-protected LDS buffer (8192 B), **not** the
  double-buffered A0/B0/A1/B1 (25088 B) the structural-emission probe emitted. The PGR1 LDS double-buffer is a
  *performance overlap*, and is RDNA3-refuted (net-negative) per the prior LDS-multiwave work — so it is
  correctly out of scope for a correctness gate. Correctness of the LDS-staged cooperative path is what this
  gate proves, and it holds.
- **Reuses a proven builder.** The candidate is the existing `build_gemm_lds`; this pass does not re-derive it
  — it *runs* it under a correctness harness to discharge this gate honestly with a real numeric result.
- **Not claimed**: performance/TFLOPS, the double-buffer overlap, bit-exact Tensile layout.

## Verdict

`PASS_GEMM_RUNNABLE_CORRECTNESS` — a runnable candidate with a real fixed-shape address model launches on the
GPU and computes `A@B` correctly at the authority shape (relative RMSE 2.1e-4). The failure branches
(`BLOCKED_GEMM_ADDRESS_MODEL`, `BLOCKED_GEMM_FRAGMENT_MAPPING`, `BLOCKED_GEMM_OUTPUT_MAPPING`,
`BLOCKED_GEMM_LAUNCH_OR_KERNARG`, `FAIL_GEMM_NUMERIC_CORRECTNESS`) were all available and none triggered.

## Next (the only remaining gate)

Correctness has passed, so — and only now — the next step is **timing** under the PTM-1 interleaved one-clock
harness: measure the candidate at the authority shape as a ratio vs the tinygrad authority WMMA row, with
sclk provenance, best-of-N, interleaved. No BEAM/search. We may now talk about TFLOPS again, because the
candidate provably computes `A @ B`.

Order completed so far: **contract → K-loop → lowering plan → emission → runnable + correctness ✓** → timing.
