# Qwen3-8B on gfx1100 — the MEASURED inference-performance map (campaign consolidation, 2026-06-19)

Authoritative consolidation of the hardware-counter + kernel-trace measurement campaign. Replaces inference with
measurement. Everything here is backed by native PMC (tinygrad HCQ), rocprofv3 kernel-trace (llama), or
clock-controlled timing — and triangulated where it matters.

## TL;DR (the one conclusion)
**tinygrad has competitive/winning GPU kernels in ISOLATION for both inference regimes, but loses the advantage
IN-MODEL. The universal bottleneck is in-model INTEGRATION, not the kernels, not codegen, not cache locality.**
Decode is HBM-bandwidth-bound; prefill is compute/WMMA-bound; they are opposite regimes needing different levers,
but the same meta-failure (isolated→in-model transfer loss). vs llama.cpp on the same GPU: tinygrad decode ~77% ,
prefill ~82%.

## The instrument (and what's trustworthy)
- **rocprofv3 CANNOT trace tinygrad's HCQ/KFD dispatches** (it bypasses the HIP runtime rocprof hooks) — confirmed
  twice. Use tinygrad's **native PMC** (`PMC=1 PROFILE=1`): it programs the gfx1100 perf counters per kernel
  (ProfilePMCEvent). Decoder + atlas: `extra/qk_pmc_capture.py`, `qk_primitive_pmu_atlas.py`,
  `qk_prefill_pmu_atlas.py` (map PMC.kern==ProgramEvent.tag→name; sum counters across SE/SA/WGP instances).
  Counters: GL2C_HIT/MISS (L2 hit%), SQ_INSTS_VALU (VALU util, viz formula), GRBM_GUI_ACTIVE, SQC_LDS_BANK_CONFLICT.
- **llama IS rocprof-traceable** (HIP/rocBLAS-linked): `rocprofv3 --kernel-trace` gives clean per-kernel GPU-time
  (its GL2C/GRBM counters return 0 — multiplexing limit; timing is the trustworthy signal).
- **VALU% is NOT a WMMA-utilization proxy** (v_wmma is one multi-cycle instruction; no WMMA counter) — use TFLOPS.
- **Per-kernel achieved-BW via GL2C_MISS×128/time is unreliable** (PMC perturbs per-kernel timing; 32-instance
  sum); model-level aggregate (bytes/token × tok/s) and clock-controlled wall timing are trustworthy.
- **CLOCK is the dominant confound** (bit us twice): the same config varied 3433→7321 tok/s across processes from
  clock alone. Only **interleaved, one-process, clock-controlled** A/B is trustworthy. min-over-warm-iters within
  one process is OK; cross-process or non-interleaved comparison is NOT.

## DECODE — HBM-bandwidth-bound (all ctx)
- **~85% of decode GPU time = weight-GEMVs** (FFN gate/up/down, qkv, attn-o, lm_head): L2 hit **3–13%**, VALU util
  **2–7%** → weights streamed once from HBM, no reuse, ~zero ALU. **At every ctx 128→4096 decode is 89–91%
  bandwidth-bound — NEVER compute/cache-bound** → **codegen (frontier #4) is DEAD for decode**, measured.
- Attention/flash = cache-served (L2 hit ~99%), small (~4%); at ctx4096 KV-cache-streaming grows to ~31% (also
  bandwidth-bound, L2 0.8–3.6% — KV too big to cache; and LESS BW-efficient than weight-GEMVs).
- Effective HBM BW: tinygrad **~38% of 960 peak** (77 tok/s), llama **~47%** (96–100 tok/s). Neither saturates HBM
  (batch-1 GEMV has arithmetic intensity ≈0 → MLP/occupancy-limited).
- **llama decode is STRUCTURALLY IDENTICAL** (~86% weight-GEMV via `mul_mat_vec_q`). So the gap is NOT structural.
- **Mechanism (triangulated):** tinygrad's int-dot GEMV is **76% peak STANDALONE (beats llama 57%)** but **~44%
  IN-MODEL** (loses 32 pts); llama 57→54% (loses 3). The gap = **int-dot e2e integration** = (1) amortize the
  activation→Q8 quant across input-sharing GEMVs (llama: `quantize_q8_1` once, reuse), (2) sustain llama's
  max-occupancy launch (`mul_mat_vec_q` grid=131072, wg=32, vgpr=24–40). Matches prior `amd-decode-kernel-beats-llamacpp`.
- **Decode levers:** (a) fused-mmvq integration (the above); (b) **spec-decode** — orthogonal multiplier,
  bandwidth-JUSTIFIED (amortizes the 4.68 GB weight read over ~2.5 accepted tokens); (c) KV-quant for long ctx.
- **Decode diagnostic update:** the prefill-style localization pass found no single transpose-like tax. Q4_K stage2
  reduce is a real `~10%` local tax but only reaches `~53-54%` on that surface; q8 lifecycle is capped by reuse `2`;
  existing env launch-shape knobs fail. The remaining large gap is **MMVQ contract preservation in-model**.

## PREFILL — compute/WMMA-bound (opposite regime)
- Dominant matmuls L2 hit **54–87%** (weights reused/cached across the 512-tile) → NOT bandwidth-bound.
- Throughput hierarchy (~2×params/tok): tinygrad fp16-WMMA **~41 TFLOPS (34% of 122 peak)** < llama int8-MMQ
  **~49 (40%)** < **Tensile fp16 66 (54%)**. tinygrad LOSES prefill to llama (82%). llama prefill = 74%
  `mul_mat_q` (own int8 quantized GEMM) + 9.2% Tensile/rocBLAS fp16 + 4.4% flash.
- **Tensile is the SAME story as decode:** 66 TFLOPS ISOLATED (shape-matrix 61–77) but the in-model route gives
  **0.999× (clock-controlled, reproduced 2×)** — the win evaporates in-model. The prior 1.27× was a clock-confound
  (RETRACTED). Amdahl predicted ~1.37×; got 1.00× → isolated speed doesn't transfer through the route.
- **Prefill lever:** the in-model integration of the fast kernel (transpose-free Tensile route OR closing the
  WMMA 34%→54% gap = the SW-pipelined-K-loop codegen, POWN-walled). NOT a kernel swap.

## The meta-pattern (the campaign's central finding)
| regime | isolated kernel | in-model | transfer loss |
|---|---|---|---|
| decode | tinygrad GEMV **76%** peak (>llama 57%) | **44%** | −32 pts |
| prefill | Tensile **66 TFLOPS** (>llama 49, tinygrad 41) | **~41 (1.00×)** | win gone |
**Both: winning kernels in isolation, advantage lost in-model → in-model integration is the universal bottleneck.**

## What's DEAD (by measurement, do not reopen without a new premise)
- Codegen/VALU improvement for **decode** (VALU idle ≤12% everywhere, bandwidth-bound at all ctx).
- LDS / locality / software-pipelining kernels (A3 refuted; decode weights uncacheable, prefill data already
  cache-served; IC-served on gfx1100).
- BEAM as a Tensile replacement (14–17 < warmstart 48; refuted) and "BEAM hangs" (false premise; it underperforms).
- Tensile prefill route AS-BUILT (0.999× in-model; the layout-transpose/in-model regime kills the isolated win).
- Sub-4-bit quant (dNLL quality wall).

## What SURVIVES (the real, measured levers)
1. **Decode: spec-decode** (frontier #3) — bandwidth-justified, A-pending (correct, ~1.3–1.4× est, needs cli-loop
   integration + clean clock-controlled measurement). Highest-confidence decode win.
2. **Decode: fused-mmvq integration** — amortize activation-quant + sustain occupancy in-model (close 44→54%).
3. **Prefill: in-model integration of the fast matmul** — transpose-free Tensile route, or the walled WMMA codegen.
4. (Both deps-bounded:) external Tensile `.co` is confirmed present/working but its isolated win doesn't transfer
   without integration work; bundling is a vendored-blob policy call (TPE-0, user decision).

## Index of supporting docs (all 2026-06-19 unless noted)
- Full atlas + decode/prefill regimes + llama refs + mechanisms: `decode-bandwidth-bound-pmu-learning-20260619.md`
- Decode integration tax ledger: `decode-integration-diagnostic-result-20260619.md`
- Tensile prefill A/B (0.999×, 1.27× retracted): `prefill-tensile-land-result-20260619.md` (+ `…-land-scope`)
- Route A (dependency-free WMMA asm): `route-a-a2-pipeline-result`, `route-a-a3-p2-p3-lds-refuted`
- Frontier scope (4 levers): `frontier-scope-beyond-route-a-20260619.md`
- Prior proof points: `amd-decode-kernel-beats-llamacpp` (memory), `prefill-own-wmma-kernel-result` (POWN),
  `spec-decode-low-sync-verdict-20260618.md`, `beam-hang-premise-audit-20260619.md`
- Tools: `extra/qk_pmc_capture.py`, `qk_primitive_pmu_atlas.py`, `qk_prefill_pmu_atlas.py`, `qk_tensile_ab_measure.py`
