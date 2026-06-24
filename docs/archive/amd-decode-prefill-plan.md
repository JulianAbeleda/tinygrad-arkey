# Prefill optimization plan (P0-P3)

> Framework: `docs/gpu-performance-first-principles.md` (the canonical bytes/math/overhead + roofline
> reference). Prefill's verdict is a textbook **§1.4 (locality/reuse/tiling)** case: the in-model matmuls do
> not tile (chained 5% of peak -> isolated 17-27%), so larger batch can't push them compute-bound. Per the
> doc's Horace-He diagnostic (flat-with-size -> overhead/tiling, not memory- or compute-bound) + the measured
> 95% GPU-busy, the binding limit is in-model scheduling/tiling, not bytes or the kernel.

Date: 2026-06-16. The decode arc deliberately excluded prefill ("keep the dense path for prefill"); it was
never scoped, profiled, or attacked. Measured this session: prefill ~65 tok/s vs llama.cpp ~3000 tok/s
(`llama-bench -ngl 99 -p 512,1024,3072 -n 0`) = **~2% of llama, ~45x behind, ~1% of fp16 peak** — by far the
worst gap vs llama (decode is ~58%). For a long prompt this is the time-to-first-token cost (~15s for 1000
tokens vs llama's ~0.3s).

## Root cause (hypothesis)
The default prefill runs the dense path (`decode_enabled = batched or (not is_prefill)` -> off during
prefill), where the **Q4_K dequant is fused into the matmul**. The GEMM operand is therefore not clean fp16
-> **RDNA3 WMMA cannot fire** -> the matmul runs scalar/untiled (~1% of peak). This is the same
"fusion-blocks-WMMA" wall as the decode W2 finding — but prefill is the regime where fixing it PAYS, because
prefill is compute-bound at batch-32 where tensor cores are the right tool (per `amd-decode-option1-result.md`:
"TC would only pay at LARGE batch — prefill, K>=64+, matmul-dominant").

## Building blocks (all built, validated in isolation, never wired to prefill)
- `extra/qk_matmul_decoded.py` — dequant Q4_K->fp16 then NATIVE matmul; dequant amortizes over the batch.
- `Q4K_UNFUSE` flag (model.py feed_forward) — casts FFN matmuls to fp16 "so RDNA3 WMMA can apply" (FFN only;
  attn projections excluded).
- the loop (N1/N2/L0/L1) — tunes native fp16 matmul to 33-98% of peak, live, 42x; its substrate IS prefill.
- TC/WMMA — the right tool at batch-32; never tested on the prefill path.

## Staged plan (gated, cheapest-first)
- **P0 — diagnose.** DEBUG profile one 32-token prefill chunk on 8B. Confirm matmuls dominate, measure
  achieved TFLOP/s vs the 83.6 fp16 peak, confirm WMMA does NOT fire. Gate: matmuls <30% of peak + no WMMA.
- **P1 — make-or-break: unfuse -> fp16 -> TC.** Route prefill linears through dequant->fp16 so WMMA fires.
  Gate: TC fires AND prefill >=5x faster. Risk: PADTO blowup (option1's 12288=256x16x3 -> pad-to-16 ~5x waste)
  at batch 32 — check pad efficiency.
- **P2 — tune.** Warm-start the loop's known-good native-matmul schedules onto the prefill matmuls (NOT native
  BEAM — it hangs gfx1100; use the curated loop / warm-start hook).
- **P3 — measure + decide.** Prefill tok/s vs llama 3000, token parity. Pre-register: >=10x (~650 tok/s,
  ~22% of llama) = ship; parity is the stretch.

## Out of scope / caveats
- Decode (separate problem, hand-asm wall, flag-exhausted).
- Not bit-exact: fp16 matmul accumulation differs from the fused path -> gate on TOKEN PARITY, not byte-identity.
- BEAM hangs gfx1100 — use warm-start, not native BEAM.

## RESULTS

### P0 — diagnose (2026-06-16): PASS, headroom confirmed
Prefill (8B, warm, 256-token) = **68 tok/s = ~1.11 TFLOP/s = 1.3% of the 83.6 fp16 peak**. WMMA does NOT
fire (`DEBUG=4` grep for wmma/v_wmma/tensor_core = 0 hits). So the matmuls run scalar fp32 (RDNA3 WMMA needs
fp16 operands) at ~1% of peak — the fused-dequant + fp32-activation path blocks tensor cores. Gate (<30% of
peak + no WMMA) cleared. The lever is real and large.

DEBUG=2 profile (warm chunk): each transformer block runs as ONE fused `function` (`FFNBlock._run`,
precompile=True) at **~86 ms for 32 tokens** (x36 blocks ~= 3.1 s/chunk). The whole-block fusion WITH the
Q4_K dequant inside is a single untiled mega-kernel -> no batch reuse, no TC.

### P1 — make-or-break (fp16/TC via flags): FAIL. No existing flag fixes prefill.
Measured prefill N=256 (warm), 8B: baseline 68 tok/s; `Q4K_UNFUSE=1` 65; `Q4K_UNFUSE=1 TC=2` 65;
`Q4K_BATCHED=1` (route prefill -> batched-GEMM primitive) 67; **`REALIZE=1` 22 (WORSE** -- materialized fp16
weights are 3.4x more bytes, and the block stays fused). `TC`/`TC_OPT` are BEAM-search actions, not applied by
the default schedule, so they no-op without BEAM (which hangs gfx1100). Gate FAILED: no flag gives >=5x. The
fix is NOT configuration -- the block-level fusion must be broken so the matmuls become clean fp16 GEMMs.

### P2 — fix direction CONFIRMED standalone (matmul_decoded), but it needs WIRING (a build, not a flag).
`extra/qk_matmul_decoded.py` (dequant Q4_K->fp16 MATERIALIZED, then NATIVE fp16 matmul) on real
prefill-shaped tensors at N=32 (the prefill batch), vs the current fused path:
| tensor | shape | native matmul | vs fused |
|---|---|---|---|
| blk.0.ffn_gate | 12288x4096 | 12.91 TF (**15.4% peak**) | **18.3x faster** |
| blk.0.ffn_down | 4096x12288 | 4.9 TF (5.9% peak) | 6.9x |
| blk.0.attn_q | 4096x4096 | 3.51 TF (4.2% peak) | 5.9x |
Even UNTUNED, dequant->fp16->native is **5-18x faster than the fused path** at batch-32 (current prefill is
1.3% of peak). Projected prefill: ~2% -> **~15-25% of llama** (clears the >=10x P3 gate); TC/loop tuning (the
33-98%-of-peak substrate) is upside on top. So the fix is proven; what remains is WIRING it into the prefill
forward.

### P2-wire attempt 1 (Linear-level fp16-contiguous): FAILED -> root cause is multi-factor
Added a gated `PREFILL_FP16` branch to `Q4K/Q6K PrimitiveLinear._fallback` (T>1): materialize the dequant
weight on a `.contiguous()` fp16 boundary + native matmul. Result: **28 tok/s (WORSE than 68)** — same as
`REALIZE=1`. So a Linear-level edit does NOT reproduce the standalone 15%-peak win. Reverted (model.py pristine).
Diagnosed why the standalone matmul tiles but the in-model one doesn't — it is multi-factor:
1. **Symbolic batch dim.** Prefill uses symbolic `T` (`v_toks`). Measured: the SAME fp16 matmul is **2.2x
   slower with a symbolic batch (1.0 TF) than concrete (2.2 TF)** — TC/tiling want concrete dims.
2. **Untuned matmul.** Even concrete, the default-scheduled matmul is only ~2-15% of peak (orientation- and
   size-dependent; the matmul_decoded 15.4% was W[out,in]@X[in,32], a favourable orientation).
3. **Per-chunk dequant.** Materializing the fp16 weight in `_fallback` re-dequants every chunk (and a blanket
   REALIZE keeps 16 GB resident -> also slower).
The current 1.3% is roughly the PRODUCT of these. No single Linear-level change fixes it.

### Remaining build (P2-wire + P3): the real work, scoped (BIGGER than first thought)
The fix is a prefill-DRIVER restructure (not a Linear edit), addressing all three factors together:
1. **Concrete-batch prefill** — pad prefill chunks to a fixed size (e.g. 32) so the matmul dims are concrete
   (TC/tiling eligible), instead of the symbolic `v_toks` chunk. Changes `generate()` chunking + the JIT
   (a concrete-T prefill graph, or pad-to-32 always).
2. **Cached/amortized dequant** — dequant each weight to fp16 ONCE per prefill (not per chunk) without keeping
   16 GB resident (per-layer streaming, or a bounded fp16 weight cache).
3. **Tuned matmul** — warm-start the loop's TC/native-matmul schedules onto the concrete-batch GEMMs (NOT
   native BEAM — hangs gfx1100).
Plus token-parity verify (fp16 accumulation differs from the fused path). This is a correctness-critical
restructure interacting with @function/precompile/JIT — a real multi-stage build, NOT a flag or a one-line
edit. The standalone win (matmul_decoded 5-18x) proves the ceiling is real; reaching it in-model is the work.

## VERDICT (2026-06-16): prefill is GPU-bound at ~1.3% peak; NO accessible lever fixes it. Schedule-transfer wall.
After exhausting every accessible lever — all measured NEGATIVE on the real in-model prefill (8B, warm):
| lever | result |
|---|---|
| `Q4K_UNFUSE` / `TC=2` / `Q4K_BATCHED` | no change (~67 tok/s) |
| `REALIZE=1` (resident fp16 weights) | 22 (worse) |
| `PREFILL_FP16` (Linear fp16-contiguous) | 28 (worse), reverted |
| concrete T=32 forward (vs symbolic) | ~same (529 vs ~470 ms) — symbolic batch is NOT the in-model cap |
| matmul orientation (`x@Wᵀ` vs `W@xᵀ`) | 1.0x (no diff) |
| chunk_size 32 / 128 / 512 | 67 / 69 / 39 — bigger batch does NOT help, hurts at 512 |
| GPU-vs-host | **95% GPU-busy** (GPU 3775 ms / wall 3957 ms) -> GPU-bound, NOT launch overhead |

ROOT CAUSE (confirmed): prefill is GPU-bound and the in-model matmul kernels run at ~1.3% of peak. The SAME
matmul as a clean top-level kernel reaches ~13 TF on the GPU (matmul_decoded), but inside the @function
precompiled block graph (with the fused Q4_K dequant) tinygrad schedules it far below peak — and NOTHING at the
driver / flag / parameter level changes that. It is the SAME class of wall as decode (good standalone kernels,
bad in-model scheduling), and worse for prefill because prefill SHOULD be compute-bound but tinygrad can't
schedule it to be. The fix is NOT a build over the existing forward — it requires either (a) transferring the
loop's tuned matmul schedules INTO the @function-compiled model forward (the unsolved transfer problem; L2
showed the loop doesn't transfer across substrates without retraining), or (b) hand-written AMD GEMM kernels
(the Writer). Both are out of the "wire an existing block" scope. Prefill optimization is therefore PARKED as a
located negative: ~2% of llama, GPU-bound, schedule-transfer-walled. The standalone matmul_decoded 13 TF proves
the silicon can do it; tinygrad's in-model scheduling is the ceiling.

## llama's prefill primitive (researched 2026-06-16) + M0/M1: the win exists in isolation, won't transfer
llama prefill = **MMQ (Matrix-Matrix Quantized)**, `ggml-cuda/mmq.cuh`, built here with `GGML_HIP_MMQ_MFMA=ON`
-> AMD WMMA int8 matrix cores (dp4a for batch<=64, WMMA above). 4 primitives: (1) quantize activations ->
Q8_1, (2) tile weight+activation into LDS, (3) int8 MMA (v_dot4 / WMMA), (4) fused dequant scale. Measured:
llama prefill = **48-50 TF (~59% of fp16 peak)** vs us 1.1 TF (1.3%) = **44x**. Every MMQ primitive already
exists in the repo (q8_1_quantize, q4k_q8_1 int-dot, q4k_gemm, the Marlin WMMA) but never composed into a tile.

**M0 (make-or-break) PASSED**: the standalone native fp16 matmul (dequant separated -> `wf16 @ B`) goes
compute-bound with batch: N=32 16% / N=128 41% / N=1024 57% / **N=2048 80% peak (66.9 TF, BEATS llama 48)**.
`@function` is EXONERATED: wrapping a clean fp16 matmul in @function(precompile=True) = 25% peak == standalone.
So the kernel-level win is real and beats llama in isolation.

**M1 (transfer) FAILED**: the win does NOT reach the in-model prefill. EVERY config measured ~1% peak / 19-49
tok/s: REALIZE=1 (clean fp16 weights) + PREFILL_FP16 (fp16 activations) + chunk 512/1024 + unfuse(contiguous) +
output-isolation, and all combinations. Factors found: in-model activations are fp32 (`x.float()`), the lazy
Q4_K dequant fuses into the matmul, and at large chunk the O(T^2) prefill ATTENTION grows too (chunk 512->25,
1024->23, worse). But even fixing weight+activation dtype + batch, the in-model matmul never tiles like the
standalone.

**VERDICT (multiply-confirmed):** the prefill kernel win EXISTS and beats llama IN ISOLATION (M0), but
tinygrad's in-model forward-graph scheduling will not produce it (M1) — the project's recurring "lever real
isolated, never translates e2e" thesis, now the prefill verdict too. Transfer requires either (a) a full
forward RESTRUCTURE (compute each matmul as a separate top-level realized op outside the fused block graph —
breaks fusion everywhere, large) or (b) a RAW MMQ custom_kernel (port mmq.cuh as raw HIP, bypassing tinygrad's
scheduler — the flash-decode approach, but a full tiled-WMMA GEMM). Both are substantial hand-kernel builds
("the Writer"); neither is a flag or a quick win. Prefill stays PARKED. model.py pristine.

## B0 bisection (2026-06-16) — the in-model penalty FACTORED + the right language (ubatch)
Reframe via measuring llama's PHYSICAL batch knob (n_ubatch, the GPU kernel-launch batch; llama default 512,
we use chunk=32). llama pp512 by ubatch: 32->1114, 128->1831, 512->3110, 2048->3112 tok/s. KEY: at ubatch=32
llama=1114 (18 TF) and OUR STANDALONE matmul N=32 = 13.7 TF -- our kernel is already ~llama's MMQ. So the 44x
is NOT a kernel gap; it's the in-model penalty (~17x at fixed batch) x the ubatch gap (~2.8x).
B0 bisects the in-model matmul (GPU TF, N=32): clean Wf16real@Xf16real = 8.4; **fp32 activation = 2.3 (3.6x)**;
**lazy Q4_K dequant weight = 2.0 (4x)**; producer rmsnorm = 10.8 (FREE); consumer silu = 12.2 (FREE); full =
1.9 (~= in-model 1.1). So the collapse is TWO named factors -- (1) the lazy Q4_K dequant FUSES into the matmul
(4x, dominant), (2) the fp32 residual stream (`x.float()`) blocks WMMA (3.6x) -- NOT neighbor fusion.
COMPLETE factored picture (all measured, all tinygrad-level, none a mysterious wall):
  prefill_slow = dequant-fused-into-matmul(4x) x fp32-stream(3.6x) x small-ubatch/memory-bound(16%->80% peak)
                 x O(T^2)-prefill-attention(the large-batch tax)
We do NOT need an MMQ kernel (our standalone matmul ~= MMQ). The fix is a coordinated PREFILL-MODE forward:
  (a) per-layer dequant Q4_K->fp16 REALIZED once (kill the 4x fusion; per-layer streaming avoids 16 GB resident),
  (b) fp16 residual stream for prefill (kill the 3.6x; enables WMMA),
  (c) large ubatch (process 512+ tokens/launch -> compute-bound, 16%->80% peak),
  (d) flash-style prefill attention (so O(T^2) doesn't dominate at large ubatch).
Each is real but now understood; (a)+(b) alone ~= 8.4 TF (~10% peak, ~7x prefill) IF (c) amortizes the bytes.
This is a distinct prefill forward path, a multi-component build -- but no longer an open wall.

## S1 (2026-06-16): the 4 components are INTERDEPENDENT -> no incremental win; flash-prefill-attn is the critical path
Tried to build it incrementally and measured the interdependency directly:
- (a) realized fp16 weights ONLY win at LARGE batch: at ubatch=32 they read 3.4x more bytes than packed Q4_K
  and the matmul is memory-bound (AI~32 < ridge 97), so REALIZE is a WASH/LOSS at small batch. (a) needs (c).
- (c) large ubatch ONLY wins if (d) flash-prefill-attention exists: REALIZE+fp16-stream+ubatch=512 measured
  **24 tok/s (still worse than 67)** -- at large T the O(T^2) SDPA prefill attention dominates and eats the
  matmul gain. (c) needs (d).
- So NO single component yields a measurable e2e win in isolation -- they only pay off TOGETHER. That
  interdependency is exactly why incremental probing (this whole thread) kept failing: each fix exposes the
  next binding factor. There is a residual in-model penalty (clean-config still ~20x off the 8.4 TF standalone
  at small batch) that the @function block-granularity profiling can't cleanly isolate.
CRITICAL PATH = (d) **flash-style PREFILL attention** (T>1 causal, online softmax over T queries -- harder than
the shipped flash-DECODE T=1 kernel). It is the unblocker: without it large ubatch is attention-bound, and
without large ubatch the matmul fixes (a)(b) don't pay. The full build = (d) flash-prefill-attn + (a) per-layer
dequant->fp16 + (b) fp16 prefill stream + (c) large ubatch, all together, token-parity gated. This is a
genuine multi-session PROJECT, not an incremental edit -- and no piece demos a win alone. model.py pristine.

## S2 (2026-06-16): the cap is CHAINED-MATMUL SCHEDULING -- isolation fixes it standalone (3.5-5.7x), but it STILL won't transfer in-model
Built a clean standalone prefill (realized fp16 weights, fp16 stream, SDPA) to find the achievable ceiling.
Findings (GPU TF / % of 83.6 peak):
- A single matmul = 80% (M0), but a CLEAN standalone transformer LAYER (7 matmuls + SDPA) = only 4-5% peak,
  flat across batch. The cap is NOT Q4_K/fp32/attention/the model.
- Isolated: 7-matmuls-only = 5% (46ms@T512); SDPA-attn-only = 2.9ms. So **the cap is the MATMUL CHAINING**,
  not attention.
- **`.contiguous()` isolation recovers it STANDALONE**: chained 4 big matmuls = 4.9% -> isolated = 17.2%
  (T512) / 27.2% (T2048). A 3.5-5.7x win from forcing each matmul to be its own tiled kernel.
- BUT it does NOT transfer in-model. Tried the FULL recipe (PREFILL_EAGER bypassing @function + REALIZE clean
  fp16 weights + PREFILL_FP16 fp16+isolate + ubatch 256/512), AND on the plain dense path (Q4K_PRIMITIVE=0):
  ALL = **22-25 tok/s / 0.4 TF / 0.5% peak**. Controlled for @function, weight dtype, activation dtype,
  per-matmul isolation, batch, primitive-vs-dense, weight realization -- every factor -- and the in-model
  matmul STILL won't tile like the identical standalone op.

## END STATE (honest): mechanism found, standalone fix found, in-model transfer UNCRACKED by experiment
The recurring thesis, now at its sharpest: standalone/isolated ops tile (17-80% peak); the same ops inside the
real forward graph run at 0.5% peak, and NO black-box lever (dtype/realize/isolate/eager/batch) bridges it.
There is a residual ~30x in-model penalty that experimentation cannot resolve -- it requires DEEP tinygrad
kernel inspection (DEBUG=6 generated code: why does the in-model matmul kernel differ from the standalone one?
not-applied TC opt? a fused symbolic-cache op? a layout/contiguity the scheduler picks differently?). That is a
tinygrad-scheduler-internals investigation, a different kind of work than the e2e experimentation done here.
Prefill is PARKED with a complete mechanism-level characterization (not a mystery, but not solved). model.py
pristine throughout. Recommend: either the DEBUG=6 kernel-diff investigation (research-grade, tinygrad
internals) or accept prefill as a characterized gap. All flag levers + the isolation/eager/realize recipe are
exhausted.

## llama KERNEL-LEVEL profile (rocprofv3, 2026-06-16): the reference target is rocBLAS Tensile WMMA GEMM
Installed the profiler the gpu-perf doc prescribes (rocprofv3 + rocprofiler-compute/omniperf + aqlprofile).
tinygrad's AMD backend bypasses ROCr (raw KFD/AQL) so rocprofv3 can't see it; its HIP backend can (but PMC
counters break tinygrad's HIP device init, and HIP/ROCr is flaky alongside the AMD backend -- so only
kernel-trace, not rich counters, on tinygrad). llama uses native HIP -> profiles cleanly. llama prefill (pp512,
8B) top kernels:
| total | kernel | VGPR | LDS | grid |
|---|---|---|---|---|
| 29.8ms x34 | `Cijk_Alik_Bljk_HB_MT128x128x16_MI16x16x16x1_SN` | 256 | 25600 | 4096 |
| 2.9ms x36 | `Cijk_..._HB_MT64x64x32_MI16x16x16x1` | 128 | 9216 | 2048 |
| -- | `quantize_q8_1`, `dequantize_block_q6_K`, `convert_unary` | | | |
**Reduced to primitives:** llama's prefill GEMM is **rocBLAS/Tensile's fp16 WMMA GEMM** (`Cijk`=Tensile, `HB`=
half-precision BLAS): `MT128x128x16` macro-tile (128x128 register/output blocking, K-tile 16), `MI16x16x16` =
the **WMMA 16x16x16 matrix-core** instruction, **25.6 KB LDS** (shared-mem tiling, the doc's S1.4), 256 VGPR.
It is the dequant->fp16->BLAS path (NOT MMQ int8 for this build/shape): `dequantize_block_q6_K`/`convert_unary`
turn weights to fp16, then rocBLAS WMMA GEMM. This CONFIRMS the matmul_decoded approach AND names the missing
piece: llama's "native GEMM" is rocBLAS Tensile (LDS-tiled + WMMA macro-tiles, ~80% peak); tinygrad's "native
GEMM" is the `r_*` reduce kernel (minimal LDS tiling, no WMMA macro-tiles -> 5% chained / 17-27% isolated).
TWO gaps to llama, both now kernel-measured: (1) tinygrad GEMM codegen 27% (isolated best) vs rocBLAS 80% =
~3x kernel-quality gap (tinygrad's matmul codegen != Tensile); (2) in-model 1% vs tinygrad-isolated 27% = ~27x
scheduling gap. To match llama, tinygrad prefill needs a rocBLAS-class tiled-WMMA GEMM (call rocBLAS/hipBLASLt,
or hand-write the Marlin LDS-staged WMMA kernel -- the W2 path that hit tinygrad's auto-tiling wall) AND
isolated large-batch calls. Profiler is now installed for future counter work; tinygrad PMC needs the
HIP-backend/ROCr conflict resolved (or use the AMD-backend SQTT path).

## ROOT CAUSE NAILED (2026-06-16, kernel-level): tinygrad's matmul emits WMMA but uses ZERO LDS tiling
rocprofv3 --pmc breaks tinygrad's HIP device init (ROCr conflict), so instead read tinygrad's OWN kernel
resources (DEBUG=4 for WMMA emission, the HIP kernel-trace for LDS/VGPR). The definitive diff vs llama's
rocBLAS Tensile GEMM:
| | tinygrad matmul `r_32_48_..._256_2` | llama rocBLAS `Cijk_MT128x128x16_MI16x16x16` |
|---|---|---|
| **LDS** | **0** | **25600 bytes** |
| WMMA emitted | YES (290 mentions, chained AND isolated) | YES |
| VGPR | 192 | 256 |
| workgroup | 32 (ONE wavefront) | large (MT128x128) |
**THE ANSWER:** tinygrad's matmul **emits the WMMA matrix-core op but does NO shared-memory cache-blocking
(LDS=0)** -- it re-reads operands from global memory per WMMA instead of staging a tile in LDS and reusing it.
So it is BANDWIDTH-bound at ~27% peak; rocBLAS stages 128x128 tiles in 25.6 KB LDS (reuse) and is COMPUTE-bound
at ~80%. This is precisely the doc's S1.4 (locality/reuse/tiling) / Simon Boehm's step-2 "shared-memory
cache-blocking" -- the ONE primitive tinygrad's GEMM codegen is missing. WMMA was never the gap (it fires);
LDS TILING is. Plus workgroup=32 (1 wavefront) = minimal occupancy per workgroup.
**So the prefill gap, fully reduced:** (a) tinygrad GEMM = WMMA-without-LDS-tiling -> bandwidth-bound 27% vs
rocBLAS 80% (the LDS-cache-blocking the matmul codegen/opt doesn't apply -- a GROUP/LOCAL-into-LDS opt that
BEAM would find but BEAM hangs gfx1100); (b) the in-model 27x collapse stacks on top. To match llama: tinygrad
needs LDS-staged matmul tiling (codegen/opt work, or call rocBLAS/hipBLASLt). Profiler + DEBUG path now
established; this is the precise, measured target -- no longer a mystery at any level.
