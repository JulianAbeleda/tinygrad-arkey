# LEARNING (measured, not inferred) — decode is HBM-bandwidth-bound; the llama gap is bandwidth EFFICIENCY

Frontier #1 (rocprof IC-served ground-truth) executed via tinygrad's **native PMC** (rocprofv3 is blind to HCQ).
Instrument: `extra/qk_pmc_capture.py` (decoder), `extra/qk_primitive_pmu_atlas.py` (in-model atlas). This replaces
the project's long chain of *inferred* bottleneck claims with hardware-counter measurement.

## Method
- `PMC=1 PROFILE=1` → tinygrad programs the gfx1100 perf counters around each kernel (ProfilePMCEvent). Decode the
  blob (sum each counter's u64 across SE/SA/WGP instances); map event→kernel via `PMC.kern == ProgramEvent.tag`.
- Atlas runs the **real Qwen3-8B decode forward eagerly** (so PMC instruments every kernel) at ctx512; aggregate
  per primitive: VALU utilization (viz formula `100·(VALU_sum/cnt)/(active·4)`), L2 hit% (`GL2C_HIT/(HIT+MISS)`),
  GPU-cycle share. Aggregate effective HBM BW computed model-level (trustworthy): weight-bytes/token × tok/s.

## Findings (ctx512)
1. **~85% of decode GPU time = bandwidth-bound weight GEMVs.** Top kernels (FFN gate/up/down at intermediate
   12288, qkv, attn-out, lm_head): **L2 hit 3–13%, VALU util 2–7%.** Weights are streamed once from HBM per token,
   no reuse, ~zero ALU.
2. **Attention/flash is cache-served and small.** Flash reduce kernels: **L2 hit ~99%** (KV cache resident),
   ~4% of GPU time. (Measured confirmation of the LDS-tiling refutation — KV is already in cache.)
3. **VALU utilization ≤12% EVERYWHERE.** No kernel in the decode forward is ALU-bound. → **frontier #4 (renderer
   codegen / VALU) cannot help decode** — measured at whole-model scale, not inferred.
4. **Production decode = 6–7 fused programs/token** (JIT), host-sync ~0% (GPU-bound). The eager atlas's ~27 kernels
   fuse into these; the per-kernel *nature* (bandwidth-bound) is unchanged by fusion.
5. **Effective HBM bandwidth ≈ 362 GB/s @ctx512 (77.4 tok/s) = ~38% of the 960 GB/s peak.** llama.cpp (~96–100
   tok/s) ≈ **47–49% of peak.** **Neither saturates HBM.** (Model 5.03 GB; ~4.68 GB weights read/token.)

(Per-kernel achieved-BW via `GL2C_MISS×128 / range-time` was attempted and discarded: PMC perturbs per-kernel
timing and `GL2C_MISS` sums across 32 L2 instances — the hit% *ratio* is sound, absolute per-kernel BW is not. The
model-level aggregate from clean W==D tok/s is the trustworthy BW number.)

## The conclusion (reframes the decode frontier)
Decode is **HBM-bandwidth-bound on streaming quantized weights**, and the gap to llama is **bandwidth EFFICIENCY**
(tinygrad ~38% vs llama ~47–49% of peak), **not** codegen quality (VALU idle) and **not** cache locality (weights
uncacheable; KV already cached). Both engines leave >50% of HBM bandwidth unused — batch-1 GEMV has arithmetic
intensity ≈0, so achieved BW is limited by memory-level parallelism (outstanding loads / occupancy), not compute.

So the **measured** decode levers, in EV order:
1. **Read the weights fewer times per token → speculative decode (frontier #3).** Since the bottleneck is the
   ~4.68 GB weight read/token, verifying K draft tokens in ONE weight-read pass amortizes the dominant cost across
   ~2.1–2.8 accepted tokens. **This is now bandwidth-justified as the #1 decode lever**, not merely "fewer passes."
2. **Raise achieved BW% on the weight GEMVs** (38→47%+): more in-flight loads / higher occupancy / fewer
   inter-kernel gaps on the bandwidth-bound GEMV kernels. This is the tinygrad→llama efficiency gap, and it's a
   memory-level-parallelism problem, not ALU codegen.
3. **Read fewer bytes** (more aggressive quantization) — REFUTED on quality (sub-4-bit dNLL, `amd-decode-sub4-refuted`).

Dead, by measurement: frontier #4 (codegen/VALU) for decode; all locality/LDS levers (A3 confirmed).

## PREFILL is the OPPOSITE regime — compute/WMMA-bound (measured, `qk_prefill_pmu_atlas.py`)
Same instrument, real 8B prefill forward (512-token chunk, `_prefill_v2=True`, fp16 TC GEMMs), ctx0:
- **Dominant matmuls show HIGH L2 hit: 54–87%** (`r_8_48_32_*` = 55% of GPU time @ 54% L2 hit; others 67–87%).
  Weights are **reused/cached across the 512-token tile** (arithmetic intensity ~512× decode) — the polar
  opposite of decode's 3–13% streaming. **So prefill is NOT bandwidth-bound.**
- A few small bandwidth-bound kernels remain (`r_128_32_4_128` = the per-token-streamed bits, <1% each).
- **VALU% is low (0.5–13%) but UNINFORMATIVE here:** `v_wmma` is a single multi-cycle instruction doing a
  16×16×16 matmul, so a WMMA kernel at full throughput issues few instructions/cycle → low instr-rate ≠ idle.
  No WMMA-specific counter exists (only SQ_INSTS_VALU/WAVE32_VALU). Use TFLOPS-timing for prefill compute, not
  VALU%. Banked TFLOPS (POWN): tinygrad prefill WMMA = **42 TFLOPS = 35% of the 122 peak**; Tensile = 66 = 54%.

**So the two halves of inference are fundamentally different bottlenecks, now MEASURED:**
| regime | counter signature | bound | measured lever |
|---|---|---|---|
| **decode** (T=1 GEMV) | L2 hit 3–13%, ~38% peak HBM BW | **HBM bandwidth** (weights streamed once) | spec-decode (amortize weight read) [#3]; raise achieved BW% |
| **prefill** (T=512 GEMM) | L2 hit 54–87%, ~35% of WMMA peak | **compute / WMMA efficiency** (weights cached, matrix-engine-fed) | WMMA scheduling (Route A capped ~32 TFLOPS) / external Tensile 66 [#2] |

The decode levers (spec-decode, bandwidth) do NOT help prefill, and the prefill levers (WMMA/Tensile) do NOT help
decode. Frontier #4 (codegen) is dead for decode (VALU idle) but LIVE for prefill (the 35%→54% WMMA-efficiency gap
= software-pipelined K-loop, the POWN-walled codegen capability).

## Files
`extra/qk_pmc_capture.py`, `extra/qk_primitive_pmu_atlas.py`, `bench/qk-primitive-pmu-atlas/result.json`. Prior:
`route-a-a3-p2-p3-lds-refuted-20260619.md` (LDS), `frontier-scope-beyond-route-a-20260619.md` (#3 spec-decode),
`spec-decode-low-sync-verdict-20260618.md`. Decode tok/s: `bench/qk-decode-runtime-overhead/result.json`.
