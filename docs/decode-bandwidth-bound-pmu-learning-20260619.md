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

## Decode across context (ctx 128/512/1024/4096) — bandwidth-bound at EVERY ctx; composition shifts
| ctx | bandwidth-bound % | cache-served % | composition |
|---|---:|---:|---|
| 128 | 89 | 11 | ~85% weight-GEMV |
| 512 | 89 | 11 | ~85% weight-GEMV |
| 1024 | 89 | 11 | weight-GEMV + growing KV reads |
| 4096 | 91 | 9 | weight-GEMV ~60% + **KV-streaming ~31%** |

- **Decode is bandwidth-bound at ALL contexts (89–91%)** — it NEVER becomes compute- or cache-bound. So frontier
  #4 (codegen) stays dead for decode at every context.
- The **KV-cache-read/attention kernels** (`r_8_4_<kvlen>`, `r_4_2_8_16_4_<kvlen>`) grow from ~12% (ctx512) to
  **~31% (ctx4096)** and are **bandwidth-bound** (L2 hit 0.8–3.6%) — the KV cache exceeds cache capacity and is
  streamed from HBM each step. The cache-served flash-*reduce* (L2 ~99%) stays small (~9–11%).
- **Efficiency signal:** at ctx4096 KV reads take ~31% of GPU time for only ~0.6 GB, vs ~60% for the 4.68 GB of
  weights → **the KV/attention kernels achieve far lower bandwidth efficiency than the weight-GEMVs.** So long-ctx
  decode has TWO bandwidth levers: amortize weight reads (spec-decode) AND (a) shrink KV bytes (KV quantization)
  and (b) raise KV-read BW efficiency. tok/s decays 84.5→68.4 (ctx128→4096), consistent with the growing KV bytes.

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

## llama.cpp reference — SAME decode structure; gap is weight-GEMV bandwidth efficiency (measured)
llama-bench is HIP/rocBLAS-linked → rocprof-traceable (unlike tinygrad's HCQ). `rocprofv3 --kernel-trace` on
llama decode (tg48, 8B Q4_K) GPU-time breakdown:
| llama kernel | GPU-time% | role |
|---|---:|---|
| `mul_mat_vec_q<Q4_K>` (×2) | **67%** | Q4_K weight GEMVs |
| `mul_mat_vec_q<Q6_K>` (×2) | **19%** | Q6_K weight GEMVs |
| `quantize_q8_1` | 3.6% | activation→Q8_1 (for int8-dot) |
| `rms_norm` (×2) | 4.5% | norms |
| `flash_attn_tile`+combine | 3.1% | attention |
| `rope_neox` (×2) | 2.0% | rope |

**→ llama decode is ~86% weight-GEMV — STRUCTURALLY IDENTICAL to tinygrad (~85%).** Both bandwidth-bound on the
same operation. So the 77 (tinygrad) vs ~96 (llama) tok/s gap is **NOT structural/algorithmic** — it is the
**bandwidth efficiency of the identical weight-GEMV**: llama reads the same ~4.68 GB/token at **~47% of peak HBM
BW**, tinygrad at **~38%** (both from tok/s×bytes). llama uses Q8_1 activation-quant + int8-dot
(`mul_mat_vec_q`); tinygrad's Q4_K GEMV reads the same bytes ~24% less efficiently. (rocprofv3 GL2C/GRBM counters
returned 0 on llama — multiplexing/collection limit, same wall the prior PMU scope hit; the kernel-trace timing
comparison is the trustworthy signal. SQ_WAVES did collect.)

**Decode gap crystallized (triangulated 3 ways — atlas counters, llama kernel-trace, tok/s×bytes):** the lever is
the **Q4_K weight-GEMV's effective HBM bandwidth (38%→47%+)** — a kernel memory-efficiency problem (access
pattern / occupancy / memory-level parallelism), NOT codegen-ALU (VALU idle) and NOT a new algorithm (llama's
structure is identical). Orthogonal multiplier: spec-decode (amortize the weight read across ~2.5 tokens).

## llama PREFILL reference — int8 quantized GEMM, not fp16 (reframes frontier #2)
`rocprofv3 --kernel-trace` on llama prefill (pp512):
| llama kernel | GPU-time% | note |
|---|---:|---|
| `mul_mat_q<Q4_K,128>` | **74.3%** | llama's OWN int8 quantized tiled GEMM (weights stay 4.5-bit) |
| `Cijk_..MT128x128x16_MI16x16x16` | **9.2%** | **Tensile/rocBLAS fp16 WMMA** (the ~10% fp16 portion) |
| `flash_attn_ext_f16` | 4.4% | prefill attention |
| `quantize_mmq_q8_1`, `dequant_q6_K`, silu, rms_norm | rest | |

- **llama prefill keeps weights quantized (int8 MMQ, 74%)**; uses Tensile/rocBLAS only for ~10% (fp16 GEMMs).
  **tinygrad PREFILL_V2 instead dequantizes Q4_K→fp16 then does fp16 WMMA** (~35% of peak / 42 TFLOPS). Different
  strategies: llama reads 4.5-bit weights (int8 dot); tinygrad reads 16-bit (after realizing fp16 weights, +VRAM).
- **Reframes frontier #2:** Tensile fp16 (66 TFLOPS) targets only the path llama uses for ~10% of prefill. To
  match llama's bulk you'd want an **int8 quantized GEMM** (like `mul_mat_q`), OR a fp16-Tensile that beats llama's
  int8 MMQ throughput — open question, needs the int8-vs-fp16 throughput measurement. **The Tensile `.co` is
  confirmed present and working on this box (llama loads `Cijk_*` from rocBLAS)** — so extraction for tinygrad is
  feasible.
- Note: prefill is compute-bound (high L2 reuse), so the byte-count difference (4.5 vs 16-bit) matters less than
  raw matmul throughput — but the int8 path has higher arithmetic density per byte loaded.

## MECHANISM (converged 3 ways) — the decode gap is int-dot e2e INTEGRATION, not the kernel
Reconciling standalone vs in-model achieved HBM BW (all % of the 960 GB/s peak):
| | standalone GEMV | **in-model** weight-GEMV |
|---|---:|---:|
| tinygrad | **76%** (banked `amd-decode-kernel-beats-llamacpp`, % of HBM peak) | **~44%** (4.68 GB / (0.85×token-time)) |
| llama | 57% | **~54%** (4.68 GB / (0.86×token-time)) |

**tinygrad's GEMV is the BETTER kernel standalone (76% > 57%) but loses 32 points going in-model (76→44%);
llama loses only 3 (57→54%).** So the decode lever is NOT a better GEMV kernel and NOT codegen (VALU idle) — it
is the **in-model integration penalty**. Three independent measurements this session converge on it, and it
matches the prior banked boundary note (`amd-decode-kernel-beats-llamacpp`: the standalone win is "e2e-neutral …
the gap is int-dot e2e integration — amortized activation-quant + sustained occupancy across ~252 launches, i.e.
llama's fused mmvq structure, NOT the kernel"):
1. **Amortized activation quantization.** llama quantizes the layer input → Q8_1 ONCE (`quantize_q8_1`, 3.6% of
   decode) and reuses it across the GEMVs sharing that input (q/k/v; gate/up). If tinygrad re-quantizes per GEMV,
   that's redundant work + extra launches.
2. **Sustained max occupancy.** llama's `mul_mat_vec_q` launches **grid=131072, wg=32 (1 wave), vgpr=24–40,
   lds=0** — a deliberately occupancy-maximizing config (tiny low-VGPR workgroups, huge grid → many waves
   resident → saturate memory-level parallelism → high achieved BW). The in-model tinygrad GEMVs evidently don't
   sustain this (76% standalone collapses to 44% in-model).

**→ Decode lever, crystallized & triangulated: replicate llama's fused mmvq integration in tinygrad — amortize
the activation→Q8 quant across input-sharing GEMVs, and sustain the max-occupancy launch config across the
in-model launches. Target: in-model weight-GEMV 44%→54%+ peak BW.** This is a model/integration change (measurable
against the 44% baseline), orthogonal to spec-decode (which multiplies on top by amortizing the weight read).

## Files
`extra/qk_pmc_capture.py`, `extra/qk_primitive_pmu_atlas.py`, `bench/qk-primitive-pmu-atlas/result.json`. Prior:
`route-a-a3-p2-p3-lds-refuted-20260619.md` (LDS), `frontier-scope-beyond-route-a-20260619.md` (#3 spec-decode),
`spec-decode-low-sync-verdict-20260618.md`. Decode tok/s: `bench/qk-decode-runtime-overhead/result.json`.
