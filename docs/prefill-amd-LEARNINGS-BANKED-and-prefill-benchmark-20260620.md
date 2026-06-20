# Prefill AMD — Learnings Banked + In-Model Prefill Benchmark

Date: 2026-06-20

## In-model prefill benchmark (the number that matters)

Warm full-forward, Qwen3-8B-Q4_K_M, gfx1100, `PREFILL_V2=1`
(`extra/qk_prefill_v2_measure.py`):

| path | tok/s | ms/512 |
|---|---:|---:|
| baseline (symbolic v_toks) | 205 | 4.89/tok |
| **PREFILL_V2 (production)** | **2,797** | **183** |
| speedup | **13.7×** | (byte-identical greedy, warmstart error 0) |

- **In-model prefill = ~45 TFLOPS effective** (8.2 GFLOP/512-tok ÷ 183 ms) = **~93% of llama** (pp512 3020 tok/s).
- This is the real end-to-end prefill throughput; it is the banked production number.

## The decisive learning: isolated kernel wins ≠ in-model throughput

| | TFLOPS |
|---|---:|
| our dependency-free GEMM, **isolated** (gold-standard GPU-time) | **~78** |
| Tensile `.co`, isolated | ~71 |
| tinygrad LLVM authority, isolated | ~55–71 (clock) |
| **in-model PREFILL_V2 forward, effective** | **~45** |

Our isolated GEMM is **~10% faster than Tensile** and **~93% of the 83.6-TFLOP fp16 peak** — a genuine kernel
win, thoroughly verified (two independent methods, 3× each). **But the in-model prefill runs at ~45 TFLOPS**,
~40% below the isolated kernel. The integration penalty (fusion boundaries, attention + non-matmul ops, KV,
host/JIT, the ffn_gate/up being only part of the FLOP) dominates. **So the isolated GEMM win does not
meaningfully transfer to prefill tok/s** — confirming the universal lesson on this hardware: *the bottleneck
is in-model integration, not the standalone kernel.*

This is why the kernel arc, while a clean technical win, does not move the production prefill number: the
production path is already at ~45 TFLOPS / 93% of llama, integration-bound.

## The dependency-free GEMM arc — what was learned (banked)

**The win:** `build_gemm_lds2(BK=32, PAD=16, PLRA=1)` square-128 @ wg2 — correct, **~10% faster than the
vendored Tensile `.co` at the GPU level** (78 vs 71 TFLOPS, two methods, 3× each), zero dependencies, on a
shape Tensile never tuned.

**Levers that worked (each measured):**
- PAD16 bank-conflict-free LDS: **+13%** (PMC bankcf 28.6→2.7/cyc) — biggest single win
- A-prefetch PLR (into dead coop-temp regs): **+9%** (PMC-confirmed latency hiding)
- wg2 occupancy (the L2-contention sweet spot); BK32 depth; square 128 tile

**Levers measured and ruled out:** full A+B PLR (VGPR-dominated), L2 locality (ours better), VALU/address
(matched Tensile, neutral — `LEANADDR`), ds_load 2× (+3% ceiling, hidden), non-square tiles (refuted).

**Corrections caught by measurement (the real value):**
1. "LDS-multiwave refuted / 3.2 TFLOPS" → occupancy-crippled at 65536 B LDS; real footprint is fine.
2. "Latency-bound, needs occupancy" → refuted; **contention** with an interior wg2 optimum.
3. "Tensile 79 TFLOPS ceiling" → an M=384 **outlier**; realistic cluster ~65.
4. "Our shape is tuned by Tensile" → it's **untuned** (M=512, N=12288 both off-grid); the `.co` is a fallback.
5. "We're ~4–8% behind Tensile" → **host launch overhead** of the `run_linear` path; at GPU level we're ~10% ahead.

## Methodology lessons (banked, reusable)

- **`wait=True` on a raw `AMDProgram` = pure GPU execution time** (on-chip signal timestamps) — the gold
  standard for kernel-vs-kernel. Single-launch wall-clock conflates host overhead; the **batch sweep
  (K=1/8/32)** separates host from GPU (ratio moves with K ⇒ host; stable ⇒ GPU).
- **Never interleave a foreign-`.co` launch with your kernel for timing** — it perturbs yours.
- **Pin the clock** (`rocm-smi --setperflevel high`) for reproducible absolutes; reset to `auto` after.
- **LDS-kernel TFLOPS is occupancy-confounded** — always report the LDS bytes launched with.
- **Name micro-inefficiencies via a disasm hot-loop diff**, but **bound each with a sensitivity probe** before
  a risky rewrite — most are hidden behind compute/occupancy.
- **Before micro-optimizing vs a vendor lib, check whether your shape is in its tuning table** — a fallback
  can be 20%+ off and contain non-reproducible outliers; verify by running *your* kernel across the shape grid.
- **Isolated kernel wins don't transfer in-model** — measure end-to-end; the integration penalty (here ~78→45
  TFLOPS) usually dominates.

## Standing

- **Production prefill: ~2797 tok/s (~93% of llama), integration-bound at ~45 TFLOPS effective.** Banked.
- **Dependency-free GEMM: ~10% faster than vendored Tensile in isolation (~78 TFLOPS, ~93% of fp16 peak),
  thoroughly verified.** A clean kernel win — but isolated; it does not move the integration-bound prefill
  number, by the universal lesson.
- All probes/docs in the `docs/prefill-amd-gemm-*-20260620.md` series; kernel in
  `extra/gemm/rdna3_wmma_matmul.py` (`build_gemm_lds2`, defaults unchanged).
