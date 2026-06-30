# Qwen3 benchmarks — AMD Radeon RX 7900 XTX (gfx1100, 24GB)

Backend: **AMD** · GPU: **AMD Radeon RX 7900 XTX (gfx1100, 24GB)** · family: **Qwen3**

**Quant matters** — decode re-reads the weights every token, so bytes-per-weight (the quant) is the dominant decode cost. Read tok/s next to its quant, not parameter count alone.

## Decode vs llama.cpp — authority (matched context)

tinygrad: clean **W==D** decode (`qk_decode_runtime_overhead.py` — `TinyJit`, device-synced, NMEAS=40, **fixed** context, shipped `FLASH_DECODE_THRESHOLD=512` so the owned flash-attention route fires at ctx≥512). llama.cpp: `llama-bench tg128` at the **matched depth** (`-d ctx`). Comparing at the same context is essential — tinygrad switches to the owned flash route at ctx≥512, and llama is ~flat across context, so a single number hides the crossover.

| Model | Quant | ctx | route | tinygrad W==D tok/s | llama tg@depth tok/s | ratio | host-sync |
|---|---|---|---|---|---|---|---|
| qwen3-0.6b | Q8_0 | 128 | non-flash | 183.2 | 279.96 | **65.4%** | 2.3% |
| qwen3-0.6b | Q8_0 | 512 | flash | 194.3 | 274.09 | **70.9%** | 1.8% |
| qwen3-8b | Q4_K_M | 128 | non-flash | 82.0 | 99.78 | **82.2%** | 0.0% |
| qwen3-8b | Q4_K_M | 512 | flash | 103.5 | 98.7 | **104.9%** | 0.0% |
| qwen3-14b | Q4_K_M | 128 | non-flash | 25.5 | 65.85 | **38.7%** | 0.0% |
| qwen3-14b | Q4_K_M | 512 | flash | 25.0 | 65.08 | **38.4%** | 0.0% |
| qwen3.5-27b | Q4_K_M | 128 | non-flash | 2.3 | 32.8 | **7.0%** | 0.0% |
| qwen3.5-27b | Q4_K_M | 512 | flash | 2.3 | 32.74 | **7.0%** | 0.0% |
| qwen3-32b | Q4_K_M | 128 | non-flash | 12.2 | 31.2 | **39.1%** | 0.0% |
| qwen3-32b | Q4_K_M | 512 | flash | 11.8 | 30.78 | **38.3%** | 0.0% |

Low **host-sync %** means the measurement is GPU-bound (not host-loop noise). At ctx≥512 the owned flash route fires; below it the non-flash path runs and is the weaker regime.

**Reading the ratios:** 8B (the size the decode kernels were tuned for) is at/above llama in the flash regime (~105% @ctx512) and ~82% on the sub-512 non-flash path. 14B/32B sit near ~40% — the larger shapes were never decode-optimized (a known, separate gap), not a measurement error. 0.6B is launch/dispatch-bound (tiny per-token GPU work), where tinygrad's per-kernel overhead costs the most.

> ⚠️ **Different architecture:** qwen3.5-27b is **Qwen3.5** (hybrid SSM/attention layers), not Qwen3. tinygrad has no tuned path for that architecture, so its ~7% is an unsupported-performance result, not a like-for-like Qwen3 comparison. Listed for completeness only.

## Prefill (pp512)

Prefill is compute-bound (a different regime from decode). tinygrad's tuned path is `PREFILL_V2` graph-gemm (needs ~+14GB VRAM, so it only fits the smaller models); where it wasn't measured the cell says so rather than showing the slow universal-path number.

| Model | Quant | tinygrad pp512 (tuned authority) | llama.cpp pp512 | ratio | route |
|---|---|---|---|---|---|
| qwen3-0.6b | Q8_0 | _not measured (VRAM / pending_) | 19263.5 | — | — |
| qwen3-8b | Q4_K_M | 4441.5 | 3003.3 | **148%** | graph-gemm (PREFILL_V2) |
| qwen3-14b | Q4_K_M | _not measured (VRAM / pending_) | 1632.0 | — | — |
| qwen3.5-27b | Q4_K_M | _not measured (VRAM / pending_) | 835.2 | — | — |
| qwen3-32b | Q4_K_M | _not measured (VRAM / pending_) | 736.2 | — | — |

## End-to-end `generate` (diagnostic, not parity)

> These are first-pass end-to-end numbers: decode is a **median over a growing-context** `model.generate` window (context-mixed + host jitter), and prefill is the **default universal path** (`PREFILL_V2=false`) via `generate` TTFT. Useful as a rough end-to-end feel; **not** a parity number — use the authority tables above for tinygrad-vs-llama.

| Model | Quant | Params | Ctx | Decode tok/s (median) | Spread | Decode GB/s | Prefill TTFT (default path) | VRAM | Load s |
|---|---|---|---|---|---|---|---|---|---|
| qwen3-0.6b | Q8_0 | 752M | 2048 | 220.38 | 28.15% | 153.4 | 1509.1 | 1.12 GB | 2.64 |
| qwen3-8b | Q4_K_M | 8.19B | 2048 | 90.58 | 14.99% | 436.6 | 68.1 | 5.36 GB | 6.56 |
| qwen3-14b | Q4_K_M | 14.77B | 2048 | 26.28 | 5.19% | 231.4 | 39.2 | 9.71 GB | 8.83 |
| qwen3-32b | Q4_K_M | 32.76B | 2048 | 12.46 | 3.86% | 247.4 | 15.6 | 20.89 GB | 17.04 |

Provenance: tinygrad commit `26edbfd87`. Regenerate with `python extra/gen_model_bench_doc.py` from the JSON artifacts in `bench/models/qwen/data/amd-gfx1100/` (artifacts are local per the bench policy; this table is the committed durable record).
