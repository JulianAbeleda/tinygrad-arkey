# llama.cpp baseline on THIS RX 7900 XTX — measured 2026-06-17

The campaign's `llama.cpp ≈ 101–106 tok/s` figure was from an uncertain/elsewhere source. Measured fresh on the
actual host (RX 7900 XTX, gfx1100, 24 GB; rocminfo-confirmed), so "% of llama" is now anchored to reality.

- llama.cpp build `ac4cddeb0 (9592)`, ROCm 7.2.4, backend **ROCm/HIP**, `-ngl 99` (all layers on GPU).
- Model: `Qwen3-8B-Q4_K_M.gguf`. Tool: `~/env/llama.cpp/build/bin/llama-bench -r 3`.

## Decode (tg128 @ context depth) vs tinygrad (hoisted/L128 default, W==D in-model)

| ctx | tinygrad tok/s | **llama.cpp tok/s** | tinygrad % of llama |
|---:|---:|---:|---:|
| ~0 (d0) | — | 99.52 | — |
| 512 | 43.5 | 98.55 | **44%** |
| 1024 | 39.1 | 97.59 | **40%** |
| 2048 | 32.7 | 95.35 | **34%** |
| 4096 | 24.8 | 92.20 | **27%** |

## Prefill

| test | tinygrad | **llama.cpp** | % |
|---|---:|---:|---:|
| pp512 | ~2486 (PREFILL_V2, warm) | **3068.99** | **81%** |

## Key findings

1. **Baseline confirmed:** short-ctx decode here is **99.5 tok/s** (the ~101–106 figure was slightly high but
   right ballpark). Prefill pp512 = **3069 tok/s**. These now replace the assumed numbers.
2. **llama decode is ~context-flat:** 99.5 → 92.2 (**−7%**) from ctx0→4096. tinygrad decays 43.5 → 24.8
   (**−43%**). The gap therefore **grows with context (44% → 27%)**, and it is **entirely attention**:
   tinygrad's `flash_partial` is 47% of decode at ctx4096; llama's attention costs ~nothing there.
3. **Implication for decode-attention work:** llama proves an **efficient, context-flat decode-attention kernel
   EXISTS on this XTX**. So decode attention is **not at a fundamental floor** — what the v3 arc refuted is that
   *our specific levers* (LDS staging, WMMA at decode-M) beat the current hoisted flash, not that a cheaper
   kernel is impossible. llama uses a tuned FA2-style flash-decode (efficient KV streaming, not LDS-staged
   re-tiling per our probes). Matching it is a harder, different kernel structure — open, not closed.
4. **Short-ctx gap (~44%) is the structural decode-block gap** (GEMV + program granularity), per the
   decode-block map. The **long-ctx extra gap is attention** specifically.

Artifact: `bench/qk-llama-baseline-xtx/result.json` (gitignored; regenerate with the llama-bench command above).
