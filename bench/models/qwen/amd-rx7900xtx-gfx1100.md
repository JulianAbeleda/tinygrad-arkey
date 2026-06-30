# Qwen3 benchmarks — AMD Radeon RX 7900 XTX (gfx1100, 24GB)

Backend: **AMD** · GPU: **AMD Radeon RX 7900 XTX (gfx1100, 24GB)** · family: **Qwen3**

Decode tok/s is the headline (decode is HBM-bandwidth bound). Numbers come from clean whole-decode `model.generate` (W==D), `PROFILE=0`, auto clock, warmed JIT, with a median over a steady-state window and the observed spread. **Quant matters** — it sets the bytes-per-weight moved each decode step, which is the dominant decode cost; compare sizes with quant in mind, not just parameter count.

| Model | Quant | Params | Ctx | Decode tok/s (median) | Decode band [min–max] | Spread | Decode GB/s | Prefill pp512 tok/s | VRAM | Load s |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-0.6b | Q8_0 | 752M | 2048 | 220.38 | 194.03–257.98 | 28.15% | 153.4 | 1509.1 | 1.12 GB | 2.64 |
| qwen3-8b | Q4_K_M | 8.19B | 2048 | 90.58 | 84.59–98.36 | 14.99% | 436.6 | 68.1 | 5.36 GB | 6.56 |
| qwen3-14b | Q4_K_M | 14.77B | 2048 | 26.28 | 25.69–27.07 | 5.19% | 231.4 | 39.2 | 9.71 GB | 8.83 |
| qwen3-32b | Q4_K_M | 32.76B | 2048 | 12.46 | 12.23–12.71 | 3.86% | 247.4 | 15.6 | 20.89 GB | 17.04 |

## Notes

- **Decode tok/s** is the steady-state median (clock-ramp/first tokens dropped). High **spread** on the smallest models is expected: they are launch/dispatch-bound (tiny per-token GPU work), so wall-clock decode is noisy — the band shows it honestly rather than hiding it behind a single number.
- **Decode GB/s** is the HBM-bandwidth proxy (bytes moved per token ÷ median token time). For a fixed quant it should rise with model size until it saturates the GPU's memory bandwidth.
- **Prefill pp512 tok/s** is time-to-first-token for a 512-token prompt on the default prefill path (prefill is compute-bound, not memory-bound — a different regime from decode).
- **VRAM** is `GlobalCounters.mem_used` after load+warmup at the listed context. Larger contexts grow the KV cache and raise this.

Provenance: tinygrad commit `26edbfd87` (dirty tree). Regenerate with `python extra/gen_model_bench_doc.py` from the JSON artifacts in `bench/models/qwen/data/amd-gfx1100/`.
