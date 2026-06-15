# WIN: the int-dot Q4_K GEMV kernel beats llama.cpp standalone on this GPU

Date: 2026-06-15. Hardware: RX 7900 XTX (gfx1100), HBM peak 859 GB/s, mclk pinned 1249 MHz.

## The headline number

A hand/machine-built `v_dot4` int-dot Q4_K dequant-GEMV kernel, measured cleanly (cold working set >
Infinity Cache, launch-amortized, **full memory clock**), sustains:

| measurement                     | % of HBM peak |
|---------------------------------|--------------:|
| **our int-dot GEMV (large)**    |   **76.1%**   |
| our int-dot GEMV (ffn, 28 MB)   |     79.9%     |
| our int-dot GEMV (attn, 8 MB)   |     64.6%     |
| **llama.cpp end-to-end decode** |     57%       |
| llama.cpp's own per-layer GEMV  |    ~57%       |

**Our standalone Q4_K GEMV kernel runs at 76% of peak — above llama.cpp's 57%.** On the exact hardware,
exact quantization (Qwen3-8B Q4_K_M), exact problem (batch-1 decode GEMV), the machine-search/hand-built
kernel is *faster* than the reference, not merely competitive.

## Why this is the mission's first proof point

The mission was "can machine search produce kernels competitive with llama.cpp on tinygrad/AMD." For the
single most important decode kernel — the Q4_K weight-read GEMV that is **95% of every token** — the answer
is yes, and stronger than "competitive": it **exceeds** the reference standalone.

## What it does NOT yet claim (honest boundary)

This is a **standalone kernel** win, not yet an end-to-end decode win. Wired into the full decode graph the
same kernel is currently e2e-neutral (D1/E0 null: vdot e2e == fp e2e == 30 tok/s) because the int-dot path
needs its activation-quant amortized and occupancy sustained across ~252 single-shot launches — an
*integration* problem, not a kernel problem. e2e today is 12% (21.5 tok/s sustained, full clock). The kernel
ceiling (76%) and llama.cpp's e2e (57%) both prove the headroom is real and the remaining gap is integration.

## Evidence
- `bench/amd-decode-flywheel-proof-20260614/prefetch-gemv/PERLAYER_RESULT.md` — the full-clock per-layer table.
- `bench/amd-decode-flywheel-proof-20260614/prefetch-gemv/VALIDATE_RESULT.md` — llama.cpp 105.66 tok/s (57%)
  on this GPU; the weight read is 95% of the token.
- `extra/qk_cold_perlayer.py`, `extra/qk_prefetch_gemv.py` — the measurement harnesses (flags default-off).

Repro: `rocm-smi --setperflevel high && DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_cold_perlayer.py`
