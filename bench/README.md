# bench/ — benchmark results index

Single source for the fork's current benchmarks: each number, where it's recorded, and the exact
command to reproduce it. Most `bench/**` output is gitignored (regeneratable); durable result
artifacts are force-added. Doc map: `../docs/README.md`; current state: `../docs/amd-decode-banked-20260616.md`.

**Setup (all commands):** `cd /home/ubuntu/tinygrad-arkey`, interpreter `.venv/bin/python`, `DEV=AMD`,
RX 7900 XTX (gfx1100), models at `/home/ubuntu/models/`. Bar: **llama.cpp ≈ 101–106 tok/s** (8B decode).
Measurement discipline (the 3 confounds: cache / launch / clock-ramp) — `../docs/amd-decode-measurement-confounds.md`.

## Current numbers

| benchmark | value | recorded in | reproduce |
|---|---:|---|---|
| **Decode 8B, default-on** | ~55 tok/s (52% llama) | `amd-decode-banked-*.md` | `cli ... --warmup --benchmark 40` (steady median, drop first ~3) |
| **Decode 8B, + ffn_down demote** | **64.3 tok/s (63%)** | `bench/qk-demote-search/search.json` | prefix `QK_DEMOTE_TENSORS=ffn_down` on the above |
| **Demotion frontier** (ffn_down/attn_v accept; lm_head 75.0 but rejected on dNLL +0.051) | see json | **`bench/qk-demote-search/search.json`** (+ `accepted-*.json`) | `python -m extra.qk_demote_search --epsilon 0.01` |
| **Decode 14B** (generated policy) | 40.6 tok/s (62%) | **`bench/qk-shared-storage-20260612/matrix-summary.md`** | harness; see that dir's README |
| **Decode 32B** (generated policy) | 17.2 tok/s (56%) | same matrix-summary | same |
| **Standalone Q4_K GEMV** (int-dot) | **76% of HBM peak** (vs llama 57%) | `amd-decode-capstone.md`, memory `amd-decode-kernel-beats-llamacpp` (raw artifact gitignored) | `extra/q4_k_bench` cold/full-clock; see capstone |
| **Flash long-context** (ctx 3072) | 9.4 → **22.7 tok/s (2.41×)** | `amd-decode-flash-attention-plan.md` (SHIPPED section), memory | `FLASH_DECODE=1` decode at long ctx; `test/external/test_qk_flash_decode.py` |
| **Sequential-tax split** | GEMV 72% / non-GEMV 28% | `amd-decode-sequential-tax-profile-*.md` | `DEBUG=2 JIT_BATCH_SIZE=1 cli --benchmark 8` → strip ANSI → `python -m extra.q4_k_profile_report` |
| **Decode dNLL (quality gate)** | baseline 2.779; ffn_down +0.0005 | `amd-decode-demotion-search-*.md` | `python -m extra.qk_nll_eval --model <gguf> --tokens 128` (±0.01 calib noise) |
| **Overlap feasibility (probe)** | 1.0× (gated: one compute ring) | `amd-decode-two-queue-probe-*.md` | `python -m extra.qk_two_queue_probe` |
| **Prefill 8B** | ~67 tok/s (**~2% llama**) — outlier | `amd-decode-prefill-plan.md` | see that doc (diagnosed, not solved) |

## Reproduce — the two most-cited

```sh
# Decode 8B, default-on (out-of-box) -> ~55 tok/s steady median
DEV=AMD .venv/bin/python -m tinygrad.llm.cli -m /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --warmup --benchmark 40

# The full demotion search (frontier + accepted policies) -> writes bench/qk-demote-search/
DEV=AMD .venv/bin/python -m extra.qk_demote_search --epsilon 0.01 --bench 24 --tokens 128
```

## Notes on the record

- **Committed (raw artifacts):** `qk-demote-search/` (this arc's search), `qk-shared-storage-20260612/`
  (8B/14B/32B matrix + policies). Reproducible from the JSON.
- **Doc-only (artifact gitignored by the prune, regeneratable):** the standalone-kernel 76%, the flash
  ctx-sweep, the prefill sweep — numbers are in the cited docs + memory; rerun the command to regenerate.
- **dNLL numbers carry ~±0.01 noise** (128-token calib set) — directionally trustworthy (the gate
  correctly rejected lm_head at +0.051), not precise to 3 decimals.
- Decode tok/s is the **steady-state median** with first ~3 tokens dropped (clock-ramp); the cold first
  token (~100 ms) is not the benchmark.
