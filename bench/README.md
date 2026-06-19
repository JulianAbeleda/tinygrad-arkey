# bench/ — benchmark results index

Single source for the fork's current benchmarks: each number, where it's recorded, and the exact
command to reproduce it. Most `bench/**` output is gitignored (regeneratable); durable result
artifacts are force-added. Doc map: `../docs/README.md`; current state: `../docs/amd-decode-banked-20260616.md`.

**Setup (all commands):** `cd /home/ubuntu/tinygrad-arkey`, interpreter `.venv/bin/python`, `DEV=AMD`,
RX 7900 XTX (gfx1100), models at `/home/ubuntu/models/`. Bar: **llama.cpp ≈ 101–106 tok/s** (8B decode).
Measurement discipline (the 3 confounds: cache / launch / clock-ramp) — `../docs/amd-decode-measurement-confounds.md`.

## Which harness for decode tok/s — READ FIRST (don't repeat the 2026-06-18 mistake)

Decode tok/s is only trustworthy from a **clean `model.generate`-class path** (device-token feedback, **no
per-step host `Tensor` creation** — that artifact *halves* the rate; `../docs/qk-runtime-overhead-arc-result-20260617.md`).
Pick the harness by what you're measuring:

| want | use | gives | notes |
|---|---|---|---|
| **production tok/s @ ctx≈0** (default headline) | **`-m tinygrad.llm … --warmup --benchmark`** (`tinygrad/llm/cli.py`) | ~86 tok/s, single point | the production default; `model.generate`, clean path |
| **decode tok/s vs ctx** (512/1024/4096…) | **`extra/qk_decode_runtime_overhead.py`** (in-model **W==D**) | 68.2/66.4/60.7 @ 512/1024/4096 | only harness that sweeps ctx on the clean path; W≈D ⇒ GPU-bound |
| flash-decode **policy** (which path is selected, off/on/auto) | `extra/qk_flash_decode_auto_bench.py` | ~54–56 flat | ⚠️ **NOT a tok/s number** — it builds a host input `Tensor` per step inside the timed loop (contaminated by design); use it for selection/correctness only |

Both clean harnesses agree on one curve: **~86 @ctx≈0 → 68.2/66.4/60.7 @ 512/1024/4096** (default stack,
2026-06-18, HEAD; matches banked within 0.2). See `../docs/qk-decode-banked-reproduce-20260618.md`,
`qk-decode-runtime-overhead/result.json`.

## Current numbers

| benchmark | value | recorded in | reproduce |
|---|---:|---|---|
| **Decode 8B, default-on** (coop stack + gqa_coop_vec + flash, HEAD 2026-06-18) | **~86 @ctx≈0; 68.2/66.4/60.7 @ctx 512/1024/4096** (~67% llama) | `qk-decode-banked-reproduce-20260618.md`, `qk-8b-decode-banked-20260617.md` | CLI `--warmup --benchmark` (ctx≈0); `extra/qk_decode_runtime_overhead.py` (ctx sweep) — see "Which harness" above |
| _superseded:_ Decode 8B pre-coop (~55) / +ffn_down demote (64.3) | historical | `amd-decode-banked-20260616.md`, `bench/qk-demote-search/search.json` | folded into the default-on stack above |
| **Demotion frontier** (ffn_down/attn_v accept; lm_head 75.0 but rejected on dNLL +0.051) | see json | **`bench/qk-demote-search/search.json`** (+ `accepted-*.json`) | `python -m extra.qk_demote_search --epsilon 0.01` |
| **Decode 14B** (generated policy) | 40.6 tok/s (62%) | **`bench/qk-shared-storage-20260612/matrix-summary.md`** | harness; see that dir's README |
| **Decode 32B** (generated policy) | 17.2 tok/s (56%) | same matrix-summary | same |
| **Standalone Q4_K GEMV** (int-dot) | **76% of HBM peak** (vs llama 57%) | `amd-decode-capstone.md`, memory `amd-decode-kernel-beats-llamacpp` (raw artifact gitignored) | `extra/q4_k_bench` cold/full-clock; see capstone |
| **Flash long-context** (ctx 3072) | 9.4 → **22.7 tok/s (2.41×)** | `amd-decode-flash-attention-plan.md` (SHIPPED section), memory | `FLASH_DECODE=1` decode at long ctx; `test/external/test_qk_flash_decode.py` |
| **Sequential-tax split** | GEMV 72% / non-GEMV 28% | `amd-decode-sequential-tax-profile-*.md` | `DEBUG=2 JIT_BATCH_SIZE=1 cli --benchmark 8` → strip ANSI → `python -m extra.q4_k_profile_report` |
| **Decode dNLL (quality gate)** | baseline 2.779; ffn_down +0.0005 | `amd-decode-demotion-search-*.md` | `python -m extra.qk_nll_eval --model <gguf> --tokens 128` (±0.01 calib noise) |
| **Overlap feasibility (probe)** | 1.0× (gated: one compute ring) | `amd-decode-two-queue-probe-*.md` | `python -m extra.qk_two_queue_probe` |
| **Prefill 8B** | ~67 tok/s (**~2% llama**) — outlier | `amd-decode-prefill-plan.md` | see that doc (diagnosed, not solved) |
| **Prefill external BLAS ceiling** (standalone fp16 GEMM control, not routed) | ffn_gate/up: hipBLASLt **69.8 TFLOPS** (1.71× tinygrad); ffn_down: rocBLAS **70.9**; attn_q/o: rocBLAS **76.7** | **`bench/qk-prefill-external-blas/ceiling.json`**, `docs/prefill-external-blas-result-20260619.md` | `g++ -std=c++17 -D__HIP_PLATFORM_AMD__=1 ... extra/qk_prefill_blas_ceiling.cpp`; see result doc |
| **Prefill pure tinygrad WMMA sweep** (not routed) | best **42.0 TFLOPS** (34% peak), gate was 62 TFLOPS; more waves/bigger tiles/BK32/noLDS regress | **`bench/qk-prefill-own-wmma/sweep.txt`**, `docs/prefill-own-wmma-kernel-result-20260619.md` | `DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_prefill_wmma_sweep.py` |

## Reproduce — the two most-cited

```sh
# Decode 8B, default-on @ctx≈0 (production headline) -> ~86 tok/s steady median (drop first ~3, clock-ramp)
DEV=AMD PYTHONPATH=. .venv/bin/python -m tinygrad.llm -m /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --warmup --benchmark 40

# Decode 8B vs ctx (the banked curve) -> 68.2/66.4/60.7 @ 512/1024/4096, in-model W==D, host-sync %
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_runtime_overhead.py

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
