# bench/ — benchmark results index

Single source for the fork's current benchmarks: each number, where it's recorded, and the exact
command to reproduce it. Most `bench/**` output is gitignored (regeneratable); durable result
artifacts are force-added. Doc map: `../docs/README.md`; current state: `../docs/amd-decode-banked-20260616.md`.

**Setup (all commands):** `cd /home/ubuntu/tinygrad-arkey`, interpreter `.venv/bin/python`, `DEV=AMD`,
RX 7900 XTX (gfx1100), models at `/home/ubuntu/models/`. Bar: **llama.cpp ≈ 98–106 tok/s** (8B decode,
depending ctx/harness) and **~3020 pp512 tok/s** (8B prefill).
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

Both clean harnesses agree on one curve: **~86 @ctx≈0 → 68.4/66.9/61.2 @ 512/1024/4096** (default stack,
rerun 2026-06-20; matches banked). See `../docs/qk-decode-banked-reproduce-20260618.md`,
`qk-decode-runtime-overhead/result.json`.
> ⚠️ **`87.6` is ambiguous — never quote it bare.** It is BOTH a real ctx≈0 decode **tok/s** (~11.4 ms) AND a real
> ctx4096 decode **ms/token** (=11.4 tok/s). The decode headline is the *curve* (~86 @ctx≈0 → ~61 @ctx4096 = **~67%
> llama** steady-state), not the ctx≈0 peak. Prefill policy (`PREFILL_V2=auto`/server) does NOT change decode (<1%,
> identical output). See `../docs/decode-prefill-headline-reconciliation-result-20260621.md`.

## Current numbers

| benchmark | value | recorded in | reproduce |
|---|---:|---|---|
| **Decode 8B, default-on** (coop stack + gqa_coop_vec + flash) | **~86 @ctx≈0; 68.4/66.9/61.2 @ctx 512/1024/4096** (~67% llama) | `bench/qk-decode-runtime-overhead/result.json`, `qk-decode-banked-reproduce-20260618.md`, `qk-8b-decode-banked-20260617.md` | CLI `--warmup --benchmark` (ctx≈0); `extra/qk_decode_runtime_overhead.py` (ctx sweep) — see "Which harness" above |
| **Decode 8B, q8 FFN research route** (default-off / opt-in) | **72.9/71.1 tok/s @ctx 512/1024** in `auto`; median `~72.0`, `~1.064×`, host-sync `0.0%` | `bench/qk-decode-primitive-transfer/decode_q8_model_route_timing_audit_rerun_20260620.json`, `docs/decode-q8-model-route-timing-audit-result-20260620.md` | `PYTHONPATH=. .venv/bin/python extra/qk_decode_q8_model_route_timing_audit.py --lanes auto,manual_peak --modes baseline,q8 --ckpts 512 1024 --nmeas 20 --warmups 8 --out bench/qk-decode-primitive-transfer/decode_q8_model_route_timing_audit_rerun_20260620.json` |
| _superseded:_ Decode 8B pre-coop (~55) / +ffn_down demote (64.3) | historical | `amd-decode-banked-20260616.md`, `bench/qk-demote-search/search.json` | folded into the default-on stack above |
| **Demotion frontier** (ffn_down/attn_v accept; lm_head 75.0 but rejected on dNLL +0.051) | see json | **`bench/qk-demote-search/search.json`** (+ `accepted-*.json`) | `python -m extra.qk_demote_search --epsilon 0.01` |
| **Decode 14B** (generated policy) | 40.6 tok/s (62%) | **`bench/qk-shared-storage-20260612/matrix-summary.md`** | harness; see that dir's README |
| **Decode 32B** (generated policy) | 17.2 tok/s (56%) | same matrix-summary | same |
| **Standalone Q4_K GEMV** (int-dot) | **76% of HBM peak** (vs llama 57%) | `amd-decode-capstone.md`, memory `amd-decode-kernel-beats-llamacpp` (raw artifact gitignored) | `extra/q4_k_bench` cold/full-clock; see capstone |
| **Flash long-context** (ctx 3072) | 9.4 → **22.7 tok/s (2.41×)** | `amd-decode-flash-attention-plan.md` (SHIPPED section), memory | `FLASH_DECODE=1` decode at long ctx; `test/external/test_qk_flash_decode.py` |
| **Sequential-tax split** | GEMV 72% / non-GEMV 28% | `amd-decode-sequential-tax-profile-*.md` | `DEBUG=2 JIT_BATCH_SIZE=1 cli --benchmark 8` → strip ANSI → `python -m extra.q4_k_profile_report` |
| **Decode dNLL (quality gate)** | baseline 2.779; ffn_down +0.0005 | `amd-decode-demotion-search-*.md` | `python -m extra.qk_nll_eval --model <gguf> --tokens 128` (±0.01 calib noise) |
| **Overlap feasibility (probe)** | 1.0× (gated: one compute ring) | `amd-decode-two-queue-probe-*.md` | `python -m extra.qk_two_queue_probe` |
| _superseded:_ **Prefill 8B old baseline** | ~67 tok/s (**~2% llama**) — obsolete/outlier | `amd-decode-prefill-plan.md` | replaced by PREFILL_V2 graph route + Branch B + Increment 0 below |
| **Prefill 8B, Branch B concrete first chunk** (`PREFILL_TC_ATTN`, default-on under `PREFILL_V2`/gfx1100) | **3394 pp512 tok/s** (`112.7%` llama) for concrete start_pos=0; byte-identical; no WMMA fired, win is fusion | `bench/qk-prefill-tc-attention/concrete_gate_result.json`, `docs/prefill-branch-b-tc-attention-result-20260620.md` | `DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_prefill_tc_attn_concrete_gate.py` |
| **Prefill 8B, Increment 0 concrete-KV** (`PREFILL_V2=1 PREFILL_CONCRETE_KV=1`, opt-in) | per-chunk forward **1.7–4.4×** faster than symbolic; **73–111% llama** across KV 512–3584; rerun A2 warm prefill **4941→343 ms** when both chunks are concrete; rerun A1 first-gen prefill **9.41→3.44 s** after precompile-at-load; byte-identical tok0 | `docs/prefill-concrete-kv-increment0-result-20260620.md`, `docs/prefill-increment0-shipped-result-20260620.md` | `DEV=AMD PREFILL_V2=1 [PREFILL_CONCRETE_KV=1] PYTHONPATH=. .venv/bin/python extra/qk_prefill_concrete_kv_a1_precompile_test.py`; same for `extra/qk_prefill_concrete_kv_a2_verify.py` |
| **Prefill flash Increment 2** (not shipped) | correct (`rel_rmse ~1e-7`) but **~15× too slow** at KV512 and worse at long KV | `docs/prefill-flash-increment2-result-20260620.md` | `extra/qk_prefill_flash.py`; `extra/qk_prefill_flash_perf.py` |
| **Prefill POLICY profiles** (shipped, gated) | default (slow long prompts) / **`PREFILL_V2=auto`** (24GB+, ~5–15× faster, VRAM-gated) / **`PREFILL_SERVER_PROFILE=1`** (best warm prefill 0.17–1.6s). `PREFILL_REMAINDER_FIX` default-on kills the 32-token trap (up to 14× on prefix-cache resume, byte-identical) | `docs/prefill-policy-integration-result-20260620.md`, `bench/qk-prefill-policy-integration/*.json` | `extra/qk_prefill_{v2_auto_policy,concrete_kv_policy,route_schedule}_probe.py` |
| **Prefill external BLAS ceiling** (standalone fp16 GEMM control, not routed) | ffn_gate/up: hipBLASLt **69.8 TFLOPS** (1.71× tinygrad); ffn_down: rocBLAS **70.9**; attn_q/o: rocBLAS **76.7** | **`bench/qk-prefill-external-blas/ceiling.json`**, `docs/prefill-external-blas-result-20260619.md` | `g++ -std=c++17 -D__HIP_PLATFORM_AMD__=1 ... extra/qk_prefill_blas_ceiling.cpp`; see result doc |
| **Prefill pure tinygrad WMMA sweep** (not routed) | best **42.0 TFLOPS** (34% peak), gate was 62 TFLOPS; more waves/bigger tiles/BK32/noLDS regress | **`bench/qk-prefill-own-wmma/sweep.txt`**, `docs/prefill-own-wmma-kernel-result-20260619.md` | `DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_prefill_wmma_sweep.py` |

## Reproduce — the two most-cited

```sh
# Decode 8B, default-on @ctx≈0 (production headline) -> ~86 tok/s steady median (drop first ~3, clock-ramp)
DEV=AMD PYTHONPATH=. .venv/bin/python -m tinygrad.llm -m /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --warmup --benchmark 40

# Decode 8B vs ctx (the banked curve) -> 68.4/66.9/61.2 @ 512/1024/4096, in-model W==D, host-sync %
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_runtime_overhead.py

# Prefill 8B concrete-KV opt-in path -> precompile concrete prefill jits at load, then byte-identical faster prefill
DEV=AMD PREFILL_V2=1 PREFILL_CONCRETE_KV=1 PYTHONPATH=. .venv/bin/python -m tinygrad.llm -m /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --warmup --benchmark 1

# The full demotion search (frontier + accepted policies) -> writes bench/qk-demote-search/
DEV=AMD .venv/bin/python -m extra.qk_demote_search --epsilon 0.01 --bench 24 --tokens 128
```

## Notes on the record

- **Committed (raw artifacts):** `qk-demote-search/` (this arc's search), `qk-shared-storage-20260612/`
  (8B/14B/32B matrix + policies). Reproducible from the JSON.
- **Doc-only (artifact gitignored by the prune, regeneratable):** the standalone-kernel 76%, the flash
  ctx-sweep, the older prefill sweep, and some prefill Increment 0 harness outputs — numbers are in the cited
  docs + committed scripts; rerun the command to regenerate.
- **dNLL numbers carry ~±0.01 noise** (128-token calib set) — directionally trustworthy (the gate
  correctly rejected lm_head at +0.051), not precise to 3 decimals.
- Decode tok/s is the **steady-state median** with first ~3 tokens dropped (clock-ramp); the cold first
  token (~100 ms) is not the benchmark.
