# bench/ — benchmark results index

Single source for the fork's **current** benchmarks: each number, where it's recorded, and the exact
command to reproduce it. Most `bench/**` output is gitignored (regeneratable); durable result
artifacts are force-added. Doc map: `../docs/README.md`; **canonical current state: `../docs/current-project-state-handoff-20260624.md`**.
Historical probe-log artifacts live under their dated `bench/<arc>/` dirs and `../docs/archive/`.

> **Canonical policy (do not re-open — guardrail `extra/qk_policy_consistency_check.py`):** Global `PREFILL_V2`
> default stays **OFF**; `PREFILL_V2=auto` / `PREFILL_SERVER_PROFILE=1` / q8 are **opt-in**. `87.6` is contextual
> (ctx≈0 tok/s vs a separate ctx4096 ms/token) — **never quote it bare**. **Decode now runs at/above llama parity**
> on the `Q4K_GEMV_WARP*` default stack (**101.6/99.8/97.4/92.9 tok/s @ctx512/1024/2048/4096 ≈ 100.6–104.0% of llama**),
> reached via the owned attention tile + buffer-identity fix; the 06-21 "~67% llama / bounded decode RESTED" framing is
> **superseded**. Prefill (eightwave default) **3597/3505/3263/2784/2217 @ctx512/1024/2048/4096/8192**. See
> `../docs/current-project-state-handoff-20260624.md`.

> **Machine-search tooling (live):** `extra/qk_decode_eval.py` (lifecycle evaluator → schema'd verdict),
> `extra/qk_lifecycle_search_loop.py` (generate→evaluate→prune loop), `extra/qk_search_spec.py` (schema authority),
> `extra/qk_nll_eval.py` (decode-path dNLL gate), `extra/qk_demote_search.py` (demotion orchestrator). Measurement
> only; no defaults changed. Backing dirs: `qk-decode-eval/`, `qk-lifecycle-search/`.

## Which harness for decode tok/s — READ FIRST (don't repeat the 2026-06-18 mistake)

Decode tok/s is only trustworthy from a **clean `model.generate`-class path** (device-token feedback, **no
per-step host `Tensor` creation** — that artifact *halves* the rate; `../docs/archive/qk-runtime-overhead-arc-result-20260617.md`).
Pick the harness by what you're measuring:

| want | use | gives | notes |
|---|---|---|---|
| **production tok/s @ ctx≈0** (default headline) | **`-m tinygrad.llm … --warmup --benchmark`** (`tinygrad/llm/cli.py`) | single point | the production default; `model.generate`, clean path |
| **decode tok/s vs ctx** (512/1024/2048/4096) | **`extra/qk_decode_runtime_overhead.py`** (in-model **W==D**) | 101.6/99.8/97.4/92.9 @ 512/1024/2048/4096 (`Q4K_GEMV_WARP` default) | only harness that sweeps ctx on the clean path; W≈D ⇒ GPU-bound |
| flash-decode **policy** (which path is selected, off/on/auto) | `extra/qk_flash_decode_auto_bench.py` | ~54–56 flat | ⚠️ **NOT a tok/s number** — it builds a host input `Tensor` per step inside the timed loop (contaminated by design); use it for selection/correctness only |

> ⚠️ **`87.6` is ambiguous — never quote it bare.** It is BOTH a real ctx≈0 decode **tok/s** (~11.4 ms) AND a real
> ctx4096 decode **ms/token** (=11.4 tok/s); a pre-parity artifact of the old curve. The decode curve is now at/above
> llama parity (101.6→92.9 @ctx512→4096), not the ctx≈0 peak. Prefill policy (`PREFILL_V2=auto`/server) does NOT
> change decode (<1%, identical output).

## Current numbers

| benchmark | value | recorded in | reproduce |
|---|---:|---|---|
| **Decode 8B, default-on** (`Q4K_GEMV_WARP` + owned attention + gqa_coop_vec + flash) | **101.6/99.8/97.4/92.9 @ctx 512/1024/2048/4096** (~100.6–104.0% llama, at/above parity) | `bench/qk-decode-runtime-overhead/result.json`, `../docs/decode-q4k-gemv-warp-promotion-result-20260624.md` | CLI `--warmup --benchmark` (ctx≈0); `extra/qk_decode_runtime_overhead.py` (ctx sweep) |
| **Decode 8B, q8 FFN** (default-off / opt-in, dNLL-gated) | ~+7% decode | `Q8_FFN_HANDWRITTEN=1`; `bench/qk-decode-primitive-transfer/` | `extra/qk_decode_q8_model_route_timing_audit.py` |
| **Prefill 8B, default** (`eightwave` promoted) | **3597/3505/3263/2784/2217 @ctx 512/1024/2048/4096/8192** | `../docs/prefill-eightwave-promotion-result-20260624.md`, `../docs/prefill-baseline-confirmed-aggressive-bound-handoff-20260624.md` | `extra/qk_prefill_emit_search.py` baseline |
| **Prefill POLICY profiles** (opt-in, gated) | default / `PREFILL_V2=auto` (24GB+, VRAM-gated) / `PREFILL_SERVER_PROFILE=1` (warm 0.17–1.6s) | `../docs/archive/prefill-policy-integration-result-20260620.md`, `bench/qk-prefill-policy-integration/*.json` | `extra/qk_prefill_{v2_auto_policy,concrete_kv_policy,route_schedule}_probe.py` |
| **Demotion frontier** (quality-gated weight demotions) | see json | `bench/qk-demote-search/search.json` (+ `accepted-*.json`) | `python -m extra.qk_demote_search --epsilon 0.01` |
| **Decode 14B / 32B** (generated policy) | 40.6 tok/s (62%) / 17.2 tok/s (56%) | `bench/qk-shared-storage-20260612/matrix-summary.md` | harness; see that dir's README |
| _control:_ **Prefill external BLAS ceiling** (standalone fp16 GEMM, not routed) | hipBLASLt 69.8 / rocBLAS 70.9–76.7 TFLOPS | `bench/qk-prefill-external-blas/ceiling.json`, `../docs/archive/prefill-external-blas-result-20260619.md` | see result doc |

## Reproduce — the most-cited

```sh
# Decode 8B, default-on @ctx≈0 (production headline)
DEV=AMD PYTHONPATH=. .venv/bin/python -m tinygrad.llm -m /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --warmup --benchmark 40

# Decode 8B vs ctx (the parity curve) -> 101.6/99.8/97.4/92.9 @ 512/1024/2048/4096, in-model W==D
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_runtime_overhead.py

# Prefill 8B concrete-KV opt-in path (precompile concrete prefill jits at load, then byte-identical faster prefill)
DEV=AMD PREFILL_V2=1 PREFILL_CONCRETE_KV=1 PYTHONPATH=. .venv/bin/python -m tinygrad.llm -m /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --warmup --benchmark 1

# The full demotion search (frontier + accepted policies) -> writes bench/qk-demote-search/
DEV=AMD .venv/bin/python -m extra.qk_demote_search --epsilon 0.01 --bench 24 --tokens 128
```

## Notes on the record

- **Committed (raw artifacts):** `qk-demote-search/`, `qk-shared-storage-20260612/` (8B/14B/32B matrix + policies).
- **Doc-only (artifact gitignored by the prune, regeneratable):** numbers are in the cited docs + committed
  scripts; rerun the command to regenerate.
- **dNLL numbers carry ~±0.01 noise** (128-token calib set) — directionally trustworthy, not precise to 3 decimals.
- Decode tok/s is the **steady-state median** with first ~3 tokens dropped (clock-ramp); the cold first
  token (~100 ms) is not the benchmark.
- **History:** the full decode-attention / prefill-WMMA-Tensile / fused-flash probe arc is archived under
  `../docs/archive/` (verdicts folded into the canonical syntheses above) and the dated `bench/<arc>/` dirs.
```
