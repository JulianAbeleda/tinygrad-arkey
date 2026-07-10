# tinygrad-arkey

A hard fork of [tinygrad](https://github.com/tinygrad/tinygrad). AMD/RDNA3 focused, aimed at fast local GGUF LLM inference and at moving the hot path toward generated code. No plans to merge upstream.

## What this is

tinygrad is a small deep learning framework by George Hotz that lowers tensor operations to GPU kernels. This fork narrows it to one job: quantized LLM inference on AMD gfx1100, with the search/audit work split between this runtime and [BoltBeam](https://github.com/JulianAbeleda/BoltBeam).

The main idea is still kernel search, but the bar is strict: a route only becomes default after it is correct, route-bound, rollback-safe, and fast enough under the authority harness. The hot default decode/prefill routes are now generated or spec-driven; owned kernels are rollback/reference material, not the target path.

## How it differs from upstream tinygrad

* **Hardware:** AMD only, currently gfx1100 / RX 7900 XTX.
* **Scope:** quantized GGUF LLM decode and prefill.
* **Search:** BoltBeam audits candidates and route policy; tinygrad runs the promoted routes.
* **Defaults:** generated where proven; owned kernels stay as rollback/reference only.
* **Instrumentation:** decode, prefill, route attribution, purity census, and compiler-lowering gates are first-class.

## Current performance state

Machine: RX 7900 XTX (24 GB), AMD gfx1100. Measured 2026-07-03 on `master` (all shipping defaults incl. the attn_v route fix). Decode = median steady-state W==D via `extra/llm/model_e2e_bench.py` (the retained `generate` artifact harness), auto clock; both runtimes measured the same way, same GGUFs, on the same box. New throughput claims should use the canonical authority entry below (`extra/qk/bench.py`); migrating this historical table to fixed-context authority artifacts is a separate methodology update.

**tinygrad now leads llama.cpp on decode across all three models at both contexts** — a few percent at ctx512, widening at ctx4096 (tinygrad decode is context-robust; llama's tg falls off with KV depth). The recent lift on 14B/8B/32B comes from the attn_v route-miss fix (`DECODE_ROUTE_ATTN_V`, +8.9% 8B / +13% 14B, byte-identical).

### tinygrad

| Model | Quant | Decode ctx512 | Decode ctx4096 | Notes |
|---|---:|---:|---:|---|
| Qwen3-8B | Q4_K_M | 103.9 tok/s | 107.9 tok/s | +3.9% / +19.4% vs llama. Generated live-split/KV_BOTH attention default at long ctx. |
| Qwen3-14B | Q4_K_M | 66.5 tok/s | 68.2 tok/s | +1.4% / +24.9% vs llama. Generated G=5 K-only attention route + attn_v fix. |
| Qwen3-32B | Q4_K_M | 31.9 tok/s | 32.6 tok/s | +2.3% / +9.8% vs llama. Fits 24 GB (20.9/24.2 GB at ctx512/4096). |

### llama.cpp reference

| Model | Quant | Decode ctx512 | Decode ctx4096 | Prefill pp512 | Notes |
|---|---:|---:|---:|---:|---|
| Qwen3-8B | Q4_K_M | 100.0 tok/s | 90.4 tok/s | 3107 tok/s | Same GGUF / RX 7900 XTX via local llama-bench (tg128). |
| Qwen3-14B | Q4_K_M | 65.6 tok/s | 54.6 tok/s | 1702 tok/s | Same GGUF / RX 7900 XTX via local llama-bench (tg128). |
| Qwen3-32B | Q4_K_M | 31.2 tok/s | 29.7 tok/s | 757 tok/s | Same GGUF / RX 7900 XTX via local llama-bench (tg128). |

Read these as current working numbers, not a universal claim. **Decode is a tinygrad win at all sizes; prefill depends on the path:**

- **Decode** — tinygrad leads llama across 8B/14B/32B, widening at long context (HBM-bound; that's the fork's headline).
- **Prefill (`PREFILL_V2` tuned graph-GEMM path)** — on **8B this EXCEEDS llama: ~4408 tok/s pp512 vs llama ~3050 (~145%)**, measured by the synced authority harness (`extra/qk/prefill_whole_synced.py`). BUT this path realizes the covered linears in fp16 (~2× params VRAM), so it **only fits 8B on 24 GB**; and it is **off by default** — the *default* prefill path is the slow universal one.
- **Prefill current state** — the canonical prefill phase ledger and current route/number live in [docs/prefill-current-state.md](docs/prefill-current-state.md) (active phase **hybrid_machine_search**, route `prefill_pipe_role_selective_generated`, pinned pp512 ~4413).
- **Prefill on 14B/32B** — the tuned path doesn't fit (fp16 overlay ~28 GB / ~64 GB), so today they fall back to the slow universal path. Closing that is a real project (a fast-prefill path that doesn't materialize the whole model in fp16).

**Measurement discipline:** report prefill/decode throughput ONLY from the authority harnesses via `extra/qk/bench.py` (below). **Never** report throughput from a `model.generate` TTFT bench — TTFT folds in generate's Python overhead + sampling + host jitter and **understates prefill by ~3×** (a hand-rolled ttft harness read 1247 tok/s for 8B where the synced authority reads ~4408).

## Running it

You need an AMD GPU, the AMD backend working, and a GGUF model file. Most current gates assume gfx1100 / RX 7900 XTX. Run from the repo root using the project's virtual environment.

`--max_context` defaults to `auto`: at load it probes free VRAM (`rocm-smi`) and admits the largest safe context for the model (weights + KV + prefill-score peak + flash scratch, held under an 0.8 fragmentation margin), capped at the model's trained context. Admission is a **tier ladder**, all driven by the memory arithmetic (no model-name checks):

1. **fp16 KV (lossless)** — used whenever it admits a useful context. 8B/14B admit their full trained context on a 24 GB card this way.
2. **int8 KV-quant (`DECODE_KV_QUANT`, ~0.6% loss)** — auto-escalated when fp16 can't fit but int8 can. The resident KV cache is int8 + a tiny per-(K/V,head,token) fp16 scale; the decode flash route dequantizes in-register (no materialized fp16 KV). This is what lets **32B run long context** (~2800 tokens where fp16 admits <2000), token-identical to fp16 in practice. It's a *capacity* lever, not a decode speedup (attention decode is compute-bound, not bandwidth-bound).
3. **refuse loud** — only when even int8 can't fit a useful context; the message points at the Q4-KV / eviction tiers (follow-ons).

An explicit `--max_context N` is still admission-checked and auto-upgrades through the ladder to honor the request, failing loud only if no tier fits. All of this relies on the seqlen-bound decode attention route (decode work scales with live context, not `max_context`), so raising the cap does not collapse decode.

```sh
# THE benchmark — the single canonical entry. Dispatches to the synced authority harnesses (prefill + decode),
# each in an isolated subprocess with the correct env. This is the ONLY sanctioned way to report throughput.
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk/bench.py --model /path/to/Qwen3-8B-Q4_K_M.gguf
#   --prefill  : prefill authority only (extra/qk/prefill_whole_synced.py, synced graph-GEMM pp@L)
#   --decode   : decode authority only (extra/qk/decode_runtime_overhead.py, W==D synced min-of-K)
```

**Do not roll your own throughput harness, and never report a `model.generate` TTFT number** — TTFT understates prefill by ~3× (it folds in generate's Python overhead + sampling + host jitter). If you need a route/purity check instead of a number, start with:

```sh
PYTHONPATH=. .venv/bin/python extra/audit/pure_machine_search_default_path_census.py --check
PYTHONPATH=. .venv/bin/python extra/audit/pure_machine_search_default_path_census.py --strict-final-default
```

## Main Files

Start with these files and the documentation map in [docs/README.md](docs/README.md):

* `tinygrad/llm/` — the core runtime (command line, model, model-file loader).
* `extra/qk/decode_runtime_overhead.py` — decode speed across context lengths.
* `extra/qk/prefill_whole_synced.py` — prefill speed.
* `extra/audit/pure_machine_search_default_path_census.py` — current generated/default-route census.
* `extra/qk/route_manifest.py` — runtime-facing route manifest, rollback flags, provenance, and refuted axes. BoltBeam owns the policy/search copy.
* `extra/qk/flash_decode.py` — generated flash/decode attention routes.
* `extra/qk/gemv_g3_codegen_lowering.py`, `extra/qk/q6k_route_spec.py`, `extra/qk/prefill_schedule_spec.py` — generated route/spec surfaces.
* `extra/qk/tg_p10_reg_scalar_repro.py` — minimal repro for the current final compiler-lowering blocker.

BoltBeam owns model facts, candidate/search schema, evaluation policy, ledgers, roofline attribution, and reports. tinygrad owns runtime execution, compiler/backend lowering, and hardware gates.

## License

MIT, inherited from tinygrad. See [LICENSE](LICENSE).
