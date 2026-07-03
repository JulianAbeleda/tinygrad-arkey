# tinygrad-arkey

A hard fork of [tinygrad](https://github.com/tinygrad/tinygrad). AMD/RDNA3 focused, aimed at fast local GGUF LLM inference and at moving the hot path toward generated code. No plans to merge upstream.

## What this is

tinygrad is a small deep learning framework by George Hotz that lowers tensor operations to GPU kernels. This fork narrows it to one job: quantized LLM inference on AMD gfx1100, with the search/audit work split between this runtime and [BoltBeam](https://github.com/JulianAbeleda/BoltBeam).

The main idea is still kernel search, but the bar is strict: a route only becomes default after it is correct, route-bound, rollback-safe, and fast enough under the authority harness. Four of the five hot default routes are now machine-authored/generated. The remaining purity debt is 8B long-context decode attention, which still uses the owned HIP two-kernel route because the generated route is close but not yet fast enough.

## How it differs from upstream tinygrad

* **Hardware:** AMD only, currently gfx1100 / RX 7900 XTX.
* **Scope:** quantized GGUF LLM decode and prefill.
* **Search:** BoltBeam audits candidates and route policy; tinygrad runs the promoted routes.
* **Defaults:** generated where proven; owned kernels stay as rollback/oracle when still fastest.
* **Instrumentation:** decode, prefill, route attribution, purity census, and compiler-lowering gates are first-class.

## Current performance state

Machine: RX 7900 XTX (24 GB), AMD gfx1100. Measured 2026-07-03 on `master` (all shipping defaults incl. the attn_v route fix). Decode = median steady-state W==D via `extra/model_e2e_bench.py` (the `generate` harness), auto clock; both runtimes measured the same way, same GGUFs, on the same box.

**tinygrad now leads llama.cpp on decode across all three models at both contexts** — a few percent at ctx512, widening at ctx4096 (tinygrad decode is context-robust; llama's tg falls off with KV depth). The recent lift on 14B/8B/32B comes from the attn_v route-miss fix (`DECODE_ROUTE_ATTN_V`, +8.9% 8B / +13% 14B, byte-identical).

### tinygrad

| Model | Quant | Decode ctx512 | Decode ctx4096 | Notes |
|---|---:|---:|---:|---|
| Qwen3-8B | Q4_K_M | 103.9 tok/s | 107.9 tok/s | +3.9% / +19.4% vs llama. Owned HIP attention default at long ctx. |
| Qwen3-14B | Q4_K_M | 66.5 tok/s | 68.2 tok/s | +1.4% / +24.9% vs llama. Generated G=5 K-only attention route + attn_v fix. |
| Qwen3-32B | Q4_K_M | 31.9 tok/s | 32.6 tok/s | +2.3% / +9.8% vs llama. Fits 24 GB (20.9/24.2 GB at ctx512/4096). |

### llama.cpp reference

| Model | Quant | Decode ctx512 | Decode ctx4096 | Prefill pp512 | Notes |
|---|---:|---:|---:|---:|---|
| Qwen3-8B | Q4_K_M | 100.0 tok/s | 90.4 tok/s | 3107 tok/s | Same GGUF / RX 7900 XTX via local llama-bench (tg128). |
| Qwen3-14B | Q4_K_M | 65.6 tok/s | 54.6 tok/s | 1702 tok/s | Same GGUF / RX 7900 XTX via local llama-bench (tg128). |
| Qwen3-32B | Q4_K_M | 31.2 tok/s | 29.7 tok/s | 757 tok/s | Same GGUF / RX 7900 XTX via local llama-bench (tg128). |

Read these as current working numbers, not a universal claim. **Decode is the tinygrad win; prefill is not** — llama's batched prefill (pp512 above, ~750–3100 tok/s) is far ahead; tinygrad's story here is decode/HBM-bound throughput, and prefill is a known compute-bound gap (the `model_e2e_bench` prefill metric is ttft-derived and includes JIT compile, so it is omitted here rather than reported misleadingly). The multi-model table in `bench/models/qwen/` is useful provenance but may lag the latest route changes.

## Running it

You need an AMD GPU, the AMD backend working, and a GGUF model file. Most current gates assume gfx1100 / RX 7900 XTX. Run from the repo root using the project's virtual environment.

```sh
# Decode benchmark
DEV=AMD PYTHONPATH=. .venv/bin/python -m tinygrad.llm -m /path/to/Qwen3-8B-Q4_K_M.gguf --warmup --benchmark 40

# Decode speed across context lengths
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_runtime_overhead.py

# Prefill speed
DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_prefill_whole_synced.py
```

Before quoting any decode number, use the fixed-context W==D authority harness or a phase artifact that states its methodology. End-to-end `generate` numbers are useful for user feel, but they mix context growth, host behavior, sampling, and warmup. For current route/purity state, start with:

```sh
PYTHONPATH=. .venv/bin/python extra/pure_machine_search_default_path_census.py --check
PYTHONPATH=. .venv/bin/python extra/pure_machine_search_default_path_census.py --strict-final-default
```

## Main Files

Start with these files and the documentation map in [docs/README.md](docs/README.md):

* `tinygrad/llm/` — the core runtime (command line, model, model-file loader).
* `extra/qk_decode_runtime_overhead.py` — decode speed across context lengths.
* `extra/qk_prefill_whole_synced.py` — prefill speed.
* `extra/pure_machine_search_default_path_census.py` — current generated/default-route census.
* `extra/qk_route_manifest.py` — runtime-facing route manifest, rollback flags, provenance, and refuted axes. BoltBeam owns the policy/search copy.
* `extra/qk_flash_decode.py` — generated flash/decode attention routes.
* `extra/qk_gemv_g3_codegen_lowering.py`, `extra/qk_q6k_route_spec.py`, `extra/qk_prefill_schedule_spec.py` — generated route/spec surfaces.
* `extra/qk_tg_p10_reg_scalar_repro.py` — minimal repro for the current final compiler-lowering blocker.

BoltBeam owns model facts, candidate/search schema, evaluation policy, ledgers, roofline attribution, and reports. tinygrad owns runtime execution, compiler/backend lowering, and hardware gates.

## License

MIT, inherited from tinygrad. See [LICENSE](LICENSE).
