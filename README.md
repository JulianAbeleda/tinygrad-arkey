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

Machine: RX 7900 XTX (24 GB), AMD gfx1100. Decode numbers below are authority-style fixed-context W==D or phase-gate numbers, not mixed-context `generate` medians.

### tinygrad

| Model | Quant | Decode ctx512 | Decode ctx4096 | Prefill pp512 | Notes |
|---|---:|---:|---:|---:|---|
| Qwen3-8B | Q4_K_M | 107.6 tok/s | 97.9 tok/s | 4434 tok/s | Current default uses owned HIP attention at long context. Generated attention is 104.0 / 93.3 tok/s and remains default-off. |
| Qwen3-14B | Q4_K_M | 53.8 tok/s | refresh needed | not measured | Current generated G=5 K-only attention route, validated at ctx512/2048. |
| Qwen3-32B | Q4_K_M | refresh needed | refresh needed | not measured | Existing committed table predates the latest route set. |

### llama.cpp reference

| Model | Quant | Decode ctx512 | Decode ctx4096 | Prefill pp512 | Notes |
|---|---:|---:|---:|---:|---|
| Qwen3-8B | Q4_K_M | 98.35 tok/s | 92.4 tok/s | 3000.9 tok/s | Same GGUF / RX 7900 XTX reference from the local benchmark notes. |
| Qwen3-14B | Q4_K_M | 64.92 tok/s | not recorded here | 1633.1 tok/s | Same local llama-bench reference. |
| Qwen3-32B | Q4_K_M | 30.74 tok/s | not recorded here | 722.7 tok/s | Same local llama-bench reference. |

Read these as current working numbers, not a universal claim. The multi-model table in `bench/models/qwen/` is useful provenance but may lag the latest route changes. The current purity state is in `bench/pure-machine-search-default-path-census/summary.md`; the final 8B attention blocker is in `bench/tg-p10-reg-scalar-combine-lowering/summary.md` and `docs/tg-p11-reduce-upcast-accumulator-widening-scope-20260701.md`.

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

## Main files

The full list of active files is in [FILE_INDEX.md](FILE_INDEX.md). The ones to start with:

* `tinygrad/llm/` — the core runtime (command line, model, model-file loader).
* `extra/qk_decode_runtime_overhead.py` — decode speed across context lengths.
* `extra/qk_prefill_whole_synced.py` — prefill speed.
* `extra/pure_machine_search_default_path_census.py` — current generated/default-route census.
* `extra/qk_route_manifest.py` — route manifest, rollback flags, provenance, and refuted axes.
* `extra/qk_flash_decode.py` — generated flash/decode attention routes.
* `extra/qk_gemv_g3_codegen_lowering.py`, `extra/qk_q6k_route_spec.py`, `extra/qk_prefill_schedule_spec.py` — generated route/spec surfaces.
* `extra/qk_tg_p10_reg_scalar_repro.py` — minimal repro for the current final compiler-lowering blocker.

More detail and the documentation map: [docs/README.md](docs/README.md).

## License

MIT, inherited from tinygrad. See [LICENSE](LICENSE).
