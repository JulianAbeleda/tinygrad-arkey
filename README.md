# tinygrad-arkey

A hard fork of [tinygrad](https://github.com/tinygrad/tinygrad). This branch is AMD/RDNA3 focused and exists to run quantized LLM inference while pushing as much of the hot path as possible toward generated code and measured route selection. It is not intended to merge upstream.

## What this is

Upstream tinygrad is a small deep learning framework by George Hotz that lowers tensor programs to kernels across many backends. This fork keeps that foundation but narrows the active target: local GGUF LLM inference on a single AMD GPU, especially Qwen-class quantized models on gfx1100.

The project started as kernel search. The current shape is more specific: BoltBeam audits profiles, candidates, evidence, and route policy; this repo carries the runtime, generated kernels, fallback oracles, and the gates that prove a route is correct before it can become default. A route is not promoted because it is elegant or generated. It is promoted only when it is route-bound, token/logit equivalent, and fast enough under the authority harness.

Today the default path is mostly generated. Four of the five hot routes are machine-authored/generated defaults: Q4_K G3 decode GEMV, Q6_K generated coop decode, the generated G=5 K-only attention tile for the validated 14B shape, and the generated role-selective prefill schedule. The remaining final purity debt is 8B long-context decode attention, which still defaults to the owned HIP two-kernel route because the generated replacement is correct but not yet fast enough. The strict census therefore still reports `TINYGRAD_DEFAULT_PURITY_FAIL`, not because the plumbing is missing, but because the final generated attention combine is blocked on a core reduce/upcast accumulator-lowering invariant.

## How it differs from upstream tinygrad

This fork is narrower than upstream. It is optimized around AMD gfx1100, quantized GGUF inference, decode and prefill measurement, and route-level search. Upstream supports many backends and workloads; this repo keeps the pieces needed for this line of work and removes or ignores broad framework surface area that is not part of the current runtime.

The hot path is also more explicit than upstream. Some kernels are ordinary tinygrad-generated kernels. Some are generated from route specs that BoltBeam can reason about. A few owned or legacy kernels remain only as rollback oracles. The project does not treat a handwritten route as a final answer unless it is still the measured best available default.

The codebase is deliberately instrumented. There are authority harnesses for decode, prefill, route attribution, purity census, compiler-lowering repros, and regression gates. The active-file map is in `FILE_INDEX.md`; the search and route state is captured in `extra/qk_route_manifest.py`, `bench/pure-machine-search-default-path-census/summary.md`, and the phase docs under `docs/`.

## Current performance state

Benchmarks in this repo are easy to misread because there are several harnesses: end-to-end `generate`, fixed-context W==D decode, prefill authority, route microgates, and diagnostic one-off traces. The old README table mixed stale numbers from different stages, so it has been removed. Use the authority artifacts below when quoting numbers.

On RX 7900 XTX / gfx1100, Qwen3-8B-Q4_K_M is at or above llama.cpp in the long-context flash-decode regime. The current 8B owned/default attention authority in `bench/tg-p9-pure-attention-primitive-route/summary.md` records owned/default W==D at 107.6 tok/s for ctx512 and 97.9 tok/s for ctx4096. The generated live-split attention candidate is close, but still not the default: it reaches 104.0 tok/s at ctx512 and 93.3 tok/s at ctx4096, or 96.7% and 95.3% of owned. That is below the 98% promotion bar, so the owned HIP attention route remains default for 8B.

For Qwen3-14B-Q4_K_M, the latest shipped generated attention improvement is the G=5 K-only block tile. `bench/gp-track/gp4_latest.json` records 53.8 tok/s at ctx512 and ctx2048 with the generated K-only staging path, up from 49.9 and 46.9 in that gate. That route is default-on for the validated G=5 shape and rolls back with `DECODE_FLASH_BLOCK_TILE_G5=0`.

For Qwen3-32B-Q4_K_M and Qwen3.5-27B, treat the committed model table under `bench/models/qwen/` as historical until it is refreshed with the latest route set. The 27B model is a different hybrid architecture, not a like-for-like dense Qwen3 comparison. The 32B path has had route work, but its current table predates the latest attention/route changes and should not be used as a headline number without a rerun.

Prefill uses the generated role-selective schedule by default. `bench/qk-prefill-pipe-role-selective/summary.md` records the role-selective path at 4434 / 4236 / 3846 / 3192 / 2532 tok/s for ctx512 / 1024 / 2048 / 4096 / 8192 on the 8B authority run, beating the prior global pipe and old LDS defaults. `bench/tg-p4-prefill-generated-schedule/summary.md` proves the emitted schedule is instruction-identical to the selected legacy schedule while moving the route into the generated/spec path.

The current north-star blocker is not benchmark noise. TG-P10 reduced the last purity blocker to one compiler invariant: reduce accumulators must widen over output upcast lanes they vary along, while staying scalar for invariant axes. See `bench/tg-p10-reg-scalar-combine-lowering/summary.md` and `docs/tg-p11-reduce-upcast-accumulator-widening-scope-20260701.md`.

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
