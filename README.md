# tinygrad-arkey

A hard fork of [tinygrad](https://github.com/tinygrad/tinygrad). AMD only, focused on quantized LLM decode and prefill on RDNA3. No plans to merge upstream.

## Repo description

tinygrad is a small deep learning framework by George Hotz. He founded comma.ai and is known for the first iPhone jailbreak. tinygrad lowers tensor operations to GPU kernels across many backends.

I forked it and plan to never merge upstream. (Hi George, if you are reading this.)

I forked tinygrad for three reasons:

* Machine code search.
* Portability over USB.
* Learn the essentials of kernels.

We do not use tinygrad's BEAM autotuner. We built our own machine search. tinygrad's BEAM searches per-kernel schedules; ours searches which primitive, route, and flag config to ship, gated by correctness and throughput. Decode now uses a search-generated G3 LaneMap route for the major Q4_K GEMV roles, speed-equivalent to the old owned warp route; Q6_K direct routing was tested and refuted. Prefill now defaults to the validated `pipe_tm2_tn2` graph-GEMM route. See `extra/qk_decode_eval.py`, `extra/qk_lifecycle_search_loop.py`, and the current docs map.

## Differences from upstream tinygrad

* Backends: AMD only (gfx1100 / RDNA3). Upstream targets AMD, NVIDIA, Metal, CPU, WebGPU, and more.
* Scope: quantized LLM decode and prefill (Q4_K / Q6_K / q8). Upstream is a general deep learning framework.
* Kernels: scheduler-generated where possible, with retained hot-path overrides and measured search-generated replacements. Q4_K decode GEMV now routes through generated G3 LaneMap under BubbleBeam/FutureSight; the owned attention tile remains the decode-attention baseline. See [docs/README.md](docs/README.md). Upstream generates all kernels from the scheduler.
* Search: our own candidate and lifecycle search over primitives and flags. We run with tinygrad's BEAM autotuner off.
* Size: stripped to the live core. Removed the example apps, the upstream test suite, and unused subsystems (about 150k LOC). Active surface is in `FILE_INDEX.md`.
* Direction: hard fork, no plans to merge upstream.

## Benchmarks

Machine: RX 7900 XTX (gfx1100, 24 GB). Model: Qwen3-8B-Q4_K_M. Clean `model.generate` path, W==D.

| benchmark | value |
|---|---|
| Decode, ctx 512 / 1024 / 2048 / 4096 | 103.9 / 102.0 / 99.7 / 94.4 tok/s (G3 speed-equivalent to owned Q4_K; Q6_K direct refuted/default-off) |
| llama.cpp reference, same ctx | 97.71 / 97.39 / 95.00 / 92.37 tok/s |
| Prefill, ctx 512 / 1024 / 2048 / 4096 / 8192 | 4291 / 4089 / 3711 / 3137 / 2423 tok/s (`pipe_tm2_tn2` default; rollback `PREFILL_GEMM_PIPELINE=0`) |
| Decode 14B / 32B | 40.6 / 17.2 tok/s |

Decode runs at or above llama.cpp parity on the default stack, and prefill has a validated TIER_A graph-GEMM route. Full index and reproduce commands: [bench/README.md](bench/README.md).

## How to use

Requires an AMD GPU (gfx1100) and the model gguf. Run from the repo root with the venv.

```sh
# Decode benchmark (production headline)
DEV=AMD PYTHONPATH=. .venv/bin/python -m tinygrad.llm -m /path/to/Qwen3-8B-Q4_K_M.gguf --warmup --benchmark 40

# Decode vs context (W==D sweep)
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_runtime_overhead.py

# Prefill default graph-GEMM route
DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_prefill_whole_synced.py

# Roll back the promoted pipe route for A/B
DEV=AMD PREFILL_GEMM_PIPELINE=0 PYTHONPATH=. .venv/bin/python extra/qk_prefill_whole_synced.py
```

Read [bench/README.md](bench/README.md) "Measuring decode tok/s" before quoting numbers. Only a clean `model.generate` path is trustworthy.

## Core scripts

The full active surface is in [FILE_INDEX.md](FILE_INDEX.md). The main ones:

* `tinygrad/llm/` core runtime (CLI, model, gguf loader).
* `extra/qk_decode_runtime_overhead.py` decode context sweep.
* `extra/qk_prefill_whole_synced.py`, `extra/qk_prefill_emit_search.py` prefill harnesses.
* `extra/qk_decode_eval.py`, `extra/qk_lifecycle_search_loop.py` machine search.
* `extra/q4_k_gemv_primitive.py`, `extra/q8_ffn_*` quant primitives.
* `extra/qk_clock_pin.py` reproducible clock pinning.
* `extra/qk_policy_consistency_check.py` docs guardrail.

Current state and the doc map: [docs/README.md](docs/README.md) and [docs/current-project-state-handoff-20260624.md](docs/current-project-state-handoff-20260624.md).

## License

MIT, inherited from tinygrad. See [LICENSE](LICENSE).
