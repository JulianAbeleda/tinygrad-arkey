# tinygrad-arkey

A hard fork of [tinygrad](https://github.com/tinygrad/tinygrad). AMD-only, focused on running quantized LLMs fast on RDNA3 GPUs. No plans to merge upstream.

## What this is

tinygrad is a small deep learning framework by George Hotz (founder of comma.ai). It turns tensor operations into GPU kernels across many backends.

This fork narrows it to one job: fast inference of quantized large language models on a single AMD GPU. I forked it to explore three things:

* Automatically searching for fast GPU kernels.
* Running over USB.
* Learning how GPU kernels actually work.

The main idea is the kernel search. Instead of hand-writing every fast kernel, we let a search try different implementations and ship the fastest one that still produces correct output. The search picks which kernel, which code path, and which settings to use, and only promotes a candidate when it is both correct and measurably faster. Today several of the hottest decode and prefill kernels are search-generated and match or beat the hand-tuned versions they replaced.

## How it differs from upstream tinygrad

* **Hardware:** AMD only (RDNA3 / gfx1100). Upstream supports AMD, NVIDIA, Apple, CPU, WebGPU, and more.
* **Scope:** quantized LLM decode and prefill only. Upstream is a general deep learning framework.
* **Kernels:** generated automatically where possible, with a few hand-tuned hot paths and search-found replacements that were measured against them.
* **Search:** our own kernel/configuration search, gated by correctness and speed. We do not use tinygrad's built-in autotuner.
* **Size:** trimmed to the parts we actually run — example apps, the upstream test suite, and unused subsystems were removed (~150k lines). The active files are listed in `FILE_INDEX.md`.

## Benchmarks

Machine: RX 7900 XTX (24 GB). Model: Qwen3-8B-Q4_K_M. Measured on a clean generation path.

| Benchmark | Result |
|---|---|
| Decode (tokens/s) — context 512 / 1024 / 2048 / 4096 | 103.9 / 102.0 / 99.7 / 94.4 |
| llama.cpp, same model and contexts | 97.7 / 97.4 / 95.0 / 92.4 |
| Prefill (tokens/s) — context 512 / 1024 / 2048 / 4096 / 8192 | 4434 / 4236 / 3846 / 3192 / 2532 |
| Decode, larger models (14B / 32B) | 40.6 / 17.2 |

Decode runs at or above llama.cpp on this machine, and prefill uses an optimized matrix-multiply path found by the search. Full numbers and commands to reproduce them: [bench/README.md](bench/README.md).

## Running it

You need an AMD GPU (gfx1100) and a model file (`.gguf`). Run from the repo root using the project's virtual environment.

```sh
# Decode benchmark
DEV=AMD PYTHONPATH=. .venv/bin/python -m tinygrad.llm -m /path/to/Qwen3-8B-Q4_K_M.gguf --warmup --benchmark 40

# Decode speed across context lengths
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_runtime_overhead.py

# Prefill speed
DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_prefill_whole_synced.py
```

Before quoting any decode number, read the "Measuring decode tokens/s" section in [bench/README.md](bench/README.md) — only a clean generation path gives trustworthy results.

## Main files

The full list of active files is in [FILE_INDEX.md](FILE_INDEX.md). The ones to start with:

* `tinygrad/llm/` — the core runtime (command line, model, model-file loader).
* `extra/qk_decode_runtime_overhead.py` — decode speed across context lengths.
* `extra/qk_prefill_whole_synced.py` — prefill speed.
* `extra/qk_decode_eval.py`, `extra/qk_lifecycle_search_loop.py` — the kernel search.
* `extra/q4_k_gemv_primitive.py` — quantized matrix-vector kernels.

More detail and the documentation map: [docs/README.md](docs/README.md).

## License

MIT, inherited from tinygrad. See [LICENSE](LICENSE).
