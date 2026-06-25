# tinygrad-arkey

A hard fork of [tinygrad](https://github.com/tinygrad/tinygrad). AMD only, focused on quantized LLM decode and prefill on RDNA3. No plans to merge upstream.

## Repo description

tinygrad is a small deep learning framework by George Hotz. He founded comma.ai and is known for the first iPhone jailbreak. tinygrad lowers tensor operations to kernels and uses machine search (BEAM) to find fast implementations across many backends.

I forked it and plan to never merge upstream. (Hi George, if you are reading this.)

I forked tinygrad for three reasons:

* Machine code search.
* Portability over USB.
* Learn the essentials of kernels.

## Benchmarks

Machine: RX 7900 XTX (gfx1100, 24 GB). Model: Qwen3-8B-Q4_K_M. Clean `model.generate` path, W==D.

| benchmark | value |
|---|---|
| Decode, ctx 512 / 1024 / 2048 / 4096 | 101.6 / 99.8 / 97.3 / 92.7 tok/s (100 to 104% of llama.cpp) |
| llama.cpp reference, same ctx | 97.71 / 97.39 / 95.00 / 92.37 tok/s |
| Prefill, ctx 512 / 1024 / 2048 / 4096 / 8192 | 3574 / 3573 / 3572 / 3571 / 3569 tok/s |
| Decode 14B / 32B | 40.6 / 17.2 tok/s |

Decode runs at or above llama.cpp parity on the default stack. Full index and reproduce commands: [bench/README.md](bench/README.md).

## How to use

Requires an AMD GPU (gfx1100) and the model gguf. Run from the repo root with the venv.

```sh
# Decode benchmark (production headline)
DEV=AMD PYTHONPATH=. .venv/bin/python -m tinygrad.llm -m /path/to/Qwen3-8B-Q4_K_M.gguf --warmup --benchmark 40

# Decode vs context (W==D sweep)
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_runtime_overhead.py

# Prefill (concrete-KV opt-in)
DEV=AMD PREFILL_V2=1 PREFILL_CONCRETE_KV=1 PYTHONPATH=. .venv/bin/python -m tinygrad.llm -m /path/to/Qwen3-8B-Q4_K_M.gguf --warmup --benchmark 1
```

Read [bench/README.md](bench/README.md) "Measuring decode tok/s" before quoting numbers. Only a clean `model.generate` path is trustworthy.

## Core scripts

The full active surface is in [FILE_INDEX.md](FILE_INDEX.md). The main ones:

* `tinygrad/llm/` core runtime (CLI, model, gguf loader).
* `extra/qk_decode_runtime_overhead.py` decode context sweep.
* `extra/qk_prefill_emit_search.py` prefill harness.
* `extra/qk_decode_eval.py`, `extra/qk_lifecycle_search_loop.py` machine search.
* `extra/q4_k_gemv_primitive.py`, `extra/q8_ffn_*` quant primitives.
* `extra/qk_clock_pin.py` reproducible clock pinning.
* `extra/qk_policy_consistency_check.py` docs guardrail.

Current state and the doc map: [docs/README.md](docs/README.md) and [docs/current-project-state-handoff-20260624.md](docs/current-project-state-handoff-20260624.md).

## License

MIT, inherited from tinygrad. See [LICENSE](LICENSE).
