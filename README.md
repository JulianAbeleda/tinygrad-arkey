# tinygrad-arkey

`tinygrad-arkey` is a lean tinygrad runtime with machine-searched AMD LLM routes. Master contains the promoted
runtime, compact selected artifacts, ordinary tinygrad fallback, and focused verification tests. Search campaigns,
handwritten qualification oracles, hardware recovery tooling, and historical evidence live on `dev` and `exp`.

## What “machine searched” means

BoltBeam plus BubbleBeam/FutureSight explored, measured, ranked, and selected the performance-sensitive route
configurations shipped here. Humans still define search spaces, compiler primitives, correctness gates, and promotion
policy. `Tensor.uop_program` is the lazy transport for a selected UOp graph; it does not by itself make a route
handwritten.

The production boundary is simple:

- exact admitted shapes use a promoted searched route;
- unsupported, disabled, or declined routes use ordinary tinygrad graph execution;
- master does not contain the handwritten direct-packed Q4/Q6 fallback kernels or research oracles;
- production code under `tinygrad/llm` does not import from `extra`.

## Install

```sh
git clone https://github.com/JulianAbeleda/tinygrad-arkey.git
cd tinygrad-arkey
uv sync
```

You need a working tinygrad AMD backend and a local GGUF for the accelerated path. Current selected routes target the
validated gfx1100 geometries and fail closed when hardware, model, memory, or artifact facts do not match.

## Run the searched hot path

```sh
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python -m tinygrad.llm \
  --model /absolute/path/to/Qwen3-8B-Q4_K_M.gguf --max_context 8192 \
  --warmup --benchmark-context 512 --benchmark 20

DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python -m tinygrad.llm \
  --model /absolute/path/to/Qwen3-8B-Q4_K_M.gguf --max_context 8192 \
  --warmup --benchmark-context 4096 --benchmark 20
```

The command prints full-prompt prefill throughput, selected candidate identities, and steady decode samples. An empty
selected-identity list is a generic fallback observation and must not be reported as searched-path performance.

For a CPU-only metadata/control check:

```sh
PYTHONPATH=. .venv/bin/python -m tinygrad.llm.bench --metadata-only --target CPU
```

See [`tinygrad/llm/README.md`](tinygrad/llm/README.md) for route interpretation, the structured control surface, and
the retained pre-migration comparison points.

## Runtime map

- `tinygrad/llm/model.py` — model construction, cache allocation, and generation.
- `tinygrad/llm/decode_routes.py` — decode admission and production route selection.
- `tinygrad/llm/decode_kernels.py` — searched Q4 G3 and Q6 decode lowerings.
- `tinygrad/llm/flash_decode_attention.py` — promoted G4/G5 live-split attention.
- `tinygrad/llm/prefill_graph_gemm.py` — searched WMMA-LDS graph-prefill executor.
- `tinygrad/llm/packed_wmma_prefill.py` — six promoted packed-WMMA configurations.
- `tinygrad/llm/prefill_candidate_runtime.py` — compact candidate-set decoding and admission.
- `tinygrad/llm/generated/` — runtime-required generated selection artifacts.

## Branch roles

- `master`: runnable promoted product surface; no handwritten specialized fallback or research archive.
- `dev`: master plus qualification oracles, development tooling, and durable handoffs.
- `exp`: active machine-search and hardware experimentation.

## Reference performance

The retained pre-migration comparison points on RX 7900 XTX/gfx1100 are:

| Model | Decode ctx512 / ctx4096 | Prefill pp512 / pp4096 |
|---|---:|---:|
| Qwen3-8B Q4_K_M | 114.19 / 103.07 tok/s | 3694 / 3236 tok/s |
| Qwen3-14B Q4_K_M | 69.70 / 62.45 tok/s | 1945 / 1785 tok/s |

These are historical comparison points, not claims about a new checkout. Publish new numbers only with the exact
commit, model hash, target, selected identities, warmups, sample count, and clean device-health record.

## License

MIT, inherited from tinygrad. See [LICENSE](LICENSE).
