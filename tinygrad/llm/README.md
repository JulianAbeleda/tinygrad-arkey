# tinygrad.llm Surface

This package keeps model execution separate from load-time policy.

## Public generic control path

From a fresh clone, use a local GGUF with the ordinary tinygrad control path:

```bash
git clone <your-tinygrad-remote> tinygrad
cd tinygrad
python -m tinygrad.llm.bench --help
python -m tinygrad.llm.bench --execute --model /absolute/path/to/model.gguf --phase prefill --context 32 > control-record.json
```

This pins model load and one public fp16 linear-fallback dispatch to **CPU** and writes a
structured JSON trace. The command imports only `tinygrad.llm` runtime modules;
it does not import a research route or accept a custom schedule. It neither
compares an output to a reference nor reports timing/throughput, so its result
is a runnable control observation—not a performance benchmark or M8 evidence.

For a no-load provenance record instead:

```bash
python -m tinygrad.llm.bench --metadata-only --model /absolute/path/to/model.gguf > control-record.json
```

Consumers must keep `authority.throughput_authoritative == false`. M8 remains
unproven until a separately verified run supplies correctness, timing, and
artifact bindings.

## AMD searched-hot-path benchmark

The public runtime can exercise the shipped searched path without importing `extra/`.
Use a local Qwen3 GGUF and the same context checkpoints as the published runs:

```bash
DEV=AMD JIT=1 PYTHONPATH=. python -m tinygrad.llm \
  --model /absolute/path/to/Qwen3-8B-Q4_K_M.gguf --max_context 8192 \
  --warmup --benchmark-context 512 --benchmark 20

DEV=AMD JIT=1 PYTHONPATH=. python -m tinygrad.llm \
  --model /absolute/path/to/Qwen3-8B-Q4_K_M.gguf --max_context 8192 \
  --warmup --benchmark-context 4096 --benchmark 20
```

On the macOS eGPU service, use the already-qualified environment prefix for that host
(including its `AM_REMOTE_DISCOVERY_PROFILE` and `AM_REMOTE_SKIP_RESIZE_BAR` values).
The command prints the full-prompt prefill rate, selected candidate identities, and one
steady decode rate per generated token. A missing searched match is visible as an empty
identity list and must not be reported as a searched-path benchmark.

The retained pre-migration reference points are 8B decode 114.19 tok/s at ctx512 and
103.07 tok/s at ctx4096; prefill was 3694 and 3236 tok/s at pp512/pp4096. The 14B
reference points are 69.70/62.45 decode and 1945/1785 prefill. These are comparison
points, not claims about a new checkout: report new numbers only with the exact commit,
model hash, target, selected identities, warmups, and sample count.

## Runtime files

- `model.py`: transformer blocks, model construction, cache allocation, and generation. It may call policy helpers, but should not grow new standalone admission or registry logic.
- `decode_routes.py`: runtime decode selection using `decode_kernels.py` and `flash_decode_attention.py`.
- `route_policy.py`: policy-file parsing and validation for generated/QK routes.
- `prefill_graph_gemm.py` and `packed_wmma_prefill.py`: searched prefill executors and frozen selected configurations.
- `prefill_candidate_runtime.py`: compact candidate-set admission for the graph-WMMA route.
- `qk_primitives.py`: Q4_K/Q6_K primitive wrappers, install-time storage policy, and GGUF-backed primitive installation.

## Control-plane files

- `admission.py`: VRAM probes and max-context/KV-tier admission arithmetic. Keep it pure and unit-testable.
- `prefill_policy.py`: pure prefill policy decisions and validation helpers. Runtime flags stay in `model.py`.
- `gguf.py`: GGUF parsing/loading. Header-only metadata readers belong here, not in `model.py`.
- `cli.py`: CLI/server wiring only. It should not own model policy arithmetic.

When adding a new feature, prefer placing the policy in a small module and threading the resolved result into `TransformerConfig`. `model.py` should consume resolved decisions, not become the source of every decision.
