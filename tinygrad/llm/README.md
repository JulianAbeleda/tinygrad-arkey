# tinygrad.llm Surface

This package keeps model execution separate from load-time policy.

## Benchmark record scaffold

From a fresh clone, create the normal development environment, then emit a
metadata-only benchmark record (this does **not** load a model or claim a
throughput result):

```bash
git clone <your-tinygrad-remote> tinygrad
cd tinygrad
python -m tinygrad.llm.bench --help
python -m tinygrad.llm.bench --metadata-only --model /absolute/path/to/model.gguf --route-id decode > bench-record.json
```

`bench-record.json` is versioned and includes the Git commit/dirty state,
model path and SHA-256, device/driver probe facts, exact command/config,
route traces, correctness status, requested warmups/samples, and throughput.
At present its route traces are deliberately `unproven`, correctness is
`not_run`, and throughput is `null`: current performance numbers remain
unverified until the exact generated plans and artifacts land. Consumers must
check `authority.throughput_authoritative`; this scaffold always sets it to
`false` rather than silently presenting metadata as a benchmark. M8 is not
complete until this command executes the verified generated route, checks
correctness, collects samples, and binds a non-null result to its artifact
digests.

## Runtime files

- `model.py`: transformer blocks, model construction, cache allocation, and generation. It may call policy helpers, but should not grow new standalone admission or registry logic.
- `decode_routes.py`: runtime decode route selection. Route-specific generated kernels live under `extra/llm_research` and are imported through `route_ops.py`.
- `route_policy.py`: policy-file parsing and validation for generated/QK routes.
- `route_ops.py`: import boundary for generated/search-produced primitives.
- `qk_primitives.py`: Q4_K/Q6_K primitive wrappers, install-time storage policy, and GGUF-backed primitive installation.

## Control-plane files

- `admission.py`: VRAM probes and max-context/KV-tier admission arithmetic. Keep it pure and unit-testable.
- `prefill_policy.py`: pure prefill policy decisions and validation helpers. Runtime flags stay in `model.py`.
- `gguf.py`: GGUF parsing/loading. Header-only metadata readers belong here, not in `model.py`.
- `cli.py`: CLI/server wiring only. It should not own model policy arithmetic.

When adding a new feature, prefer placing the policy in a small module and threading the resolved result into `TransformerConfig`. `model.py` should consume resolved decisions, not become the source of every decision.
