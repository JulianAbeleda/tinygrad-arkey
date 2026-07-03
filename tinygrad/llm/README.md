# tinygrad.llm Surface

This package keeps model execution separate from load-time policy.

## Runtime files

- `model.py`: transformer blocks, model construction, cache allocation, and generation. It may call policy helpers, but should not grow new standalone admission or registry logic.
- `decode_routes.py`: runtime decode route selection. Route-specific generated kernels live under `extra/qk` and are imported through `route_ops.py`.
- `route_policy.py`: policy-file parsing and validation for generated/QK routes.
- `route_ops.py`: import boundary for generated/search-produced primitives.
- `qk_primitives.py`: Q4_K/Q6_K primitive wrappers, install-time storage policy, and GGUF-backed primitive installation.

## Control-plane files

- `admission.py`: VRAM probes and max-context/KV-tier admission arithmetic. Keep it pure and unit-testable.
- `prefill_policy.py`: pure prefill policy decisions and validation helpers. Runtime flags stay in `model.py`.
- `gguf.py`: GGUF parsing/loading. Header-only metadata readers belong here, not in `model.py`.
- `cli.py`: CLI/server wiring only. It should not own model policy arithmetic.

When adding a new feature, prefer placing the policy in a small module and threading the resolved result into `TransformerConfig`. `model.py` should consume resolved decisions, not become the source of every decision.
