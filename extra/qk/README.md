# QK Research Surface

`extra/qk` is the research and generated-route workspace. It is intentionally separate from `tinygrad/llm`, which owns the shipped runtime surface.

Use these roles when adding or reading files:

- `promoted`: files used by the runtime path or route manifests.
- `active`: current gates, audits, and microgates that decide whether a route can be promoted.
- `support`: reusable helpers, layouts, caches, lowering utilities, and harness contracts.
- `refuted`: historical probes that explain a closed path and should not be copied into runtime code.
- `scratch`: one-off experiments that should either be deleted or moved into BoltBeam once resolved.

Rules:

- New runtime-visible routes must enter through `tinygrad/llm/route_ops.py` or `tinygrad/llm/decode_routes.py`.
- New probes must have a clear role in `surface_audit.py`; unclassified files fail the audit.
- Refuted or scratch files are evidence, not defaults. Do not route to them from `tinygrad/llm`.
- Prefer adding reusable helpers to support files rather than copy-pasting kernels into new probes.

Run:

```sh
PYTHONPATH=. python3 extra/qk/surface_audit.py
```
