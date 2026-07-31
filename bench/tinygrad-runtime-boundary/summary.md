# Tinygrad Runtime Boundary Audit (R0)

Verdict: **R0_PASS_BOUNDARY_PINNED**

Entrypoint: `extra/llm/cli.py :: main() --serve (class LLMServer / Handler)`

Surface present: 20/20 checks

| check | present | owner |
|---|---|---|
| GET /v1/models route exists | yes | runtime |
| POST /v1/chat/completions route exists | yes | runtime |
| POST /v1/completions route exists | yes | runtime |
| model load path (RuntimeState.load+from_gguf) | yes | runtime |
| max_context handling (from_gguf caps by ctx) | yes | runtime |
| prompt/KV prefix reuse (get_start_pos) | yes | runtime |
| SSE streaming behavior (stream_json) | yes | runtime |
| oversized-prompt guard before generate | yes | runtime |
| GET /runtime/status control | yes | runtime |
| GET /runtime/models control | yes | runtime |
| GET /runtime/metrics control | yes | runtime |
| GET /runtime/cache control | yes | runtime |
| compile-cache hit/miss + kernel-count observability | yes | runtime |
| POST /runtime/load control | yes | runtime |
| POST /runtime/unload control | yes | runtime |
| POST /runtime/warmup control | yes | runtime |
| POST /runtime/cancel control | yes | runtime |
| one-generation-at-a-time busy policy (gen_lock) | yes | runtime |
| structured JSON error contract | yes | runtime |
| model registry (build_registry) | yes | runtime |

Client-concern leakage into runtime: none

Live probe: skipped or no server running.
