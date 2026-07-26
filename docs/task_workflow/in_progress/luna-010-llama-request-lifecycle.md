# LUNA-010: llama-bench request lifecycle source map

## Status

`PASS (static source map)` for source revision `ac4cddeb0dbd778f650bf568f6f08344a06abe3a` at `/home/ubuntu/env/llama.cpp`. This source checkout is dirty (`?? .rocprofv3/`, `?? bench/`); LUNA-002 must still establish that the traced binary was built from this revision. No GPU work was dispatched.

## Owners and phase map

| Phase | Source owner | Inputs / behavior | Proposed observable boundary |
|---|---|---|---|
| CLI and case construction | `tools/llama-bench/main.cpp`; `tools/llama-bench/llama-bench.cpp` (`cmd_params`, `llama_bench`, benchmark loop) | Bench parameters carry prompt length (`n_prompt`), generated length (`n_gen`), repetitions, `n_batch`, `n_ubatch`, flash-attention and GPU-layer/offload settings into context/model parameters. | Preserve emitted bench case line and argv; trace begins immediately before the measured decode/encode call. |
| Context/model creation | `tools/llama-bench/llama-bench.cpp`; public context construction in `src/llama-context.cpp` | Builds model/context, backend scheduler and KV cache using the case parameters. `n_gpu_layers` determines model-layer placement; the chosen backend must be positively proven by LUNA-002. | Backend initialization logs plus first HIP API event. |
| Prompt token construction | `tools/llama-bench/llama-bench.cpp` benchmark case setup | Creates a `llama_batch` for the configured prompt length and routes it to `llama_encode` or `llama_decode` according to model architecture. | First `llama_decode`/`llama_encode` host call and its first device launch. |
| Prompt evaluation | `src/llama-context.cpp` (`llama_context::decode`, `llama_decode`; graph compute path); `src/llama-graph.cpp` graph builder | Prompt is split into ubatches by `n_ubatch`; graph shape has `n_tokens > 1`, so linear layers are GEMM-like. KV writes occur for every prompt token. | Per-ubatch graph compute / scheduler submission; delimiter after the final prompt synchronization. |
| First generated token | Same decode entry and graph builder | A one-token batch consumes the prompt KV state and appends the first generated token. It is a decode/GEMV-like shape even though it immediately follows prompt. | One separate `llama_decode` call after prompt; record its launch interval. |
| Steady decode | Same decode entry and graph builder, repeated by `llama-bench.cpp` | Each generated token is a one-token decode using an increasing KV length. | One host decode-call interval per token, with HIP API correlation IDs. |
| Completion/timing boundary | `tools/llama-bench/llama-bench.cpp`; `src/llama-context.cpp` synchronization/graph compute path | Bench timing must include the explicit backend synchronization after the measured region; asynchronous queue submission alone is not a device-complete boundary. | `hipStreamSynchronize` / backend synchronize API event, then host timer stop. |

## Flag semantics to retain in the LUNA-020 command matrix

- Prompt/generation/repetition: `-p`/prompt tokens, `-n`/generation tokens, and benchmark repetition option must be copied from the exact `llama-bench --help` for the selected binary rather than inferred from this map.
- Batching: `n_batch` controls logical evaluation batch capacity; `n_ubatch` controls graph micro-batches and can split prompt work into multiple graph submissions.
- Flash attention: the bench/context flash-attention setting reaches graph construction; admission is model-shape and backend-capability dependent, not a guarantee of a single flash kernel.
- GPU offload: model `n_gpu_layers` controls where tensors/layers are placed. `-ngl 99` is only a requested placement; LUNA-002 must prove HIP/ROCm runtime selection.

## Trace protocol

Use one process and one benchmark case. Record host markers around: context creation, each prompt ubatch, first token, each steady token, and final synchronization. Join HIP API correlation IDs to kernel dispatch timestamps. This gives a non-overlapping prompt / transition / steady-decode ledger even if graph execution is asynchronous.

## Controls and blockers

- Positive control: LUNA-021 must observe a HIP launch between a marked `llama_decode` interval and the corresponding backend synchronization.
- Negative control: do not classify host enqueue time as kernel time.
- Blocker: source-to-binary identity is unproven pending LUNA-002; this map must not be used to assert runtime kernel names until then.
