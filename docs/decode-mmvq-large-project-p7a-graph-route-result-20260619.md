# Decode MMVQ large project P7a graph route result - 2026-06-19

> **SUPERSEDED by `decode-mmvq-large-project-p7b-raw-kernarg-rebind-result-20260619.md`.** The P7a failure was real
> for the first wrapper, but P7b fixed the raw-kernarg rebind/call path and the Q4 graph route now passes.

Purpose: execute the next build after P5/P6: make the imported Q4_K MMVQ route graph-safe enough for an in-model
decode route.

Artifacts:

- `extra/qk_decode_mmvq_graph_route.py`
- `extra/qk_decode_mmvq_p7_q4_graph_route.py`
- `bench/qk-decode-mmvq-large-project/p7a_q4_graph_route_failure.json`

## What was built

P7a implemented the graph-route adapter:

- `q8_quant_stub` establishes a tinygrad `custom_kernel` graph node for the `block_q8_1` producer;
- `q4_mmvq_stub` establishes a tinygrad `custom_kernel` graph node for the imported llama Q4_K consumer;
- runtime-cache swaps install the actual q8 producer and imported llama MMVQ runner;
- the imported runner patches the captured `144` byte llama kernarg with current tinygrad buffer VAs;
- queue launch dimensions are forced through the same `AMDComputeQueue.exec` wrapper pattern used by the Tensile and q8
  artifact probes.

Two graph variants were attempted:

1. Hidden temporary q8/output buffers inside the TinyJit route.
2. Persistent q8/output side buffers passed as explicit TinyJit arguments.

## Result

Both variants faulted during TinyJit replay:

| variant | result |
|---|---|
| temporary q8/out buffers | MMU fault on replay synchronize (`NotPresent`, around `0x717DD84FD000`) |
| persistent side buffers | MMU fault on replay synchronize (`NotPresent`, around `0x717DD8598000`) |

Verdict: **REDIRECT_RUNTIME_CAPABILITY**.

## Interpretation

This does **not** invalidate the imported Q4_K consumer:

- P3/P4 direct HCQ launch remains correct and fast;
- P5 eager lifecycle remains correct;
- P6 Q4 shape matrix remains correct and fast.

What failed is narrower: a raw captured llama kernarg, patched inside a runtime-cache wrapper, is not safely replayable
through TinyJit/HCQGraph. tinygrad's normal kernels have rebindable argument metadata; the imported llama runner has a
raw byte buffer whose pointer fields are opaque to graph replay.

The next build is therefore not model wiring. It is one of:

1. **First-class raw-kernarg rebind support**: teach the AMD graph/runtime path that specific offsets in an imported
   kernarg correspond to specific buffer arguments, so graph replay can patch them like normal kernel args.
2. **Single-kernel wrapper thunk**: compile a tiny AMD wrapper with normal `(out, q4, q8)` args that jumps/calls the
   imported function or otherwise materializes a rebindable ABI. This may not be practical across code-object
   boundaries.
3. **Native tinygrad lowering**: transfer the llama MMVQ schedule into tinygrad codegen so the graph sees ordinary
   arguments from the start.

Until one of those exists, the imported route remains an eager research primitive, not a graph-safe decode route.
