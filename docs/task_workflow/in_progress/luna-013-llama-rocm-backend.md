# LUNA-013: ROCm/HIP backend source map

## Status

`PASS (static source-to-dispatch map; known-kernel confirmation deferred to LUNA-021)`. This checkout routes AMD through the CUDA-named backend compiled with HIP, rather than a distinct ROCm implementation: `ggml/src/ggml-hip/CMakeLists.txt` configures HIP and `ggml/src/ggml-cuda/vendors/hip.h` supplies HIP vendor bindings.

## Backend pipeline

| Layer | Source owner | Responsibility | Trace/join key |
|---|---|---|---|
| Graph construction | `src/llama-graph.cpp` | Emits ggml nodes such as `MUL_MAT`, RoPE, copy, softmax and `FLASH_ATTN_EXT`. | Node op, tensor types/dimensions, layer index, host phase marker. |
| Graph scheduling / placement | ggml backend scheduler (`ggml/src/ggml-backend*.cpp`) plus llama context compute path | Assigns graph portions to backend buffers/devices and computes graph splits. | Backend name/device, split boundary, tensor buffer ownership. |
| Allocation | CUDA/HIP backend buffer types and allocator in `ggml/src/ggml-cuda/ggml-cuda.cu` and ggml backend buffer interfaces | Allocates device buffers and maintains tensor extra data/workspace. | HIP allocation pointer/size, backend buffer identity. |
| Queue / stream | `ggml/src/ggml-cuda/ggml-cuda.cu` context and operation dispatch | Owns HIP stream(s), enqueues operations and performs backend synchronization through the HIP vendor layer. | HIP stream handle and HIP API correlation ID. |
| Kernel launch | `ggml-cuda.cu` op dispatcher calls operation implementations (`mmq.cu`, `mmvq.cu`, `fattn.cu`, `rope.cu`, `cpy.cu`, `softmax.cu`) | Forms dimension-dependent launch geometry and calls HIP-compatible launch wrappers. | Demangled kernel symbol, grid/block, dynamic shared memory, stream. |
| Compile/load | `ggml/src/ggml-hip/CMakeLists.txt`, `ggml/src/ggml-cuda/CMakeLists.txt`, HIP compiler toolchain | Kernels are ahead-of-time compiled into the ggml HIP shared object/code objects; runtime loads that object through normal dynamic/HIP module mechanisms. | Library path/build ID, code-object hash, `hipModuleLoad` only if the build/runtime uses module APIs. |

## Where launch identity is formed

- Kernel *family* is selected in `ggml-cuda.cu` from the ggml op and tensor types/dimensions.
- Template instantiations fix quant/attention specializations: `template-instances/mmq-instance-q4_k.cu`, `mmq-instance-q6_k.cu`, and attention instantiations included by the `fattn` sources.
- Grid, block, and dynamic shared-memory arguments are formed at the operation launch sites in the corresponding CUDA/HIP `.cu` implementation; they are runtime shape-dependent and must be taken from HIP trace data.
- HIP traces may expose mangled C++ template symbols. Demangle and retain the template arguments, compilation unit, stream, grid/block and HIP API correlation ID as the source-to-dispatch join record.

## Predicted positive control for LUNA-021

For an offloaded Q4_K or Q6_K prompt with a multi-token ubatch, a launch should resolve to a kernel instantiated from either `mmq-instance-q4_k.cu` or `mmq-instance-q6_k.cu`, with the correlated graph operation `GGML_OP_MUL_MAT`. For one-token decode, `mmvq`/`vecdotq` is a stronger candidate but is not asserted as a required outcome because the device/shape dispatcher may choose MMQ or fallback.

A second independent control is available when flash attention is admitted: a HIP kernel from `fattn.cu` correlates to `GGML_OP_FLASH_ATTN_EXT`; otherwise the decomposed attention op sequence is the expected negative control.

## Limitations and handoff

- Static source alone cannot prove which HIP backend was selected by a particular binary, nor prove actual code-object residency, launch geometry, or kernel spelling.
- LUNA-021 must retain HIP API + kernel trace, backend logs, executable/library build IDs and code-object identifiers, then confirm at least the MMQ positive control above.
- No GPU dispatch, profiling, binary inspection, or installation was performed by this task.
