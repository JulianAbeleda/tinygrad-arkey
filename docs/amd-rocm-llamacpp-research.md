# AMD ROCm and llama.cpp research for tinygrad-arkey

This note compares the local `tinygrad-arkey` AMD path with the AMD-relevant parts of the forked `llama.cpp` and `ROCm` repos. The goal is to understand what those projects solve, why their design works, and what we can test from first principles to improve TinyGPU/tinygrad inference on the RX 7900 XTX.

## Current local path

Our active path is not the normal Linux ROCm stack:

```text
tinygrad-arkey Python
  -> tinygrad AMD runtime
  -> RemotePCIDevice RPC
  -> extra/remote/serve.py
  -> TinyGPU DriverKit / UT4G
  -> AMD RX 7900 XTX
```

The relevant tinygrad pieces are:

- `tinygrad/runtime/support/system.py`: `RemotePCIDevice` opens a TCP socket, sends raw PCI/MMIO/sysmem commands, tracks bulk bytes, and can print RPC roundtrip statistics.
- `extra/remote/serve.py`: the remote server probes devices, lazily opens the PCI device, maps BARs, allocates system memory with `MAP_SYSMEM`, and keeps allocations in a process-local list.
- `tinygrad/llm/gguf.py`: GGUF quantized weights are decoded into tinygrad Tensor expressions. It supports Q4_K/Q5_K/Q6_K/IQ variants, including Q4_K_M model files.
- `tinygrad/llm/model.py`: transformer blocks currently run with implicit weights and a comment says the GGUF is unpacked on the fly.

That means our current bottlenecks can live in two different places:

1. Runtime/transport reliability: device discovery, BAR setup, sysmem mapping, MMIO, command queue setup, and reconnect behavior.
2. Kernel quality: what code runs after the model is loaded, especially quantized matmul/dequant, attention, KV cache access, and per-token scheduling.

The observed failures fit the first category first. We have seen `MAP_SYSMEM` errors, empty replies from the HTTP server after device loss, stale bridge state after model failure, and cases where the UT4G is still visible but the AMD GPU path is gone. The performance gap against expected ROCm paths probably lives in both categories.

## What llama.cpp does for AMD

`llama.cpp` uses `ggml` backends. For AMD, the important path is HIP:

- Build flag: `-DGGML_HIP=ON`.
- GPU target: `-DGPU_TARGETS=gfx1100` for Radeon RX 7900 XTX/XT/GRE.
- Optional fast attention path: `-DGGML_HIP_ROCWMMA_FATTN=ON` for RDNA3+ or CDNA when rocWMMA headers are available.
- Device selection and compatibility are delegated to ROCm/HIP environment behavior such as `HIP_VISIBLE_DEVICES` and `HSA_OVERRIDE_GFX_VERSION`.

The useful idea is that `llama.cpp` keeps CUDA-like kernel code but routes it through HIP for AMD. In `ggml/src/ggml-cuda/vendors/hip.h`, CUDA/cuBLAS names are mapped to HIP/hipBLAS equivalents. That lets one backend family share a lot of code while still compiling for AMD.

The second useful idea is that `llama.cpp` does not treat all AMD GPUs the same. In `ggml/src/ggml-cuda/common.cuh`, AMD architectures are separated into GCN, CDNA, RDNA1, RDNA2, RDNA3, RDNA3.5, and RDNA4. RX 7900 XTX is RDNA3 / `gfx1100`. In `ggml/src/ggml-cuda/mmvq.cu`, quantized matvec behavior branches on those architecture classes and chooses RDNA3-specific limits. This is directly relevant because single-token generation is dominated by repeated quantized matvec/matmul work.

`llama.cpp` also has an RPC backend, but it operates at a higher level than our remote PCI path. Its RPC commands include buffer allocation, tensor transfer, tensor copy, graph compute, graph recompute, and device memory queries. The server exposes `ggml` devices, and the client offloads computation to those devices. It can cache large tensors so they do not need to be resent. That is a very different design from sending raw PCI/MMIO/sysmem operations from tinygrad.

The practical lesson from `llama.cpp` is:

- Use AMD as an explicit architecture target, not just a generic GPU.
- Keep quantized weights packed as long as possible.
- Fuse dequant and matvec/matmul where the hot path needs it.
- Keep model tensors and KV cache resident near the device.
- Move remote APIs up toward graph/tensor operations when possible, not down toward repeated raw bus operations.

## What ROCm solves

ROCm is the full AMD software stack: compilers, HIP, runtimes, profilers, debuggers, and tuned libraries. HIP is the main runtime and kernel language for AMD GPU programming. It is designed as a CUDA-portable layer, so projects can port CUDA-style kernels to AMD without rewriting the whole runtime model.

The relevant ROCm components for our goal are:

- HIP: kernel launch, runtime API, memory management, streams/queues, and CUDA portability.
- rocBLAS / hipBLAS / hipBLASLt: optimized GEMM and matrix paths.
- rocWMMA: mixed-precision matrix multiply-accumulate support, relevant to RDNA3+/CDNA paths.
- Composable Kernel: performance-critical ML kernels across architectures.
- ROCProfiler / ROCTracer / ROCm Bandwidth Test: measurement tools for runtime calls, async activity, copy bandwidth, and kernel behavior.

ROCm also documents BAR and MMIO behavior. AMD GPUs expose PCIe BARs for GPU memory, doorbells, I/O, and MMIO. The doorbell BAR tells the GPU that new work is in its queue. BAR placement and physical address limits matter; if a BAR address is outside what a peer device can address, DMA/P2P access can fail. This is very close to our TinyGPU/UT4G path because we are doing explicit BAR and sysmem work instead of letting the ROCm kernel driver hide it.

The practical lesson from ROCm is:

- A stable AMD runtime is mostly a lifecycle problem: discover device, establish address spaces, map memory, create queues, ring doorbells, launch kernels, handle errors, and clean up.
- A fast AMD runtime is mostly a residency and specialization problem: avoid unnecessary host/device transfers, avoid repeated setup, use arch-specific kernels, and measure queue/copy/kernel time separately.
- Our remote server is currently responsible for pieces that ROCm normally owns. That makes instrumentation and error recovery more important.

## First-principles hypotheses

### Hypothesis 1: the remote PCI lifecycle is the first reliability bottleneck

If the device drops or the bridge gets into a dirty state, the model server can still appear alive while generation fails. This explains the pattern where `/v1/models` can answer but generation returns an empty reply or hangs. From first principles, the HTTP server and Python process are not the source of truth; the source of truth is whether the AMD device can still respond to BAR, sysmem, queue, and kernel operations.

How to leverage ROCm/llama.cpp:

- Add a tinygrad health check that validates the remote AMD path before serving model requests.
- Treat `MAP_SYSMEM`, BAR mapping, and queue setup as explicit lifecycle states with clear logs.
- Make the bridge fail closed when the GPU disappears instead of leaving the API server half-alive.
- Copy the design principle from ROCm: device liveness and allocation ownership are runtime invariants, not incidental side effects.

Test:

- Before and after each model load, run a one-op AMD tensor sanity check through `REMOTE=127.0.0.1:6667 DEV=AMD`.
- Log every `MAP_SYSMEM` request size, contiguous flag, paddr count, and returned handle.
- On failure, compare whether the UT4G is visible, the AMD GPU is visible, and whether a fresh bridge process fixes it without power cycling.

### Hypothesis 2: raw remote operations are costing too much per token

USB4/DriverKit can work, but every remote roundtrip is expensive compared with a local ROCm runtime. If per-token generation causes too many MMIO reads/writes, sysmem reads/writes, or repeated setup operations, token/sec will fall even if kernels are decent.

How to leverage ROCm/llama.cpp:

- Use llama.cpp RPC as the design reference: remote APIs should move tensors/graphs, not every low-level operation.
- Cache model tensors and KV state close to the device.
- Coalesce remote commands where possible.
- Avoid redoing sysmem/BAR/queue setup after the model is already resident.

Test:

- Run inference with `DEBUG=1` so `RemotePCIDevice` prints sent MB, received MB, and roundtrip count at exit.
- Compare 0.6B, 1.7B, and 4B for roundtrips per generated token.
- If roundtrips/token scales with model depth or token count, optimize the remote path first.
- If roundtrips/token is low but token/sec is still low, focus on kernel quality.

### Hypothesis 3: Q4_K_M performance is limited by generic unpack/dequant instead of packed quantized kernels

Qwen 1.7B and larger Q4_K_M models depend heavily on quantized matvec/matmul. tinygrad currently decodes GGUF quant blocks through tensor expressions in `gguf.py`, and `model.py` notes that weights are unpacked on the fly. llama.cpp has dedicated quantized kernels and RDNA3-specific choices for MMVQ/MMQ. That is probably a major reason ROCm/llama.cpp paths can outperform our tinygrad path.

How to leverage llama.cpp:

- Study `ggml/src/ggml-cuda/mmvq.cu`, `mmq.cu`, and quant dequant helpers as algorithms, not as code to copy blindly.
- Identify the exact Q4_K_M layout tinygrad sees in `gguf.py`.
- Add or pattern-match a fused path for Q4_K_M dequant plus matvec/matmul on AMD.
- Specialize for RDNA3 / `gfx1100` assumptions where tinygrad can express them.

Test:

- Benchmark a single representative Q4_K_M linear layer with the same input shape as decode.
- Compare generic unpack-then-matmul against a fused quantized path.
- Track generated token/sec separately from prompt prefill token/sec, because decode stresses matvec more heavily.

### Hypothesis 4: the best long-term remote design is graph-level, not PCI-level

The current TinyGPU path proves we can reach the AMD card through DriverKit, but it exposes a very low-level contract. llama.cpp RPC exposes a higher-level compute contract: allocate buffers, set tensors, compute graph, get memory. That reduces host involvement during the hot path.

How to leverage llama.cpp:

- Treat its RPC backend as a reference architecture for remote tensor execution.
- Keep the existing remote PCI path for bring-up and hardware access.
- Experiment with a higher-level tinygrad remote executor that ships compiled kernels/graphs and persistent buffers, then performs token steps with minimal host chatter.

Test:

- Measure whether one decode step can be represented as a small number of remote commands after warmup.
- Prototype a resident session object: model buffers, KV cache, compiled kernels, queue state, and a single step command.

## External research map

arXiv is useful, but it should not be the only source. Papers usually explain the kernel principle, while repos and benchmark threads show what actually works on RDNA3 / `gfx1100`.

Where to look:

- Official AMD ROCm docs first, especially HIP, hipBLAS/hipBLASLt, rocWMMA, Composable Kernel, profiling, and BAR/MMIO docs. These tell us what AMD expects the runtime and kernel stack to do.
- llama.cpp and ggml code/issues/PRs second. This is the closest production codebase to our workload: GGUF, Q4_K_M, decode-heavy inference, HIP, RDNA3 branches, and RPC.
- ROCm library repos third: `ROCm/composable_kernel`, `ROCm/rocBLAS`, `ROCm/hipBLASLt`, and `ROCm/rocWMMA`. These are useful when we need to understand AMD tiling, WMMA, architecture-specific GEMM, and why some kernels only target Instinct/CDNA.
- arXiv fourth, for kernel ideas we can translate into tinygrad: packed quantized matvec, dequant fusion, shared-memory staging, offline layout transforms, and avoiding bank conflicts.
- Community benchmark and patch reports fifth. These are noisy, but they expose the real 7900 XTX failure modes: missing `gfx1100` targeting, ROCm version regressions, Vulkan vs HIP differences, and per-shape MMVQ tuning.

Current online leads:

- AMD's llama.cpp ROCm documentation lists the relevant ROCm libraries: hipBLAS for matrix/vector operations, hipBLASLt for fused GEMM features and integer tensor cores, and rocWMMA for mixed-precision matrix multiply/accumulate and flash-attention acceleration.
- AMD's Composable Kernel docs show the first-principles pattern we care about: fuse the operations around quantized linear layers into one GPU kernel instead of materializing intermediate tensors. The same page warns that some XDL instances are Instinct-specific and do not run on Radeon, so RX 7900 XTX work must be checked for RDNA3 support.
- QUICK (`arXiv:2402.10076`) is relevant because it optimizes quantized LLM inference by interleaving quantized weights offline and avoiding shared-memory write-back after dequantization. Even though it is CUDA/NVIDIA-focused, the principle maps to our Q4_K_M path: layout and dequant strategy matter.
- Fast NF4 Dequantization Kernels (`arXiv:2604.02556`) is relevant because it frames dequantization itself as the bottleneck and improves it through memory hierarchy use. The exact NF4 format differs from GGUF Q4_K_M, but the principle is directly applicable.
- Recent 7900 XTX reports keep pointing at explicit RDNA3 targeting (`gfx1100` / `HSA_OVERRIDE_GFX_VERSION=11.0.0`) and model-shape-specific MMVQ tuning. Those are not canonical sources, but they match the local hypothesis that generic AMD kernels leave decode performance on the table.

The practical conclusion: use arXiv to learn the kernel principles, but use llama.cpp/ROCm to decide what to implement first. For `tinygrad-arkey`, the best next paper topic is not "LLM inference" broadly; it is "weight-only quantized matvec/dequant fusion for decode on AMD RDNA3."

## Parallel search findings

The parallel search was split into runtime, libraries, llama.cpp implementation, papers, and real-world benchmarks.

Runtime findings:

- ROCm BAR documentation is directly relevant to TinyGPU. AMD GPUs expose framebuffer BARs, doorbell BARs, optional BARs, and MMIO BARs. The doorbell BAR signals queued GPU work. That maps to why our bridge can fail even before kernel quality matters: if BAR setup, physical addressing, or sysmem mapping is wrong, the runtime cannot reliably submit work.
- ROCm and llama.cpp reports mention VMM and allocation failures on Radeon/HIP paths. Even with the normal ROCm stack, memory management can be version-sensitive. For our custom DriverKit path, this makes `MAP_SYSMEM` logging and allocation ownership a first-class test target.

Library findings:

- hipBLAS and rocBLAS matter for dense matrix paths, but llama.cpp's decode hot path is often custom quantized kernels rather than plain BLAS.
- hipBLASLt and rocWMMA are important, but support must be checked per architecture. Some AMD docs and community notes warn that certain high-performance instances are Instinct/CDNA-oriented, not Radeon/RDNA3-oriented.
- rocWMMA explicitly lists RDNA3 `gfx1100/gfx1101/gfx1102` support in newer documentation, so it is worth checking for flash-attention and matrix paths, but not assuming it covers our Q4_K_M decode bottleneck.

llama.cpp findings:

- Official llama.cpp supports HIP as an AMD backend and RPC as a backend, but its RPC is graph/tensor-level rather than raw PCI-level.
- The AMD path needs explicit RDNA3 targeting. Online benchmarks repeatedly mention `gfx1100`, `AMDGPU_TARGETS=gfx1100`, or `HSA_OVERRIDE_GFX_VERSION=11.0.0` as the difference between a correct HIP path and a slow/fallback path.
- The most relevant implementation area remains `ggml/src/ggml-cuda/mmvq.cu` and adjacent quantized matrix kernels. Community tuning reports claim large gains by tuning MMVQ parameters per model shape on 7900 XTX, which matches our hypothesis that shape-specific decode kernels matter.

Paper findings:

- QUICK argues for offline layout/interleaving of quantized weights to reduce dequant/shared-memory overhead. The useful principle is not CUDA-specific: quant layout can determine whether the kernel wastes memory traffic.
- Fast NF4 Dequantization frames dequantization as the bottleneck and improves it by exploiting shared memory. The format differs from GGUF Q4_K_M, but the memory-hierarchy lesson applies.
- CodeGEMM and LUT-GEMM style papers are useful because they ask whether dequantization can be avoided or replaced with lookup/precomputed partial products. These are longer-term ideas, not first patches.

Benchmark findings:

- Public 7900 XTX numbers vary heavily with ROCm version, build flags, backend, model, context length, and quant. That means they are useful for order-of-magnitude targets, not exact pass/fail numbers.
- Several reports put 7B/8B Q4-class llama.cpp HIP decode far above our current tinygrad 1.7B/4B behavior. That supports the view that our current path is not just "AMD is slow"; it is either remote overhead, generic quant kernels, or both.
- Vulkan sometimes beats ROCm for llama.cpp on RDNA3 in community tests. That does not directly help TinyGPU, but it is a warning that "ROCm" alone is not automatically the fastest path on Radeon.

## Implementation delta after audit

This section separates existing scaffolds from the next implementation deltas. It is the working plan after the initial runtime-health commits.

Already present:

- `extra/remote/bench.py` is the starting health scaffold. It now checks probe, PING latency, config read, BAR0 mapping, `MAP_SYSMEM`, sysmem read/write throughput, optional tiny AMD tensor sanity, and classifies `healthy`, `dirty`, or `dead`.
- `tinygrad/runtime/support/system.py` already has global remote counters through `RemotePCIDevice.stats()` and `reset_stats()`, plus optional `REMOTE_RPC_TIMEOUT`.
- `extra/remote/serve.py` already logs `PROBE`, lazy device open, `MAP_BAR`, and `MAP_SYSMEM`, and tracks process-local command counts/errors.
- `tinygrad/llm/gguf.py` already supports Q4_K/Q4_K_M correctness through generic tensor dequantization.
- `extra/q4_k_bench.py` is the current standalone Q4_K baseline scaffold.
- This document is the source research note for the AMD 7900 XTX optimization thesis.

Slice 1 delta: bridge lifecycle and health semantics.

- Add per-RPC phase timing on the client side, not only server cumulative counters. The minimum output should include command name, count, total ms, average ms, sent bytes, received bytes, and failures.
- Add a remote health query command to the bridge protocol. The request/response framing should stay compatible with the existing fixed request/response protocol; add a new `RemoteCmd` rather than changing the wire shape of existing commands.
- Add server-side dirty-state behavior. After a device-level failure in probe/open/BAR/sysmem/MMIO/sysmem access, mark the server dirty and reject subsequent device commands with a clear error until restart or explicit reset.
- Add a model-server preflight health check before `/v1/models` and generation responses are treated as usable.

Slice 2 delta: per-generation remote pressure metrics.

- Add per-generation stats reset hooks in the LLM inference path, not just global process counters. The likely integration point is `tinygrad/llm/cli.py` around request handling and `tinygrad/llm/model.py` around prefill/decode boundaries.
- Report prefill and decode separately: tokens, elapsed time, remote roundtrips, MB sent, MB received, roundtrips/token, and MB/token.
- Keep the existing low-level RPC protocol for now. Roundtrip reduction in this slice should come from caching, residency, and avoiding repeated calls, not from redesigning the wire format.
- Place session-local mapping/allocation caches in the client runtime first (`RemotePCIDevice` or its owning AMD runtime objects), because `serve.py` allocations are currently process-local and not durable across client reconnects.

Slice 3 delta: Q4_K_M layer benchmark before kernel changes.

- Extend `extra/q4_k_bench.py` so it can benchmark one selected Qwen Q4_K_M layer on CPU and AMD, list tensor names, and emit JSON/CSV-friendly metrics.
- Measure three timings separately: Q4_K decode, decoded matvec, and end-to-end decode-plus-matvec.
- Use a validated model target first: Qwen 1.7B Q4_K_M is the initial gate because it has previously loaded on this remote AMD path. Qwen 4B is the stress gate. Qwen 8B/7B is not a first acceptance target until the runtime is stable.
- Keep `test/unit/test_gguf.py` as correctness coverage and add performance benchmarking separately; do not turn unit tests into GPU performance tests.

Slice 4 delta: packed Q4_K_M fused decode matvec.

- Prototype a packed Q4_K_M decode+matvec path gated behind AMD/gfx1100. The generic GGUF tensor path must remain the fallback for all other devices and quant types.
- Verify that tinygrad's custom-kernel path can express Q4_K block memory access patterns before committing to a broad rewrite. Existing `extra/llama_kernels/` examples are useful for custom-kernel mechanics but do not prove Q4_K_M coverage.
- Use llama.cpp `mmvq.cu` as the algorithmic reference: packed block layout, dequant inside the matvec, RDNA3 wave32 assumptions, and per-shape tuning.
- Start with one Qwen layer shape, prove correctness against current `gguf.py` dequant, then tune rows-per-block/vectorization and expand only if the benchmark wins.

Comparison baseline:

- Standard ROCm/KFD (`KFDIface`) remains a useful comparison path when available. The active target is the remote TinyGPU/DriverKit path, but KFD can help distinguish remote-bridge overhead from AMD kernel quality.

## Recommended order of work

1. Finish Slice 1 health semantics: client-side RPC phase timings, health query, dirty-state gate, and model-server preflight.
2. Add Slice 2 per-generation metrics: prefill/decode roundtrips per token and MB per token.
3. Finish Slice 3 Q4_K_M layer benchmark on Qwen 1.7B and Qwen 4B.
4. Start Slice 4 only after the benchmark proves the current decoded matvec is the bottleneck.
5. Consider a graph-level remote API after the PCI path is reliable and per-generation metrics show remote chatter remains material.

## Relevant local references

- tinygrad remote client: `/Users/julianabeleda/env/tinygrad-arkey/tinygrad/runtime/support/system.py`
- tinygrad remote server: `/Users/julianabeleda/env/tinygrad-arkey/extra/remote/serve.py`
- tinygrad GGUF quant decode: `/Users/julianabeleda/env/tinygrad-arkey/tinygrad/llm/gguf.py`
- tinygrad LLM execution: `/Users/julianabeleda/env/tinygrad-arkey/tinygrad/llm/model.py`
- llama.cpp HIP build docs: `/Users/julianabeleda/env/llama.cpp/docs/build.md`
- llama.cpp AMD arch definitions: `/Users/julianabeleda/env/llama.cpp/ggml/src/ggml-cuda/common.cuh`
- llama.cpp quantized matvec path: `/Users/julianabeleda/env/llama.cpp/ggml/src/ggml-cuda/mmvq.cu`
- llama.cpp HIP compatibility layer: `/Users/julianabeleda/env/llama.cpp/ggml/src/ggml-cuda/vendors/hip.h`
- llama.cpp RPC docs: `/Users/julianabeleda/env/llama.cpp/tools/rpc/README.md`
- ROCm overview: `/Users/julianabeleda/env/ROCm/docs/what-is-rocm.rst`
- ROCm HIP programming guide: `/Users/julianabeleda/env/ROCm/docs/how-to/programming_guide.rst`
- ROCm BAR memory doc: `/Users/julianabeleda/env/ROCm/docs/how-to/Bar-Memory.rst`
- ROCm API libraries: `/Users/julianabeleda/env/ROCm/docs/reference/api-libraries.md`
- AMD llama.cpp ROCm docs: `https://rocm.docs.amd.com/projects/llama-cpp/en/docs-26.02/install/llama-cpp-install.html`
- AMD Composable Kernel inference docs: `https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html`
- ROCm BAR access documentation: `https://rocm.docs.amd.com/en/develop/how-to/Bar-Memory.html`
- ROCm HIP compiler docs: `https://rocm.docs.amd.com/projects/HIP/en/latest/understand/compilers.html`
- rocWMMA API docs: `https://rocm.docs.amd.com/projects/rocWMMA/en/docs-6.4.3/API_Reference_Guide.html`
- llama.cpp ROCm allocation issue: `https://github.com/ggml-org/llama.cpp/issues/14178`
- llama.cpp TurboQuant HIP discussion: `https://github.com/ggml-org/llama.cpp/discussions/21526`
- QUICK quantized-kernel paper: `https://arxiv.org/abs/2402.10076`
- Fast NF4 dequantization paper: `https://arxiv.org/abs/2604.02556`
- CodeGEMM quantized GEMM paper: `https://arxiv.org/abs/2512.17970`
- LUT-GEMM quantized matrix multiplication paper: `https://arxiv.org/abs/2206.09557`
- RDNA3 MMVQ tuning report: `https://www.reddit.com/r/ROCm/comments/1s7jgvl/kernelanvil_2x_decode_speedup_on_7900_xtx_by/`
- 7900 XTX ROCm/Vulkan comparison report: `https://www.reddit.com/r/ROCm/comments/1s1vo37/rocm_on_7900_xtx_significantly_slower_than_vulkan/`
