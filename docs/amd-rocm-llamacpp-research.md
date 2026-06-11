# AMD ROCm and llama.cpp research for tinygrad-arkey

This note compares the local `tinygrad-arkey` AMD path with the AMD-relevant parts of the forked `llama.cpp` and `ROCm` repos. The goal is to understand what those projects solve, why their design works, and what we can test from first principles to improve TinyGPU/tinygrad inference on the RX 7900 XTX.

For the current hardware/runtime dropout investigation, see `docs/amd-remote-dropout-investigation.md`.

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

## 2026-06-10 status update: hypotheses re-scored, optimization campaign defined

Context that changed since the hypotheses were written: the KDB blocker is
solved (`docs/amd-kdb-root-cause.md`), the card boots and computes from the
Mac, and upstream's Apple-approved TinyGPU driver publicly achieves
**18.5 tok/s on Qwen 3.5 27B with a Mac mini M4 + RX 7900 XTX over USB4** —
the same GPU and machine class as this rig, on stock tinygrad kernels.
That number is now the calibration target: it proves what this hardware does
when the path is healthy, and it re-scores the hypotheses:

- **H1 (lifecycle/reliability): VALIDATED.** Predicted the dirty-bridge and
  dropout work. Completed by the idle keepalive in serve.py (idle ASPM/CLx
  transitions drop the link; a 1Hz config read suppresses them).
- **H2 (per-token roundtrip cost): STRENGTHENED, culprit sharpened.** Prior
  inference runs went through the serve.py TCP shim — a debugging layer the
  upstream product path does not have. The shim doubles serialization on
  every operation. H2's own test (DEBUG=1 roundtrips/token) is the first
  measurement of the campaign.
- **H3 (fused Q4_K kernels): DEMOTED.** Upstream's 18.5 tok/s uses the same
  stock kernels this fork has. Kernels cannot explain a gap vs upstream;
  H3 is only live again if roundtrips/token are low and tok/s still lags on
  the same model+quant. `extra/q4_k_bench.py` stands ready for that case.
- **H4 (graph-level remote): HALF-MOOTED.** The expensive "remote" was our
  own shim; the direct path (`DEV=AMD` -> APLRemotePCIDevice -> TinyGPU) is
  already in-process. H4 remains relevant only for the Ubuntu-as-GPU-server
  use case.

### Optimization campaign (measurement-first)

Known taxes, in expected order of magnitude:

1. **serve.py TCP shim** — removable by running the direct path. Measure
   first: tok/s direct vs through-shim on the same small model.
2. **Physical link health** — May captures show ACIO Gen2/3 retraining
   storms on the bare UT4G board. Watch link errors during weight upload;
   if present, try certified TB4 cable / other port / powered dock before
   any software work.
3. **Conservative mitigations** — `AMD_REMOTE_ALLOC_CAP_MB=2` (fragmented
   allocations), `REMOTE_MMIO_CHUNK`/`REMOTE_MMIO_FENCE_EVERY` (slow bulk
   MMIO, serve-path only). Relax one at a time, measuring each.
4. **Kernels (H3)** — only after 1-3, and only if a gap vs upstream remains.

Decision rule (unchanged from H2, now with a target): measure
roundtrips/token and tok/s against the upstream-proportional expectation for
the model under test. Optimize the layer the measurement convicts, never the
layer the intuition suspects.

Session plan:
- Session A: direct-path tok/s on a small model (0.6B/1.7B), DEBUG=1
  roundtrip stats, ACIO error watch during load. One power cycle.
- Session B: relax mitigations one at a time; hardware swaps if link errors
  appeared in Session A.
- Session C: scale model size toward the 27B-class target; Qwen end-to-end.

### Goal refinement (2026-06-10): the target is ROCm-native speed, not upstream parity

Operator-defined goal: the AM driver on the Mac mini should reach what this
same card delivers under ROCm/llama.cpp on native Linux PCIe. Upstream's
18.5 tok/s (Qwen 3.5 27B) is a waypoint, not the destination — ROCm-class
llama.cpp on a 7900 XTX is roughly 30-40 tok/s for that size class, i.e.
upstream tinygrad-over-USB4 runs at ~50-60% of ROCm. Closing that gap has
three distinct phases with different physics:

- **Phase 1 — reach upstream parity (transport taxes).** Remove the serve.py
  shim, fix link health, relax mitigations. Bounded, mostly done-or-known.
- **Phase 2 — close the kernel gap (H3 revived).** The upstream-vs-ROCm gap
  is dominated by kernel quality: llama.cpp has fused MMVQ/MMQ quantized
  kernels with RDNA3 tuning; tinygrad unpacks quant blocks generically.
  Levers: tinygrad BEAM search (cheap, try first), then a fused Q4_K
  dequant+matvec path specialized for gfx1100 (`extra/q4_k_bench.py` is the
  harness). This phase is identical for Mac and native Linux — develop and
  measure it on the Ubuntu boot where iteration is fast, deploy to the Mac.
- **Phase 3 — dispatch amortization.** Decode is GPU-memory-bound, so USB4
  only costs what per-token host chatter costs. TinyJit/graph batching must
  collapse each token step to ~one submission; measure roundtrips/token and
  drive it toward O(1). If achieved, the USB4 asymptote is within ~10% of
  native for decode.

Required baseline before any optimization: **measure actual ROCm/llama.cpp
tok/s on this exact card** (Ubuntu normal boot, card in the PCIe slot) for
the model sizes under test (1.7B/7B/27B-class, Q4_K_M, decode and prefill
separately). That number — not a literature estimate — is the finish line,
and it also calibrates how much of the gap is kernels (Phase 2) vs
dispatch (Phase 3).

Feasibility assessment: decode-speed parity with native ROCm is plausible
(memory-bound work happens entirely on-card); weight-load time will always
lose to native PCIe (USB4 ~1/4 the bandwidth); prefill sits in between.
"ROCm speed" should therefore be scored on steady-state decode tok/s.

## Layer 1 / Layer 2 campaign scope (2026-06-10)

Goal restated as arithmetic: decode tok/s = effective_GB/s ÷ weight_bytes.
Today's stack ~300 GB/s effective; llama.cpp/ROCm ~530-690; streaming ceiling
on a 7900 XTX ~820-860 (85-90% of 960 spec). Layer 1 closes tinygrad->llama.cpp.
Layer 2 exploits what llama.cpp's architecture cannot: whole-graph fusion and
layout freedom. All hypotheses are scored on the same metric: effective GB/s
per decode layer, and end-to-end tg128.

Hypotheses continue the H1-H4 numbering above.

### Layer 1 — reach ROCm-class effective bandwidth (target: >=530 GB/s)

**H5 — schedule gap (BEAM).** A measurable share of the gap is generic
scheduling, recoverable by search with zero code changes.
- Prediction: BEAM=2..4 improves quantized decode tok/s 10-30% over BEAM=0.
- Test: Ubuntu native boot, same GGUF, tg128 with BEAM=0/2/4; per-kernel
  GB/s via DEBUG=2.
- Confirmed if >=15% gain. Falsified if <5% — then the gap is ~all idiom (H6).

**H6 — idiom gap (fused quantized matvec).** The dominant gap is instruction
selection: dequant lowered to generic ALU instead of RDNA3 packed integer dot
(v_dot4/v_dot8), plus unfused dequant-matvec memory traffic.
- Prediction: a fused Q4_K dequant+matvec path emitting packed-dot ops on one
  representative decode layer reaches >=500 GB/s effective (within ~15% of
  llama.cpp's per-layer number measured on the same shape).
- Test: extra/q4_k_bench.py — bench the same layer shape three ways:
  current generic path, BEAM-tuned path, fused-idiom path; compare against
  llama.cpp MMVQ on identical shape/quant.
- Known unknown: tinygrad's RDNA3 renderer may need a v_dot intrinsic or
  assembly pattern added before the idiom is expressible. Scope that first;
  it is the long pole of Layer 1.
- Confirmed if >=500 GB/s. Partially confirmed if packed-dot emission works
  but GB/s stalls 30%+ below llama.cpp — then the residual is memory-pipeline
  idioms (LDS swizzle, software pipelining), iterate within H6.

**H7 — transport neutrality.** With Layer-1 kernels, the Mac/USB4 path decodes
within 10% of Ubuntu-native tinygrad on the same model.
- Prediction: TinyJit batching collapses per-token host traffic to O(1)
  submissions; USB4 latency then costs <10% at decode.
- Test: identical model both machines; DEBUG=1 roundtrips/token; tg128 ratio.
- Falsified if roundtrips/token scales with layer count after JIT warmup —
  then dispatch work (H4 territory) precedes further kernel work.

### Layer 2 — beat llama.cpp (target: >=700 GB/s effective end-to-end)

**H8 — fusion headroom.** llama.cpp's fixed kernel boundaries force VRAM
round-trips (norms, rope, residuals, attention glue) costing >=15% of decode;
whole-graph compilation can eliminate most of them.
- Prediction: bytes-moved-per-token (measurable from per-kernel DEBUG stats)
  exceeds weight-bytes by >=25% on the current path; fusing norm/rope/residual
  into matvec prologues/epilogues brings it within 10% of weight-bytes, and
  end-to-end tok/s then exceeds llama.cpp on the same model.
- Test: instrument bytes/token before and after enabling/adding fusions;
  end-to-end tg128 vs the ROCm baseline.

**H9 — layout freedom.** GGUF's fixed block layout is suboptimal for RDNA3
wavefront coalescing; re-tiling weights at load (fork-only freedom; llama.cpp
cannot) buys an additional 5-10%.
- Prediction: an A/B of GGUF-native vs re-tiled layout on the H6 fused kernel
  shows >=5% GB/s improvement.
- Test: q4_k_bench layout variants on the same layer.

### Sequencing and gates

1. ROCm baseline (in progress) — converts all targets above from estimates
   to measured numbers; recalibrate H5-H9 predictions when it lands.
2. Per-kernel gap audit: DEBUG=2 decode profile on Ubuntu-native tinygrad;
   rank kernels by (bytes x deficit). Apportions the 2x between H5 and H6.
3. H5 (free) -> H6 scoping (renderer expressibility) -> H6 build -> H7 check
   on Mac -> H8 -> H9.
4. Decision rule unchanged: optimize the layer the measurement convicts.
   Each hypothesis has a falsifier; record kills in this doc, not just wins.

Risks: BEAM search wall-clock on large models (mitigate: bench single layers);
renderer work in H6 may be deeper than expected (scope before building);
H8 fusion may fight the scheduler (incremental fusions, measure each).

## ROCm baseline MEASURED (2026-06-11, Ubuntu native, Ryzen 5800X, ROCm 7.2.4, llama.cpp ac4cddeb0)

Qwen3 family, Q4_K_M (1.7B is Q8_0 — no Q4_K_M published). Effective decode
bandwidth = tg128 x file_GB. This is the finish line; tinygrad is scored
against the GB/s column, not raw tok/s (which is model-size dependent).

| model | file GB | tg128 fa0 | eff GB/s | pp512 fa0 |
|---|---|---|---|---|
| 1.7B Q8_0 | 1.70 | 230.2 | 391 | 11330 |
| 4B Q4_K_M | 2.32 | 152.8 | 354 | 4789 |
| 8B Q4_K_M | 4.68 | 101.2 | 473 | 3108 |
| 14B Q4_K_M | 8.38 | 65.8 | 551 | 1751 |
| 32B Q4_K_M | 18.40 | 30.8 | 567 | 750 |

**Effective bandwidth rises monotonically and asymptotes at ~567 GB/s (59% of
960 peak) on large models.** Small models are dispatch/overhead-bound (354-473);
large models are the true memory-bound ceiling. The clean monotonic curve is
itself the validity proof — a non-boosting card or broken build would show a
flat low ceiling. Card confirmed boosting (post-run sclk 2854 MHz).

Correction: the earlier "8B Q4_K_M should be 110-150 t/s" sanity gate was
anchored to Llama2-7B **Q4_0** published numbers. Qwen3-8B Q4_K_M is a larger,
heavier-to-dequant model; 101-106 t/s is in line, NOT a failure. Baseline valid.

Anomaly: **32B fa1 collapses to 16.8 t/s (309 GB/s) with huge variance
(+/-1.04)** vs 30.8 t/s fa0. Host RAM was nearly exhausted at capture
(free 398Mi of 31Gi); the 18.4GB model + FA buffers likely forced swap
mid-run. Use fa0 for the 32B finish line (30.8); re-run 32B fa1 with more
free RAM before trusting it. FA helps at <=14B (+2-6%), so this is setup,
not a kernel result.

### Recalibrated targets (replacing the literature estimates in H5-H9)

Finish line (llama.cpp eff GB/s, large-model regime): **~567 GB/s**.
Roofline (85-90% of 960): **~820-860 GB/s**.

- **Layer 1 target (H5+H6): tinygrad decode eff GB/s -> >=520** (within ~8%
  of llama.cpp's 567). Per-model tg128 to match within 10%: 8B>=93, 14B>=61,
  32B>=28.
- **Layer 2 target (H8+H9): >=650 GB/s** end-to-end (beat llama.cpp by ~15%
  via fusion + layout, exploiting the 567->820 headroom llama.cpp leaves).

CRITICAL NEXT MEASUREMENT: run these SAME GGUFs through tinygrad on the
Ubuntu native boot (DEV=AMD, local PCIe, no USB4). That gives tinygrad's
kernel-only effective GB/s with zero transport confound — the real H5/H6
starting point. The 18.5 tok/s figure is Mac/USB4 and conflates kernels with
transport; it cannot be the kernel baseline. Same machine, same card, same
files = pure kernel delta vs the 567 above.

## tinygrad-native MEASURED (2026-06-11, same card/models/machine, BEAM=0)

The clean kernel-only baseline (Ubuntu native, local PCIe, no USB4). The
transport theory is now dead: native tinygrad is barely faster than the
USB4 number, so the gap is KERNELS, not transport.

| model | tg tok/s | llama.cpp | gap | tg eff GB/s | llama GB/s | tg % of peak |
|---|---|---|---|---|---|---|
| 4B | 18.82 | 152.8 | 8.1x | 44 | 354 | 4.5% |
| 8B | 15.77 | 101.2 | 6.4x | 74 | 474 | 7.7% |
| 14B | 9.09 | 65.8 | 7.2x | 76 | 551 | 7.9% |
| 32B | 4.41 | 30.8 | 7.0x | 81 | 567 | 8.5% |

**The gap is ~7x, not the ~2x literature-extrapolation. tinygrad reaches
~8% of memory peak vs llama.cpp's ~59%.** (Estimate correction: the earlier
"40-70% of llama.cpp" guess was 3-4x too optimistic. Recorded as a miss.)

### Cause, verified in code (not asserted)

`tinygrad/llm/gguf.py:57-67` — Q4_K dequant (ggml_type 12) is a generic
multi-op tensor expression (stack/cat/bitwise/reshape across several passes)
returning **float32**:
`return (d * sc.unsqueeze(-1) * q - dmin * mn.unsqueeze(-1)).flatten(-2)`

This is H6 confirmed at the source: dequant is generic tensor algebra, not a
fused in-register kernel hitting RDNA3 packed-dot ops. DEBUG=2 corroborates —
individual graph chunks run 300-410 GB/s, but the model achieves only ~74-81
GB/s effective on the quantized size, i.e. far more bytes are moved than the
Q4 weights require (fp32 dequant materialization + unfused passes).

### TWO caveats before concluding 7x is fundamental

1. **BEAM=0.** This is tinygrad's known-pathological default; BEAM=2/4 kernel
   search routinely recovers 2-5x on individual kernels. A large fraction of
   the 7x may be absent search, not absent idiom. **BEAM sweep is the single
   most important next measurement** — it apportions the gap between H5
   (free) and H6 (engineering).
2. **fp32 dequant.** dequant casts to float32, not fp16/bf16; even before a
   fused kernel, dequantizing to half (or keeping quant resident and fusing)
   cuts dequant-side traffic.

### Revised next steps (supersede prior sequencing)

1. **BEAM=2 then BEAM=4 sweep** on 8B + one large model, native Ubuntu. Record
   tok/s and eff GB/s. This is the gate: if BEAM gets 8B from 15.77 toward
   ~50+, the gap is mostly schedule (H5) and large; if it barely moves, the
   gap is the fused-dequant idiom (H6) and the kernel-build work starts.
2. Confirm whether dequantized weights are materialized once vs recomputed
   per token (inspect model.py weight handling / realize). Recompute-per-token
   would be a separate, cheaper win than the fused kernel.
3. Only after BEAM is known: scope H6 (can the RDNA3 renderer emit packed-dot
   from a fused dequant-matvec pattern?).

## H6 build-order correction (2026-06-11, fact-checked)

Verified against llama.cpp ROCm data: on RDNA3/gfx1100, WMMA (matrix units)
is NEUTRAL-to-HARMFUL for inference; the standard TILE/VEC path is as fast or
faster (neutral on ROCm 6.3.1, regression on 7.2.1). Implication for H6:

- DECODE (tg128) is GEMV, memory-bound. Matrix instructions are irrelevant to
  it by construction. The decode lever is bytes-moved: kill fp32 dequant
  materialization (gguf.py casts Q4_K -> float32), fuse dequant into the GEMV,
  schedule vector loads well. This is layer-1 + fusion, not a missing matrix
  primitive — cheaper and more BEAM-reachable than the "packed-dot renderer
  patch" framing assumed.
- Matrix/dot instructions (WMMA, and possibly v_dot4 via dp4a) belong to the
  PREFILL/GEMM battle (pp512), a separate compute-bound problem.
- Open: whether llama.cpp's gfx1100 MMVQ decode kernel uses v_dot4 or plain
  vectorized FMA — unresolved by web search; read the actual kernel before
  committing any renderer work.

Revised build order: (1) eliminate fp32 dequant materialization, (2) BEAM
sweep as wall-locator, (3) inspect llama.cpp's real decode kernel to name the
residual primitive — do NOT pre-commit to a matrix/dot renderer change.

## THE OPTIMIZATION BET (2026-06-11) — H-OPT

Synthesis of all measured + fact-checked data into one falsifiable bet, with
decomposed pre/post estimates. Estimates are bounded by the 960 GB/s roofline
and decomposed from measured contributions, not hero-guessed. (Caveat: this
session's point estimates have run optimistic 3-4x; trust the DECOMPOSITION
and SCENARIOS, treat single numbers as midpoints of wide bands.)

### The bet, in one sentence

tinygrad's ~7x decode gap vs llama.cpp on gfx1100 is dominated by BYTES MOVED
(fp32 dequant materialization + unfused multi-pass dequant), NOT by a missing
hardware instruction; therefore most of it is layer-1 + fusion reachable
(cheap), with a residual idiom wall that — if it exists — is ONE templated
primitive (the proven AutoTVM/CUTLASS blend), not a research program.

### Gap decomposition (32B asymptote, measured)

| source | factor | mechanism | layer |
|---|---|---|---|
| A: bytes bloat | 4.4x | kernels run ~355 GB/s actual but model gets ~81 eff on Q4 size; fp32 dequant + unfused passes move ~4.4x the Q4 bytes | 1 + fusion |
| B: scheduling | 1.6x | kernels at 37% of peak vs llama's 59%; BEAM=0 | 1 (BEAM) |
| total | 7.0x | matches measured 567/81 | |

The decomposition is the load-bearing claim: most of the gap (4.4x) is bytes,
which is dtype + fusion, the cheap/reachable end — NOT the 1.6x scheduling that
BEAM alone addresses, and NOT a missing instruction.

### Pre / post estimates (8B Q4_K_M, decode tok/s)

| state | tok/s | % of llama (101) | eff GB/s | what it takes |
|---|---|---|---|---|
| PRE (measured) | 15.8 | 16% | 74 | BEAM=0, fp32 dequant |
| floor | ~24 | ~23% | ~110 | BEAM only (no byte fix) |
| mid | ~62 | ~61% | ~290 | BEAM + fp16 dequant + partial fusion |
| ceiling | ~100 | ~parity | ~470+ | full fused-Q4-dequant-GEMV idiom + BEAM |

Decode is transport-neutral (H7), so the SAME numbers should hold on the Mac
within ~10% — i.e. ROCm-parity-class decode over USB4 is the ceiling outcome.

### How we get there (ordered, each gates the next)

1. **BEAM=2/4 sweep** (free, native Ubuntu). Locates the wall, delivers the
   floor number. If 8B lands ~24 and flattens, confirms bytes (not schedule)
   is the gap — the central bet.
2. **Kill fp32 dequant materialization**: dequant to fp16/bf16 in gguf.py;
   confirm matvec accumulation dtype. Cheap; attacks Source A directly.
3. **Confirm dequant caching**: does tinygrad recompute dequant per token or
   once? (inspect model.py / realize). Recompute-per-token is a separate cheap
   win.
4. **Fuse dequant into the GEMV** so Q4 weights are read once in-register.
   May be reachable by BEAM's existing fusion moves (test) or need a rewrite-
   rule hint. This is the bulk of Source A and the campaign's center of mass.
5. **Residual wall only**: read llama.cpp's actual gfx1100 decode kernel
   (MMVQ / TILE-VEC) to name the exact missing primitive; add it as a
   templated primitive + BEAM tune. Do NOT pre-build this.

### Things to look into (open questions, ranked)

1. Does the matvec accumulate in fp32 or can it be fp16/bf16? (gguf.py dequant
   is fp32-out — the single most checkable byte-bloat source)
2. Is dequant materialized once or recomputed per token? (model.py weight path)
3. Is dequant->GEMV fusion in BEAM's current move set, or does it need a rule?
4. What does llama.cpp's gfx1100 decode kernel actually use — plain vectorized
   FMA, or dp4a/v_dot4? (decides whether step 5 is even needed)
5. Can tinygrad's RDNA3 renderer emit v_dot4 if step 4 says it's needed?
6. BEAM wall-clock on large models (mitigate: tune one layer, cache schedules,
   the "separation in time" resolution of the search-cost contradiction).

### Falsifiers

- If BEAM alone reaches >50% of llama (8B > ~50 tok/s), the bytes-bloat
  decomposition is wrong (scheduling was the gap) — revise.
- If killing fp32 + fusion does NOT move eff GB/s above ~150, bytes were not
  the dominant source — the wall is lower/different than the model predicts.
- If the ceiling stalls 2x+ below llama after a fused idiom, there IS a
  hardware-primitive wall (step 5 real) — the layer-2 hole is instruction-deep.
