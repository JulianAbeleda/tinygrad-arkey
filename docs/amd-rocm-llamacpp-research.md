# AMD ROCm and llama.cpp research for tinygrad-arkey

Current decode decision state: see `docs/amd-decode-current-verdicts.md`. This
file is the detailed research log and includes historical hypotheses and
falsifications.

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

## Correction (2026-06-11): "fused-Q4-GEMV" is fusion+search, NOT hand-writing

Grounded in the code: the dequant is ALREADY a lazy graph designed to fuse.
- gguf.py: weights load as a LAZY ggml_data_to_tensor() expression on the raw
  Q4 bytes; only the Q4 blob is realized, not the dequant.
- model.py:131-136: the forward is jitted with the comment "we unpack the GGUF
  on the fly" — the dequant lives INSIDE the forward graph, meant to fuse into
  the matmul so Q4 weights are read once and dequant'd in-register.

So the 4.4x byte bloat is the fusion BREAKING (complex dequant expression with
.contiguous()/transpose/stack barriers + BEAM=0 + fp32 output), not a missing
hand-written kernel. The fused-Q4-GEMV is expressible in tinygrad's EXISTING
ops (elementwise dequant + reduce matmul) — therefore reaching it is a
graph-rewrite + scheduling + search problem, i.e. the MACHINE's domain.

Three tiers of "let the machine do it" (increasing human input):
- A. BEAM on — machine searches schedules. Addresses the 1.6x scheduling.
- B. Make dequant fuse — restructure the lazy graph (fp16 not fp32, simpler
     ops, drop materialization barriers) so tinygrad's EXISTING fusion pushes
     dequant through the matmul. Still the machine (rewrite+schedule); human
     role is minimal "representation gardening", not kernel authorship.
     Captures most of the 4.4x.
- C. Hand-add a primitive (packed-dot) — only if A+B wall. Fact-checked as a
     PREFILL/GEMM concern, likely NOT needed for decode.

Key consequence for the "machine takes most" goal: for DECODE the idiom needs
NO new op, so the machine can in principle take ~all of it; the irreducible
human residue shrinks from "write a kernel" to "make the dequant fusion-
friendly + turn on search". That is the machine-maximal path and it is the
recommended strategy. The one genuine layer-2 hole (a new instruction) belongs
to prefill, a separate battle.

Revised step 4/5 of H-OPT: step 4 is "remove fusion barriers so the existing
machinery fuses dequant into GEMV" (gardening, not authoring); step 5 (hand
primitive) is decode-unlikely and prefill-only.

## Q4_K expression-vectorization probe (2026-06-11)

Scope: steps 1-3 of `docs/amd-decode-optimization-plan.md` final plan, native
Ubuntu only (`DEV=AMD`, local PCIe), no BEAM. Goal was to test the cheap
machine-side shot before adding a new primitive: can the Q4_K dequant expression
be rewritten so codegen emits wider/vectorized quant loads while remaining
bit-exact?

### Microbench harness

`extra/q4_k_bench.py` now selects representative Qwen3-8B Q4_K decode GEMV
shapes from GGUF metadata instead of guessed dimensions. Model config read from
the GGUF:

| key | value |
|---|---:|
| architecture | qwen3 |
| embedding_length | 4096 |
| feed_forward_length | 12288 |
| block_count | 36 |
| attention.head_count | 32 |
| attention.head_count_kv | 8 |

Representative Q4_K tensors/shapes:

| tensor | GEMV W shape N x K | Q4 bytes |
|---|---:|---:|
| `blk.0.ffn_gate.weight` | 12288 x 4096 | 28.31 MB |
| `blk.4.ffn_down.weight` | 4096 x 12288 | 28.31 MB |
| `blk.0.attn_q.weight` | 4096 x 4096 | 9.44 MB |
| `blk.0.attn_k.weight` | 1024 x 4096 | 2.36 MB |

Correctness gate: before timing each tensor, the active `ggml_data_to_tensor`
Q4_K path is compared bit-exact against a frozen copy of the previous Q4_K
expression in the benchmark. No timing is emitted if this fails.

### Baseline scalar path

Command:

```bash
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/q4_k_bench.py \
  /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --device AMD --all-shapes \
  --iters 5 --activation ones --format json
```

Baseline fused decode+matvec bandwidth, counted as Q4 weight bytes / kernel
time:

| tensor | kernels | ms | Q4 eff GB/s |
|---|---:|---:|---:|
| `blk.0.ffn_gate.weight` | 1 | 2.18 | 12.97 |
| `blk.4.ffn_down.weight` | 1 | 2.19 | 12.95 |
| `blk.0.attn_q.weight` | 1 | 3.46 | 2.73 |
| `blk.0.attn_k.weight` | 1 | 2.21 | 1.07 |

The decoded-fp16 matvec-only control is much faster for the large FFN shapes
(~65 GB/s by Q4-byte denominator), but it reads fp16 weights, so it is not a
viable decode path by itself.

Generated-code baseline (`DEBUG=4`) for `blk.0.ffn_gate.weight`:

- fused kernel: `r_128_32_3_16_4_2_32`
- launch bound: `amdgpu_flat_work_group_size(1, 32)`
- quant load width: scalar `unsigned char`
- activation load/store: `half4`

### Variant: `GGUF_Q4K_WIDE=1`

Change: in `tinygrad/llm/gguf.py`, the Q4_K scale bytes and quant bytes are
bitcast through `uint32` words and unpacked with vector bit operations. The
change is flag-gated with `GGUF_Q4K_WIDE=1`; baseline remains the default.

Correctness: PASS, bit-exact against the frozen benchmark reference on all
representative tensors.

Performance:

| tensor | kernels | ms | Q4 eff GB/s |
|---|---:|---:|---:|
| `blk.0.ffn_gate.weight` | 2 | 5.64 | 5.02 |
| `blk.4.ffn_down.weight` | 2 | 5.62 | 5.04 |
| `blk.0.attn_q.weight` | 2 | 5.70 | 1.66 |
| `blk.0.attn_k.weight` | 2 | 5.76 | 0.41 |

Generated-code result: NO vectorized quant loads. The renderer reconstructs
`uint32` values from scalar `unsigned char` loads, for example:

```c
unsigned char val0 = (*(data1 + ...));
unsigned char val1 = (*(data1 + ...));
unsigned int alu = (((unsigned int)(val0))<<0u) + ...
```

It also introduces a small `unsigned int*` constant-shift buffer and splits the
path into two kernels on the microbench. This is both slower and still scalar.

### Verdict

NO-GO for the expression-vectorization path.

The preset acceptance gate was bit-exact correctness plus movement toward at
least 200 GB/s. The only attempted pure-expression variant was bit-exact but
regressed from ~13 GB/s to ~5 GB/s on the dominant FFN shapes and did not emit
wider loads. Full 8B decode was not run because the microbench gate failed.

Conclusion: the current gather/slice/bitcast expression is outside the useful
span of tinygrad's existing vectorization/codegen. The next step is the layer-2
work: introduce a real packed Q4_K GEMV primitive/candidate that represents
wide packed loads + dequant + dot, then let search tune around that primitive.

## Q4_K primitive scaffold probe (2026-06-11)

Follow-up after the expression-vectorization no-go. Goal: verify whether the
custom-kernel/UOp primitive path can emit wide loads at all, independent of the
current `gguf.py` Tensor expression.

Added `extra/q4_k_primitive_probe.py`, which:

1. Reads the same real Qwen3-8B Q4_K tensor metadata.
2. Creates an explicit `uint32` view of the Q4 bytes.
3. Calls a minimal UOp custom kernel over that `uint32` buffer.
4. Checks the copy is bit-exact.
5. Dumps generated code under `DEBUG=4`.

Command:

```bash
DEV=AMD DEBUG=4 PYTHONPATH=. .venv/bin/python extra/q4_k_primitive_probe.py \
  /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --device AMD \
  --tensor blk.0.ffn_gate.weight --iters 5
```

Result:

| stage | generated load | note |
|---|---|---|
| `raw_u8.bitcast(uint32).contiguous()` prep | scalar `unsigned char` loads | Tensor expression still byte-packs |
| `q4k_u32_copy_probe` custom kernel | `unsigned int val0 = (*(data1+gidx0))` | primitive path can issue 32-bit loads |

Correctness passed: copied `uint32` words equal the prepared `uint32` view.

Important implication: the primitive path is viable only if it can reinterpret
or receive the packed Q4 buffer as word-typed storage without first materializing
a scalar byte-pack Tensor kernel. A real Q4_K GEMV primitive therefore needs one
of:

- a custom lowering that treats the raw `uint8` Q4 pointer as aligned `uint32*`
  internally, or
- a GGUF/model-load representation that stores Q4_K packed buffers in a
  word-typed backing Tensor while preserving the generic `uint8` fallback.

This is the next representation boundary. The expression rewrite failed because
bitcasting inside the graph does not change load width. The custom primitive
can emit the desired load width, but only after the input storage type/path is
fixed.

### Follow-up: direct GGUF `uint32` storage path

Updated `extra/q4_k_primitive_probe.py` with `--source disk-u32`.

Instead of opening the GGUF as a `uint8` Tensor and bitcasting inside the graph,
this opens the file as `Tensor(path, dtype=dtypes.uint32)`, slices the aligned
Q4_K tensor range while it is still DISK-backed, and then copies only that slice
to AMD:

```bash
DEV=AMD DEBUG=4 PYTHONPATH=. .venv/bin/python extra/q4_k_primitive_probe.py \
  /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --device AMD \
  --tensor blk.0.ffn_gate.weight --source disk-u32 --iters 5
```

Result:

| stage | observation |
|---|---|
| DISK view | 27 MB slice at the Q4_K tensor offset, not the whole GGUF |
| DISK -> AMD copy | 27 MB copied, matching the selected tensor's packed bytes |
| raw word check | first 16 `uint32` words match file bytes |
| custom kernel | `unsigned int val0 = (*(data1_7077888+gidx0))` |
| scalar byte-pack kernel | absent |

This resolves the representation staging problem for the primitive prototype:
the first real Q4_K GEMV primitive should consume a word-typed packed buffer
created this way. The remaining work is the actual packed-load + scale/min
unpack + dot kernel and its correctness gate.

## Q4_K correctness-only GEMV primitive scaffold (2026-06-11)

Added `extra/q4_k_gemv_primitive.py` as the first step-6 primitive scaffold.
This is intentionally a correctness harness, not a speed candidate.

What it does:

1. Opens the GGUF Q4_K tensor through the direct `uint32` storage path.
2. Copies only the selected packed row slice to AMD.
3. Runs a custom UOp kernel that unpacks Q4_K scales/mins and nibbles from
   `uint32` words.
4. Computes a GEMV against fp16 activations.
5. Compares against the frozen `q4_k_reference` dequant path plus dot product.

Commands:

```bash
DEV=AMD DEBUG=2 PYTHONPATH=. .venv/bin/python extra/q4_k_gemv_primitive.py \
  /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --device AMD \
  --tensor blk.0.ffn_gate.weight --rows 2 --iters 2

DEV=AMD DEBUG=2 PYTHONPATH=. .venv/bin/python extra/q4_k_gemv_primitive.py \
  /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --device AMD \
  --tensor blk.0.ffn_gate.weight --rows 16 --iters 1
```

Results:

| rows | Q4 bytes | correctness |
|---:|---:|---:|
| 2 | 4.5 KB | max_abs `2.88486e-4` |
| 16 | 36 KB | max_abs `8.53777e-4` |

Generated-code check (`DEBUG=4`, rows=2): the custom primitive kernel
`q4k_gemv_ref_2_4096` loads Q4_K packed data as `unsigned int`, for example:

```c
unsigned int val0 = (*(data1_1152+(alu1+1)));
unsigned int val3 = (*(data1_1152+alu1));
```

The `unsigned char` loads still visible in the same debug log come from the
reference dequant path used for correctness comparison, not from the primitive.

Verdict: step 6 is partially complete. The Q4_K layout arithmetic is correct
enough to proceed, and the primitive consumes the right word-typed storage. It
is still serial per output row and therefore not a performance candidate. Next:
parallelize the reduction/work distribution and expose tunable parameters
before comparing against `extra/q4_k_bench.py`.

## Q4_K partial parallel primitive pass (2026-06-11)

Extended `extra/q4_k_gemv_primitive.py` with:

- `--mode partial`: writes per-row/per-part partial sums, then reduces them.
- `--parts N`: splits the K-block reduction into `N` partitions.
- device-time reporting from `GlobalCounters.time_sum_s`, separate from Python
  wall time.
- `--schedule auto`: leaves custom-kernel opts open for scheduler experiments.
- deterministic random fp16 activations for GEMV correctness, replacing the
  earlier all-ones vector.
- a direct unpacked-weight gate (`--unpack-check-rows`) that compares the
  primitive's decoded weights element-wise against `q4_k_reference`.

Correctness gates:

```bash
DEV=AMD DEBUG=2 PYTHONPATH=. .venv/bin/python extra/q4_k_gemv_primitive.py \
  /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --device AMD \
  --tensor blk.0.ffn_gate.weight --rows 16 --mode partial --parts 16 --iters 1

DEV=AMD DEBUG=2 PYTHONPATH=. .venv/bin/python extra/q4_k_gemv_primitive.py \
  /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --device AMD \
  --tensor blk.0.ffn_gate.weight --rows 12288 --mode partial --parts 1 --iters 3
```

Results:

| shape | mode | parts | correctness | device time | Q4 eff GB/s |
|---|---|---:|---:|---:|---:|
| 16x4096 | partial | 16 | unpack max_abs `0`; GEMV max_abs `5.90324e-4` | 0.017 ms | 2.19 |
| 12288x4096 | partial | 1 | unpack max_abs `0`; GEMV max_abs `0.00123835` | 0.732 ms | 38.67 |

Full FFN `--parts` sweep:

| parts | correctness | device Q4 GB/s | kernels |
|---:|---:|---:|---:|
| 1 | max_abs `0.00177777` | 38.58 | 1 |
| 2 | max_abs `0.00177777` | 38.80 | 2 |
| 4 | max_abs `0.00177777` | 39.00 | 2 |
| 8 | max_abs `0.00177777` | 38.82 | 2 |
| 16 | max_abs `0.00177777` | 37.82 | 2 |

Comparison anchor from `extra/q4_k_bench.py` on the same tensor:

- existing fused graph `decode_q4_k_plus_matmul` device kernel: ~0.34-0.36 ms,
  about ~80 Q4-GB/s by packed Q4 bytes.
- custom primitive partial scaffold: ~0.73 ms, about ~39 Q4-GB/s.

Correctness-gap follow-up: the remote audit was right that `x=ones` was too
weak because a sum is permutation-insensitive. The harness now uses random fp16
activations and an exact decoded-weight comparison. The direct unpack gate
passes with max_abs `0`, so the Q4_K scale/nibble ordering is now tested
directly instead of inferred from a loose GEMV tolerance.

Verdict: the first parallel primitive pass is correct but not fast enough.
Splitting K into partials does not help yet; the bottleneck is still the custom
kernel's schedule/codegen shape. Pinned custom kernels use poor row-local shape
(`flat_work_group_size(1, 1)` in the simplest generated form). `--schedule auto`
can render a better-looking row-local/upcast shape, but normal AMD compilation
fails, so broad BEAM is still premature. Next work is scheduler-safe opts or a
custom UOp shape that gets local/upcast parallelism without compile failures.

## Q4_K scheduler-safe opt sweep (2026-06-11)

Added `extra/q4_k_opt_sweep.py`, a subprocess-based sweep harness for explicit
primitive opts. Each candidate runs `extra/q4_k_gemv_primitive.py` and is
classified as:

- `pass`: unpack exact, random-GEMV correct, timed.
- `illegal-opt`: tinygrad rejected the Opt shape.
- `compile-fail`: AMD compiler failed.
- `wrong`: compiled but failed correctness.
- `error`: renderer/runtime error outside the above buckets.

Command:

```bash
.venv/bin/python extra/q4_k_opt_sweep.py \
  /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --repo /home/ubuntu/tinygrad-arkey --rows 12288 --iters 2 \
  --timeout 60 --json /home/ubuntu/q4k-opt-sweep-full.json
```

Full FFN shape results (`blk.0.ffn_gate.weight`, 12288x4096):

| candidate | status | Q4 GB/s | device ms | correctness |
|---|---|---:|---:|---|
| baseline | pass | 38.51 | 0.735 | unpack 0, GEMV `0.00123835` |
| auto | compile-fail | | | unpack 0 |
| `LOCAL:0:2` | pass | 53.43 | 0.530 | unpack 0, GEMV `0.00123835` |
| `LOCAL:0:4` | pass | 100.78 | 0.281 | unpack 0, GEMV `0.00123835` |
| `LOCAL:0:8` | pass | 182.91 | 0.155 | unpack 0, GEMV `0.00123835` |
| `LOCAL:0:16` | pass | 301.44 | 0.094 | unpack 0, GEMV `0.00123835` |
| `LOCAL:0:32` | pass | 403.64 | 0.070 | unpack 0, GEMV `0.00123835` |
| `UPCAST:0:2` | pass | 27.20 | 1.041 | unpack 0, GEMV `0.00123835` |
| `UPCAST:0:3` | pass | 33.84 | 0.837 | unpack 0, GEMV `0.00123835` |
| `UPCAST:0:4` | pass | 28.58 | 0.991 | unpack 0, GEMV `0.00123835` |
| `UPCAST:0:5` | illegal-opt | | | |
| `UNROLL:*` | compile-fail / illegal / renderer error | | | unpack 0 before failure |
| `GROUP:0:{4,8,16}` | wrong | | | GEMV error ~4.3-4.5 |
| `GROUPTOP:0:16` | wrong | | | GEMV error ~4.4 |
| `GROUPTOP:0:32` | illegal-opt | | | |
| auto-like `UPCAST:0:3 UNROLL:2:0 LOCAL:0:32` | compile-fail | | | |
| `LOCAL:0:32 UPCAST:0:3` | pass | 165.74 | 0.171 | unpack 0, GEMV `0.00123835` |
| `LOCAL:0:32 UPCAST:0:4` | pass | 110.70 | 0.256 | unpack 0, GEMV `0.00123835` |

Stable rerun of the winning candidate:

```bash
DEV=AMD DEBUG=2 PYTHONPATH=. .venv/bin/python extra/q4_k_gemv_primitive.py \
  /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --device AMD \
  --tensor blk.0.ffn_gate.weight --rows 12288 --mode partial --parts 1 \
  --opt LOCAL:0:32 --iters 10
```

Result: unpack max_abs `0`, random-GEMV max_abs `0.00123835`, device `0.077`
ms, `368.66` Q4-GB/s, one kernel.

Generated-code check (`DEBUG=4`) for `LOCAL:0:32`:

- primitive kernel: `q4k_gemv_partial_12288_4096_1`
- opts: `Opt(op=OptOps.LOCAL, axis=0, arg=32)`
- launch: `amdgpu_flat_work_group_size(1, 32)`
- packed Q4 loads: `unsigned int val0 = (*(data1_7077888+...))`

Verdict: scheduler-safe local row parallelism is found. This clears the
standalone primitive speed gate: the custom primitive now beats the existing
fused graph microbench anchor (~80 Q4-GB/s) by a wide margin on this layer. The
next risk is integration: wire this tuned primitive into `extra/q4_k_bench.py`
or an equivalent lowering flag so it is not a standalone orphan, then compare
the same tensor/activation contract before touching full decode.

Search framing update: this result fits the corrected Welder/Mirage framing.
The contribution is not "search over schedules" or "search over graph
partitions"; those exist. The useful boundary is exposing Q4_K as a verified
packed tile primitive so schedule/search systems have a representation to tune
instead of opaque scalar byte math.

## Q4_K primitive integrated into microbench (2026-06-11)

Step 10 result: the tuned primitive is no longer an orphan script. Added
`--primitive` to `extra/q4_k_bench.py`, wired to the `LOCAL:0:32` custom Q4_K
GEMV primitive under `TinyJit`. The integrated bench now uses deterministic
random fp16 activations by default, checks exact primitive unpack against
`q4_k_reference`, and checks primitive GEMV against the decoded matmul reference
before timing. Device-time bandwidth is reported when `DEBUG=2`; with lower
debug levels it prints `n/a` instead of a fake zero.

Repro note: this changes `extra/q4_k_bench.py`'s default activation from
all-ones to random. Use `--activation ones` only when intentionally reproducing
older timing tables; primitive correctness runs should keep random activations.

Wall-time command:

```bash
DEV=AMD DEBUG=0 PYTHONPATH=. .venv/bin/python extra/q4_k_bench.py \
  /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --device AMD \
  --all-shapes --primitive --iters 10 --format text
```

All primitive correctness gates passed:

| tensor | primitive unpack | primitive GEMV max_abs |
|---|---:|---:|
| `blk.0.ffn_gate.weight` | 0 | 0.00179243 |
| `blk.4.ffn_down.weight` | 0 | 0.00262928 |
| `blk.0.attn_q.weight` | 0 | 0.00239778 |
| `blk.0.attn_k.weight` | 0 | 0.00180006 |

Wall-time results with JIT replay (`DEBUG=0`, Q4 bytes / wall time):

| tensor | shape | fused graph GB/s | primitive GB/s | primitive ms |
|---|---:|---:|---:|---:|
| `blk.0.ffn_gate.weight` | 12288x4096 | 24.38 | 204.82 | 0.138 |
| `blk.4.ffn_down.weight` | 4096x12288 | 24.06 | 209.77 | 0.135 |
| `blk.0.attn_q.weight` | 4096x4096 | 8.11 | 68.42 | 0.138 |
| `blk.0.attn_k.weight` | 1024x4096 | 2.00 | 17.23 | 0.137 |

Device-time command:

```bash
DEV=AMD DEBUG=2 PYTHONPATH=. .venv/bin/python extra/q4_k_bench.py \
  /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --device AMD \
  --all-shapes --primitive --iters 3 --format text
```

Device-time results (`DEBUG=2`, Q4 bytes / GPU event time):

| tensor | shape | fused graph GB/s | primitive GB/s | verdict |
|---|---:|---:|---:|---|
| `blk.0.ffn_gate.weight` | 12288x4096 | 81.40 | 379.85 | primitive wins |
| `blk.4.ffn_down.weight` | 4096x12288 | 15.76 | 193.97 | primitive wins |
| `blk.0.attn_q.weight` | 4096x4096 | 15.56 | 176.24 | primitive wins |
| `blk.0.attn_k.weight` | 1024x4096 | 111.43 | 49.90 | fused graph wins on raw kernel time |

Verdict: the tuned primitive is integrated enough to compare under the same
tensor/activation contract and clears the microbench speed gate for the large
decode matrices. It should not be wired into model execution as a blanket
replacement yet: the small KV projection shows that the choice must be
shape-aware. Next step is a safe parameter search over primitive knobs per
shape, using subprocess classification first; broad scheduler/BEAM auto-search
remains gated by compile/fault containment.

## Q4_K shape-aware primitive policy sweep (2026-06-11)

Step 11 result: added `extra/q4_k_policy_sweep.py`, a subprocess-contained
policy harness that runs the existing integrated microbench per tensor and per
primitive candidate. It parses the same correctness-gated bench output and
chooses the primitive only when it beats the fused graph by the selected metric.
Default metric is `device_q4_eff_gbs`, so the policy is based on raw kernel
quality rather than DEBUG logging wall time.

Also changed `extra/q4_k_bench.py` to stage only the selected Q4_K tensor slice
from GGUF instead of copying the full model file to AMD before slicing. This
makes per-shape subprocess search practical and does not change the timed
matvec kernels.

Command:

```bash
DEV=AMD DEBUG=2 PYTHONPATH=. .venv/bin/python extra/q4_k_policy_sweep.py \
  /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --repo /home/ubuntu/tinygrad-arkey --iters 3 --timeout 60 \
  --json /home/ubuntu/q4k-policy-full.json
```

Candidate set: explicit local sizes `8/16/32/64`, selected `parts` values
`1/2/4`, and two combined local+upcast candidates. No broad `--schedule auto`
or BEAM was used.

Best policy by device-time Q4 bandwidth:

| tensor | shape | fused graph GB/s | best primitive | primitive GB/s | ratio | choice |
|---|---:|---:|---|---:|---:|---|
| `blk.0.ffn_gate.weight` | 12288x4096 | 81.19 | `local64_p1` | 415.86 | 5.12x | primitive |
| `blk.4.ffn_down.weight` | 4096x12288 | 15.72 | `local32_p4` | 273.19 | 17.38x | primitive |
| `blk.0.attn_q.weight` | 4096x4096 | 15.53 | `local64_p1` | 186.70 | 12.02x | primitive |
| `blk.0.attn_k.weight` | 1024x4096 | 117.89 | `local32_p4` | 59.32 | 0.50x | fused graph |

Selected candidate details:

| candidate | opts | parts | notes |
|---|---|---:|---|
| `local64_p1` | `LOCAL:0:64` | 1 | best for `12288x4096` and `4096x4096` |
| `local32_p4` | `LOCAL:0:32` | 4 | best for `4096x12288`, but still loses on `1024x4096` |
| fused graph | existing Q4_K expression path | | keep for small KV projection |

Failures were contained as normal subprocess failures. Example:
`local32_upcast3_p1` became `illegal-opt` on several shapes instead of
terminating the sweep.

Verdict: step 11 produces a usable shape policy. The next model-path work must
be selective: preserve packed Q4_K storage and dispatch the primitive only for
policy-winning shapes. A blanket replacement would regress the small KV path by
raw device time.

## Q4_K primitive model-path flag (2026-06-11)

Step 12 result: added an off-by-default real model path behind
`Q4K_PRIMITIVE=1`.

Implementation:

- `tinygrad/llm/gguf.py` now has `gguf_load_with_metadata`, returning the GGUF
  tensor table (`data_start`, tensor names/dims/types/offsets) alongside the
  normal decoded state dict for single-file GGUFs.
- `tinygrad/llm/model.py` installs `Q4KPrimitiveLinear` wrappers after normal
  state loading when `Q4K_PRIMITIVE=1` is set.
- The wrappers keep the ordinary loaded weight as fallback, but carry packed
  `uint32` Q4_K word storage for policy-selected tensors.
- Decode-vs-prefill is controlled at `Transformer.__call__`: prefill captures
  the normal fallback graph; rollout captures the primitive graph. This avoids
  branching on a symbolic token dimension inside the block.
- Packed word storage and the wrapper registry are hidden from state traversal
  with `__slots__`, so CLI parameter counting does not double-count the packed
  buffers.

Policy installed for Qwen3-style dense blocks:

| role | primitive policy |
|---|---|
| `ffn_gate`, `ffn_up` | `LOCAL:0:64`, `parts=1` |
| `ffn_down` | `LOCAL:0:32`, `parts=4` |
| `attn_q`, `attn_output` | `LOCAL:0:64`, `parts=1` |
| `attn_k`, `attn_v` | fallback fused graph |

Construction smoke:

```bash
DEV=AMD Q4K_PRIMITIVE=1 PYTHONPATH=. .venv/bin/python - <<'PY'
from tinygrad import nn
from tinygrad.llm.model import Transformer, Q4KPrimitiveLinear
m, kv = Transformer.from_gguf('/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf', max_context=8)
count = sum(isinstance(getattr(blk, name), Q4KPrimitiveLinear)
            for blk in m.blk for name in ('ffn_gate','ffn_up','ffn_down','attn_q','attn_output','attn_k','attn_v')
            if hasattr(blk, name))
params = sum(p.numel() for p in nn.state.get_parameters(m))
print(count, params)
PY
```

Result: `162` primitive linears installed; parameter count remains
`8,190,735,360` instead of double-counting packed Q4 buffers.

Short 8B decode comparison (`--warmup --benchmark 4`):

| mode | steady tok/s | note |
|---|---:|---|
| baseline | ~15.9-16.1 | existing fused graph |
| `Q4K_PRIMITIVE=1` | ~29.6-30.1 | selective primitive policy |

DEBUG=2 confirmation: flagged rollout emits real model kernels including
`q4k_gemv_partial_4096_4096_1`, `q4k_gemv_partial_12288_4096_1`, and
`q4k_gemv_partial_4096_12288_4`. This verifies step 12 is not just a standalone
microbench path.

Next: run sustained 8B decode (`--benchmark 128`) with and without the flag.
The short run shows a real full-model gain, but the decision point is sustained
tok/s and the remaining dominant kernels.

## Q4_K primitive sustained 8B decode (2026-06-11)

Step 13 result: sustained 8B decode confirms the model-path primitive gain.

Commands:

```bash
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python -m tinygrad.llm \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --warmup --benchmark 128 2>&1 | tee /home/ubuntu/tg-8b-baseline-128.log

DEV=AMD Q4K_PRIMITIVE=1 JIT=1 PYTHONPATH=. .venv/bin/python -m tinygrad.llm \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --warmup --benchmark 128 2>&1 | tee /home/ubuntu/tg-8b-q4k-primitive-128.log
```

Parsed tok/s:

| mode | samples | avg tok/s | avg last 32 | median last 32 | last |
|---|---:|---:|---:|---:|---:|
| baseline | 128 | 15.44 | 15.40 | 15.42 | 15.40 |
| `Q4K_PRIMITIVE=1` | 128 | 28.74 | 28.32 | 28.93 | 28.66 |

Full-model gain: `28.74 / 15.44 = 1.86x`.

Representative final steady lines:

- baseline: ~`64.9-65.2 ms`, ~`15.3-15.4 tok/s`, ~`75 GB/s`.
- `Q4K_PRIMITIVE=1`: usually ~`34.6-35.0 ms`, ~`28.6-28.9 tok/s`,
  ~`140-142 GB/s`, with one observed late outlier around `24 tok/s`.

Verdict: the primitive is no longer just a microbench win. It moves actual 8B
decode from ~15% of llama.cpp's cited ~101 tok/s to ~28% on this machine. The
remaining gap is still large, but the project crossed an important boundary:
the packed primitive can be wired through model execution and produce sustained
full-decode speedup.

Next: repeat on Qwen3-14B. If 14B underperforms, run a 14B-specific policy
sweep before changing the role policy.

## Q4_K primitive sustained 14B decode (2026-06-11)

Step 14 result: the 8B role policy generalizes to Qwen3-14B and produces a
large sustained full-decode gain.

Commands:

```bash
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python -m tinygrad.llm \
  --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf \
  --warmup --benchmark 128 2>&1 | tee /home/ubuntu/tg-14b-baseline-128.log

DEV=AMD Q4K_PRIMITIVE=1 JIT=1 PYTHONPATH=. .venv/bin/python -m tinygrad.llm \
  --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf \
  --warmup --benchmark 128 2>&1 | tee /home/ubuntu/tg-14b-q4k-primitive-128.log
```

Parsed tok/s:

| mode | samples | avg tok/s | avg last 32 | median last 32 | last |
|---|---:|---:|---:|---:|---:|
| baseline | 128 | 8.88 | 8.88 | 8.98 | 8.88 |
| `Q4K_PRIMITIVE=1` | 128 | 14.90 | 14.48 | 15.72 | 15.72 |

Full-model gain: `14.90 / 8.88 = 1.68x` by all-sample average. The flagged
run has visible outliers, including one late ~`10 tok/s` sample, while normal
steady samples are mostly `15.3-15.8 tok/s`.

Representative final steady lines:

- baseline: ~`111-113 ms`, ~`8.9 tok/s`, ~`79 GB/s`.
- `Q4K_PRIMITIVE=1`: usually ~`63-65 ms`, ~`15.3-15.8 tok/s`, ~`136-140 GB/s`.

Verdict: the role-based primitive policy is not just overfit to Qwen3-8B. It
moves Qwen3-14B from ~`8.9 tok/s` to ~`14.9 tok/s` average, about 22.6% of the
cited llama.cpp ~66 tok/s reference instead of ~13.5%. The remaining gap is
still specialized-kernel and graph/dispatch quality, but this is a real
end-to-end decode improvement on both target models.

## Q4_K BEAM/search containment (2026-06-11)

Step 15 result: added `extra/q4_k_beam_containment.py`, a small subprocess
harness for risky primitive scheduler/search paths. It deliberately runs the
`extra/q4_k_gemv_primitive.py --schedule auto` path with `PARALLEL=0`,
`BEAM_DEBUG=2`, `BEAM_STRICT_MODE=0`, and `BEAM_DEV_TIMEOUT=1`, classifies the
failure, then immediately runs a known-good `LOCAL:0:64` primitive kernel as a
GPU health check.

Command:

```bash
DEV=AMD PYTHONPATH=. .venv/bin/python extra/q4_k_beam_containment.py \
  /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --repo /home/ubuntu/tinygrad-arkey \
  --log-dir /home/ubuntu/q4k-beam-containment \
  --timeout 60
```

Result:

| path | status | elapsed | note |
|---|---:|---:|---|
| risky `--schedule auto` | `compile-fail` | `1.508s` | normal `tinygrad.device.CompileError`, contained in subprocess |
| health `LOCAL:0:64` | `pass` | | `0.069 ms`, `408.18 Q4-GB/s`, unpack max_abs `0` |

The harness reported `contained: true`: the risky scheduler path failed without
leaving the AMD device unusable for the subsequent known-good kernel. This does
not make broad search fast or useful yet; it only establishes a safer failure
boundary for primitive-level tuning.

I also ran a deliberately tiny graph-level BEAM probe on the existing fused
microbench path, not the full model:

```bash
DEV=AMD BEAM=1 PARALLEL=0 BEAM_DEBUG=2 BEAM_STRICT_MODE=0 DEBUG=2 \
  PYTHONPATH=. .venv/bin/python extra/q4_k_bench.py \
  /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --device AMD --tensor blk.0.attn_k.weight --iters 1 --format text
```

It completed safely and reported:

```text
BEAM_SEARCH: final tm=19.44us, applied_opts=[Opt(op=OptOps.GROUPTOP, axis=0, arg=16), Opt(op=OptOps.GROUP, axis=0, arg=0), Opt(op=OptOps.LOCAL, axis=0, arg=2)]
```

The measured fused microbench summaries from that run were:

| path | device Q4 GB/s |
|---|---:|
| decoded matmul | `13.80` |
| fused Q4_K expression + matmul | `130.49` |

Verdict: step 15 is a containment result, not a performance result. Risky
search must stay subprocessed and health-checked on native Ubuntu only. No live
BEAM/search path should be run on the Mac/TinyGPU bridge.

## Q4_K Mac/deployment boundary (2026-06-11)

Step 16 result: the deployment rule is recorded. This is not a Mac transport
neutrality benchmark; no Mac/TinyGPU run was performed in steps 11-16.

Native Ubuntu is the only approved place for search/tuning:

- primitive opt sweeps;
- shape policy sweeps;
- `--schedule auto`;
- live `BEAM=*` search;
- containment probes that intentionally compile unknown kernels.

The Mac/TinyGPU path should receive only fixed artifacts that already passed on
native Ubuntu. For the current implementation, that artifact is the explicit
role/shape policy behind `Q4K_PRIMITIVE=1`; it does not depend on live BEAM.

Allowed Mac/TinyGPU smoke command:

```bash
DEV=AMD Q4K_PRIMITIVE=1 JIT=1 PYTHONPATH=. .venv/bin/python -m tinygrad.llm \
  --model /path/to/Qwen3-8B-Q4_K_M.gguf --warmup --benchmark 4
```

Allowed sustained command after the smoke passes:

```bash
DEV=AMD Q4K_PRIMITIVE=1 JIT=1 PYTHONPATH=. .venv/bin/python -m tinygrad.llm \
  --model /path/to/Qwen3-8B-Q4_K_M.gguf --warmup --benchmark 128
```

Fallback is immediate: unset `Q4K_PRIMITIVE` and run the existing tinygrad fused
graph path.

Prohibited on Mac/TinyGPU:

- `BEAM=*`;
- `--schedule auto`;
- `extra/q4_k_policy_sweep.py`;
- `extra/q4_k_opt_sweep.py`;
- `extra/q4_k_beam_containment.py`;
- any subprocess search that compiles/runs unknown AMD kernels.

Reason: a bad search candidate has already shown it can destabilize the AMD
path. On native Ubuntu that can be contained and health-checked; over the
Mac/TinyGPU bridge it risks dropping the bridge/PCIe path. The search cost and
risk should be paid once on native Ubuntu, then the fixed result deployed later.

## Q4_K residual decode profile (2026-06-11)

Step 17 result: added `extra/q4_k_profile_report.py` and recorded residual
decode profiles in `bench/q4k-profile-20260611/`.

There are two profile modes:

- `batched`: normal `JIT=1 DEBUG=2` graph-batched runtime. This is the real
  tok/s and residual-overhead profile.
- `named`: `JIT=1 DEBUG=2 JIT_BATCH_SIZE=1`. This disables graph batching so
  DEBUG=2 exposes individual kernel names. It is attribution-only; its wall
  time includes deliberate launch overhead and is not a throughput number.

Commands used the same pattern for 8B/14B and baseline/primitive:

```bash
DEV=AMD JIT=1 DEBUG=2 PYTHONPATH=. .venv/bin/python -m tinygrad.llm \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --warmup --benchmark 32

DEV=AMD Q4K_PRIMITIVE=1 JIT=1 DEBUG=2 PYTHONPATH=. .venv/bin/python -m tinygrad.llm \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --warmup --benchmark 32

DEV=AMD JIT=1 JIT_BATCH_SIZE=1 DEBUG=2 PYTHONPATH=. .venv/bin/python -m tinygrad.llm \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --warmup --benchmark 32
```

Report generation:

```bash
PYTHONPATH=. .venv/bin/python extra/q4_k_profile_report.py \
  --out bench/q4k-profile-20260611/report.md \
  --json bench/q4k-profile-20260611/report.json \
  bench/q4k-profile-20260611/*debug2-batched.log \
  bench/q4k-profile-20260611/*debug2-jitbs1.log
```

Normal graph-batched runtime summary, steady state after dropping the first
benchmark token:

| model | mode | tok/s | wall ms/tok | AMD kernel ms/tok | residual ms/tok |
|---|---|---:|---:|---:|---:|
| 8B | baseline | `15.69` | `63.75` | `63.07` | `0.67` |
| 8B | `Q4K_PRIMITIVE=1` | `29.06` | `34.46` | `33.76` | `0.70` |
| 14B | baseline | `9.09` | `110.03` | `109.31` | `0.72` |
| 14B | `Q4K_PRIMITIVE=1` | `15.77` | `63.59` | `62.89` | `0.71` |

This confirms the primitive win under DEBUG=2 and shows the steady-state
residual is small (`~1-2%` wall), so the remaining gap is not primarily
host/dispatch/runtime overhead. No >1.5x token outliers appeared in the
32-token profile.

Named attribution summary, using `% AMD kernel` as the basis:

| model | mode | primitive GEMV | primitive reduction | generic/fallback dense Q4-style |
|---|---|---:|---:|---:|
| 8B | baseline | `0.00%` | `0.00%` | `93.40%` |
| 8B | `Q4K_PRIMITIVE=1` | `14.51%` | `1.10%` | `70.56%` |
| 14B | baseline | `0.00%` | `0.00%` | `94.69%` |
| 14B | `Q4K_PRIMITIVE=1` | `14.06%` | `9.56%` | `66.79%` |

Interpretation caveat: the named rows are attribution-only. They use
`JIT_BATCH_SIZE=1` to expose individual kernel names, which deliberately changes
wall time. Use the batched rows for throughput and residual overhead, and use
the named rows only for the `% AMD kernel` ownership split.

Parser-hardening check: after the strict classifier and the
`fallback_quant_fused` rename, these attribution numbers did not materially
move. The original claim still holds on the stricter report: primitive GEMV is
about `14%` of named AMD kernel time, and the remaining generic/fallback quant
bucket is about `71%` on 8B and `67%` on 14B.

Top remaining named kernels after the primitive flag:

| model | top remaining kernel | named ms/tok | note |
|---|---|---:|---|
| 8B | `r_32_32_4_48_2_2_2_32` | `50.61` | generic dense Q4-style kernel, not `q4k_gemv_partial_*` |
| 14B | `r_40_32_4_68_2_2_2_32` | `78.68` | generic dense Q4-style kernel, not `q4k_gemv_partial_*` |
| 14B | `r_8_32_4_20_4_2_32` | `26.07` | generic dense Q4-style kernel |

Verdict: do not start with primitive GEMV v2. The current primitive GEMV is only
~14% of named AMD kernel time, and reductions are not the dominant 8B problem.
The next target is mapping the remaining anonymous generic kernels back to
model ops and policy coverage holes, then deciding whether to extend primitive
coverage, revise the role policy, or add a fused FFN/intermediate lowering.

## Q4_K correctness and safety gates (2026-06-11)

Step 18 added `extra/q4_k_output_ab.py`, `extra/q4_k_safety.py`, guarded the
risky auto-schedule entry points, and added `Q4K_PRIMITIVE_DEBUG=1` install
diagnostics.

Greedy output A/B results:

| model | tokens | result | baseline elapsed | primitive elapsed |
|---|---:|---|---:|---:|
| Qwen3-8B-Q4_K_M | 32 | exact token match | `29.067s` | `32.727s` |
| Qwen3-14B-Q4_K_M | 32 | exact token match | `37.542s` | `41.835s` |

The harness runs baseline and primitive in separate subprocesses and compares
generated token IDs exactly. These timings include model load/JIT/wall overhead
and are not used as speed measurements.

Safety checks:

| check | result |
|---|---|
| direct primitive `--schedule auto` without override | refused before risky path |
| default opt sweep containing the `auto` candidate | refused before risky path |
| BEAM containment harness without override | refused before risky path |
| policy sweep with synthetic `PCI+AMD` device label | refused before model metadata/GPU work |
| policy sweep with synthetic `CUDA` device label | refused as non-native AMD |
| explicit fixed opt sweep candidate (`baseline`) | passes |
| fixed primitive smoke (`LOCAL:0:64`, 64 rows) | passes correctness |

Diagnostics check:

```text
Q4K_PRIMITIVE_DEBUG installed=162 skipped_total=237 not_q4_k=182 policy_fallback=55
```

Verdict: the model-level "fast garbage" risk is closed for the tested 8B and
14B greedy path, and the BEAM/auto-schedule safety rule is now enforced in code.
The remaining optimization target is still the step 17 residual: map anonymous
generic kernels back to model ops and decide whether coverage/policy expansion
or a new fused lowering is the next move.

## Q4_K profile-report hardening (2026-06-11)

Follow-up to the profiling audit: `extra/q4_k_profile_report.py` now fails loud
instead of silently producing a plausible-but-wrong report.

Changes:

- strict UTF-8 log reads; no `errors="replace"`;
- strict filename labels: basename must identify `8b`/`14b`, exactly one of
  `baseline`/`primitive`, and exactly one of `batched`/`jitbs1` or `named`;
- malformed AMD DEBUG lines and malformed token-summary lines raise immediately;
- zero-token or zero-AMD-line inputs raise immediately;
- parse-health counters are emitted into `report.md`/`report.json`;
- kernel classification is centralized in a rule table and overlapping bucket
  matches raise instead of relying on predicate order;
- `test/external/test_q4_k_profile_report.py` asserts representative kernel
  buckets, split-K reduction followups, strict labels, and parser failures.

The regenerated profile report kept the same performance conclusion. The added
guardrail is methodological: a changed log format, misnamed file, or ambiguous
kernel signature now stops the analysis instead of defaulting to a quiet bucket.

## Q6_K ffn_down primitive and residual mapping (2026-06-11)

Follow-up to the step 17/18 residual: the largest anonymous "fallback Q4-style"
kernels were misnamed by the profiler. Mapping model metadata against kernel
shapes showed the dominant residual is type-14 `Q6_K`, not missing Q4_K
coverage:

- `ffn_down.weight`: half of the Qwen3 blocks are `Q6_K` (`ggml_type=14`);
- `output.weight`: `Q6_K`, huge vocab projection;
- some `attn_v.weight`: `Q6_K`, small KV projection.

The profiler bucket is now renamed to `fallback_quant_fused`, and
`q6k_gemv_partial_*` gets its own bucket.

Scope note: `extra/q4_k_profile_report.py` is explicitly a Qwen3 8B/14B
Q4_K_M AMD `DEBUG=2` decode classifier. The dense fallback signatures are not a
general tinygrad kernel taxonomy; foreign models or devices need new boundary
tests before their bucket attribution should be trusted.

Microbench results, real shapes, random activations, DEBUG=2 device time:

| tensor | fused graph | Q6 primitive | verdict |
|---|---:|---:|---|
| 8B `blk.0.ffn_down.weight` `(4096,12288)` | `1.945 ms` | `0.319 ms` | install |
| 14B `blk.0.ffn_down.weight` `(5120,17408)` | `2.760 ms` | `0.474 ms` | install |
| 8B `output.weight` `(151936,4096)` | `4.286 ms` | `5.409 ms` | fallback |
| 8B `blk.0.attn_v.weight` `(1024,4096)` | `0.040 ms` | `0.110 ms` | fallback |

Implementation:

- added `extra/q6_k_gemv_primitive.py`, with bit-exact unpack gate and random
  GEMV correctness gate;
- added `Q6KPrimitiveLinear`, enabled only by `Q6K_PRIMITIVE=1`;
- Q6 policy is intentionally narrow: only `*.ffn_down.weight`;
- `output.weight` and `attn_v.weight` stay on the generic fused graph because
  the measured primitive loses there.

Install diagnostics on 8B with `Q4K_PRIMITIVE=1 Q6K_PRIMITIVE=1
Q4K_PRIMITIVE_DEBUG=1`:

```text
Q4K_PRIMITIVE_DEBUG installed=162 skipped_total=237 not_q4_k=182 policy_fallback=55
Q6K_PRIMITIVE_DEBUG installed=18 skipped_total=381 not_q6_k=362 policy_fallback=19
```

End-to-end correctness:

| model | mode | tokens | result |
|---|---|---:|---|
| Qwen3-8B-Q4_K_M | `Q4K_PRIMITIVE=1 Q6K_PRIMITIVE=1` | 32 | exact match vs baseline |
| Qwen3-14B-Q4_K_M | `Q4K_PRIMITIVE=1 Q6K_PRIMITIVE=1` | 32 | exact match vs baseline |

End-to-end decode, full 128-token benchmark:

| model | Q4 primitive prior | Q4+Q6 primitive | vs llama.cpp ref |
|---|---:|---:|---:|
| 8B | `~28.7 tok/s` | `58.17 tok/s` avg (`58.54` drop-first) | `~57.6%` of `101 tok/s` |
| 14B | `~14.9 tok/s` | `28.27 tok/s` avg (`28.43` drop-first) | `~42.8%` of `66 tok/s` |

Named 8B Q4+Q6 profile:

| bucket | ms/tok | % AMD kernel |
|---|---:|---:|
| `q4k_primitive_gemv` | `11.36` | `33.18%` |
| `q6k_primitive_gemv` | `5.96` | `17.41%` |
| `fallback_quant_fused` | `4.92` | `14.37%` |

Verdict: the remaining gap moved. The old dominant FFN-down fallback is fixed.
The current residual is mostly primitive quality (`q4k` + `q6k` GEMV now about
half of named AMD kernel time) plus the Q6 output projection, where this
primitive is not the right shape. Next work should tune/build v2 for primitive
GEMV and separately investigate an output-projection-specific path; do not turn
on Q6 output/attn_v with the current primitive.

## Q4+Q6 profile completion and automated primitive sweep (2026-06-11)

Executed the scoped Step 1-3 pass: complete Q4+Q6 profiling, automated primitive
knob sweep, and Q6 output projection decision. Artifacts are in
`bench/q4q6-profile-20260611/`.

Step 1 profile:

| model | mode | tok/s | AMD ms/tok | residual | key attribution |
|---|---|---:|---:|---:|---|
| 8B | batched | `58.77` | `16.32` | `4.28%` | runtime residual low |
| 8B | named | `5.42` | `33.94` | `81.63%` | Q4 primitive `33.02%`, Q6 primitive `17.50%`, fallback quant `14.34%` |
| 14B | batched | `28.79` | `34.06` | `1.98%` | runtime residual low |
| 14B | named | `4.01` | `78.11` | `68.71%` | Q4 primitive `26.53%`, Q6 primitive `12.73%`, reductions `17.80%`, fallback quant `24.13%` |

This confirms the batched residual is not the immediate bottleneck. The named
rows remain attribution-only; they justify primitive tuning, but throughput
acceptance is based on batched 128-token decode.

Step 2 was automated, not manual:

- Existing `extra/q4_k_policy_sweep.py` swept Q4 gate/up, square attn/output,
  and down representatives for 8B and 14B.
- Added `extra/q6_k_policy_sweep.py` for Q6 `ffn_down` and `output.weight`,
  with unpack and random-GEMV correctness gates preserved.

Microbench results found apparent candidates:

| area | best microbench candidate | result |
|---|---|---|
| 8B Q4 | current policies | already best in sweep |
| 14B Q4 gate/up | `parts=1 LOCAL:0:32` | faster than current in microbench |
| 14B Q4 down | `parts=2 LOCAL:0:32` | faster than current in microbench |
| Q6 `ffn_down` 8B/14B | `parts=2 LOCAL:0:32` | `201.95` / `218.13` quant-GB/s, faster than current primitive |
| Q6 `output.weight` 8B/14B | `parts=1 LOCAL:0:16` | only `1.10x` / `1.09x` over fused graph |

The diagnosis is mixed: the primitive paths are not simply saturating DRAM.
Q6 `ffn_down` best reaches only about `202-218` quant-GB/s and about
`0.49-0.53` dot TFLOP/s, so unpack arithmetic, occupancy, and reduction shape
remain plausible limiters.

Step 3 full-decode gates rejected the candidates:

| variant | 8B avg | 8B last16 | 14B avg | 14B last16 | verdict |
|---|---:|---:|---:|---:|---|
| previous Q4+Q6 policy | `58.17` | `55.98` | `28.27` | `27.80` | stable baseline |
| output enabled + sweep policies | `53.85` | `25.21` | `28.87` | `28.39` | reject, 8B collapse |
| no output + sweep policies | `59.98`, then `15.12` rerun | `54.39`, then `14.43` rerun | `27.13`, then `28.77` rerun | `17.43`, then `28.22` rerun | reject, unstable |
| reverted policy rerun | `57.45` | `54.89` | not rerun | not rerun | stable range restored |

Verdict: do not carry any runtime policy change from this sweep. Q6 output stays
fallback. Q6 `ffn_down` stays at the last stable policy (`parts=1
LOCAL:0:64`). The useful result is negative but valuable: automated knob search
found microbench wins, and full-decode gates showed they are not production
wins. Next optimization should be a real primitive-v2 design change, not just
retuning the current primitive knobs.

## Vdot premise check: roofline + llama.cpp MMVQ read (2026-06-12)

Executed the cheap gate before any renderer/core packed-dot lowering. Artifacts
are in `bench/vdot-premise-20260612/`.

Representative v1 measurements:

| model | format | tensor | policy | quant GB/s | logical TFLOP/s | memory peak |
|---|---|---|---|---:|---:|---:|
| 8B | Q4_K | `blk.0.ffn_gate.weight` | `parts=1 LOCAL:0:64` | `421.10` | `1.50` | `43.9%` |
| 8B | Q4_K | `blk.4.ffn_down.weight` | `parts=4 LOCAL:0:32` | `270.56` | `0.96` | `28.2%` |
| 8B | Q6_K | `blk.0.ffn_down.weight` | `parts=1 LOCAL:0:64` | `131.63` | `0.32` | `13.7%` |
| 14B | Q4_K | `blk.0.ffn_gate.weight` | `parts=1 LOCAL:0:64` | `360.89` | `1.28` | `37.6%` |
| 14B | Q6_K | `blk.0.ffn_down.weight` | `parts=1 LOCAL:0:64` | `154.55` | `0.38` | `16.1%` |

The accepted v1 kernels are memory/schedule-bound by roofline. Their logical
dot intensity is only about `2.4-3.6` ops per packed quant byte, far below the
RX 7900 XTX FP32 ridge point of about `64` ops/byte. Their logical dot
throughput is also only `0.3-1.5` TFLOP/s, so the remaining gap is not explained
by a saturated dot/compute pipeline.

`DEBUG=4` confirms the accepted v1 kernels do not emit `v_dot4`/`dp4a`; they use
packed Q4/Q6 loads, half activation loads, bit/nibble extraction, and scalar
fp32 accumulation. That fact alone does not justify packed-dot lowering, because
the kernels are not compute-bound.

llama.cpp was read at pinned commit
`ba1df050f3dc7827fc64936b2e24fe499c9f74eb`:

- MMVQ maps Q4_K/Q6_K to q8_1 vecdot helpers.
- The helpers call `ggml_cuda_dp4a`; on HIP RDNA3 this maps to
  `__builtin_amdgcn_sudot4(...)`.
- The activation side is staged into q8_1 before MMVQ.
- RDNA3 scheduling is type-specific rather than a generic "more warps" rule.

Verdict: llama.cpp agrees that packed dot is part of a fast design, but it is
part of a larger representation/schedule package. Do not start isolated
renderer/core `v_dot4` lowering as the next default task. If compiler research
continues, target semantic packed-layout plus schedule/codegen generation; if
local inference speed is the goal, keep the consolidated Q4/Q6 v1 path.

## Semantic stop-gated generated search (2026-06-12)

Executed the next compiler-research slice after the roofline premise check.
Artifacts are in `bench/qk-semantic-20260612/`.

Code changes:

- `extra/qk_ansor.py` now estimates each generated candidate's minimum global
  bytes, ops/byte, q8 staging bytes, and semantic stop reason.
- `--skip-stopped` skips isolated packed-dot candidates when the v1 roofline
  premise already says the shape is memory/schedule-bound.
- Runtime policy selection now separates the research winner from the
  model-supported policy winner. q8 winners can be reported without being
  emitted as a runtime policy.
- `tinygrad/llm/model.py` accepts generated Q4/Q6 primitive-family candidates
  by family (`q4_k_packed_u32`, `q6_k_packed_u16`), not only by the historical
  `v1_q*_packed` names.
- `extra/q4_k_output_ab.py` can run the greedy output A/B against a generated
  policy artifact.
- `extra/qk_profile_pmc.py` parses tinygrad AMD PMC profile events.

Full-shape generated-search highlights:

| model | tensor | research winner | runtime policy | winner GB/s | best q8 GB/s | stopped vdot |
|---|---|---|---|---:|---:|---:|
| 8B | `blk.0.ffn_gate.weight` Q4 | `q4_local64_p1` | `q4_local64_p1` | `422.31` | `243.07` | `4` |
| 8B | `blk.4.ffn_down.weight` Q4 | `q4_local32_p4` | `q4_local32_p4` | `268.74` | `257.77` | `4` |
| 8B | `blk.0.attn_k.weight` Q4 | `fused_graph` | `fused_graph` | `103.21` | `37.98` | `4` |
| 8B | `blk.0.ffn_down.weight` Q6 | `q6_local64_p2` | `q6_local64_p2` | `198.19` | n/a | `0` |
| 14B | `blk.0.ffn_gate.weight` Q4 | `q4_local32_p1` | `q4_local32_p1` | `366.16` | `318.69` | `4` |
| 14B | `blk.5.ffn_down.weight` Q4 | `q8_1_q4_intdot` | `q4_local32_p2` | `328.89` | `328.89` | `4` |
| 14B | `blk.0.attn_k.weight` Q4 | `v1_q4_packed` | `v1_q4_packed` | `64.05` | `47.86` | `4` |
| 14B | `blk.0.ffn_down.weight` Q6 | `q6_local64_p2` | `q6_local64_p2` | `212.32` | n/a | `0` |

Policy parity:

| model | generated installed | generated unsupported | effective mismatches | interpretation |
|---|---:|---:|---:|---|
| 8B full policy | `180` | `0` | `18` | generated policy changes some Q6 split choices, but not enough to win full decode |
| 14B full policy | `280` | `0` | `200` | generated policy materially expands primitive coverage |

Full decode gates:

| model | mode | avg tok/s | last64 tok/s | last16 tok/s | verdict |
|---|---|---:|---:|---:|---|
| 8B | explicit Q4/Q6 flags | `51.36` | `50.23` | `49.10` | stable baseline for this rerun |
| 8B | generated full policy | `50.94` | `46.80` | `51.89` | correct but flat; do not prefer |
| 14B | explicit Q4/Q6 flags | `23.44` | `23.11` | `22.83` | same-commit comparison |
| 14B | generated full policy | `40.50` | `39.54` | `38.62` | accepted |
| 14B | generated full policy rerun | `40.09` | `39.88` | `39.09` | accepted repeat |

Greedy output A/B:

| model | baseline | candidate | tokens | result |
|---|---|---|---:|---|
| 8B | generic fused graph | generated full policy | `32` | `match=True` |
| 14B | generic fused graph | generated full policy | `32` | `match=True` |

PMC smoke for `q4k_gemv_partial_12288_4096_1`:

| GL2 hit rate | VALU / busy | SALU / busy | SQ busy | VALU inst |
|---:|---:|---:|---:|---:|
| `0.1613` | `1.2584` | `0.0508` | `16411721` | `20653056` |

Interpretation:

- The stop gate correctly prevents another isolated vdot chase.
- The useful generated-search win is 14B Q4/Q6 coverage and split-policy
  selection using existing runtime-supported primitive families.
- q8 remains a research signal only. The one 14B q8 research winner is not a
  runtime policy until there is a q8 wrapper plus full-decode correctness and
  speed gates.
- Generated policy should stay opt-in and artifact-pinned; do not make it a
  global default.

## 14B generated-policy remeasure audit (2026-06-12)

The 14B generated-policy result was remeasured because it had plausible artifact
signals: a low explicit baseline, a surprisingly high percent of the llama.cpp
reference, and a large explicit-to-generated jump.

Artifacts: `bench/qk-14b-remeasure-20260612/`.

Repeated fresh-process decode:

| mode | runs | avg tok/s mean | range | note |
|---|---:|---:|---:|---|
| prior `c3315d6ad` explicit Q4/Q6 | 3 | `22.78` | `22.04-23.16` | prior code also low |
| current `a5ee7f65a` explicit Q4/Q6 | 3 | `23.27` | `23.18-23.36` | stable |
| current `a5ee7f65a` generated policy | 3 | `39.68` | `39.42-40.05` | stable |

This rules out the proposed explanation that the `model.py` generated-family
matching refactor regressed explicit 14B. The older `~28 tok/s` 14B number was
not reproduced on either commit in this audit.

Install/debug and policy parity:

| mode | Q4 wrappers | Q6 wrappers | total generated/explicit policy installs |
|---|---:|---:|---:|
| explicit Q4/Q6 flags | `180` | `20` | `200` |
| generated policy | `240` | `40` | `280` |

Generated policy adds real coverage and schedule changes:

- Q4 `attn_k`: 40 tensors move from fused graph to primitive;
- Q4 `attn_v`: 20 tensors move from fused graph to primitive;
- Q6 `attn_v`: 20 tensors move from fused graph to primitive;
- Q4 `ffn_gate` and `ffn_up`: 40 tensors each change `LOCAL:0:64` to
  `LOCAL:0:32`;
- Q4 `ffn_down`: 20 tensors change split from `parts=4` to `parts=2`;
- Q6 `ffn_down`: 20 tensors change split from `parts=1` to `parts=2`.

DEBUG=2 profile:

| mode | tok/s | wall ms/tok | AMD kernel ms/tok | residual ms/tok | residual |
|---|---:|---:|---:|---:|---:|
| explicit Q4/Q6 batched | `24.07` | `41.55` | `40.84` | `0.71` | `1.71%` |
| generated policy batched | `42.22` | `23.69` | `22.95` | `0.74` | `3.11%` |

Named attribution, AMD-kernel ms/tok:

| bucket | explicit | generated | movement |
|---|---:|---:|---|
| Q4 primitive GEMV | `20.63` | `21.64` | similar |
| Q6 primitive GEMV | `9.91` | `8.60` | slightly lower |
| Q4 primitive reductions | `13.86` | `1.14` | major reduction overhead removed |
| fallback quant fused | `18.75` | `5.34` | major coverage win |
| other AMD | `10.34` | `1.28` | anonymous leftovers mostly removed |

Verdict: the 14B generated policy survives the audit. The win is explained by
coverage plus schedule-policy selection over existing Q4/Q6 primitive families,
not by q8/vdot and not by residual/dispatch noise. Keep it opt-in and
artifact-pinned.
