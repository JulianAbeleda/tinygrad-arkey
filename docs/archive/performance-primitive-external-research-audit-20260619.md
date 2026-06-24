# External research audit - performance primitive lifecycle (second pass)

Purpose: consolidate the second-round arXiv/OpenReview/ChinaXiv scan into this project's primitive map. This is not a
literature review for its own sake. Each entry records the external claim, whether it appears true/relevant, how it
maps to our tinygrad/llama.cpp evidence, and whether it changes the current build queue.

Scope: LLM inference kernels and adjacent lifecycle work: quantized decode, attention, prefill/decode scheduling,
kernel generation/search, persistent/dynamic kernels, and backend portability. Date: 2026-06-19.

## Search sources checked

- arXiv / recent 2026 arXiv query set for LLM inference kernels, quantized decode, FlashAttention, serving, and
  automated kernel generation.
- OpenReview query set for LLM inference kernels, FlashInfer/attention, quantized inference, kernel agents, KV cache,
  and serving.
- ChinaXiv / Chinese-source check. `ChinaXiv` is the closest Chinese arXiv equivalent, but direct ChinaXiv searches
  for LLM inference kernels were low-signal for this topic. Highest-signal Chinese ecosystem work still appears via
  arXiv/OpenReview/company repos/docs rather than ChinaXiv.

## Applicability table

| external work | claim we care about | true / applicable here? | project consequence |
|---|---|---|---|
| FlashAttention-4, "Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling" (arXiv 2603.05451), https://arxiv.org/html/2603.05451v1 | New hardware generations require attention algorithm + kernel pipeline redesign, not only retuning old kernels. | **True, but NVIDIA/Blackwell-specific.** The general lesson applies; the exact async-MMA/tensor-memory techniques do not transfer directly to gfx1100. | Supports keeping long-prompt attention as a separate deep primitive. Does not reopen the refuted reuse-free flash-prefill kernel. |
| Event Tensor, "A Unified Abstraction for Compiling Dynamic Megakernels" (arXiv 2604.13327), https://arxiv.org/html/2604.13327v1 | Launch boundaries and inter-kernel synchronization can be first-class bottlenecks; dynamic megakernels can fuse larger lifecycles. | **Plausible and relevant after TPE-6.** Our current decode host-overhead path is refuted, but routed external prefill could expose new boundary overhead. | Add as a future audit after one-block transfer: persistent/block-level lifecycle fusion, not a current replacement for TPE-6. |
| KernelBench-X (arXiv 2605.04956), https://arxiv.org/html/2605.04956v1 | Kernel generation needs correctness + hardware-efficiency evaluation, with category-aware benchmarks. | **Applicable to process, not immediate code.** This validates our gated machine-search rows. | Use as support for turning primitive rows into an automated search harness only after the primitive boundary is explicit. |
| "Towards Automated Kernel Generation in the Era of LLMs" (arXiv 2601.15727), https://arxiv.org/html/2601.15727v3 | LLM/agent kernel generation is emerging but fragmented; feedback loops and benchmarks matter. | **Applicable as methodology.** It does not provide a ready tinygrad/AMD solution. | Reinforces: search over legal primitive knobs + measured artifacts, not free-form one-off generated kernels. |
| GPU Kernel Scientist, "An LLM-Driven Framework for Iterative Kernel Optimization" (OpenReview), https://openreview.net/pdf?id=K4XSvet59a | LLM agents can iteratively optimize kernels using hypotheses and measurement feedback. | **Applicable to future tooling.** Our current workflow is manual but structurally similar: measure, hypothesize, patch, gate. | Candidate for automating bounded search rows; not a replacement for source-of-truth docs or in-model gates. |
| CudaForge, "An Agent Framework with Hardware Feedback for CUDA Kernel Optimization" (OpenReview), https://openreview.net/forum?id=f4GtuI2blh | Hardware feedback can guide multi-agent CUDA kernel optimization. | **CUDA-specific but conceptually applicable.** Needs AMD/tinygrad metrics and stable candidate generation first. | Future machine-search harness should ingest hardware counters/device timings where available; currently gfx1100 counter support remains a blocker. |
| FlashInfer, "Efficient and Customizable Attention Engine for LLM Inference Serving" (OpenReview), https://openreview.net/pdf?id=RXPofAsL8F | Attention should be a reusable configurable inference engine across paged/ragged/speculative/prefix-sharing scenarios. | **Highly applicable to our missing attention lifecycle, less to pp512 matmul.** | Long-prompt attention audit should be framed as an attention-engine primitive, not one bespoke score-free kernel. |
| KVQuant, "Towards 10 Million Context Length LLM Inference with KV Cache Quantization" (OpenReview), https://openreview.net/pdf/14defcf80798b0426d9bd05b25ab492c11727c8a.pdf | KV cache quantization with custom CUDA kernels can extend context and speed K/V matvecs. | **Relevant but not yet in our target regime.** Our dominant pp512 frontier is matmul; long-context decode/prefill attention is deferred. Lossy quality gates required. | Add to future long-context attention/KV lifecycle audit. Do not mix into current Q4_K weight-side MMVQ conclusions. |
| FastTree, "Optimizing Attention Kernel and Runtime for Tree-Structured LLM Inference" (OpenReview), https://openreview.net/forum?id=BwvHcHZ3kJ | Prefix/tree sharing needs co-designed attention kernels and runtime grouping. | **Serving-workload applicable, not single-request benchmark applicable.** | Useful if the project pivots to shared-prefix serving; not relevant to current local single-stream tok/s gates. |
| CodeGEMM, "A Codebook-Centric Approach to Efficient GEMM in Quantized LLMs" (OpenReview), https://openreview.net/pdf?id=OH7U836jKk | Codebook quantization can avoid repeated dequant by using precomputed partial sums / code-centric GEMM. | **Potentially novel but outside current model format.** Qwen3-8B-Q4_K_M uses GGUF Q4_K/Q6_K, not a codebook format. | Research-only future row if we choose a codebook-quant model. Not applicable to current byte-identical llama/tinygrad comparison. |
| W4A8 GEMM, "Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Inference" (arXiv 2509.01229), https://arxiv.org/html/2509.01229v1 | W4A8 can be bottlenecked by dequant on CUDA cores; efficient kernels must fuse/deal with dequant and Tensor Core use. | **Directly supports our q8/MMVQ lifecycle framing.** Exact kernel is CUDA-focused; the activation-format lesson applies. | Confirms q8 side-channel is a lifecycle primitive, not a dot-instruction tweak. Does not bypass Q8L-2 producer expressibility failure. |
| Dynamic activation sparsity in quantized LLM inference (arXiv 2511.04477), https://arxiv.org/html/2511.04477v1 | Activation sparsity + special layouts/load balancing can accelerate quantized GEMV. | **Potentially applicable but lossy/dynamic and not mapped.** We have not audited activation sparsity distributions or quality. | Future optional row after q8 lifecycle, with dNLL gates and activation histogram/profiling first. Not current highest EV. |
| BitDecoding, "Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache" (arXiv 2503.18773), https://arxiv.org/html/2503.18773v1 | Low-bit KV cache can use Tensor Cores for decode attention when context is long. | **Relevant to long-context attention only.** Not a pp512 matmul answer and not a Q4_K weight MMVQ answer. | Future long-context KV/attention lifecycle audit. Requires quality gates and cache-format route. |
| POD-Attention (arXiv 2410.18038), https://arxiv.org/html/2410.18038v2 | Prefill/decode overlap can improve serving utilization because the two phases have different bottlenecks. | **Serving applicable, not current single-run benchmark authority.** | Do not use for current tok/s microbenchmarks; keep for multi-request serving roadmap. |
| RAPID-Serve (arXiv 2601.11822), https://arxiv.org/html/2601.11822v1 | Concurrent prefill/decode on the same GPU can improve SLO/throughput tradeoffs. | **Serving applicable.** Requires scheduler/runtime work and request mix; not relevant to isolated pp512/decode comparisons. | Future serving track only. |
| "Harmonizing Prefill and Decode..." (arXiv 2511.04791), https://arxiv.org/html/2511.04791v2 | Adaptive GPU scheduling/SM-resource control can reduce prefill/decode interference and launch-order stalls. | **Potentially applicable to serving; backend/runtime heavy.** Not applicable to single-request TPE-5/TPE-6. | If serving becomes a goal, add SM/resource partitioning audit. |
| "Memory-Bound but Not Bandwidth-Limited: The Physical AI Inference Gap in Batch-1 LLM Decode" (arXiv 2605.30571), https://arxiv.org/abs/2605.30571 | Batch-1 decode can be memory-dominated but still fail to scale with HBM because runtime/launch/kernel lifecycle overheads surface on faster GPUs. | **Conceptually applicable, especially for NVIDIA.** In this repo, current AMD decode host overhead was measured/refuted, but CUDA/RTX 5090 may differ. | Supports a separate NVIDIA portability audit; does not reopen AMD host-overhead-as-decode-bottleneck without new evidence. |
| TileFuse, "A Fused Mixed-Precision Kernel Library for Efficient Quantized LLM Inference" (arXiv 2606.11357), https://arxiv.org/html/2606.11357v1 | Fused unpack/dequant/execute plus layout/dataflow co-design is needed for practical low-bit LLM inference on AMD XDNA2 NPUs. | **Hardware-specific, principle applicable.** XDNA2 NPU is not gfx1100 GPU. | Supports primitive-lifecycle framing; not directly buildable here. |
| ChinaXiv, https://www.chinaxiv.org/ and repository profile https://doapr.coar-repositories.org/repositories/chinaxiv/ | China has a national preprint platform covering scientific fields in Chinese/English. | **True as a source, low signal for this niche so far.** Direct ChinaXiv searches did not surface useful LLM kernel papers. | Keep ChinaXiv as a discovery source, but prioritize arXiv/OpenReview/GitHub/company technical reports for LLM inference kernels. |

## What this changes

The external research does **not** invalidate the current local verdicts:

- bounded decode primitive space remains exhausted for this fork;
- q8/MMVQ remains deferred behind producer/codegen capability;
- pure-tinygrad prefill WMMA bounded knobs remain refuted;
- Tensile extraction remains the live prefill path after TPE-5 PASS;
- spec decode remains closed as a single-kernel shortcut.

It does add four future research categories that are underrepresented in our current docs:

1. **Long-context KV/attention lifecycle:** FlashInfer, KVQuant, BitDecoding, FastTree, and FlashAttention-4 all point
   to attention as a configurable lifecycle primitive. This matters only when the target regime makes attention large.
2. **Persistent/dynamic block lifecycle:** Event Tensor-style megakernels could matter after TPE-6 if routing overhead
   or inter-kernel boundaries eat the isolated Tensile win.
3. **Hardware-feedback machine search:** KernelBench-X / GPU Kernel Scientist / CudaForge validate our row/gate design
   but imply a future automated harness should feed back device timings/counters. The canonical principle is now the
   hardware-feedback hierarchy in `what-makes-a-performance-primitive-efficient-20260618.md`: correctness and timing
   can decide gates, while counter-free root-cause claims must be labeled as inferred. The concrete follow-up scope is
   `primitive-local-observability-search-scope-20260619.md`: build primitive-local observability/ledger tooling before
   any broad agentic search.
4. **Alternative quantization representations:** CodeGEMM and activation-sparsity work are novel, but require model
   format or quality-policy changes; they are not byte-identical paths for current Q4_K_M.

## Current priority after this audit

No priority inversion. The next local step remains:

1. **TPE-6 one-block transfer** for extracted Tensile prefill kernels, because TPE-5 predicts ~1.40x pp512 if routing
   overhead does not erase the win.
2. If TPE-6 passes, decide external-artifact policy vs codegen-transfer target.
3. Separately, scope a long-context attention/KV lifecycle audit if the benchmark target expands beyond pp512 and
   batch-1 decode.

The external papers mainly sharpen the language: the unit of optimization should be a **primitive lifecycle**, and
machine search should optimize only after the lifecycle contract is explicit.
