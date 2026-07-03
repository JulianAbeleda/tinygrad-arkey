# Performance Primitive Research Principles

This document is a focused supplement to [Coding Principles](coding-principles.md).

It does not replace the core project principles. It applies only to GPU performance work,
quantized inference primitives, machine-search experiments, and kernel/codegen research
where isolated wins can easily mislead.

The core principles still govern the work:

- centralize authority
- modularize execution
- abstract for simplicity
- keep concerns orthogonal
- encode invariants
- contain dangerous power
- test behavior at the boundary

This supplement explains how those rules apply when the work is performance-primitive
research instead of ordinary product code.

## Core Rule

A performance primitive is not just an operation.

A performance primitive is:

```text
math
+ data layout
+ activation format
+ memory path
+ work decomposition
+ reduction strategy
+ compiler lowering
+ scheduling/register behavior
+ integration boundary
```

Do not declare a primitive understood until all required parts are included in the
measurement.

## Measure The Whole Primitive

Isolated kernels are useful for diagnosis, not final truth.

A benchmark is incomplete if it excludes required work such as:

- activation quantization
- q8 packing
- score materialization
- dequant/unpack work
- scale/min decode
- partial reductions
- graph/JIT dispatch behavior
- data layout conversion
- quality loss

Examples from this repo:

- `READRAW` reached llama-class bandwidth because it skipped dequant/unpack.
- `sudot4` improved the Q4_K kernel but lost whole-linear once q8 pack was included.
- TC prefill attention won in isolation but failed in-model because score materialization dominated.
- ring overlap worked in microbenchmarks but decode remained HBM-bound.
- speculative decode acceptance was strong but the runtime loop made the real path slower.

Rule:

```text
If required work is excluded, call the result diagnostic, not a win.
```

## Prefer In-Model Gates Over Proxy Wins

The final gate for runtime performance work is the real model path.

Use this order:

1. source/render audit
2. microkernel diagnosis
3. isolated role benchmark
4. whole-linear or whole-primitive benchmark
5. in-model W==D benchmark
6. quality/correctness gate
7. default decision

Do not skip directly from a microkernel win to a model route.

A path may ship only when the relevant boundary passes:

- decode: in-model token/s, W==D, greedy correctness or explicit quality gate
- prefill: pp throughput, warm forward, dNLL where lossy, no decode regression
- quantized matvec: q8/dequant/pack/reduction costs included
- runtime overlap: application workload, not only independent-kernel overlap

## Separate Diagnostic, Candidate, And Shipped States

Every performance result should be labeled as one of:

- diagnostic: explains a bottleneck, not a candidate
- candidate: passes an isolated or whole-primitive gate, not model-shipped
- shipped: passes in-model gate and has fallback/control
- refuted: failed the relevant gate with a recorded reason
- deferred: promising but blocked by a specific missing capability

Do not let diagnostic artifacts become implicit authority.

## Use Reference Implementations As Oracles, Not Ceilings

llama.cpp, CUDA, ROCm, Metal, BitBLAS, Marlin, AWQ, SmoothQuant, and similar systems are
references.

They are not ceilings and not specifications.

Use them to answer:

- what math is actually performed?
- what data layout is used?
- when is activation quantized?
- how is work mapped to lanes/warps/blocks?
- where are reductions performed?
- which instructions are emitted?
- how much required work is avoided, fused, or amortized?

Do not copy a surface feature without auditing the full primitive.

Bad:

```text
llama uses dp4a, so add dp4a.
```

Good:

```text
llama uses packed q4 extract, q8 activation, signed dot4, qsum/min correction,
per-group scale, and a specific scheduler. Which of those are missing here?
```

## A Primitive Name Must Include Its Dataflow

Do not name search rows by one instruction.

Bad names:

- `dp4a`
- `TC attention`
- `ring2`
- `flash`

Better names:

- `q4k_ffn_mmvq_q8_with_pack_cost`
- `prefill_attention_tc_materialized_scores`
- `decode_attention_gqa_coop_vec`
- `q6k_lm_head_coop_k_partial_sum`
- `q4k_mmvq_sudot4_128row_with_q8_pack`

The name should make the required data movement and integration boundary visible.

## Audit Before Building Deeper

When a primitive fails, do not immediately build the next variant.

First ask which layer failed:

- math/formula
- data layout
- memory coalescing
- activation format lifecycle
- reduction strategy
- instruction selection
- compiler lowering
- register scheduling
- graph/runtime integration
- quality tolerance

Build the next probe only after the failing layer is named.

The MMVQ campaign followed this pattern:

- coalescing-bound Q6_K roles shipped
- Q4_K FFN local coop refuted
- q8/dp4a refuted until signedness was audited
- signed dot4 fixed
- 128-thread scheduler tested
- whole-linear q8 pack refuted the route
- next frontier became activation format lifecycle, not another dot kernel

## Activation Format Is Part Of The Primitive

A fast int-dot kernel is not useful if the activation format is expensive to produce.

For W4A8-style paths, include:

- q8 pack time
- q8 scale computation
- q8 layout conversion
- reuse count across linears
- quality error
- whether the graph can common or cache the packed value
- whether the previous operation can emit q8 as an epilogue or side-channel

Rule:

```text
Kernel speed minus activation-pack cost is the primitive speed.
```

If q8 pack cancels the kernel win, the target becomes activation lifecycle, not another
matvec kernel.

## Coalescing, Register Tightness, And Reduction Are Tradeoffs

Do not optimize one in isolation.

Common tradeoff:

```text
one-thread-per-row -> register-tight but uncoalesced
cooperative lanes -> coalesced but needs partial reduction and shared state
hand-tuned backend -> balances both with scheduler/register control
```

When testing a scheduler shape, record:

- threads per row
- rows per block
- K split
- reduction location
- output writes
- scale decode location
- q8 reuse
- native instruction emission
- occupancy implications

A variant that improves one side but regresses the other may be useful diagnostically but
should not be routed.

## Decode T=1 Must Preserve Parallelism

Decode has a different shape than prefill. Prefill has a multi-query/token axis. Decode usually
has `T=1`, so a primitive must create enough parallel work from other axes.

Rule:

```text
For decode T=1, a primitive must manufacture enough parallel work from KV splits and
GQA/query-head columns while preserving reuse. Fusion or LDS reuse that reduces workgroups
is harmful, even if it removes kernels or memory reads.
```

This rule was added after the llama decode audit corrected the attention hypothesis:

- llama's T=1 decode attention path is a non-WMMA vector `flash_attn_tile`, not the WMMA prefill path;
- its win comes from many KV-split parallel blocks, LDS K/V staging, GQA query-head column packing, and register
  online softmax;
- tinygrad's raw fused flash and fused LDS+GQA scalar tiles were byte-exact but slower because they collapsed the
  wrong axes or failed to preserve enough parallel work;
- prefill-proven LDS tile structure did not transfer because decode lacks the prefill multi-query axis.

Do not treat these as sufficient objectives for decode:

- fewer kernels;
- fusion alone;
- LDS reuse alone;
- GQA reuse by serializing G heads;
- beating a weak global-reread baseline.

Before timing a decode attention candidate, answer:

- how many workgroups does it create at ctx512/1024/4096?
- does it preserve query-head/GQA parallelism instead of serializing it?
- does reuse reduce redundant memory traffic without collapsing occupancy?
- is online-softmax/combine work hidden under enough parallel KV work?
- does it compare against the current winner, not a weaker baseline?

For this repo, the current decode-attention comparator is `gqa_coop_vec`, and a new vector-tile candidate must beat
that comparator before any in-model route work is justified.

```text
Apply this principle to the existing winning primitive and its split parameters before building a new hand-tile.
```

This was learned the hard way (2026-06-21): new hand-tiles (scalar fused LDS+GQA, warp-cooperative) were
byte-exact but slower than `gqa_coop_vec`'s **matmul** q·k. The principle's lever — more KV-splits — was instead
cheapest and most effective applied to the existing winner: lowering `FLASH_L` (128→64) gave more workgroups and a
~1.08× attention win @ctx1024 (passing the standalone gate), where a from-scratch tile could not. **But that win
did not promote:** whole-decode W==D was +1.8%@1024 and **−1.2%@4096** — below the ≥5% bar and regressing long
context, because tinygrad's split-combine cost caps the useful split count well below llama's many-split regime.
So: the bounded decode-tile lane is rested; the remaining lever is the full llama-style `flash_attn_tile`
lifecycle with an **efficient many-split / stream-k combine** (north-star), not another bounded tile or flag
sweep. Do not promote `FLASH_L=64` by default. See `docs/decode-vector-flash-tile-realigned-result-20260621.md`.

## Decode Attention Must Be Literature-Grounded

Do not design a decode attention candidate as "a fused tile" in the abstract.

A decode attention candidate must say which hardware-aware attention principle it implements:

- **FlashAttention / IO-aware tiling:** reduce HBM traffic by keeping attention tiles and online-softmax state in
  on-chip memory where possible.
- **Flash-Decoding / T=1 parallelism:** manufacture parallel work from KV splits because decode has no token/query
  axis to fill the GPU.
- **FlashDecoding++ / adaptive dataflow:** avoid synchronized partial-softmax overhead, avoid under-utilized flat
  GEMMs, and do not use one static dataflow for shapes with different bottlenecks.
- **FlashInfer-style phase separation:** prefill, decode, and append are different workloads. A primitive that wins
  for prefill does not automatically transfer to decode.

Relevant references:

- Dao et al., **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** (2022),
  https://arxiv.org/abs/2205.14135.
- **Flash-Decoding for long-context inference** (2023),
  https://crfm.stanford.edu/2023/10/12/flashdecoding.html.
- PyTorch, **Flash-Decoding for long-context inference**,
  https://pytorch.org/blog/flash-decoding/.
- Hong et al., **FlashDecoding++: Faster Large Language Model Inference on GPUs** (2024),
  https://arxiv.org/abs/2311.01282.
- FlashInfer, **Accelerating Self-Attentions for LLM Serving with FlashInfer**,
  https://flashinfer.ai/2024/02/02/introduce-flashinfer.html.

Rule:

```text
A decode attention candidate must be IO-aware, decode-aware, softmax-aware, dataflow-aware, resource-aware, and
comparator-aware before it is timed.
```

Before implementing a fused decode attention candidate, document:

- IO-aware plan: what HBM materialization is removed or avoided?
- decode-aware plan: where does T=1 parallelism come from?
- softmax-aware plan: where do `m`, `l`, and `acc` live, and how are partials combined?
- dataflow-aware plan: how does it avoid redundant exp/PV work and flat-GEMM underutilization?
- resource-aware plan: expected LDS bytes, VGPRs, workgroups, occupancy, and vector/dot instructions.
- comparator-aware plan: why it should beat `gqa_coop_vec`, not only a weaker baseline.

This section explains why recent failures were useful:

- Path A had a valid online-softmax idiom but violated the dataflow rule by recomputing exp across output lanes.
- Matmul-PV validated the tiled-PV idea at long context but exposed a symbolic-count layout/codegen blocker.
- The llama oracle validated that the target is real: a fused, LDS-staged, vectorized decode attention tile is much
  faster standalone than the current tinygrad route.

## Split-KV Reduction Economics Are Part Of The Decode Primitive

A Flash-Decoding tile manufactures T=1 parallelism by splitting the KV cache into `S` chunks (`Hkv·S` workgroups),
each writing a partial `(m, l, PV[D])`. A separate **combine** kernel then does the log-sum-exp merge. That combine
is part of the primitive — and it is a separate **lifecycle/economics tax** that a tile A/B never measures.

This was learned the hard way (2026-06-21, Route B B4): an owned hand-AMDGCN tile passed the local A/B (2.35×
GPU-busy vs `gqa_coop_vec`) and the external-kernel-as-JIT-graph-node capability, then **missed W==D** —
not from a bug, a graph-overhead defect, or a kernel mismatch (greedy byte-identical), but because the split-KV
combine gives back part of the tile win every layer.

Three facts that a tile-only benchmark hides:

- **Low combine bytes do NOT imply a cheap combine.** B4's combine moves ~0.8 MB yet costs ~12–16 µs because it is
  **latency/occupancy-bound** (~64 GB/s ≈ 6.7% of HBM peak; only 32 workgroups on 96 CUs), not bandwidth-bound.
- **The combine is a flat floor in context, scaling only with `S`** — a fixed ~12–16 µs is paid every token. As a
  share of attention it is large at short/mid context (44%/35%/26% @ctx512/1024/2048) and shrinks long
  (17% @ctx4096).
- **Amdahl can make a real local win non-promotable.** Attention is ~17% of the decode step; even a free combine
  caps the achievable W==D. A cheaper combine is the lever only when the projection shows it clears the gate.

Rule:

```text
A split-KV decode-attention candidate (KV-splits + a separate combine/reduction) must report split-KV economics
BEFORE any W==D promotion work: tile_us, combine_us, combine_fraction, combine effective bandwidth, tile/combine
workgroup counts (occupancy vs CU count), the per-ctx optimal split S, and an Amdahl projection of W==D for
measured / half / free combine. A tile A/B win alone is NOT W==D-ready.
```

The audit is `extra/qk/split_kv_economics_audit.py` → `bench/qk-split-kv-economics-audit/latest.json`
(`split_kv_economics_audit_v1`); BoltBeam owns the promotion binding/policy for this contract. It classifies each
candidate:

- `COMBINE_TAX_DOMINATES` — a cheaper/fused combine is projected to clear the gate (the actionable next lever);
- `COMBINE_SMALL_AMDAHL_LIMIT` — even a free combine cannot clear it (attention's Amdahl share is the ceiling;
  attack FFN/GEMV, not the combine);
- `POLICY_ONLY` — a ctx-gated opt-in already clears the gate (route policy is the lever);
- `MEASUREMENT_UNSTABLE` — no trustworthy W==D anchor; tighten the harness first.

B4 classifies `COMBINE_TAX_DOMINATES`: the combine is latency-bound and a cheaper combine is projected to move
ctx4096 from +5.41% (measured) to ~+7.0% (half) / ~+8.6% (free), so the next bounded lever is a cheaper combine
(B5), not another tile. See `docs/split-kv-economics-audit-result-20260621.md` and
`docs/b4-split-kv-combine-tax-result-20260621.md`.

When a combine optimization is scoped from this audit, its target must include margin over the W==D gate. A
`half-combine` projection that lands around the +7% ctx4096 threshold is **not enough margin** for promotion once
model noise, integration overhead, and policy constraints are included. For B4/B5-class split-KV routes:

```text
combine <= 8 us is a useful diagnostic/local gate, but borderline for promotion.
combine <= 6-7 us is the preferred W==D target.
combine ~= 5 us is the stretch target that gives real gate margin.
```

Do not start the W==D promotion loop for a cheaper-combine variant unless local attribution shows it plausibly reaches
the preferred target at the operative long-context split (`S≈56-64`) and preserves correctness.

## Name The Primitive Class: llama-style, vLLM-style, Silicon-style, DeepSeek-style

This repo uses multiple reference families. Name which one a candidate is borrowing from.

They are complementary, not mutually exclusive:

```text
llama-style primitive quality
+ vLLM-style lifecycle/search system
+ silicon-style hardware/compiler co-design awareness
+ DeepSeek-style lowest-layer escape hatches
```

### llama-style local primitive

A llama-style primitive is a local-execution primitive optimized for low-latency, small-batch decode. For this repo,
the key case is single-stream `T=1` decode on Qwen3-8B-Q4_K_M.

Use this name for candidates that target:

- tight per-token latency;
- direct local decode rather than serving throughput;
- high-occupancy work decomposition when the token axis is gone;
- KV-split parallelism, GQA/query-head column packing, LDS/SRAM reuse, and efficient partial combine;
- exact whole-model W==D token/s as the authority.

Relevant references:

- FlashAttention defines the IO-aware attention principle: tile attention to reduce HBM traffic by using on-chip
  memory deliberately. See Dao et al., **FlashAttention: Fast and Memory-Efficient Exact Attention with
  IO-Awareness** (2022), https://arxiv.org/abs/2205.14135.
- Flash-Decoding states the decode-specific split/combine idea: split K/V loading across parallel workers, then
  rescale and combine partial attention outputs. See **Flash-Decoding for long-context inference** (2023),
  https://crfm.stanford.edu/2023/10/12/flashdecoding.html.
- FlashDecoding++ reinforces that decode performance depends on staged pipeline/dataflow choices and hardware-aware
  adaptation, not only fewer kernels. See Hong et al., **FlashDecoding++: Faster Large Language Model Inference on
  GPUs** (2024), https://arxiv.org/abs/2311.01282.
- llama.cpp is the implementation oracle for the current local decode comparator. Its exact source behavior must be
  audited because it is not fully specified by a paper. In this project, the measured llama decode path is a non-WMMA
  vector `flash_attn_tile`, not the prefill WMMA path.

Rule:

```text
If the candidate is trying to close the single-stream decode gap, call it llama-style and prove it preserves
T=1 parallelism against the current tinygrad winner.
```

### vLLM-style lifecycle primitive

A vLLM-style primitive is not just a kernel body. It is a serving/lifecycle primitive: how requests, KV cache,
candidate routes, batching, graph capture, and policy interact so the system can choose and maintain fast paths.

Use this name for candidates that target:

- paged/block KV cache layout and prefix/cache sharing;
- continuous or iteration-level batching;
- route/backend selection by workload shape;
- graph capture/replay and scheduling policy;
- candidate registry, evaluator contract, search loop, promotion/pruning rules;
- serving throughput or maintainable route selection rather than one isolated kernel.

Relevant references:

- vLLM/PagedAttention defines the block-paged KV cache primitive and serving system built around it. See Kwon et al.,
  **Efficient Memory Management for Large Language Model Serving with PagedAttention** (2023),
  https://arxiv.org/abs/2309.06180.
- Orca defines iteration-level scheduling and selective batching for generative-model serving. See Yu et al.,
  **Orca: A Distributed Serving System for Transformer-Based Generative Models** (OSDI 2022),
  https://www.usenix.org/conference/osdi22/presentation/yu.

Rule:

```text
If the candidate changes how work is admitted, routed, cached, batched, evaluated, promoted, or pruned, call it
vLLM-style and evaluate the full lifecycle, not only the kernel.
```

### Silicon-style hardware/compiler co-design primitive

A silicon-style primitive treats hardware, compiler, runtime, memory hierarchy, and serving policy as one design
surface. This repo is not building a chip, but it should still learn from systems where the accelerator is designed
around the workload instead of treating hardware as a fixed black box.

Use this name for candidates that target:

- hardware-native tensor/dataflow shapes, not only source-level kernels;
- compiler lowering that owns layout, tiling, and movement across memory levels;
- runtime/graph scheduling that matches the accelerator's execution model;
- observability hooks that expose the real hardware bottleneck;
- route policies that change because the hardware prefers a different batch, tile, or communication shape.

Relevant references:

- AWS Trainium/Inferentia are exposed through Neuron, a stack containing compiler, runtime, profiling/debugging, and
  framework integrations. See **AWS Neuron SDK**,
  https://aws.amazon.com/ai/machine-learning/neuron/.
- Google TPUs are ASICs designed for ML matrix workloads and are accessed through compiler/runtime stacks such as
  JAX, PyTorch, and Cloud TPU infrastructure. See **TPU architecture**,
  https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm.
- Google's original TPU architecture writeup describes the systolic array as the central hardware primitive for dense
  matrix work. See **An in-depth look at Google's first Tensor Processing Unit**,
  https://cloud.google.com/blog/products/ai-machine-learning/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu.

Rule:

```text
If the candidate depends on hardware execution structure, name the hardware/compiler contract explicitly instead of
pretending the source kernel alone owns performance.
```

### DeepSeek-style lowest-layer escape hatch

A DeepSeek-style primitive is used when the standard framework/library abstraction does not expose the control needed
to use the hardware efficiently. The response is not to handwave "bypass CUDA" or abandon libraries everywhere. The
response is to identify the exact missing control surface, drop to the lowest responsible layer for that primitive,
and keep the result behind a measured lifecycle gate.

The common public claim that DeepSeek simply "did not get NVIDIA libraries" is too loose. The DeepSeek-V3 report says
they trained on NVIDIA H800 GPUs, built their own HAI-LLM framework, used custom all-to-all communication kernels,
customized PTX instructions, FP8 fine-grained quantization, and overlapped computation with communication. The point is
not no-library purity; the point is owning the layer where the default stack was insufficient.

Use this name for candidates that target:

- custom low-level instructions or backend-specific assembly/IR when the library path cannot express the needed
  schedule;
- explicit SM/work partitioning between compute and communication;
- custom collective/dispatch/combine kernels tied to routing and topology;
- precision or accumulation behavior not supported by standard kernels;
- online quantization, dequantization, or format movement as part of the primitive;
- algorithm/framework/hardware co-design where routing, data format, and kernel body are inseparable.

Relevant references:

- DeepSeek-V3 reports an H800 training cluster, an in-house HAI-LLM framework, DualPipe computation/communication
  overlap, and custom all-to-all kernels. See **DeepSeek-V3 Technical Report**, Sections 3.1-3.2,
  https://arxiv.org/html/2412.19437v1.
- The same report says their communication kernels use customized PTX instructions and autotuned communication chunk
  size to reduce L2 usage and interference with compute kernels. See Section 3.2.2,
  https://arxiv.org/html/2412.19437v1.
- The FP8 section describes fine-grained quantization, higher-precision accumulation by promotion to CUDA cores, and
  online quantization as part of the training primitive. See Section 3.3,
  https://arxiv.org/html/2412.19437v1.

Rule:

```text
If a library/framework path hides the control surface that determines performance, define a DeepSeek-style escape
hatch: exact missing control, lowest layer needed, correctness/quality gate, and default-off lifecycle boundary.
```

### Combined target for this project

The project is complete only when the relevant classes work together:

```text
llama-style decode primitive:
  beats the local llama.cpp W==D reference on the target benchmark

vLLM-style lifecycle/search primitive:
  finds, gates, routes, and preserves that win through a closed evaluator/search loop

silicon-style co-design awareness:
  explains which hardware/compiler/runtime contract the primitive relies on

DeepSeek-style escape hatch:
  uses lower-level control only when the normal stack cannot express the winning primitive
```

Do not collapse these categories:

- A llama-style kernel win without lifecycle search is a hand patch, not the project goal.
- A vLLM-style search loop without a llama-beating primitive is infrastructure, not completion.
- A serving-throughput win does not automatically solve single-stream `T=1` decode.
- A local decode win does not automatically solve paged KV, batching, or policy.
- A hardware-aware explanation is not enough unless the compiler/runtime path can use it.
- A low-level escape hatch must be justified by a missing control surface, not by preference for hand-written code.

## Value Semantics Beat Source Emission

A lowering test must validate computed values, not just generated source or disassembly.

This matters for architecture-specific instructions.

Example from this repo:

- a helper labeled signed dot4 emitted a native dot4-looking instruction
- source/disassembly checks passed
- value tests later proved it behaved unsigned
- several conclusions were distorted until value-level tests were added

Rule:

```text
Every new lowering or intrinsic needs value tests with signed, unsigned, negative, zero,
and edge-case lanes.
```

## Harnesses Are Performance Primitives Too

A benchmark harness is part of the primitive. If the harness changes the lifecycle, clock state, comparator, compile
tax, quality gate, or dispatch path, it changes the result.

This repo treats harness design as a first-class primitive because several wrong conclusions came from weak harnesses:

- the prefill "no e2e change" result was a short-prompt harness bug;
- the bare `87.6` decode number was ambiguous until context/unit authority was reconciled;
- auto-clock per-kernel timing was volatile even when whole-decode W==D was stable;
- q8 looked unstable until the evaluator used the clock-controlled lane;
- local raw-dispatch timing was misleading until throughput and graph/JIT authority were separated.

Relevant references:

- **MLPerf Inference Benchmark** defines inference benchmarking around representative workloads, quality targets,
  comparable scenarios, and reproducible submissions. See Reddi et al., **MLPerf Inference Benchmark** (2019),
  https://arxiv.org/abs/1911.02549.
- **Methodological Principles for Reproducible Performance Evaluation in Cloud Computing** emphasizes preserving
  experiment artifacts and environment details because complex systems cannot be described by a single timing number.
  See SPEC Research Group (2019),
  https://research.spec.org/fileadmin/user_upload/documents/rg_cloud/endorsed_publications/SPEC_RG_2019_Methodological_Principles_for_Reproducible_Performance_Evaluation_in_Cloud_Computing.pdf.
- **Ansor / TVM AutoScheduler** grounds the generate -> measure -> feedback loop: generate candidate programs,
  measure on real hardware, and feed the results back into the search. See Zheng et al., **Ansor: Generating
  High-Performance Tensor Programs for Deep Learning** (2020), https://arxiv.org/abs/2006.06762, and the TVM
  auto-scheduler overview, https://tvm.apache.org/2021/03/03/intro-auto-scheduler.
- **Triton** grounds the idea that high-performance GPU kernels need an explicit tiled programming model plus
  autotuning, not only ordinary tensor expressions. See Tillet et al., **Triton: an intermediate language and
  compiler for tiled neural network computations** (2019),
  https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf.

Rule:

```text
A performance claim is only valid when the evaluator captures workload, comparator, correctness/quality, timing
authority, environment, repeats/noise, candidate metadata, and promotion policy.
```

A valid benchmark artifact must include:

- workload shape and context;
- candidate id and primitive class;
- comparator id and why it is the current winner;
- exact command and env;
- git commit and dirty status;
- hardware and clock/perf state;
- warmup/compile handling;
- repeats, median, spread, and noise/reproducibility band;
- correctness or quality gate;
- local diagnostic timing vs in-model W==D authority;
- pass/fail threshold;
- final verdict and stop reason;
- links to ledger/refutation rows.

Do not promote from:

- DEBUG/print timing alone;
- PROFILE-on timing as the final authority;
- one-off raw-dispatch timings when the real route uses a graph/JIT lifecycle;
- results without a comparator;
- results without correctness/quality;
- results whose margin is inside the noise band.

## Machine Search Is Generate, Evaluate, Prune, And Remember

Machine search is not "try random kernels."

The closed-loop shape is:

```text
template space
-> generated candidate
-> structural/policy pruning
-> reproducible evaluator
-> artifact
-> lifecycle verdict
-> ledger/refutation
-> next candidate
```

Ansor's lesson is that search needs a candidate space, real hardware measurements, and feedback. MLPerf's lesson is
that measurements need workload, scenario, and quality authority. Triton's lesson is that the search space must expose
hardware-relevant tiling/dataflow knobs.

For this repo, a machine-search system must have:

- template definitions, not only prose scopes;
- generated candidate specs;
- closed-lane pruning before benchmarking;
- evaluator bindings;
- machine-readable artifacts;
- policy/default rules;
- refutation memory;
- a path from local A/B to W==D promotion.

Rule:

```text
If a candidate cannot be generated, evaluated, pruned, and remembered, it is still a manual experiment, not a
machine-search row.
```

## Machine Search Needs Rows, Not Wishes

A machine-search row must state:

- primitive name
- current implementation
- reference implementation
- required dataflow
- legal knobs
- correctness/quality gate
- isolated gate
- in-model gate
- expected Amdahl impact
- known refutations
- required fallback

Do not search a space until the row names the full primitive boundary.

Example shape:

```json
{
  "primitive": "q4k_ffn_mmvq_sudot4_with_q8_lifecycle",
  "phase": "decode",
  "required_work": ["q8_pack", "packed_q4_extract", "signed_dot4", "qsum", "scale_epilogue"],
  "ship_gate": "in-model W==D >=5% and quality accepted",
  "kill_gate": "whole-linear fails after pack cost included"
}
```

## Stop Conditions Must Match The Goal

There are two valid modes.

Shipping mode:

```text
If the scoped gate fails, stop and do not route.
```

Research mode:

```text
If the scoped gate fails, identify the next layer and decide explicitly whether to fund it.
```

Do not confuse the two.

A failed shipping gate does not prove the research space is exhausted.

A research continuation does not justify shipping or defaulting a failed candidate.

## Keep Fallbacks And Authority Centralized

If a performance path ships:

- route it from one authority point
- add an explicit fallback flag
- test that unsupported shapes fall back
- test that unrelated roles are untouched
- update the current-state docs/cache
- mark older diagnostics as superseded or refuted

Do not leave multiple unofficial ways to activate the same experimental path.

## Quality Is A First-Class Gate

Lossy paths need quality gates even when speed passes.

Examples:

- q8 activation paths need dNLL or equivalent validation before defaulting
- demotion policies need measured quality budgets
- fp16/TC prefill paths need dNLL, not only greedy smoke

If speed fails, quality can be skipped for that candidate because it will not ship.

If speed passes, quality is mandatory before default.

## Document Refutations As Assets

A refutation should explain:

- what was tested
- what passed
- what failed
- why the old hypothesis changed
- what should not be reopened
- what next layer, if any, remains

Refutations are part of the search map. They prevent repeated work.

## Summary

For GPU primitive research in this repo:

1. Define the full primitive boundary.
2. Include required data movement and format work.
3. Audit reference implementations before copying them.
4. Use source, value, micro, whole-primitive, and in-model gates in order.
5. Treat activation lifecycle as part of matvec primitives.
6. Do not ship lossy or proxy-only wins.
7. Separate shipping discipline from research continuation.
8. Convert findings into machine-search rows.
9. Keep authority centralized and fallbacks explicit.
10. Record refutations as durable knowledge.
11. Name whether a candidate is llama-style local primitive work, vLLM-style lifecycle work, silicon-style
    hardware/compiler co-design, DeepSeek-style lowest-layer escape hatch, or an explicit combination of these.
12. **Buffer-identity ABI rule (precompiled graph-node kernels).** Pass BUFFER-IDENTITY inputs across a precompiled
    `custom_kernel`/graph-node boundary; do NOT pass sliced/cache views when whole-buffer + in-kernel offset math is
    possible. `callify.transform_precompiled_call` force-`.contiguous()`s every input except `Ops.AFTER`/`Ops.BIND`,
    and `_precompiled_output_redirect` reads a `BUFFER`/`MULTI` with `has_buffer_identity()` directly (no copy) but
    **materializes** a `SLICE`/`RESHAPE`. A `RESHAPE` *on top of* the `AFTER` also materializes (the redirect accepts
    only `BUFFER`/`MULTI`). Bad: `cache_kv[0,layer]`, `cache_kv.reshape(flat).after(store)`. Good: whole
    `cache_kv.after(store)` (no reshape) + kernel-computed K/V offsets. This single rule was the entire +13–19 %
    decode-to-parity win (2026-06-23): the "+11 % KV materialization tax" mis-diagnosed for ~10 tasks as a core
    Runtime-KV persistence problem was just the owned tile reading sliced cache views. See
    `docs/owned-tile-buffer-identity-kv-read-result-20260623.md`.
13. **Learned models propose primitive search spaces; deterministic lifecycle gates decide.** A LoRA/SFT adapter (or
    any learned policy) is a *primitive-space proposer*: it emits a bounded search spec (`SearchRow`: lane, primitive,
    hypothesis, knobs+bounds, required evidence, stop rules) — never source code and never a promotion decision. The
    deterministic runner stays the authority: harness contract, route/materialization, ISA/resource, correctness, and
    W==D/whole-prefill transfer decide every outcome. LoRA/SFT comes first (structured supervised primitive-space
    generation); RLVR/RL is deferred until the strict-JSON schema and a deterministic reward are stable and shown
    useful in shadow mode. `PRIMITIVE_SPACE_PROPOSER_NOT_KERNEL_JUDGE` / `LORA_FIRST_FOR_PRIMITIVE_SPACE_LEARNING` /
    `RLVR_DEFERRED_UNTIL_SCHEMA_AND_REWARD_STABLE` / `DETERMINISTIC_HARNESS_REMAINS_AUTHORITY`. See
    `docs/primitive-space-learning-loop-lora-first-result-20260623.md`.
