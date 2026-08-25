# What makes inference fast

Date: 2026-07-31
Updated: 2026-08-24 (kernel lifecycle, token lifecycle, dense-token ledger, and causal promotion loop)

This is a principles document. It is not organised around beating any particular competitor —
llama.cpp appears only as an external datapoint that validates the frame. The principles below are
irreducible: they follow from how the hardware works, and they hold on any target.

This is the canonical repo answer to **"what makes inference fast?"** It has two nested answers: first
make each selected kernel fast, then make the complete token path preserve those local gains.
`docs/beating-llama-first-principles-20260731.md` applies an earlier version of these principles to one
campaign; `docs/pure-machine-search.md` defines authorship and promotion provenance. Neither replaces this
document. New performance theory belongs here first, then campaign scopes may cite it without restating it.

Every number is cited. Numbers that are **measured** and numbers that are **projected** are marked as
such, because this codebase has repeatedly been misled by projections presented as facts (§9).

### Scope

The token ledger added below is for **dense autoregressive transformers**. Every token traverses every
layer, so each layer's weights and boundaries belong in the accounting. It must not be transferred to
mixture-of-experts, recurrent, speculative, or other conditional-compute architectures without first
rebuilding their route ledger. Those architectures can change which weights execute, how routing costs
enter the critical path, and which work may overlap.

Campaign measurements and competitor comparisons intentionally do not live in this principles section.
They belong in dated evidence and campaign ledgers, where machine state, binaries, synchronization, and
sampling protocol can be audited.

---

## 0. Fast kernels and fast tokens are two different achievements

Inference performance has two lifecycles:

```text
kernel lifecycle                         token lifecycle
----------------                         ---------------
identify one operation                   inventory the whole token route
        ↓                                        ↓
find its binding resource                rank total repeated/critical cost
        ↓                                        ↓
build and validate a local candidate     compose selected kernels and contracts
        ↓                                        ↓
prove the kernel itself is faster        remove copies, boundaries, and serial tails
        ↓                                        ↓
declare its exact output contract ─────→ prove the complete production token is faster
```

The left lifecycle creates a locally efficient primitive. The right lifecycle decides whether inference
benefits. A kernel can win in isolation while the token stays flat because its saved time is replaced by
a conversion, materialization, launch, synchronization, or slower consumer. Conversely, a modest kernel
change can create a large token win when its output contract also deletes repeated downstream work.

This distinction explains why a system may already have highly competitive arithmetic kernels and still
have a slower token path. The remaining problem is then not necessarily to invent better matrix math. It
is to stop losing that kernel speed at the joins between operations and across the repeated layer graph.

### 0.1 The kernel lifecycle: create local speed

Every kernel candidate should pass through the same lifecycle:

| stage | required question | evidence |
| --- | --- | --- |
| identify | What exact semantic operation, shapes, formats, and multiplicity does this kernel serve? | selected production route and census |
| attribute | Which resource binds: compulsory bytes, route bytes, instruction issue, compute rate, occupancy, reduction, or latency? | byte/operation ledger, profiler counters, and disassembly where needed |
| design | Which mechanism attacks that bound without moving cost elsewhere? | explicit causal claim |
| validate | Is the candidate bit-exact or within the declared numerical contract? | boundary-level exactness test |
| isolate | Does the intended mechanism make the kernel faster under representative cache and input conditions? | repeated kernel or subgraph bracket |
| contract | Does it emit the dtype, layout, ownership, and destination required by the real consumer? | producer/consumer contract audit |
| compose | Is the candidate still selected, and does its gain survive its immediate production neighborhood? | production subgraph measurement |
| promote | Does the complete token wall improve without an unexplained ledger displacement? | clean repeated end-to-end bracket |

The isolated result answers **“did we build a faster kernel?”** It does not answer **“did inference get
faster?”** Isolation is intentionally useful: it proves or rejects the local mechanism cheaply. But the
candidate remains provisional until the token lifecycle accepts it.

### 0.2 The token lifecycle: preserve and compose local speed

A token becomes faster only when the production route completes sooner. Kernel instruction count,
launch count, node count, bandwidth, and isolated microbenchmark time are evidence about that route;
none is the definition of success by itself. The working identity is:

```
device_union = node_sum - genuine_overlap
token_wall   = device_union + non_overlapped_host_and_boundary_time
```

The identity is an accounting model, not permission to subtract independently measured quantities.
`node_sum`, `device_union`, and wall must come from compatible runs and clock domains before their
difference is interpreted.

The token lifecycle is:

1. Inventory every semantic region, physical node, boundary, and repetition in the selected route.
2. Rank total critical contribution rather than isolated per-call time.
3. Insert locally validated kernels with explicit producer/consumer contracts.
4. Remove hidden copies, materializations, redundant preparation, graph breaks, and serial tails.
5. Measure device node sum, genuine overlap, device union, and production wall in compatible domains.
6. Accept only gains that remain visible at wall, then rebuild the ledger because the bottleneck moved.

The token result answers **“did the user receive the next token sooner?”** That is the authority for
inference speed.

### 0.3 The route ledger

For a conventional decoder-only dense transformer, inventory the whole production token in dependency
order. Implementations may fuse or rename rows, but no semantic work may disappear from the ledger:

| route region | semantic work to account | common speed levers |
| --- | --- | --- |
| token entry | token lookup, input movement, first normalization | retain graph-resident values; remove conversions and materializations |
| attention projections | Q, K, and V quantized matrix-vector products | reduce compulsory/route bytes; coalesce loads; raise achieved bandwidth; share activation preparation |
| position and cache production | Q/K normalization, positional transform, K/V cache writes | absorb epilogues into producers; write the final cache representation directly |
| attention | score formation, masking, reduction/softmax, value accumulation | tile for cache reuse; fuse reductions legally; avoid score/probability intermediates |
| attention output | output projection and residual update | keep the residual live; produce the consumer's declared type/layout |
| feed-forward entry | normalization and activation quantization/preparation | share one prepared activation across consumers when exact semantics permit |
| gate and up projections | two large quantized matrix-vector products | stream weights near the sustainable memory rate; pair only when shared work or topology is actually removed |
| activation and down projection | gated activation, down projection, residual update | fuse the activation/epilogue; avoid materialized hidden states and redundant header or scale work |
| token exit | final normalization, vocabulary projection, selection/sampling | eliminate serial tails; keep reductions and selection device-native where possible |
| runtime boundary | launches, graph segmentation, synchronization, dispatch, copies | capture stable topology; remove true dependency boundaries; do not count already-overlapped launches |

Each physical row in a measured ledger should record at least:

| field | question it answers |
| --- | --- |
| semantic role and multiplicity | What work is this, and how often does one token execute it? |
| selected implementation | Which production kernel or fused route actually ran? |
| input/output contract | Which dtype, shape, layout, and ownership cross the boundary? |
| compulsory bytes | What must cross the binding memory tier for any legal implementation? |
| route bytes | What additional traffic comes from rereads, intermediates, spills, or conversion? |
| body time and achieved rate | Is the row limited by bytes, instructions, occupancy, latency, or compute? |
| predecessor/successor edges | Is the row critical, removable, fusible, or able to overlap? |
| node sum, union contribution, and wall evidence | Did local movement survive composition into the token? |
| exactness evidence | Did the candidate preserve the required token semantics? |

This is the entire ledger in two senses: every semantic region is present, and every cost is assigned to
body work, transported bytes, a topology boundary, overlap, or an unaccounted residual. A faster kernel
that merely moves cost into a conversion or copy has not improved the ledger.

### 0.4 The only general ways to make a dense token faster

Every valid win reduces at least one term below without increasing another by more:

1. **Move fewer route bytes.** Keep quantized weights packed, delete intermediate write/read pairs,
   share prepared activations, and make producers emit the exact representation consumers require.
2. **Move compulsory bytes faster.** Improve coalescing, vector width, lane mapping, occupancy, and
   memory-level parallelism so a streaming row approaches the target's sustainable rate. Wider loads
   can help here even when they cannot reduce DRAM bytes; the question is whether the production row's
   achieved rate, then token wall, improves.
3. **Do less body work.** Remove redundant unpacking, address arithmetic, reductions, and repeated
   metadata work, provided that work lies on the binding path rather than behind memory latency.
4. **Delete a real boundary.** Fuse an epilogue, absorb a residual, sink a cache write into its producer,
   pair compatible producers, or remove a host/device round trip. The output contract is part of the
   optimization: an undeclared type or layout can recreate the deleted boundary as a hidden copy.
5. **Create genuine overlap.** Schedule independent work concurrently and measure the resulting union.
   A comparator's overlap is not automatically available to this graph, and overlap mass is not a pile
   of removable body time.
6. **Shorten the serial tail.** Device-native selection and sampling matter when they terminate the
   dependency chain, even if their arithmetic volume is small.

Fewer nodes are useful only when they represent less critical-path work or fewer boundaries. More nodes
can still form a faster token if their union is smaller. Likewise, an isolated bandwidth or instruction
win is only a mechanism; wall reduction is the composed result.

### 0.5 Why the two-lifecycle ledger makes progress faster

The productive loop joins the two lifecycles and is deliberately asymmetric: testing is cheap,
promotion is demanding.

1. Capture a fresh production wall and a full device ledger.
2. Rank rows by total token contribution, not by per-call ugliness or visual complexity.
3. State the candidate's accounting claim before coding: fewer bytes, higher rate, less body work,
   fewer boundaries, more overlap, or a shorter tail.
4. Run the smallest causal gate that can falsify that claim, with exactness checked at the relevant
   boundary.
5. If facts say the lever should work but the test is flat, treat the result as an information wall:
   inspect route selection, hidden materializations, output contracts, composition, synchronization,
   and clock-domain alignment before declaring the mechanism exhausted.
6. Promote only after the exact production route wins under a clean repeated wall bracket and the
   ledger explains where the time went.
7. Re-capture the entire ledger. Promotion changes topology and ranking, so yesterday's next target may
   no longer be today's next target.

This loop compounds because it prevents three expensive mistakes: optimizing a row that is not on the
critical path, abandoning a sound mechanism after testing an incomplete route, and booking a local win
whose cost reappears at the next boundary. Dated campaign documents retain the numerical history; this
document retains the general rule that made that history reproducible.

---

## 1. A token's bulk-work speed is set by two lower bounds, and the route adds boundaries

A token cannot be produced faster than the larger of the route's bulk-work bounds:

```
T_bytes,route = B_route / BW   bytes moved by the selected route, over achievable bandwidth
T_flops       = F / R          required operations, over the achievable rate of the unit
                              that performs them; for a dense transformer F ~= 2·M·P
```

`M` is the batch (tokens processed together), `P` the parameter count, `BW` achievable bandwidth, and
`R` the achievable rate of the relevant execution unit. Two byte quantities must not be conflated:

```
B_min    compulsory bytes across the binding memory tier under an ideal legal schedule
B_route  bytes actually moved across that tier by the selected route: weights, activations, materialized
         intermediates, spills, repeated reads, and output traffic
```

For weight-dominated decode, `B_min` is approximately the packed weights touched by the token. A fused
route may leave those compulsory weight bytes unchanged while reducing `B_route` by sharing activation
loads, keeping an epilogue resident, or deleting an intermediate write/read.

The bulk-work lower bound is:

```
T_bulk >= max(T_bytes,route, T_flops)
```

The complete route also has dependency boundaries that cannot always overlap with bulk work:

```
T_token >= T_bulk + T_boundary,critical
```

`T_boundary,critical` includes only launch, synchronization, dispatch, conversion, and dependency gaps on
the token's non-overlapped critical path. It is not a license to add every launch cost to a Roofline
projection; graph capture and asynchronous execution can overlap or remove some boundaries, so this term
must be measured.

Tiling, occupancy, LDS layout, instruction mapping, representation, graph topology, and boundary placement
are means of reducing `B_route`, increasing achieved `R` or `BW`, or shrinking the non-overlapped critical
path. A fast isolated kernel is therefore necessary but not sufficient for a fast token route.

---

## 2. The regime crossover, and why model size does not appear in it

Set the two bulk-work bounds equal to find the batch size where the binding constraint flips. In the
weight-dominated idealization used for this crossover, weights stored at `w` bits each contribute
`B_min,weights = P·w/8`:

```
2·M·P / R  =  P·w / (8·BW)

⇒  M* = (w / 16) · (R / BW)
```

**`P` cancels.** The crossover batch does not depend on model size at all — only on the machine's
compute-to-bandwidth ratio and the storage density of the weights.

Two consequences:

- A given machine has essentially one crossover point for a given quantisation format. It applies to an
  8B model and a 14B model identically.
- Decode (`M=1`) and prefill (`M=512`) sit on opposite sides of it by wide margins on every device
  measured here. They are not two settings of one problem; they are two different problems (§3, §4).

Computing `M*` for a target requires measured `R` and `BW` for that target. On AMD gfx1100 both exist
(§5). **On Metal, `R` is now measured (§10, 2026-07-31): ≈3.78 TFLOPS.** `BW` is still unmeasured
for Metal, so `M*` there is reported as a function of `BW` rather than a single number (§10).

## 2A. What the industry actually composes into a fast kernel

There is no single "fast GPU kernel" paper and no single optimization from which the rest follows.
Production kernels compose several independent theories. Volkov's lower-occupancy argument explains one
execution layer; it does not explain the representation, the IO lower bound, the arithmetic transformation,
or whole-route performance.

| Layer | Governing principle | What production systems do | Consequence for this repo |
| --- | --- | --- | --- |
| Workload geometry | GEMV and GEMM have different reuse | Dispatch different kernel families by `M/N/K`, batch and quant type | Decode and prefill must not share an undifferentiated search family |
| Lower bound | Roofline / arithmetic intensity | First decide whether bytes or execution throughput bind | Attribute the gap before opening a search |
| Representation | Block/scalar/affine quantization | Store low-bit weights with block metadata; sometimes quantize activations | Representation and metadata placement are part of the kernel design |
| Algebra | Distribute scales/zero-point corrections around the dot product | Accumulate integer dots and block sums; apply scales outside the element loop | Do not materialize dequantized decode weights |
| Instruction mapping | Match the encoded dot to native hardware | Use packed integer dot, MMA/WMMA, vector loads and native conversion forms | ISA availability and achieved rate are candidate facts, not GPU-name guesses |
| Memory hierarchy | Hierarchical tiling and IO-aware algorithms | Reuse through registers/shared memory; coalesce global transactions | Search legal tiles and layouts under measured resource limits |
| Latency hiding | Balance ILP and TLP; occupancy is not the objective | Unroll, prefetch, pipeline and hold independent accumulators in registers | Search NACC/unroll/warps while rejecting spills and resource cliffs |
| Reduction/writeback | Parallel reduction and cooperative layout conversion | Warp/tree reductions and coalesced epilogues | A fast inner dot with a serialized reduction is not a fast kernel |
| Graph topology and reuse | Shared producers, sole consumers and immediate reductions change the legal IO schedule | Compute compatible reductions together, retain producer state into epilogues, fold indexed outputs into their consumer reduction | Describe candidates by graph facts and account for eliminated traffic versus added live state |
| Boundary fusion | Eliminate materialization and launches where producer/consumer schedules are compatible | Fuse dequant, bias, activation, gate, normalization or attention epilogues | Rank fusion by eliminated wall/bytes, including any occupancy regression |
| Specialization | No one schedule wins every shape/device | Maintain finite variants or generate and autotune them on real hardware | Search by stable shape/quant/device facts and promote measured winners |
| End-to-end economics | Amdahl's law | Optimize the selected route, including setup, quantize, fixup and epilogue | Isolated kernel speed is diagnostic; same-run endpoint time is authoritative |

These layers are visible in current public implementations:

- NVIDIA CUTLASS maps GEMM through hierarchical threadblock/warp/instruction tiles, register/shared-memory
  reuse, software pipelining, and a cooperative/fused epilogue. Its own documentation explicitly notes
  that these kernels may accept lower occupancy because register tiling and double buffering provide the
  required concurrency: [CUTLASS efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html).
- FlashAttention changes the algorithm's IO complexity by tiling exact attention so intermediates remain
  on-chip rather than being materialized to HBM: [FlashAttention](https://arxiv.org/abs/2205.14135).
- TensorRT uses weight-only INT4 kernels that read compressed weights and dequantize them in the compute
  path, and TensorRT-LLM performs post-load fusion of quantized linears, MoE, normalization, activation and
  RoPE patterns: [TensorRT weight-only quantization](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html),
  [TensorRT-LLM fusion](https://nvidia.github.io/TensorRT-LLM/features/auto_deploy/transforms/post_load_fusion.html).
- Triton exposes tile sizes, group order, stages and warp count as shape-keyed autotune parameters, while
  keeping the accumulator resident for a fused activation:
  [Triton matrix multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html).
- TVM Ansor/MetaSchedule construct or explore schedules and measure them on real hardware rather than
  treating a hand-selected tile as universal: [Ansor](https://www.usenix.org/system/files/osdi20-zheng.pdf),
  [MetaSchedule](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html).

This industry comparison does **not** imply that this repo should copy any one implementation. It says the
same constraints recur across independent systems: control bytes, map to native instructions, create enough
legal concurrency, avoid unnecessary boundaries, and measure the complete selected route.

### 2A.1 Graph topology is a performance input, not a model name

Roofline analysis of one contraction does not determine the best schedule for a chain of contractions. The
producer/consumer graph can expose reuse and remove legal external traffic even when the underlying matmul
primitive is unchanged. Three recurring motifs are:

```
shared-input multi-reduction
    a = reduce(Wa, x)
    b = reduce(Wb, x)             -> load/stage x once, carry two accumulator sets

producer with a sole pointwise consumer
    a = reduce(W, x)
    y = epilogue(a)               -> apply the epilogue before writing a

indexed producers with an immediate reduction
    z[e] = reduce(W[e], x[e])
    y = sum_selected(z[e])        -> accumulate selected producer outputs directly into y
```

The portable opportunity is identified by graph facts: common inputs, consumer count, indexed axes,
selection cardinality, reduction algebra, and legal reduction order. `w1+w3`, `top_k=6`, a particular
quant type, or a model name are specialization facts, not the principle itself.

One semantic transformation may still require different schedule families:

- Decode may use a paired GEMV with a fused activation, followed by a direct small-`top_k` down reduction.
- Prefill may group token/expert pairs by expert, use a tiled paired GEMM, and select a different segmented
  reduction strategy.
- A large batch or wide accumulator set may make separate vendor GEMMs faster because fusion loses
  occupancy, spills, duplicates work, or prevents the best matrix-unit tile.

The profitability test is therefore not "fewer kernels." A dimensionally valid first-pass ledger converts
each claimed saving or cost to time at the relevant measured ceiling:

```
benefit_time ~= saved_binding-tier_bytes / BW
              + saved_operations / R
              + saved_non-overlapped_boundary_time

cost_time    ~= added_binding-tier-bytes / BW
              + added_operations / R
              + resource/occupancy/primitive-throughput penalty
```

These terms are a candidate-ranking ledger, not an additive runtime predictor: compute, memory and boundaries
may overlap, and register/LDS pressure changes achieved `BW` and `R`. Both sides therefore require measurement
in the selected route. The current NV Q4_K `w1+w3` decode result is a useful
bounded example: the fused semantic pattern is valid, but the measured scalar schedule improves whole-token
wall time by only 1.7-2%, while a wider standalone-optimal load style regresses in-loop by 5%
(`docs/task_workflow/input/q4k-w1w3-fused-qv-implementation-record-20260803.md`). The graph transformation
transfers; the winning schedule does not transfer automatically.

DwarfStar demonstrates the same separation at a larger scope. Its CUDA prefill substrate vendors
llama.cpp's MMQ/MMVQ kernels, while its DeepSeek routes specialize the graph around those primitives with
paired gate/up epilogues, expert grouping, and direct selected-expert down reductions:
[vendored-kernel record](https://github.com/antirez/ds4/blob/main/cuda/mmq/VENDOR.md),
[CUDA routed-MoE paths](https://github.com/antirez/ds4/blob/main/ds4_cuda.cu), and
[Metal paired/SwiGLU and sum kernels](https://github.com/antirez/ds4/blob/main/metal/moe.metal).
This is evidence that primitive quality and route quality are distinct, composable layers; it is not
evidence that one model-specific handwritten kernel should become the abstraction.

### 2A.2 The theory stack demonstrated by llama.cpp `mul_mat_vec_q`

llama.cpp's decode GEMV is a useful concrete decomposition because all of the layers are visible without a
large GEMM framework.

For one output vector,

```
y = W x
work         ~= 2*N*K
weight bytes ~= (b/8)*N*K
intensity    ~= 16/b operations per byte
```

At an effective `b ~= 4.5` bits/weight, the lower-bound intensity is only about `3.56 operations/byte`.
That is far below the compute-to-bandwidth ratio of a modern discrete GPU, so reading the weights is the
first-order cost. This is the Roofline result, not a claim about occupancy.

The Q4_K/Q8_1 dot then uses block-affine algebra. In schematic form,

```
w_i ~= d_w*q_i - m_w
x_i ~= d_x*a_i

sum(w_i*x_i)
  ~= d_w*d_x*sum(q_i*a_i) - m_w*d_x*sum(a_i)
```

The hot loop therefore needs an integer dot and, for affine blocks, an activation sum. It does not need to
construct an fp16 weight vector. llama.cpp quantizes each fp32 activation block to 32 int8 values and stores
its scale and sum in Q8_1; its Q4_K/Q6_K dot helpers unpack weights into byte lanes, use packed dot products,
and apply block scales/corrections outside the individual multiply:

- [`quantize_q8_1`](https://github.com/ggml-org/llama.cpp/blob/ac4cddeb0dbd778f650bf568f6f08344a06abe3a/ggml/src/ggml-cuda/quantize.cu)
- [quantized dot implementations](https://github.com/ggml-org/llama.cpp/blob/ac4cddeb0dbd778f650bf568f6f08344a06abe3a/ggml/src/ggml-cuda/vecdotq.cuh)
- [`mul_mat_vec_q`](https://github.com/ggml-org/llama.cpp/blob/ac4cddeb0dbd778f650bf568f6f08344a06abe3a/ggml/src/ggml-cuda/mmvq.cu)

On NVIDIA, `dp4a` performs four packed byte multiplies and an int32 accumulation per instruction
([PTX ISA](https://docs.nvidia.com/cuda/archive/12.5.1/pdf/ptx_isa_8.5.pdf)). That is instruction-set
matching: the representation is arranged so useful mathematical work maps onto a native packed operation.
The same principle may map to a different instruction or even a direct fp16 route on another target; the
portable fact is the match, not the opcode name.

Activation quantization is an amortization decision:

```
activation quantization = O(K)
matrix-vector use       = O(N*K)
relative setup          = O(1/N)
```

It may be profitable when one quantized activation is reused across thousands of output rows. It is not
free: the quantize kernel, boundary, and any reuse failure belong in the route time. Our direct-fp16
activation route can beat llama's complete path if it reaches comparable raw GEMV efficiency without paying
that setup; it cannot claim the win from omission alone.

### 2A.3 What is specifically Volkov, and what is not

Volkov's result is that occupancy is a means of supplying concurrency, not the goal. Little's-law reasoning
requires enough independent work to cover latency:

```
required concurrency ~= latency * throughput
```

That concurrency can come from thread-level parallelism (more resident warps) or instruction-level
parallelism (independent loads and accumulators within each thread). Register blocking, unrolling,
prefetching, multiple output accumulators and accepting lower occupancy when they improve useful throughput
are the Volkov layer: [Better Performance at Lower Occupancy](https://www.nvidia.com/content/gtc-2010/pdfs/2238_gtc2010.pdf).

In `mul_mat_vec_q`, the Volkov-shaped mechanisms are the unrolled per-thread `tmp` accumulator array,
multiple rows/columns per block, prefetched fusion operands, compile-time warp counts and launch bounds.
Warp partitioning of K and tree reduction are parallel-algorithm structure; Q8_1 and the correction term are
quantization algebra; `dp4a` is hardware mapping; keeping Q4_K packed is Roofline/IO reasoning. Calling the
whole kernel "Volkov" erases the actual design decisions.

### 2A.4 Contract for generated kernels in this repo

Theory should constrain the search; measurement should select within it.

**Candidate invariants (reject before timing):**

1. State the operation shape and reuse regime (`M/N/K`, quant type, role, batch/depth class), plus the
   semantic graph motif when the candidate crosses an operator boundary: shared producers, consumer counts,
   indexed/selection axes, reduction axes, and required reduction order.
2. State the predicted binding resource from measured `BW` and relevant instruction-rate ceilings.
3. In bandwidth-bound decode, read packed weights once and never materialize an expanded weight tensor.
4. Express dequantization as register-local unpack/scale/correction or prove why another representation wins.
5. Use coalesced/aligned global transactions and a native useful dot/MMA/vector instruction where the target
   provides one; verify the emitted ISA rather than inferring it from source.
6. Use a cooperative reduction/writeback with no serialized or duplicate full-K work.
7. Reject spills, illegal resource use, incomplete coverage, hidden fallback and incorrect numerics.
8. Include quantization, routing/grouping, fixup, boundary copies, reductions and epilogues in the candidate's
   route cost.
9. State before/after `B_route`, materialized intermediates, launch count, and reuse cardinality, with each
   quantity labeled measured, statically counted, projected, or unavailable. Do not claim a byte win when
   compulsory weight traffic is unchanged; name the activation, intermediate or boundary traffic actually
   removed.
10. Keep semantic legality separate from schedule profitability. Compare against the best unfused primitive
    route, and require independent schedule selection for decode, prefill, quant and target regimes.

**Search dimensions (measure, do not derive once):**

- direct-fp16 activation versus activation quantization plus integer dot;
- lane/warp ownership of K and output rows;
- rows or columns accumulated per thread/block;
- vector load width, alignment contract and packed-weight lane layout;
- tile sizes, warps/waves, stages, unroll and independent-accumulator count;
- register versus shared/LDS staging, prefetch distance and double buffering;
- warp/tree/cooperative reduction shape;
- fusion set, materialization boundaries, and number of coupled outputs/accumulator sets;
- indexed-work organization: direct selected slots versus sort/group/permute by weight identity;
- selected-output reduction arity and placement: materialize then reduce, atomic/segmented reduce, or direct
  accumulation into the final destination;
- activation transform/quantization reuse scope across projections, experts and heads;
- split-K/stream-K/fixup strategy;
- separate schedule families by stable shape, quant and measured target capabilities.

**Promotion order:** correctness and coverage → emitted-ISA proof → resource/spill facts → isolated kernel and
component-wall measurement → same-session selected-route measurement → whole-model endpoint. Node-sum
projections may rank experiments only after their provenance is stated; they never promote a route.

The practical objective is therefore not "generate arbitrary kernels and hope search rediscovers CUDA
folklore." It is:

> Generate legal compositions of reusable compiler primitives that already respect IO and algebraic lower
> bounds; search the target-dependent schedule and route choices; promote only measured endpoint winners.

---

## 3. Decode is bandwidth-bound: minimise bytes

At `M=1` every weight byte is read exactly once and used once. There is no second consumer of a byte
after it is loaded, so `T_bytes,route` dominates and ALU headroom is close to free
(`docs/HANDOFF_14b_decode_depth_decay_20260726.md:22-24`).

**What follows:** keep the weights in their smallest representation and unpack them in registers.
Q4_K is ≈4.5 bits/weight against fp16's 16, so materialising an fp16 copy would inflate the binding
quantity by ≈3.5× **with no reuse to amortise it against**. That is why no decode strategy anywhere in
this corpus materialises fp16.

Measured, Metal, Qwen3-8B-Q4_K_M: decode 5.386 → **17.24 tok/s**, byte-identical output, versus
llama.cpp 20.34 (84.8%).

---

## 4. Prefill is compute-bound: maximise the rate of the multiply unit

At `M=512` each weight byte is read once and reused across all 512 rows of the GEMM. FLOPs scale with
`M`; bytes do not. `T_flops` dominates (`docs/measurement-regime-audit-llama-prefill-20260715.md:20-27`).

**What follows, and it inverts §3:** the question stops being "how few bytes" and becomes "which unit
performs 512× more multiply-accumulates." Spending *more* bytes once — to buy entry into a faster
multiply unit — is a good trade, because the one-time cost amortises over 512 reuses.

This is the single most common source of confusion in this codebase. The right answer for decode is the
wrong answer for prefill, and vice versa. A kernel can be simultaneously bandwidth-optimal and
compute-catastrophic.

---

## 5. Which unit does the multiply is a discrete choice worth ~10–20×

Not a tuning parameter. A route that runs the multiply-accumulate on vector ALUs is capped roughly an
order of magnitude below one that runs it on the matrix unit **of the same silicon**.

Measured, AMD gfx1100, 14B prefill pp512
(`docs/8b-vs-14b-prefill-regression-20260721.md:19-28`):

| route | multiply unit | pp512 tok/s |
| --- | --- | ---: |
| `DIRECT_PACKED_FALLBACK` | vector ALU | ~354 |
| `BOUNDED_PACKED_TILES` (packed-WMMA) | WMMA | 1829–1948 |

The entire ~5× is this switch. Nothing else in that table changed.

**Until this is settled, no other optimisation matters.** Tuning a vector-ALU route is polishing
something capped an order of magnitude below the alternative.

**This principle is target-conditional, not universal — it does not hold on Apple M4.** Measured
directly (`extra/llm_research/microbench/fma_peak_metal.py`, 2026-07-31, same 10-core M4 as `R`):
plain fp16 FMA on ordinary vector ALUs, no `simdgroup` op anywhere, disassembly-verified pure-FMA
hot loop, reaches **3445–3909 GFLOPS depending on precision variant** (fp32→fp32 3445, fp16→fp32
mixed-accumulate 3528, fp16→fp16 3909) — **0.91×–1.03× `R`'s 3781 GFLOPS**, i.e. the same order of
magnitude, not the ~10–20× gap AMD gfx1100 shows between `DIRECT_PACKED_FALLBACK` and
`BOUNDED_PACKED_TILES`. Full sweep and verdict in §10. The reason §5's gap exists on gfx1100 and
not here is architectural, not a measurement artifact of this campaign: gfx1100 has a physically
separate WMMA execution pipe alongside its vector ALUs, so routing onto the wrong one strands most
of the silicon idle; on this M4, `simdgroup_multiply_accumulate` appears to lower onto the same FP
ALUs plain FMA already uses — there is no second, faster pipe to strand work off of. **Before
invoking this principle for a new target, check whether that target's matrix path is measured
separately from its plain-ALU path (as gfx1100's WMMA-vs-vector split was) — if the two numbers
converge, as here, the 10–20× lever isn't available, and any strategy built around chasing "the
matrix unit" over "the ALU" needs a different premise (see §10 for what does remain true on
Metal).**

---

## 6. Which strategy you may use is decided by memory arithmetic, not preference

There are three ways to feed the multiply unit. Each has a precondition, and the policy is fail-closed:
if the precondition is not met, the strategy declines and control falls through to the next one
(`prefill_policy.py::_EXECUTING_STRATEGIES`).

| strategy | uses matrix unit | needs resident fp16 |
| --- | --- | --- |
| `DIRECT_PACKED_FALLBACK` | no — vector ALU | no |
| `FULL_RESIDENT_OVERLAY` | yes | **yes** |
| `BOUNDED_PACKED_TILES` | yes | no — dequant fused into the operands |

Eligibility is arithmetic, not judgement:

| model / device | fp16 copy | budget | overlay eligible? |
| --- | ---: | ---: | --- |
| AMD 8B | 16.4 GB | 24 GB | ✅ yes |
| AMD 14B | 29.5 GB | 24 GB | ❌ no |
| **Metal 8B** | **16.4 GB** | **12.7 GB** | ❌ **no** |

(`docs/8b-vs-14b-prefill-regression-20260721.md:40-51`, `:55-64`)

**Metal-8B is structurally 14B-shaped.** It cannot pay the overlay's entry cost, so it falls through to
`DIRECT_PACKED_FALLBACK` — the vector-ALU floor. Confirmed in production today: `prefill_route =
DIRECT_PACKED_FALLBACK` (`bench/prefill-whole-synced/t2-metal-pp512.json`).

The historical answer for that shape is `BOUNDED_PACKED_TILES`: reach the matrix unit *without*
materialising fp16, by fusing the dequant into the operands.

---

## 7. Speed lives in the tile geometry, not in the list of transforms

The recipe that produced a win on one configuration is not the cause of the win, and does not transfer.

Measured counter-example (`docs/8b-vs-14b-prefill-regression-20260721.md:64-76`): applying 8B's
`UPCAST/UNROLL` warmstart to a 14B contiguous fp16 weight gave **6.6 TFLOP/s** against packed-WMMA's
**9.5** — 31% *slower*. 14B's winning configuration used **TC only**; the geometry
(`tm/tn/tk/waves/LDS`) did the work.

The same doc records the failure this caused: a projected 14B "ceiling" of ~1940 tok/s that was
extrapolated from 8B's overlay speed — **a path 14B structurally cannot run** (§6).

---

## 8. The remaining freedom is a legal space to be searched, not a formula to be derived

Once the strategy is fixed by §6, what is left is geometry, and it is not analytically derivable. Three
owners, and the separation is load-bearing
(`extra/llm_research/bubblebeam_futuresight.py`, module docstring):

- **BubbleBeam** — proposes target-neutral legal dimension values from declared target facts.
- **FutureSight** — statically rejects and orders candidate payloads.
- **BoltBeam** — owns candidate schema, identity, finite expansion, measured ranking, and promotion,
  alone.

Hand-derived geometry and hand-extrapolated ceilings are the failure mode this structure exists to
prevent. §7's 31%-slower result and §9's traps are what it looks like when the structure is bypassed.

---

## 9. Measurement traps, each one paid for

1. **Quoting the spec sheet instead of measuring the achievable rate.** AMD's real WMMA peak is
   **105 TF**, measured by an isolated microbenchmark with zero loads in the loop
   (`extra/llm_research/microbench/wmma_peak.cpp`; `docs/prefill-roofline-first-principles-20260724.md:9-18`).
   Quoting the 122.8 spec figure understates efficiency by 17%; quoting 61.4 flatters it 1.7× and
   produced a false "we are at 94% of peak, nothing left" reading.

2. **Counting FLOPs with a shortcut.** For a 512-token chunk of Qwen3-8B: `2·P·T` gives 8.19 TFLOP
   (+15%, counts embed and lm_head which are not per-token matmuls); promoted role shapes give 4.48
   TFLOP (−37%, covers only 63% of in-layer params); the config-derived figure is **7.11 TFLOP**
   (`docs/prefill-roofline-first-principles-20260724.md:20-36`).

3. **Comparing non-commensurable units.** A single-GEMM GFLOPS figure is not a whole-model tok/s
   figure. This session spent most of a day treating a 2070 GFLOPS number as comparable to 221 tok/s.

4. **Comparing across sessions.** `docs/prefill-current-state.md:105-116` supersedes every earlier
   cross-session llama figure in this corpus; re-running llama.cpp in the same session with the same GPU
   state gives materially different short-context numbers.

5. **Dividing by the wrong model's comparator.** An earlier claim in this session that we beat llama by
   1.88× divided 8B's achieved number by a *14B* llama comparator. The real same-session margins are
   §10's — single digits.

6. **Timing enqueue instead of execution.** Metal is asynchronous. Without `Device.synchronize()` before
   stopping the clock, an M4 measured 63,583 GFLOPS.

7. **Treating an unsearched default as a ceiling.** A measurement of one configuration bounds that
   configuration, not the machine.

---

## 10. Where each target stands, measured

**Validation against llama.cpp** — same-session, paired, `flock`-serialised
(`docs/prefill-current-state.md:109-116`). These are the honest margins:

| | pp | ours | llama, same session | margin |
| --- | ---: | ---: | --- | ---: |
| AMD 8B | 512 | 3727 | 3347 ± 242 | +11.4% (llama's noisiest point, 7% stdev — soft) |
| AMD 8B | 4096 | 3262 | 3158 ± 17 | **+3.3%** |
| AMD 14B | 512 | 1948 | 1845 ± 86 | **+5.6%** |
| AMD 14B | 4096 | 1787 | 1642 ± 9 | **+8.8%** |

**Metal, measured 2026-07-30/31:**

| quantity | value | note |
| --- | ---: | --- |
| decode | 17.24 tok/s | llama 20.34 → 84.8%; byte-identical output |
| prefill pp512 | 54.2 tok/s | llama 221.23 → 24.5%; route `DIRECT_PACKED_FALLBACK` |
| fp16 GEMM, `(512,12288,4096)` | 2694–2753 GFLOPS | mean 2733, 8 reps |
| dequant→fp16→GEMM, precompiled | 2293 GFLOPS | 5 reps, `max_abs_error` 0.0 |
| fused Q4_K → simdgroup, generic path | 544 GFLOPS | correct: 0.0 error, 100% coverage, deterministic |
| fused Q4_K, hand-authored precontract path | — | **incorrect**: 18.75% write coverage, non-deterministic |

Metal prefill sits at 24.5% of llama on the vector-ALU floor — the same position, and nearly the same
ratio, as AMD 14B at 0.19× before its fix (§5). This is a known configuration with a known answer.

**The 2026-07-31 BubbleBeam campaign** — first time TC was in the candidate space on Metal at the
prefill shape (`m=512`, `phase: prefill`, `ffn_gate_up`, 18 candidates):

- **Every TC candidate BLOCKED** — 5 of 5, all `provider_compile:provider_failure`.
- The 11 measured candidates carry only `LOCAL`/`UPCAST`/no transform, i.e. **vector-ALU only**. Best
  1061 GFLOPS. Correctness passed against `canonical_packed_reference`.
- All candidates were emitted with `compute.family = generic_matvec` — a decode-shaped family — despite
  the request specifying `m=512`.

So the campaign measured the vector-ALU floor at prefill shape. **It did not test tensor cores**, and by
§5 the floor is the thing that needs escaping. The block reason is the finding.

**Metal `R`, measured 2026-07-31** (`extra/llm_research/microbench/wmma_peak_metal.py`,
`extra/llm_research/microbench/README.md`), on the same 10-core M4 (Mac16,10) as the rest of this
table — an isolated `simdgroup_multiply_accumulate` microbenchmark mirroring `wmma_peak.cpp`: zero
loads in the hot loop, independent accumulators, runtime trip count, never-taken keep-alive,
disassembly-verified (`xcrun metal -c` + `metal-objdump`: zero `addrspace(1)`/`addrspace(3)`
references inside the loop; operands `mat_a`/`mat_b` constant-folded directly into the intrinsic
call, never loaded).

Grid-size sweep plateaus; a swept NACC (2/4/8/16) shows the *opposite* shape from gfx1100 — this
hardware needs almost no independent accumulators to hide matrix-op latency (`nacc=1` reaches
3718 GFLOPS, only 1.6% below the `nacc=2` peak), and throughput **falls** as NACC grows past 2
(2380 GFLOPS at nacc=4, 1742 at nacc=16) because register pressure costs occupancy faster than
extra ILP buys anything. The true plateau, found by re-sweeping grid size at the winning
`nacc=2`, is:

**R ≈ 3781 GFLOPS ≈ 3.78 TFLOPS** (`nacc=2, blocks=32768, tpb=256`, 262144 simdgroups; spread
<1 GFLOPS across 5 reps at the plateau; insensitive to threadgroup shape 32–1024).

This is now the achievable denominator for any Metal matrix-unit efficiency claim: the 2733 GFLOPS
fp16-GEMM "ceiling" above is 72.3% of it (as expected — a full kernel bundling load/address/epilogue
cost sits below the isolated rate); the 2293 and 544 GFLOPS figures are 60.6% and 14.4% of it. No
Apple-published TFLOPS spec exists for this instruction or for the base 10-core M4 GPU to compare
against; the only external figures found (`chsasank/device-benchmarks`, web search 2026-07-31) are
third-party FP16 ALU benchmarks of the **M4 Max (40-core)**, ~13.3–14.2 TFLOPS — 4x this die's core
count and not a matrix-unit-specific number — so no "fraction of spec" figure is reported as
authoritative.

**Plain FP16 FMA peak, measured 2026-07-31** (`extra/llm_research/microbench/fma_peak_metal.py`),
same M4, same harness (calibrated `iters` so wall time sits ≥50× above a probed dispatch-overhead
floor — a first version of this run reported up to 1.78M GFLOPS with no grid plateau because
`iters` was too small relative to `blocks`, timing host overhead rather than the GPU; retracted
before being reported, root-caused, and fixed by calibration; see the module docstring). Directly
answers §5's question for this hardware: is `simdgroup_multiply_accumulate` a separate, faster
matrix unit, or does it lower onto the same ALUs plain FMA uses?

Swept vector width (scalar `half` through `half4`×2 as an emulated width-8 — this MSL toolchain has
no `half8`/`float8`; naming one fails `xcrun metal -c` with "incomplete type"), NACC (1/2/4/8/16),
and grid size, for three precision variants, since `simdgroup_float8x8` accumulates fp16×fp16→fp32,
not fp16→fp16:

| variant | plateau (GFLOPS) | width, nacc, blocks | vs `R` (3781.3) |
| --- | ---: | --- | ---: |
| fp16→fp16 | 3908.7 | width=8, nacc=16, blocks=16384 | **1.034×** |
| fp16→fp32 (matches matmul numerics) | 3527.8 | width=8, nacc=8, blocks=16384 | **0.933×** |
| fp32→fp32 | 3444.9 | width=8, nacc=8, blocks=16384 | **0.911×** |

All three land within +3%/−9% of `R` — the same order of magnitude, not a separate unit worth
10–20× (§5). fp16→fp32 is the primary comparator (it is what the matrix op actually computes) and
sits at 0.933× `R`. Packed-math check: fp16→fp32 / fp32→fp32 = 1.024× — essentially no packed-fp16
throughput doubling; fp16→fp16 / fp32→fp32 = 1.135× — a modest edge, far short of 2×. Apple's ALUs
here do not reward fp16 with a compute-throughput multiplier the way AMD/NVIDIA packed-fp16 paths
do; fp16 buys bandwidth/storage, not FLOP/s.

NACC behaviour differs sharply by variant, and differently from `R`: fp16→fp16 improves
monotonically out to nacc=16 (3702→3842→3883→3890→3898 GFLOPS), but fp16→fp32 and fp32→fp32 both
peak at nacc=8 and then **collapse** at nacc=16 (3517→688 and 3437→606 GFLOPS respectively, a
>80% drop) — a register-pressure cliff, since float accumulators cost more registers per lane than
half ones and 16 of them blow the budget. Disassembly (`xcrun metal -fno-fast-math -c` — flag-
matched to `MetalCompiler.compile()`'s actual `-fno-fast-math` runtime flag; an unmatched-flags
disassembly earlier in this investigation showed spurious `fma fast`/`fadd fast` from default
fast-math and was not evidence about what was measured) confirms all three:
`air.compile.fast_math_disable` present, hot loop is `@air.fma.v4f16`/`@air.fma.v4f32` only with zero loads/converts
inside it (the fp16→fp32 widening conversion is loop-invariant and correctly hoisted to a one-time
preheader, not repeated per iteration), zero `simdgroup` references anywhere, one load (`iters`)
and one gated store (the never-taken sentinel) total.

**Verdict: one shared unit, not two.** `simdgroup_multiply_accumulate` does not access a separate,
faster matrix pipe on this hardware — it lowers onto (or performs comparably to) the ordinary FP
ALUs that plain scalar/vector FMA already uses. §5's "which unit — worth 10–20×" principle, while
correct for gfx1100 (§5's WMMA-vs-vector split, ~5× measured), **does not apply to Metal on M4**:
there is no faster unit to route onto. Metal prefill's remaining headroom (§10, 24.5% of llama) is
therefore a **routing/tiling problem, not a units problem** — closer to llama's 81%-of-`R` decode
ratio than to AMD 14B's pre-fix 0.19×-of-`R` state, and the fix looks like §7/§8 (tile geometry,
searched not derived), not like "reach the matrix unit instead of the ALU."

**Crossover, `M* = (w/16)·(R/BW)`, `w = 4.5` bits/weight (Q4_K):** no measured `BW` exists for this
M4 anywhere in this corpus (checked: no `GB/s` figure tied to Metal/M4 in `docs/`), so `M*` is
reported as a function of `BW` rather than substituting a spec figure — the exact error this frame
exists to prevent:

```
M*(BW) = (4.5/16) · (3.78e12 / BW_bytes_per_s) = 1063 / BW_GBps
```

| `BW` (GB/s) | `M*` (tokens) |
| ---: | ---: |
| 50 | 21.3 |
| 100 | 10.6 |
| 200 | 5.3 |
| 500 | 2.1 |
| 800 | 1.3 |

Across this entire plausible range for a unified-memory device, `M*` stays in the low single/double
digits. **Decode (`M=1`) sits below `M*` for every value in the table except the most extreme
(≥800 GB/s), and prefill (`M=512`) sits far above `M*` for all of them** — the classification in §3/§4
is robust to the unmeasured `BW`, even though the precise crossover point is not yet known.

### Open gaps

0. **The precontract prefill kernel is correct and measured as of 2026-07-31** (see
   `docs/qwen3-8b-prefill-metal-precontract-campaign-20260731.md`). Four lowering defects were fixed --
   a fragment-read row extent hardcoded to AMD's `tc.dims[0]`, dropped leftover-lane K groups, a
   lane->row/K correspondence assuming RDNA3's low-bit split, and a loop-carried write-after-read race.
   `max_abs_error` 0.0, coverage 96.67%, bit-identical rounds. First measured campaign: best geometry
   **3610 GFLOPS sustained** (stress-tested `f0cb8c58d`: full-output coverage proven, flat within
   ±0.5% from m=256 to m=8192; the earlier 2558 "sustained" figure was measurement-harness overhead) against a **1063** control, 87 of 87 candidates correct.
   **Not promoted** -- QUALIFY and POLICY remain blocked, so production still runs
   `DIRECT_PACKED_FALLBACK` at 54.2 tok/s.

1. ~~`R` for Metal is unmeasured.~~ **Resolved 2026-07-31: R ≈ 3.78 TFLOPS**, above. `M*` still needs
   a measured `BW` for this M4 to pin down exactly (see table above for the shape of that dependency).
2. **Why every TC candidate fails to compile through the provider.** This is now the load-bearing
   blocker for Metal prefill.
3. **Why the hand-authored precontract path writes 18.75% of its output non-deterministically.** Four
   hypotheses tested and refuted: lane permutation, C-fragment width overcount, multi-wave
   decomposition, device-blind admission. Unexplained.
4. **Whether the candidate space should be emitting a GEMM family rather than `generic_matvec`** at
   `m=512`.
5. **No measured `BW` for Metal/M4.** Needed to turn `M*(BW)` above into a single number.
