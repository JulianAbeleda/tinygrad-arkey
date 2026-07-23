# Shared attention: development sequence and observed pattern

Date: 2026-07-23

## End goal

Move both the fp16-derived and quantized-model prefill routes closer to their applicable AMD rooflines using one centralized, scheduler-native, score-resident attention implementation. The two model routes must reuse the same compiler and kernel assets rather than carry route-specific attention implementations.

## What progressed in sequence

The implementation advanced through this dependency chain:

```text
composite reduction
  -> automatic attention rewrite
  -> multi-output online-softmax state
  -> fused QK + softmax + PV
  -> tensor-core lowering for QK and PV
  -> exact 8B and 14B GQA integration
  -> real AMD correctness proof
  -> corrected replay benchmarking
  -> bottleneck isolation
  -> roofline optimization
```

### 1. Prove the fused attention math

The online-softmax combine was separated from the devectorizer and implemented as a general composite reduction. The reduction carries the running maximum, normalization sum, and PV accumulator while applying the correction formula across KV tiles.

### 2. Make attention score-resident

Rangeify recognizes the ordinary attention graph and emits the composite reduction. QK scores and probabilities are consumed inside the reduction rather than materialized as full `Q x KV` buffers.

### 3. Lower both contractions to tensor cores

The native path contains attributed QK and PV WMMA operations. The proof ledger distinguishes the two roles even when UOp interning deduplicates physical nodes.

### 4. Integrate both target model geometries

One path serves:

- 8B: `B=1, Hq=32, Hkv=8, G=4, Hd=128`
- 14B: `B=1, Hq=40, Hkv=8, G=5, Hd=128`

Both the fp16-derived route and the bounded packed-weight route converge on the same fp16 Q/K/V attention implementation. K/V heads are addressed directly with `kv_head = q_head // G`; expanded K/V tensors are not created.

### 5. Prove correctness on real AMD hardware

First-position and prefix-position captures passed for both model profiles. The proof verifies causal behavior, final output ownership, QK/PV roles, absence of score/probability materialization, resources, numeric records, and source/graph/ISA hashes.

### 6. Correct the performance measurement

The initial eager timing included graph construction, scheduling, and output allocation. It was superseded by a TinyJit replay-only protocol with prebuilt graphs, synchronized replay, reused allocations, full-output numeric comparison, warmup, and repeated samples.

The corrected candidate is approximately `4x` to `26x` faster than ordinary GQA attention across the measured 8B/14B KV512-KV4096 matrix, with maximum absolute error around `6.1e-05`.

### 7. Isolate the remaining bottleneck sequentially

Each optimization experiment was treated as a falsifiable hypothesis:

```text
workgroup synchronization
  -> removed the single-wave barrier
  -> replay changed by less than 0.2%
  -> rejected as the main bottleneck

repeated K/V traffic
  -> implemented G2 multi-wave LDS sharing
  -> load sites fell substantially
  -> LDS, barriers, and private state offset the saving
  -> up to 1.65% slower; performance path reverted

high physical register numbers
  -> implemented a compact attention-only VGPR partition
  -> compiled without spill and remained numerically correct
  -> 1.43% slower at KV512 and 2.46% slower at KV4096
  -> experimental mapping reverted

simultaneous live state
  -> current leading hypothesis
  -> not yet tested by a structural lifetime-reduction variant
```

## Pattern observed across the work

The repeated pattern is that local mechanical improvements do not improve elapsed time when they preserve or increase the kernel's constrained resource.

1. Removing synchronization reduced instructions but did not expose more throughput.
2. Sharing K/V reduced memory-load sites but introduced LDS, barrier, and private-memory costs while register pressure stayed high.
3. Moving values to lower VGPR numbers changed placement but not the quantity or lifetime of live state, so occupancy did not improve.
4. The candidate scales nearly linearly with KV, while 8B and 14B timings remain unexpectedly close despite the 14B route having 25% more query heads. This is consistent with a saturated resource or latency-limited kernel rather than simple operation-count scaling.
5. The single-wave kernel is spill-free but reports roughly 250 allocated VGPRs. Spill-free therefore does not mean occupancy-efficient.
6. QK, running softmax state, the full 128-wide PV accumulator, and WMMA fragments coexist. The most durable remaining explanation is that this simultaneous live state restricts resident waves and latency hiding.

The practical lesson is:

```text
instruction reduction != throughput improvement
traffic reduction     != throughput improvement
register remapping    != lifetime reduction

The next variant must reduce the constrained resource itself.
```

## Current state

- Fused score-resident attention implementation: complete.
- Shared compiler/kernel architecture for both routes: complete.
- Exact 8B/14B attention integration: complete.
- Attention-level real-AMD correctness proof: complete.
- Corrected replay benchmark matrix: complete.
- Whole-model corrected prefill benchmark matrix: not complete.
- Roofline optimization: not complete.

Functionally, the attention path is complete and proven. The remaining project is performance architecture and whole-model promotion, not basic attention correctness.

## Literature alignment

The project has completed the problem addressed by the original FlashAttention paper: avoid materializing the score and probability matrices by using an IO-aware, tiled, online-softmax algorithm.

The remaining problem matches the transition described by FlashAttention-2. A correct IO-aware attention kernel can remain far below GEMM efficiency because work is partitioned poorly between blocks and warps, producing low occupancy or unnecessary shared-memory communication. FlashAttention-2 improves this by parallelizing even a single head across more blocks and by assigning warps more independent work.

The Hopper FlashAttention-2 case study also treats Q/K/V tile choice as a joint balance between register pressure and shared-memory use. This explains why the G2 experiment was directionally correct about reuse but unsuccessful as implemented: it reduced K/V load sites while increasing LDS, barrier, and private-state costs.

FlashAttention-3 addresses the stage after work partitioning by overlapping data movement, tensor-core work, and softmax through specialization and interleaving. Its Hopper-specific mechanisms do not transfer directly to RDNA3, but the scheduling order does: establish sufficient residency first, then hide the serial recurrence and load latency.

Relevant papers:

- FlashAttention: <https://arxiv.org/abs/2205.14135>
- FlashAttention-2: <https://arxiv.org/abs/2307.08691>
- FlashAttention-2 Hopper/CUTLASS case study: <https://arxiv.org/abs/2312.11918>
- FlashAttention-3: <https://arxiv.org/abs/2407.08608>
- RegDem, selective shared-memory register demotion: <https://arxiv.org/abs/1907.02894>

The concise problem statement is:

> The score-resident kernel is limited by register residency and work partitioning. Long-lived PV output accumulators coexist with QK fragments, online-softmax state, and WMMA temporaries, reducing resident-wave parallelism and latency hiding.

This is an evidence-backed working diagnosis, not yet a direct counter-level proof of the exact live object responsible.

## Realigned sequential test ladder

Do not assume output-dimension slicing is the production answer. First determine which ownership axis lowers the allocated resource while preserving useful tensor-core work.

### Test 1: live-state and residency ledger

Attribute compiler-reported VGPR use to the persistent PV accumulator, QK fragments, `m/l` softmax state, WMMA A/B fragments, address state, and temporary values. Record theoretical resident waves from the emitted resource counts. This establishes the baseline that every variant must change.

### Test 2: query-row ownership microvariant

Reduce the number of query rows owned by one wave or block while retaining the full output dimension. Give additional waves or blocks disjoint query rows so no output reduction is required. Measure VGPR allocation, resident-wave capacity, scratch/private memory, WMMA role counts, numeric output, and corrected replay.

This is the closest RDNA3 analogue to FlashAttention-2's work-partitioning direction and is the preferred first production-shaped experiment.

### Test 3: output-dimension ownership diagnostic

If query-row tiling cannot materially reduce the persistent PV state, accumulate a smaller value/output slice, such as 32 of 128 dimensions. Initially permit QK/softmax recomputation so the experiment cleanly answers whether smaller PV live state changes occupancy and throughput.

This is a diagnostic before it is a production design. Promote it only if the occupancy gain exceeds recomputation cost.

### Test 4: selective state demotion

If reduced ownership proves the bottleneck but recomputation loses, deliberately stage selected inactive state in LDS. Do not rely on compiler spilling and do not reproduce G2's private-memory path. Admit the variant only when resource metadata proves a residency gain.

### Test 5: grouped-query reuse after residency

Retry K/V sharing only after per-wave state is smaller. Waves should own independent query/output partitions and communicate only the K/V data whose reuse exceeds synchronization cost. G2 disproved the previous partition, not the value of reuse in every partition.

### Test 6: latency overlap

Only after residency improves, interleave next-tile loads, QK WMMA, online-softmax work, and PV WMMA. This is the transferable FlashAttention-3 principle; RDNA3 requires a software schedule rather than Hopper TMA/WGMMA mechanisms.

### Test 7: final geometry and whole-model promotion

Sweep tile geometry only after the architecture passes the preceding gates. Then run corrected whole-prefill measurements for both model routes and compare against their applicable compute/bandwidth rooflines.

## Falsification rules

Each step must change the resource it claims to address before replay time is interpreted:

- A register experiment must reduce allocated VGPRs or increase calculated residency.
- A sharing experiment must reduce measured or statically attributable K/V traffic without adding spills/private memory.
- A pipeline experiment must preserve residency and expose independent operations across the recurrence.
- A kernel improvement must survive full-output numeric checks and whole-model timing.

If a variant changes instructions but not residency, traffic, or dependency overlap, it is not evidence for or against the architectural theory.

## Promotion boundary

Do not claim completion from an isolated faster kernel. Production promotion requires:

- full numeric correctness for 8B and 14B;
- causal first and prefix positions;
- KV512, KV1024, KV2048, and KV4096;
- proof-gated QK and PV WMMA attribution;
- no score/probability materialization;
- acceptable VGPR, LDS, scratch, and spill resources;
- corrected replay improvement;
- corrected whole-model prefill improvement;
- separate applicable roofline accounting for fp16 attention and quantized whole-model prefill.
