# Dense projection service strategy

## Decision

The low apparent bandwidth of the dense Q, O, K, and V projections is a
token-lifecycle problem that requires a kernel substrate.

The right development order is:

1. Prove a new service mechanism in a small, exact one-layer kernel gate.
2. Integrate only mechanisms that materially improve the complete dependent
   span.
3. Book recovery only after a full-token control/candidate/control wall test.

An isolated kernel win is necessary evidence, but it is not the optimization
target. The target is a faster token.

## Why the small projections report low bandwidth

The current measurements fit a size-aware service model of the form:

```text
projection time = fixed stream/ramp cost + payload / asymptotic stream rate
```

Large dense projections run long enough to amortize the fixed term. Q, O, K,
and V are shorter, so they finish while the memory service is still ramping.
Their low aggregate rates therefore do not mean that the SMs merely need more
warps or wider loads.

This distinction matters:

- **Kernel occupancy** asks whether enough warps are resident.
- **Memory occupancy** asks whether enough independent misses are in flight.
- **Service continuity** asks whether the device remains fed long enough to
  amortize launch, scheduling, and DRAM-ramp latency.

The installed kernels already have broad row parallelism. The remaining
distance is primarily service continuity across short physical streams.

Consequently, assigning every independent projection the long-stream rate is
not a valid standalone roofline. Reaching that rate requires changing the
number or lifetime of physical service episodes.

## What the existing experiments prove

### Aggregation can improve device service

The exact ordinary QKV full-grid gate combined sibling projections into one
longer episode. It recovered about 2.38 microseconds per tested group in the
rotated-cold primitive and reduced timestamped GPU work in production.

The corresponding full-token bracket nevertheless regressed. The gain was
lost at the changed program/output/replay boundary. This is evidence that
aggregation is physically useful but that the present integration does not
translate it into token latency.

Authority:
`docs/task_workflow/output/nv-ordinary-q4-qkv-full-result-20260825.md`.

### Local instruction and latency techniques are closed for the current format

The tested Q4_K transpose, cache lookahead, and one- and two-stage asynchronous
staging constructions were exact and changed the intended mechanisms. They
did not improve cold service. In particular, asynchronous staging reduced
long-scoreboard exposure but added enough copy, barrier, shared-memory, and
address work to make the kernel slower.

Authority:
`docs/task_workflow/output/nv-qokv-exhaustive-test-invest-result.md`.

This closes those spellings, not all possible hardware-native quantized
pipelines. A new weight layout and matching compute primitive would be a new
mechanism.

### Ordinary overlap is not the general answer

Running two bandwidth-bound streams concurrently does not remove their bytes
and can make them contend for the same memory service. Overlap is valuable
when the two sides use complementary resources, or when fine-grained
readiness allows useful work to replace a real idle interval.

The llama launch-ahead construction mostly hides dispatch and preamble tails.
It does not turn O into a long, concurrent DRAM stream. Tinygrad already has
gapless Flash-combine-to-O dispatch, so copying that policy has only marginal
remaining exposure.

## Mechanism ranking

| Rank | Mechanism | What changes | Current disposition |
| ---: | --- | --- | --- |
| 1 | Cluster-level QKV-attention-O fusion | Keeps intermediate collectives on-chip and schedules producer/consumer tiles together | Exact current-Q4_K O ownership gate failed; do not build the full span from this spelling |
| 2 | Static SM-level persistent task graph | Amortizes physical stream and launch boundaries across dependent operators without polling | First-class bounded O gate completed; current exact ownership loses when launched ahead |
| 3 | Artifact-plus-kernel quantized pipeline | Co-designs stored layout, dequantization, asynchronous movement, and hardware math | Promising, but changes the format and may change numerics |
| 4 | Lower material weight bytes or structured sparsity | Reduces compulsory DRAM traffic | Strongest physical lever; requires model-quality qualification |
| 5 | Cross-request nano-batching | Supplies independent work from other requests | Serving-throughput lever, not strict one-token latency |

Static stripes, wider loads, cache hints, row-local `cp.async`, prefilled task
queues, and naive resident polling are not priority reopenings. They have
already reached exact negative gates for the current work and data layout.

## Relevant published constructions

- [ClusterFusion](https://arxiv.org/abs/2508.18850) introduces cluster-level
  gather and reduction primitives and directly fuses QKV projection,
  attention, and output projection while retaining intermediates on-chip.
- [Mirage Persistent Kernel](https://arxiv.org/abs/2512.22219) lowers a model
  into an SM-level task graph executed by an in-kernel runtime, enabling
  fine-grained cross-operator scheduling without a launch per operator.
- [Ada-MK](https://arxiv.org/abs/2605.11581) uses a compile-time-selected
  megakernel DAG to remove runtime branch and launch costs in decode.
- [MARLIN](https://arxiv.org/abs/2408.11743) co-designs quantized layouts,
  asynchronous movement, pipelining, and striped work assignment. It also
  observes that startup latency and imperfect partitioning matter more for
  smaller matrices and higher-bandwidth devices.
- [LiquidGEMM](https://arxiv.org/abs/2509.01229) changes the quantized format
  and dequantization spelling so load, dequantization, and matrix arithmetic
  can form one fine-grained pipeline.
- [FlashDecoding++](https://arxiv.org/abs/2311.01282) selects different
  implementations for GEMV, flat GEMM, and larger GEMM shapes instead of
  assuming one static dataflow is optimal.

These papers validate mechanism classes. Their reported speedups are not
forecasts for this runtime, weight format, model population, or device.

## Kernel versus token approach

The project should use a **kernel-first experiment inside a token-first
design**.

The kernel gate is needed because it cheaply answers the physics questions:

- Can the mechanism keep a longer service episode active?
- Are material weight bytes unchanged or reduced?
- Is the output exact under the installed arithmetic contract?
- Does it preserve enough row and CTA parallelism?
- Does it avoid spinning, scratch round trips, and extra global reductions?

The token gate is decisive because it answers the lifecycle questions:

- Does producer readiness improve rather than move later?
- Are output ownership and replay boundaries cheaper?
- Does the GPU-union improvement survive host and graph execution?
- Does the change disturb Flash, norms, cache state, or downstream consumers?

The prior QKV result demonstrates why neither level can be skipped: the kernel
physics passed, while the token translation failed.

## Executed cluster discriminator

The cluster path was decomposed into three gates so that implementation cost
was charged before a full fused layer was built.

| gate | result | reading |
| --- | --- | --- |
| Cluster capability and DSM handoff | Exact and operational for cluster sizes 2, 4, and 8 | The RTX 5090 and CUDA toolchain support the mechanism. A naive remote-DSM handoff was competitive with a global scratch handoff only at cluster size 2. |
| ClusterFusion-style O join | Published FP16 atomic join was neither bit-exact nor cheap; deterministic FP32 scratch join was exact but added a global round trip | The paper's output ownership cannot be copied directly under the installed exact arithmetic contract. |
| Production-shaped Q4_K O ownership | Bit-exact, finite, and spill-free for cluster sizes 2 and 4; both lost in hot and rotated-cold timing | The current Q4_K lane ownership requires enough DSM reconstruction and synchronization to erase the service-continuity benefit. |

The promotion-grade rotated-cold medians for one 4096-by-4096 O layer were:

| arm | time | recovery versus installed |
| --- | ---: | ---: |
| Installed one-row/one-warp control | 9.430 us | -- |
| Two-block cluster | 11.228 us | -1.798 us |
| Four-block cluster | 11.642 us | -2.212 us |

Hot timing lost by still more, which rules out an instruction-side win hidden
by cold noise. Both candidates used 45 registers, 512 bytes of shared memory,
and zero spills; the installed control used 43 registers and no shared memory.

This is a closure of the exact tested ownership topology, not of CUDA clusters
or cross-operator fusion in general. It also means the full
QKV-attention-O integration is not currently justified: its mandatory O
building block failed before graph and token-boundary costs were added.

Authority:

- `docs/task_workflow/evidence/nv-cluster-projection-service/gate-r9.json`
- `docs/task_workflow/evidence/nv-cluster-projection-service/o-exactness-r9.json`
- `docs/task_workflow/evidence/nv-cluster-projection-service/o-cluster-r9.json`

## Remaining exact test ladder

The next work should proceed in the following order.

1. **First-class persistent O emitter.** Completed. The UOp emitter is exact,
   finite, spill-free, and reaches standalone O body parity at 1,024 workers.
   Launching it ahead of Flash regresses the complete span because the waiting
   population slows the producer. Do not integrate it into production.
2. **One-layer static task graph.** Closed for the current full-readiness O
   ownership. A same-grid spelling cannot expose material full-row O work
   before all Flash heads are ready; the exact partial-readiness construction
   already lost to scratch and synchronization tax.
3. **Full-token reverse bracket.** Not justified for persistent O because the
   one-layer prerequisite failed.
4. **Artifact-plus-kernel co-design.** Now the next strict-token mechanism:
   test a genuinely new stored layout and matching dequantization/compute
   pipeline. Do not reopen byte-preserving qdata transpose, cache hints, or
   row-local asynchronous staging; their exact cold gates already failed.
5. **Byte reduction.** Qualify a smaller representation or structured
   sparsity for model quality, then measure compulsory DRAM-byte reduction.
   This is the strongest remaining physical lever, but it changes the model
   artifact and therefore needs a separate acceptance contract.

Cross-request nano-batching remains a throughput test, not a strict
single-token latency test.

## Next exact discriminator

The next high-information test is now a one-layer statically scheduled
persistent span with these properties:

1. Cover Q, K, V, decode attention, and O as a dependent task graph.
2. Retain the installed weight formats and exact row-reduction associations.
3. Preserve the production-sized Q/K/V and O worker populations.
4. Use event/barrier publication, not resident spin polling.
5. Keep attention intermediates on-chip when ownership permits it.
6. Measure the complete span against the installed multi-kernel chain under
   hot and rotated-cold conditioning.
7. Stop before production integration unless the exact gate predicts a
   material full-token recovery.
8. Promote only after an exact full-token reverse bracket beats both controls.

This discriminator separates two outcomes cleanly:

- A pass means the missing lever was service topology and runtime ownership.
- A fail means the exact current-format path is near its construction wall,
  and the remaining large levers require byte or representation changes.

## Arithmetic exposure

The recorded Q/O/K/V pool contains roughly 0.87 GB of material weights and
currently costs roughly 0.84 ms per token. If the complete pool sustained 80%
of the fitted long-stream rate, the arithmetic exposure would be roughly
0.21 ms/token.

Near the present single-token latency, full translation of that exposure would
be on the order of fourteen additional tokens per second. This is a ceiling,
not a forecast: the present independent stream population cannot reach it by
kernel-local tuning alone.

For any measured recovery `dT` from an endpoint latency `T`, the token-rate
translation is:

```text
gain_tok_s = 1 / (T - dT) - 1 / T
```

Only a complete token-wall result may populate `dT` in the booked ledger.
