# Generic dense striped projection substrate

## Status

**Deferred.** The idea is worth retaining as a generic dense-model substrate,
but it is not the current optimization target. The active ledger points to
flash attention as the largest immediate dense-decode loss.

No performance recovery is booked by this document.

## Decision

Do not start with a whole-token persistent megakernel. If this direction is
reopened, first test a small, generic striped projection-group microgate. Build
the reusable planner and production route only if that discriminator raises
the complete Q+K/V service rate enough to predict material token-wall recovery.

The desired abstraction is reusable across dense Llama-, Qwen-, Mistral-, and
similar decoder families. It must be keyed by shapes, quantization formats,
arithmetic contracts, output ownership, and device capability—not by model
name or individual layer number.

## Motivation from the current ledger

The current projection rates are:

| family | payload | time | aggregate payload rate |
|---|---:|---:|---:|
| Q | 340 MB/token | 297.216 us/token | 1.143 TB/s |
| O | 340 MB/token | 304.976 us/token | 1.114 TB/s |
| K/V | 189 MB/token | 233.568 us/token | 0.811 TB/s |

These low aggregate rates do not imply proportional standalone-kernel
headroom. The measured size-aware model is approximately:

```text
body time = 3.27 us of physical stream/launch ramp
          + payload / 1.75 TB/s asymptotic rate
```

At 80% of the fitted asymptote, or 1.40 TB/s, Q and O would each need about
6.74 us per layer. Their size-aware reconstruction is about 8.66 us. K/V would
need about 2.94 us per physical launch, below the measured fixed ramp before
any weight payload is charged. Therefore 80% is not a coherent standalone
target while the current stream count remains unchanged.

The substrate must change the fixed term by continuously assigning work across
multiple projections. It is not another wider-load or isolated-GEMV project.

## Proposed generic abstraction

Conceptually:

```python
ProjectionGroupSpec(
  input_width=K,
  projections=[
    ProjectionTask(output_rows=N0, weight_format=F0,
                   reduction=R0, output_ownership=O0),
    ProjectionTask(output_rows=N1, weight_format=F1,
                   reduction=R1, output_ownership=O1),
    ProjectionTask(output_rows=N2, weight_format=F2,
                   reduction=R2, output_ownership=O2),
  ],
  activation_format=A,
  schedule=STRIPED_STATIC,
)
```

The generated route key should contain:

```text
backend and architecture
input width
ordered output widths
ordered weight formats
activation representation
row-reduction contracts
output ownership and sinks
schedule descriptor
```

It must not contain a model identity or a manually selected block list. A
shape-specialized generated artifact is acceptable; shape specialization is
not model hard-coding.

## Execution model

A fixed CTA population receives virtual row tiles from the concatenated set of
compatible projections:

```text
shared activation
      |
      +-- projection 0 row tiles
      +-- projection 1 row tiles
      +-- projection 2 row tiles
```

The initial spelling should use a deterministic static stripe rather than a
runtime atomic work queue. Each tile retains its format-specific load/dequant
primitive, lane ownership, reduction association, and output ABI. Workgroup
regions must remain uniform. The scheduler changes physical task assignment;
it does not silently change row arithmetic.

The first useful consumer is dense attention Q/K/V, especially GQA shapes
where Q has many more rows than K or V. The same abstraction can later express
dense MLP gate/up, which is another group of sibling projections sharing one
activation. O and down are single-consumer projections and do not independently
benefit from grouping, though they can reuse the row-task specification.

## Existing substrate

The repository already contains most semantic components:

- multi-output program execution;
- Q4_K and Q6_K row decoders;
- exact row-reduction implementations;
- shared-Q8 activation providers;
- separate caller-owned Q/K/V outputs;
- producer-owned KV-cache sinks;
- mixed Q4/Q6 handling;
- workgroup-uniform conditional regions;
- qualification records and full-token reverse brackets.

The current implementations are still fixed to the production 4096/1024
geometry and explicit admission flags. Relevant sources are:

- `tinygrad/llm/q4k_kv_pair.py`
- `tinygrad/llm/shared_q8_attention.py`
- `tinygrad/llm/kernel_program.py`
- `tinygrad/llm/model.py`

The required work is a refactor and new task scheduler, not a greenfield
quantized-kernel stack.

## Expected engineering lift

| package | rough lift | disposition |
|---|---|---|
| generic striped QKV microgate | small to medium | first and mandatory |
| reusable task descriptors, lowering, planner admission, and tests | medium to large | only after microgate passes |
| persistent attention-layer megakernel | large | later generation |
| whole-token persistent runtime | very large compiler/runtime project | not justified by current evidence |

As an order-of-magnitude estimate, the discriminator should be a few hundred
lines. A production-quality generic projection-group substrate, including
planner integration and a meaningful shape/format test matrix, is likely on
the order of one thousand lines. A whole-token megakernel is several thousand
lines plus residency, synchronization, deadlock, resource-model, and compiler
work; it should not be treated as a continuation of the small experiment.

## Potential token translation

The current Q+K/V pool is approximately:

```text
payload       529 MB/token
time          530.784 us/token
rate          0.997 TB/s
endpoint      4.166708 ms/token = 239.998 tok/s
```

If a grouped scheduler changes only this pool, the gross exposure is:

| grouped QKV rate | pool recovery | projected latency | projected throughput | projected gain |
|---:|---:|---:|---:|---:|
| 1.05 TB/s | 26.905 us | 4.139803 ms | 241.557 tok/s | +1.560 tok/s |
| 1.10 TB/s | 49.809 us | 4.116899 ms | 242.901 tok/s | +2.904 tok/s |
| 1.15 TB/s | 70.721 us | 4.095987 ms | 244.141 tok/s | +4.144 tok/s |
| 1.20 TB/s | 89.890 us | 4.076818 ms | 245.289 tok/s | +5.292 tok/s |
| 1.25 TB/s | 107.526 us | 4.059182 ms | 246.355 tok/s | +6.357 tok/s |
| 1.30 TB/s | 123.805 us | 4.042903 ms | 247.347 tok/s | +7.349 tok/s |
| 1.40 TB/s | 152.875 us | 4.013833 ms | 249.138 tok/s | +9.141 tok/s |

These are arithmetic exposure ceilings. Previous full-grid producers proved
that device-body improvement need not translate into token wall. Only a
same-session full-token bracket may book recovery.

The practical bands are:

- 1.10 TB/s: useful substrate result; about one third of the retained llama
  throughput gap in gross arithmetic;
- 1.20 TB/s: strong result; approximately +5.3 tok/s if it translates;
- 1.30 TB/s: near-parity-sized exposure;
- 1.40 TB/s: 80% of the fitted asymptote and slightly beyond the retained llama
  endpoint if fully translated.

## Test-before-invest gate

The first microgate must:

1. Accept a list of projection descriptors, not Q/K/V-specific arguments.
2. Exercise at least Q4/Q4/Q4 and Q4/Q4/Q6 projection groups.
3. Preserve each installed row's arithmetic and output exactly for exact
   routes; shared-Q8 routes retain their existing full-logit semantic gate.
4. Produce separate caller-owned outputs and support an optional direct
   KV-cache sink.
5. Read no additional material weight bytes and introduce no transport output.
6. Compare the complete grouped span against the installed Q plus K/V span
   under matched hot and cold conditions.
7. Reach at least 1.10 TB/s aggregate before planner investment; 1.20 TB/s is
   the preferred production-investment threshold.
8. Predict or measure at least about 50 us/token of complete-population device
   recovery before a production route is built.
9. Pass a reps-at-least-9 control/candidate/control full-token bracket and beat
   both controls before any recovery is booked.

Decision tree:

```text
generic striped microgate
        |
        +-- below 1.10 TB/s
        |      stop: insufficient service-rate change
        |
        +-- 1.10 to 1.20 TB/s
        |      run full-population device profile
        |      invest only if union/critical span translates
        |
        +-- at least 1.20 TB/s
               build generic planner integration
               run semantic and full-token wall gates
```

An incomplete output contract, missing route population, or opaque transport
is an accounting wall and must be repaired before judgment. A complete
wall-negative bracket closes that scheduler construction.

## Research lineage

The design is informed by, but does not directly transplant, these systems:

- FlashDecoding++, flat-GEMM hardware adaptation and double buffering:
  `https://arxiv.org/abs/2311.01282`
- MARLIN, asynchronous mixed-precision pipelines and striped/Stream-K-like
  task partitioning: `https://arxiv.org/abs/2408.11743`
- LiquidGEMM, implicit load/dequant/MMA pipelining:
  `https://arxiv.org/abs/2509.01229`
- Mirage Persistent Kernel, SM-level task graphs and persistent execution:
  `https://arxiv.org/abs/2512.22219`
- Ada-MK, compile-time fixed decode megakernels:
  `https://arxiv.org/abs/2605.11581`
- AutoMegaKernel, batch-one persistent decode including an RTX 5090 result:
  `https://arxiv.org/abs/2606.09682`

The published systems use different models, formats, batch regimes, and GPU
generations. Their speedups are evidence for mechanisms, not performance
forecasts for this repository.

## Reopen condition

Reopen this document after the current flash campaign, or earlier only if
flash reaches a complete mechanism wall. Begin with the generic striped
microgate; do not begin by refactoring production routes or building a
whole-token megakernel.
