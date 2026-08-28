# Dense quantized representation and kernel co-design scope

## Decision

The next dense-decode campaign should co-design the stored weight
representation and its consuming kernel. This is a real path because it can
change the two quantities that current scheduling experiments could not:

1. compulsory bytes streamed once per token; and
2. the dependency chain needed to turn those bytes into dot products.

This is not authorization to invent one model-specific packed kernel. The
deliverable is a reusable dense-linear substrate driven by tensor facts,
format descriptors, and target capabilities.

The development rule remains test then invest:

1. prove byte economics and numerical admissibility without a production
   route;
2. prove one representative cold kernel including all provider work;
3. cover the dense role and shape matrix;
4. qualify recurrent model quality;
5. integrate only after those gates pass; and
6. book recovery only from a complete token reverse bracket.

## Scope boundary

Included:

- batch-one autoregressive decode;
- dense transformer projection weights;
- attention Q, K, V, and O projections;
- FFN gate, up, and down projections;
- dense vocabulary projection when it uses a compatible format;
- source Q4_K and Q6_K artifacts;
- NVIDIA Blackwell as the first measured target;
- target-neutral format and route metadata;
- correctness-compatible prefill fallback.

Explicitly excluded from the first campaign:

- MoE expert routing and expert residency;
- cross-request batching as a latency claim;
- speculative decoding;
- KV-cache compression;
- attention score/value kernel changes;
- training a new base model from scratch;
- replacing GGUF as the source-model interchange format;
- claiming performance on unmeasured backends.

Models must be verified dense from model facts before entering the test
matrix. A filename or parameter count is not sufficient evidence.

## Current factual baseline

### Storage geometry

The installed source formats are block formats rather than ordinary dtypes:

| format | elements/block | bytes/block | effective bits/weight | current storage role |
| --- | ---: | ---: | ---: | --- |
| Q4_K | 256 | 144 | 4.5000 | `uint32 words` |
| Q5_K | 256 | 176 | 5.5000 | research conversion support |
| Q6_K | 256 | 210 | 6.5625 | `uint16 halfs` |
| IQ4_XS | 256 | 136 | 4.2500 | generic GGUF dequant only |
| MXFP4 | 32 | 17 | 4.2500 | generic GGUF dequant only |

Q4_K's 144-byte block contains two FP16 super-scales, twelve bytes of packed
six-bit scale/min metadata, and 128 bytes of four-bit codes. The installed
kernel must unpack both the codes and the affine metadata before issuing the
FP32 accumulation.

### Loader and route substrate already available

The repository already separates several concerns that the new path should
reuse:

- `QuantFormat` owns block geometry and storage roles.
- `Q4KPrimitiveStorage` and `Q6KPrimitiveStorage` own packed sidecars and
  persistent-memory accounting.
- GGUF metadata identifies tensor name, source type, shape, byte offset, and
  backing storage.
- `ModelRoutePlan` assigns role and shape without hard-coding a model class.
- route admission separates target capability from measured promotion.
- model facts bind logical tensor identity to concrete packed allocations.
- research `KernelProgram` execution provides a closed-default kernel gate.

The new implementation should extend these boundaries instead of adding a
second independent loader or another environment-selected production path.

### Closed prior work

The scope must not silently repeat the following experiments:

- byte-preserving Q4_K qdata transposition reduced hot instructions but did
  not improve rotated-cold service;
- wider loads removed cache-hit instruction work but not compulsory DRAM;
- cache hints and row-local asynchronous staging added more issue, barrier, or
  shared-memory work than they hid;
- exact CTA, stripe, cluster, and persistent ownership variants did not raise
  the complete cold dependent span;
- post-hoc Q6_K to Q4_K reduced bytes but failed local quality badly;
- post-hoc Q6_K to Q5_K passed its kernel/byte gate but failed recurrent
  full-logit quality, including every single FFN-down placement tested;
- coarse row and end-to-end shard selectors did not produce an admissible
  Q5_K subset;
- Q8 activation ownership was numerically inadmissible under the current
  recurrent logit contract.

These results close those constructions. They do not close a new format
trained or calibrated from higher-precision weights with a matching kernel.

## Two distinct work lanes

Representation co-design contains two different scientific questions. They
must not share a verdict.

### Lane A: simpler dependency graph

Lane A may retain the same effective bytes as Q4_K but encode values so the
kernel performs materially less metadata decode and has a shorter load-to-use
chain. Its acceptance contract is a measured cold service-rate increase.

This lane is numerical unless the new encoding reproduces every Q4_K decoded
weight exactly. A simple repack is lossless; a new symmetric quantizer is not.

### Lane B: fewer material bytes

Lane B reduces effective stored bytes, including every scale, bitmap, index,
alignment pad, correction, and tied-weight duplicate. Its acceptance contract
is model quality plus a measured compulsory-DRAM reduction.

Lane B has the larger roofline value. It also requires a new model artifact
and cannot inherit the exact-token contract automatically.

## Candidate representation families

### R0: lossless Q4_K physical re-layout

R0 preserves every decoded Q4_K value and all 144 bytes per 256 weights. It
may only reorder fixed metadata or code planes offline so the consumer can
issue a better-aligned or less dependent load sequence.

This is the exact control for Lane A, but it is low priority. The existing
qdata-transpose gate already proved that fewer load instructions can be a hot
win and a cold loss. R0 may be reopened only for a new layout with a static
SASS argument showing a materially different cold load-to-use graph—not for
another arbitrary transpose. Its contract is bitwise output, identical DRAM
bytes, and higher rotated-cold service rate.

### R1: plane-aligned symmetric Q4, group 32

Provisional identifier: `S4_G32_P256`.

One 256-weight block contains:

```text
16 bytes   eight FP16 scales, one per 32 weights
128 bytes  signed four-bit codes in a kernel-selected code plane
---------
144 bytes  4.5 bits/weight
```

The decode is:

```text
weight = scale[group] * signed_nibble
```

This deletes Q4_K's minimum correction, packed six-bit scale/min extraction,
and corresponding integer-to-float dependency chain. The metadata plane is
one aligned 16-byte vector load. The code plane layout is selected jointly
with the warp consumer and is created offline—never transposed per token.

Why it ranks first:

- same byte count isolates dependency/service-rate value;
- no variable-length decoding or index traversal;
- easy independent oracle;
- can consume FP16 activations first, avoiding the already-failed Q8
  assumption;
- provides the base grammar for lower-byte symmetric variants.

It is not expected to win merely because it has fewer static instructions.
It must raise rotated-cold achieved rate or it stops.

### R2: plane-aligned symmetric Q4, group 64

Provisional identifier: `S4_G64_P256`.

```text
8 bytes    four FP16 scales, one per 64 weights
128 bytes  signed four-bit codes
---------
136 bytes  4.25 bits/weight
```

This removes 5.56% of Q4_K bytes while retaining a simple fixed-size block.
Its larger group increases quantization error, so activation-aware scale
selection and recurrent quality are mandatory.

R2 is tested only after R1 establishes that the simple symmetric kernel is a
competitive consumer. Otherwise a lower-byte version of a slow grammar has
no production path.

### R3: symmetric Q5 for sensitive Q6_K roles

Provisional identifier: `S5_G32_P256`.

```text
16 bytes   eight FP16 scales
160 bytes  signed five-bit codes
---------
176 bytes  5.5 bits/weight
```

The byte count matches Q5_K but the decode grammar is designed for the target
consumer rather than GGML compatibility. This is the first lower-byte
candidate for Q6_K FFN-down, attention-V, and vocabulary tensors when Q4 is
too destructive.

The prior Q6-to-Q5 result proves that bytes and a direct kernel can win while
recurrent quality fails. R3 therefore starts with calibration or fine-tuning;
it may not use ordinary post-hoc per-block rounding as its final candidate.

### R4: symmetric base plus bounded correction

Provisional family: `S4_G64_CK` or `S3_G32_CK`.

A small base format is paired with a bounded, fixed-layout correction such as:

- a fixed number of outlier positions and residual values per block;
- a uniform per-tensor low-rank residual;
- a small dense residual only for calibration-selected tensors; or
- a second bit-plane enabled at tensor granularity.

All correction bytes count toward effective bits/weight. The first version
must be uniform per tensor, not variable per row or block, so warps do not
diverge and the route remains descriptor-driven.

This family is accepted only if the correction restores recurrent quality
while leaving a material byte advantage after indices, padding, and extra
loads are charged.

### R5: hardware-native FP4 plus activation co-design

This lane targets Blackwell tensor-core-compatible FP4 storage and a matching
activation representation. It is deliberately later because its full cost
includes:

- activation quantization;
- activation scales and any sums/zero points;
- output accumulation and conversion;
- padding to hardware tile shapes;
- batch-one utilization; and
- recurrent numerical error.

A weight-only FP4 microbenchmark is not an admissible result. The gate must
include the activation provider and prove that the selected instructions are
actually emitted. Existing generic MXFP4 dequant support is useful as an
oracle, not proof of a fast decode route.

### R6: structured sparsity

N:M sparsity can remove more bytes than format tuning but requires
sparsification-aware fine-tuning and a matching sparse execution primitive.
It is a separate training lane because post-hoc zeroing is unlikely to retain
the installed quantized model's quality.

Sparse metadata, tile padding, unsupported dense fallbacks, and any dense
residual all count against its effective byte ratio. This lane begins only
after a native sparse batch-one kernel gate proves meaningful device value.

### Deprioritized families

- General entropy coding: random-access GPU decompression and variable-length
  blocks conflict with short GEMV service unless a fixed-rate tile cache is
  demonstrated first.
- Runtime transposition: already fails the cold accounting and cannot reduce
  stored bytes.
- Arbitrary per-row mixed formats: prior row sensitivity was non-monotonic and
  divergent consumers are expensive. Tensor-level selection comes first.
- Expanded dequantized sidecars: may help prefill but increase decode DRAM and
  are outside this strict-token objective.

## Source-weight and calibration contract

There are two valid source modes:

1. **Lossless-layout research:** the existing GGUF packed bytes are the source
   authority. The candidate must reproduce every decoded weight or kernel
   output under its declared exactness contract.
2. **Numerical-format research:** use the original FP16/BF16 checkpoint, or a
   documented higher-precision training authority. The existing Q4_K GGUF is
   the quality baseline, not the quantizer's source.

Dequantizing Q4_K and requantizing it is permitted only as a cheap plumbing
smoke test. It cannot establish the production quality ceiling because the
discarded source information is unrecoverable.

Calibration requirements:

- calibration prompts and evaluation prompts are disjoint;
- activations are captured at every candidate linear input;
- the objective measures `W*x`, not weight error alone;
- sensitive placements are evaluated through recurrent logits, not ranked by
  local relative error;
- scale/correction selection is deterministic from a recorded seed and
  configuration;
- the source model, tokenizer, prompt set, converter, and candidate manifest
  all receive content hashes.

A minimal activation-aware objective is:

```text
minimize over encoded Wq: sum_x ||(W - Wq) x||^2
```

More advanced reconstruction may propagate calibration activations through
the already-quantized preceding layers, but its exact procedure must be part
of the artifact manifest.

## Artifact architecture

### Sidecar first

The first implementation should produce an immutable sidecar rather than a
new GGUF type. GGUF remains the source and fallback. The sidecar is admitted
only when its source-model hash matches.

Suggested logical manifest:

```text
schema/version
source_model_sha256
source_tensor_table_sha256
converter_commit/config_hash
calibration_manifest_hash or null
format descriptors
target layout family
tensors[]:
  name
  source quant/type
  role
  rows, cols
  format id and parameters
  byte offset, payload bytes, padded bytes, alignment
  payload sha256
  tied-weight owner
```

The binary payload is fixed-offset and alignment-explicit. No Python object,
pickle, executable code, or device pointer is serialized.

### Logical quantization versus physical layout

The design must distinguish:

- logical values: bit width, signedness, group size, scale type, zero-point or
  correction semantics; and
- physical layout: scale plane ordering, code plane swizzle, row tile,
  K-block tile, alignment, and target grammar.

One logical format may have several target layouts. A physical layout is not
allowed to masquerade as a new numerical quantizer.

### Memory ownership

Derived packed buffers are immutable model parameters and must use the
existing model-parameter allocation owner. The ledger records:

- source payload bytes;
- candidate payload and padding bytes;
- duplicate persistent bytes;
- shared/aliased bytes;
- temporary conversion bytes; and
- fallback storage retained at runtime.

Production promotion should not require both full source and candidate GPU
copies indefinitely. A rollback may reopen from the source file, but the
steady-state memory claim must charge the actual resident set.

## Software substrate

### Proposed production-neutral types

The scope calls for a generic descriptor rather than one class per experiment:

```text
DenseQuantFormat
  logical format identity
  block elements and block bytes
  group geometry
  scale/zero/correction semantics
  storage planes

DenseQuantLayout
  target capability requirements
  tile and swizzle geometry
  plane alignment
  kernel grammar id

DenseQuantStorage
  packed tensors by named plane
  source/candidate byte accounting
  artifact provenance
  memory ownership
```

These may extend `qk_layout.py` and the existing primitive-storage model, but
must not turn block formats into element-addressable `DType` values.

### Converter boundary

Research converters belong under `extra/llm_research/quant/` and consume a
common streamed tensor-source interface. Required adapters are:

- GGUF packed source for lossless/plumbing gates;
- GGUF dequantized source, labeled post-hoc research only;
- FP16/BF16 checkpoint source for numerical candidates.

The converter must support one tensor, one role population, and whole-model
output with the same encoding code. Test-only NumPy references are kept
separate from the production loader.

### Loader boundary

The loader performs:

1. manifest/schema validation;
2. source hash and tensor-table validation;
3. tensor shape/role/tied-weight validation;
4. target capability and measured-promotion lookup;
5. immutable packed-view or sidecar materialization;
6. model-fact binding; and
7. fail-closed fallback to the source GGUF route.

Unknown format versions, missing tensors, hash mismatches, unsupported target
layouts, memory-cap failures, and partial sidecars must never silently select
the candidate.

### Route boundary

Route choice is keyed by:

```text
(logical format, physical layout, role, rows, cols, target capabilities,
 measured promotion record)
```

It is not keyed by model filename or a global environment switch. Research
leases may select a candidate explicitly; production uses a generated route
record with a rollback.

## Kernel architecture

### Common decode atom

Each format supplies one block-dot grammar:

```text
load metadata plane
load fixed code plane
decode lane-owned values
multiply by FP16 activation or selected activation atom
accumulate FP32 in a declared order
```

Role kernels compose this atom with descriptors for:

- output rows and K extent;
- rows/warp and warps/CTA;
- paired gate/up or K/V ownership;
- direct output versus partial reduction;
- FP16 cast, residual add, SiLU/multiply, and cache-store epilogues; and
- activation provider identity.

The format owns data interpretation. The role owns topology and epilogue.
Neither should clone the other.

### First kernel lane

R1 initially consumes the existing FP16 activation and produces FP32 output.
This isolates the weight-format dependency graph. It must support:

- one 4096-by-4096 Q/O-shaped tensor;
- one narrow K/V-shaped tensor discovered from model facts;
- one wide gate/up-shaped tensor; and
- one down-shaped tensor with its true K extent.

Shapes are parameters to the emitter. The concrete first fixtures may come
from the installed 8B model, but no `4096`, `1024`, or `12288` constant may
become the format API.

### Hardware-native lane

R5 begins only after a standalone instruction-admission probe proves the
target renderer/toolchain can issue the intended FP4/FP8 operation. The full
gate includes activation quantization and scale movement. A scalar dequant
fallback is required for correctness but cannot be used as performance
evidence.

### Numerical association

Every kernel declares one of two contracts:

- `exact_source_association`: bitwise against the source-format oracle; or
- `numerical_candidate`: finite output under a predeclared error and recurrent
  model-quality contract.

No candidate may switch contracts after timing results are known.

## Dense role and shape matrix

Every promoted logical format must be evaluated for the roles it claims:

| role family | logical shape | special lifecycle responsibility |
| --- | --- | --- |
| attention Q | `D x D` | often consumes normalized FP16 and feeds Q norm/RoPE |
| attention K | `KVD x D` | feeds K norm/RoPE/cache |
| attention V | `KVD x D` | may currently be Q6_K; feeds V cache |
| attention O | `D x D` | consumes Flash output and adds residual |
| FFN gate/up | `F x D` twice | paired weights, SiLU and multiply |
| FFN down | `D x F` | often Q6_K; adds residual |
| vocabulary | `V x D` | large stream plus sampler boundary |

`D`, `KVD`, `F`, and `V` come from model facts. Group size divisibility,
padding, tied tensors, GQA ratio, and vocabulary tails are validated rather
than assumed.

The initial campaign may promote formats per role. It must not require one
numerical format to fit every role equally well.

## Test-invest ladder

### Gate 0: accounting feasibility

Inputs:

- real model tensor table;
- source and proposed bytes/block;
- padding/alignment;
- correction/index bytes;
- current per-role cold duration and compulsory bytes.

Outputs:

- effective bits/weight per tensor and role;
- bytes saved per token;
- optimistic DRAM-time ceiling;
- same-byte service-rate requirement; and
- required quality coverage.

Stop when:

- effective bytes do not materially beat the source for a byte candidate;
- padding erases the nominal saving;
- only tensors outside the measured critical path are covered; or
- the maximum token exposure is below the agreed investment floor.

For tensor `i`:

```text
source_bytes_i    = N_i * K_i / block_elems_source * block_bytes_source
candidate_bytes_i = payload_i + metadata_i + correction_i + index_i + padding_i
byte_floor_i      = (source_bytes_i - candidate_bytes_i) / measured_long_rate
```

This floor is exposure, not booked recovery.

### Gate 1: deterministic artifact and oracle

Required tests:

- deterministic converter output from identical inputs;
- manifest and payload hashes;
- bounds and alignment validation;
- random legal block decode against an independent CPU oracle;
- real tensor sampled decode;
- tied-weight and split-GGUF handling;
- corruption and source-hash rejection;
- no hidden runtime transpose;
- exact byte ledger including padding and corrections.

Stop on any ambiguous format interpretation or nondeterministic artifact.

### Gate 2: one-tensor semantic kernel

Use at least three real weight tensors and recorded real activations. Require:

- finite outputs;
- exact agreement for an exact contract;
- predeclared local error for a numerical contract;
- guard zones and read-only input checks;
- deterministic output at fixed inputs;
- no spills;
- expected SASS instruction family and load widths;
- included activation-provider and epilogue work.

Timing uses R9 hot and independently rotated-cold brackets. NCU records:

- DRAM read bytes;
- L2 bytes/hit rate;
- duration and achieved DRAM throughput;
- long-scoreboard stall;
- executed instructions;
- registers, shared memory, and spills; and
- selected tensor-core or integer-dot instructions when claimed.

Advance rules:

- R1 must show a material cold duration/rate win, not merely fewer hot
  instructions.
- R2--R6 must show the promised DRAM-byte reduction and a cold duration win.
- provider cost is included before comparison.
- a hot-only win or unchanged cold rate stops that spelling.

### Gate 3: role population census

Run every claimed role and shape discovered from at least one complete dense
model. Report population-weighted results, not only the best tensor.

Require:

- no material regression in any high-mass role;
- total predicted body recovery remains positive after provider and epilogue
  costs;
- graph census shows exactly the intended candidate population;
- fallback covers unsupported shapes; and
- resident candidate storage matches the byte ledger.

Formats may be admitted per role if quality and performance differ.

### Gate 4: progressive quality

Quality is evaluated before a full timing integration.

Level A — weight and local projection:

- weight-domain statistics for diagnosis only;
- projection relative L2/cosine/max error on recorded real activations;
- per-role and per-layer distributions, not only aggregate means.

Level B — one-block recurrent logits:

- one candidate block at a time;
- several recurrent steps after a fixed prefix;
- full logits saved for control and candidate;
- finite logits, relative L2, top-k membership/order, argmax, and decision
  margin.

Level C — progressive dose:

- least-sensitive measured placements first;
- multiple layers and then the full claimed role population;
- error checked at every recurrent step;
- no extrapolation from local weight error.

Level D — dense-model evaluation:

- disjoint prompt suites and context depths;
- perplexity or task metrics appropriate to each verified dense model;
- deterministic greedy traces as diagnostics;
- comparison against the source GGUF baseline and, when available, the
  higher-precision source.

Thresholds and dose order are frozen before timing. Short greedy-token
agreement alone is never sufficient.

Stop when a single economically meaningful dose fails and there is no
predeclared calibration/training action that changes the representation.
Reclassify the work as a training project rather than continuing kernel
tuning on an inadmissible artifact.

### Gate 5: one-block lifecycle

Install the sidecar on one real block with ordinary graph execution. Require:

- graph census and kernel identity;
- no source/candidate double materialization outside the declared memory
  mode;
- all adjacent casts, providers, epilogues, and cache stores accounted;
- identical outputs for exact formats or qualified outputs for numerical
  formats;
- complete block-span win under cold production conditioning.

A faster isolated kernel that loses at the program/output boundary stops.

### Gate 6: full-token reverse bracket

Only a Gate-5 pass justifies whole-model installation.

Protocol:

- installed control A;
- candidate B;
- installed control C;
- at least seven, preferably nine, balanced brackets;
- locked clock/power/thermal protocol;
- same prompt, depth, delivery policy, and graph settings;
- token hashes for exact formats;
- saved full logits and quality authority for numerical formats;
- device and host wall reported separately;
- candidate graph census and resident-memory ledger.

Promotion requires B to beat both controls or a predeclared robust midpoint
criterion. Node-sum recovery is explanatory only.

### Gate 7: generalization

Before calling the substrate generic, repeat on:

- at least three verified dense models;
- at least two hidden widths/FFN ratios;
- all claimed role families;
- several context depths, including a Flash geometry boundary;
- the initial NVIDIA target plus correctness fallback on another target when
  available; and
- prefill correctness, even if prefill performance uses the source route.

One model may receive a promotion record before the generic claim, but the
format API and artifact schema must already be shape-generic.

## Performance accounting and token translation

For an installed token latency `T` and a measured complete-token recovery
`dT`:

```text
gain_tok_s = 1 / (T - dT) - 1 / T
```

Only Gate 6 may supply booked `dT`.

Before Gate 6, three ceilings are kept separate:

1. **byte ceiling:** saved material bytes divided by measured long-stream
   rate;
2. **kernel ceiling:** sum of valid cold per-call recoveries across the exact
   installed population; and
3. **lifecycle ceiling:** complete one-block span recovery after providers and
   epilogues.

The smallest ceiling is the honest forecast. No rate assumption may assign a
short stream the long-stream asymptote without changing its measured service
episode.

## Required test matrix

Each candidate record includes:

| axis | required values |
| --- | --- |
| source | higher-precision authority; post-hoc source only for smoke |
| logical format | R1--R6 identifier and exact parameters |
| physical layout | scale/code/correction planes and swizzle id |
| role | Q, K, V, O, gate, up, down, vocabulary as claimed |
| shape | fact-derived `N x K` |
| activation | FP16 first; Q8/FP8 only with provider included |
| epilogue | plain, residual, FP16/cache store, gate/up fusion as applicable |
| cache state | hot and independently rotated-cold |
| quality dose | singleton, progressive role dose, whole claimed population |
| context | short, authority depth, and geometry boundary |
| model | at least three verified dense models before generic claim |
| target | measured target plus correctness fallback |

## Failure modes to design out

- Measuring a transcode while excluding its load-time or per-token cost.
- Reporting nominal code bits while ignoring scale, index, correction, and
  alignment bytes.
- Keeping source and candidate resident while claiming candidate memory size.
- Using dequantized Q4_K as if it were the original training weight.
- Selecting rows or blocks from local weight error after downstream
  sensitivity proved non-monotonic.
- Timing a hardware-native weight consumer without its activation provider.
- Treating emitted PTX as proof that the desired SASS instruction survives.
- Letting candidate/control share warmed weights in a cold comparison.
- Changing accumulation association without changing the declared numerical
  contract.
- Comparing only greedy tokens after recurrent full-logit drift has failed.
- Hard-coding layer counts, hidden sizes, GQA ratios, or tensor names into the
  format.
- Promoting a one-shape result as a generic dense substrate.
- Modifying production admission before primitive, quality, and lifecycle
  gates pass.

## Work packages and deliverables

### WP0 — economics and source readiness

Deliver:

- verified dense-model inventory;
- higher-precision source availability and hashes;
- current source byte census by role;
- candidate effective-byte calculator;
- calibration/evaluation prompt manifests;
- predeclared quality and performance thresholds.

Exit: one candidate has material exposure and a valid source/calibration path.

### WP1 — format descriptor and sidecar

Deliver:

- logical/physical format descriptors;
- versioned manifest schema;
- deterministic one-tensor and whole-model converter;
- independent CPU decode oracle;
- corruption, hashing, tying, and alignment tests.

Exit: Gate 1 passes without production routing.

### WP2 — R1 kernel discriminator

Deliver:

- FP16-activation `S4_G32_P256` block-dot emitter;
- plain and residual-add direct-output kernels;
- real Q/O, K/V, gate/up, and down fixtures;
- SASS, NCU, hot, and rotated-cold evidence.

Exit: material cold service-rate win. If R1 fails, stop the symmetric grammar
before building lower-byte variants.

### WP3 — lower-byte calibrated candidates

Deliver:

- R2 and R3 converters from higher-precision weights;
- activation-aware calibration;
- optional R4 bounded correction search;
- byte and local projection ledger;
- progressive recurrent quality results.

Exit: at least one economically meaningful population passes quality.

### WP4 — complete role kernels

Deliver:

- descriptor-driven topology variants;
- direct outputs and required epilogues;
- paired gate/up or K/V only when their complete span wins;
- role-population performance census;
- unsupported-shape fallback tests.

Exit: positive population-weighted cold recovery with no major role regression.

### WP5 — loader and one-block integration

Deliver:

- sidecar validation/materialization;
- model-fact and allocation-owner bindings;
- closed-default research admission;
- memory ledger and graph census;
- one-block lifecycle bracket.

Exit: Gate 5 passes.

### WP6 — quality and token promotion

Deliver:

- full claimed-dose recurrent quality;
- multi-prompt/context evaluation;
- full-token A/B/C bracket;
- installed population and memory census;
- route-policy record and rollback;
- cross-model generalization report.

Exit: measured token win under the declared quality contract.

### WP7 — optional hardware-native and sparse lanes

R5 and R6 are independent follow-ups. They reuse the artifact, quality,
accounting, and route substrate but require their own instruction and training
gates.

## First decisive experiment

The first experiment is intentionally smaller than building a new model
artifact:

1. Implement the `S4_G32_P256` CPU converter and independent decoder.
2. Convert three real tensors from dequantized Q4_K only as a plumbing and
   performance smoke; label the artifact post-hoc and non-promotable.
3. Emit the FP16-activation block-dot kernel with the same row reduction and
   epilogue as the installed route.
4. Validate finite output and the predeclared numerical error.
5. Compare SASS dependency classes and R9 hot/rotated-cold timing.
6. Run NCU and require unchanged 144-byte/block DRAM plus a material duration
   or achieved-rate win.
7. Stop the grammar if the result is hot-only or cold-neutral.
8. If it passes, acquire/use higher-precision weights and proceed to
   calibrated R2/R3 quality—not before.

This gate answers the key uncertainty cheaply: does removing Q4_K's affine
metadata dependency create a better cold kernel, or is the stream already so
memory-dominated that only fewer bytes can matter?

## Decision tree

```text
R1 simpler format wins cold?
  no  -> stop same-byte co-design; pursue byte reduction only
  yes -> build calibrated R2/R3

R2/R3 pass recurrent quality at useful dose?
  no  -> try one bounded R4 correction design
          no pass -> stop numerical artifact lane or begin explicit fine-tuning project
  yes -> complete role census

role population wins cold after providers/epilogues?
  no  -> stop kernel/layout spelling
  yes -> one-block lifecycle

one-block lifecycle wins?
  no  -> stop integration
  yes -> full-token reverse bracket

full token wins under quality contract?
  no  -> retain research artifact; book zero
  yes -> promote per target/role with rollback, then generalize
```

## Definition of completion

The scope is complete when every candidate family is in one of four explicit
states:

- `PROMOTED`: full-token win, quality pass, memory/accounting pass, rollback;
- `STOP_PERFORMANCE`: correct/qualified but fails cold or lifecycle timing;
- `STOP_QUALITY`: byte/kernel value exists but recurrent quality fails;
- `TRAINING_REQUIRED`: no admissible post-training artifact exists and the
  remaining work is explicitly a model-training project.

An implemented converter, a faster hot kernel, unchanged greedy tokens, or a
theoretical byte ceiling is not completion by itself.

## Existing authority to carry forward

- `docs/task_workflow/output/nv-numerical-byte-reduction-result.md`
- `docs/task_workflow/output/nv-qokv-exhaustive-test-invest-result.md`
- `docs/task_workflow/output/nv-bounded-persistent-o-result.md`
- `docs/task_workflow/output/dense-projection-service-strategy.md`
- `tinygrad/llm/qk_layout.py`
- `tinygrad/llm/qk_primitives.py`
- `tinygrad/llm/model_route_plan.py`
- `tinygrad/llm/model_facts.py`
