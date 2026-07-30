# Metal replay and generated-route parity scope

Date: 2026-07-29

Status: scoped, not implemented. This is the EXP-only performance follow-up to
`boltbeam-metal-compatibility-scope-20260729.md`. The earlier scope proved that the portable search/provider loop works
and correctly rejected its losing isolated candidate. This scope starts from the later matched cross-runtime trace and
from the historical AMD sequence that turned a slow, GPU-bound path into a llama-competitive path.

Branch boundary: all investigation and implementation begin on tinygrad `exp`. This scope does not authorize AMD/eGPU
hardware work, a hand-authored Metal kernel, weakening Metal's offset-safety guard, or promotion to `dev`/`master`.

## 1. End goal

Recover a maintainable, machine-search-selected Metal decode path for Qwen3-8B Q4_K_M without copying AMD ISA,
hardcoding model names into runtime policy, or confusing graph-launch cleanup with kernel optimization.

The work has two separately measured outcomes:

1. **Replay parity:** every captured and uncaptured decode program has an attributable graph-admission result; Metal
   replay uses the smallest safe grouping supported by the backend; no buffer offset is truncated or silently wrapped.
2. **Generated-route parity:** the important optimization *concepts* proven during the AMD campaign become portable
   workload/candidate descriptors, are searched and measured on Metal, and may bind only through the existing generated
   plan lifecycle after a matched whole-model win.

The final success condition is not “make Metal look like AMD.” It is:

```text
semantic model role + shape + quant/layout/dtype
  -> target-neutral candidate population
  -> tinygrad-generated Metal program
  -> measured hardware result
  -> BoltBeam whole-model policy
  -> exact, fail-closed selected-plan binding or explicit refutation
```

No runtime change is retained merely because it reduces a command-buffer count, and no isolated kernel result is
promoted merely because it wins a microbenchmark.

## 2. Pinned evidence and what it means

### 2.1 Current Metal trace

The durable trace is in BoltBeam at `bench/metal-qwen3-8b-trace-20260729/`. Its matched identity is:

- model: `Qwen3-8B-Q4_K_M.gguf`;
- model SHA-256: `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`;
- workload: one decode token at fixed depth 128, after one warmup;
- target: Apple M4 10-core GPU / Metal;
- tinygrad EXP: `1f9a3c48ddfe9a095573c51457dfc76651e045c6`;
- llama.cpp: `4f0e43da6f8f6e9390d88409610098ec2d2dc5c7`;
- BoltBeam portable-trace implementation: `9b0e7e9` or a pinned full hash in the first new run manifest.

| Metric | llama.cpp | tinygrad EXP |
| --- | ---: | ---: |
| decode | 19.688 tok/s | 11.210 tok/s |
| whole-step wall | 50.792 ms | 89.240 ms |
| selected command buffers | 3 | 101 |
| selected GPU interval union | 49.108 ms | 87.737 ms |
| host/unattributed | 1.684 ms | 1.503 ms |

The tinygrad command buffers consist of 24 `batched` buffers plus 77 individually submitted buffers. The 24 graph
batches contain 726 underlying dispatches with observed batch sizes
`32,64,128,256,188,3,2,5,3,2,3,3,2,3,3,2,3,3,2,3,3,2,3,8`. The direct buffers are 45 `r_` labels and 32 `E_`
labels. Therefore “101 command buffers” must never be restated as “101 kernels.”

This single-sample trace proves that both runtimes are overwhelmingly GPU-bound during the selected interval. It does
not prove why the 77 calls were excluded from Metal graph replay, which semantic roles own those calls, or whether
reducing outer submissions will materially reduce the 87.737 ms of GPU work.

### 2.2 Historical AMD control

The AMD history shows why launch count cannot be treated as the complete explanation:

| Commit/date | Context | tinygrad | Outer programs/token | Host-sync |
| --- | ---: | ---: | ---: | ---: |
| `58e33b595`, 2026-06-17 | 128 | 49.1 tok/s | 6 | 0.0% |
| `58e33b595`, 2026-06-17 | 512 | 36.9 tok/s | 6 | 0.0% |
| `45fd317dc`, 2026-06-18 | 128 | 71.5 tok/s | 6 | 0.0% |
| `45fd317dc`, 2026-06-18 | 512 | 68.2 tok/s | 7 | 0.0% |
| `de81166eb`, 2026-06-24 | 512 | 101.6 tok/s | 6 | 0.0% |
| `de81166eb`, 2026-06-24 | 4096 | 92.9 tok/s | 6 | 0.0% |

The outer replay structure stayed approximately constant while throughput roughly doubled. The gains came from
generated/cooperative quant routes, attention routing and shape coverage, and later route-miss fixes—not from reducing
six launches to fewer launches.

The matched authority at `dc4c9f9f8` recorded Qwen3-8B at 85.1 tok/s versus llama 99.4 at depth 128, and 107.4 versus
98.35 at depth 512. The 2026-07-03 state at `8f9339865` recorded tinygrad ahead of llama at depths 512 and 4096 for
8B, 14B, and 32B after further generated route and attention coverage. Those numbers are AMD historical controls, not
Metal performance promises.

### 2.3 Current Metal representability boundary

`tinygrad/runtime/graph/metal.py::MetalGraph.supports_uop` rejects a call when any sliced Metal buffer has a byte
offset greater than `0xFFFFFFFF`. Metal indirect command-buffer binding takes a 32-bit offset at this boundary. The
loaded Qwen3-8B weights exceed 4 GiB in aggregate, so this guard is a plausible reason that later weight slices become
direct calls.

This is a **hypothesis**, not yet an attribution. The first implementation packet must report the exact reason for
every graph admission and rejection. It is forbidden to remove the guard, cast the offset to uint32, or describe all
77 direct calls as offset failures before that census exists.

### 2.4 Working hypothesis stack

| Id | Hypothesis | Required falsification/confirmation |
| --- | --- | --- |
| H0 | The trace is contaminated by mismatched model, depth, runtime, cache, or thermal state | Repeated, interleaved matched controls with immutable identities |
| H1 | Most/all 77 direct programs are rejected because an ICB buffer offset exceeds 32 bits | Per-call admission census with stable reason and exact offending offset/resource identity |
| H2 | Safe rebasing/segmentation can capture those programs and reduce replay fragmentation | Correctness-preserving replay A/B using identical compiled programs |
| H3 | Replay compaction alone explains most of the Metal gap | Whole-token GPU/wall A/B; reject if device time remains materially unchanged |
| H4 | The residual gap is generated schedule/route quality, analogous to AMD history | Semantic role attribution plus isolated and whole-model candidate evidence |
| H5 | AMD optimization concepts can be expressed portably and searched for Metal | Transfer matrix and target-neutral candidate validation; no AMD route/ISA dependency |

H1 and H4 may both be true. The workflow must not force a single-cause story.

## 3. Architectural boundaries

### 3.1 One authority per concern

| Concern | Authority |
| --- | --- |
| graph batching and generic admission interface | `tinygrad/engine/jit.py` |
| Metal ICB capabilities, encoding, resources, and safety limits | `tinygrad/runtime/graph/metal.py` and `tinygrad/runtime/ops_metal.py` |
| semantic model role, tensor facts, and generated-plan attachment | existing `tinygrad/llm` facts/plan/route modules |
| experimental probes and transfer analysis | `extra/llm_research` on EXP |
| target and workload identities, candidate population, measured rank, evidence, promotion/refutation | BoltBeam |
| legal dimension proposals and static assessment | BubbleBeam/FutureSight |

Do not add a second Metal graph splitter, target table, candidate registry, route selector, benchmark schema, or
promotion rule.

### 3.2 Required reuse

- Extend `GraphRunner.supports_uop` through a typed admission result while preserving the existing boolean-compatible
  behavior for backend callers during migration.
- Reuse the existing portable trace profile and provider capability path in BoltBeam.
- Reuse `tinygrad.llm.model_facts`, `model_route_plan`, route attachment/admission, generated catalog, and runtime route
  observation rather than parsing kernel names into policy.
- Reuse `boltbeam.full_kernel_candidate.v2`, the target registry, finite population/hashing, correctness evidence,
  result ledger, and whole-model policy.
- Reuse the shared BubbleBeam/FutureSight proposal/assessment engine; add supplied compiler capabilities rather than a
  Metal fork.
- Reuse ordinary tinygrad Metal as the control and rollback.

### 3.3 Separation of facts

These dimensions remain orthogonal in every candidate and trace:

- model family/revision and immutable model hash;
- phase (`decode` initially; prefill is out of scope);
- semantic role and exact logical M/N/K;
- quant storage format and block layout;
- input, output, scalar compute, and accumulator dtype;
- value lanes/vectorization;
- address space and buffer-base/offset representation;
- schedule transforms and reduction strategy;
- backend, exact target, compiler, and binary identity;
- graph admission/grouping and kernel program identity;
- isolated timing versus whole-model timing.

A large weight offset must not imply a model route. `Q4_K` must not imply an AMD lane map. A Metal schedule must not
implicitly choose an accumulator dtype. A command-buffer label must not become a route authority.

## 4. Evidence contracts

Prefer extending the existing portable trace schemas. Introduce a new schema only when the existing envelope cannot
represent the fact without ambiguity.

### 4.1 Graph admission census

Proposed payload: `tinygrad.graph_admission_census.v1`.

Required fields per logical call:

- stable run-local call index and generated program/source hash;
- semantic metadata when present: phase, role family, layer/shape/quant/dtypes;
- device/backend and graph implementation;
- decision: `admitted`, `rejected`, or `batch_boundary`;
- reason code, not a free-form string;
- relevant capability/limit and observed value;
- buffer argument index, base allocation identity, byte offset, and byte span when the reason is resource-related;
- resulting batch index/size or direct-call index;
- explicit `metadata_unavailable` fields rather than guessed names.

Minimum reason vocabulary:

- `admitted`;
- `no_graph_backend`;
- `unsupported_call_op`;
- `mixed_device`;
- `backend_buffer_offset_width`;
- `backend_resource_limit`;
- `explicit_graph_barrier`;
- `batch_size_limit`;
- `graph_constructor_failure` (post-split failure, captured separately);
- `unknown` (invalidates completion until explained).

The generic layer owns the normalized result type. A backend adapter owns capability-specific detail. Normal
execution must not pay trace serialization cost unless observation is enabled.

### 4.2 Replay A/B evidence

Extend the cross-runtime trace/run manifest with:

- replay strategy id and implementation revision;
- logical call count, graph-batch count, direct-call count, and underlying dispatch count;
- admission-reason histogram and exact census hash;
- compiled program/source/binary identity set proving the A/B changed grouping only;
- raw wall and selected GPU-union samples;
- correctness/token/logit check;
- allocation count/bytes and working-set facts before/after;
- thermal/order metadata and any invalidated samples.

Do not divide a Metal graph duration evenly across entries to infer a winning semantic role.

### 4.3 Route attribution and search evidence

Each searched program must carry a stable role/shape/quant/dtype identity from graph construction through JIT metadata,
Metal label/evidence, candidate hash, and runtime census. A human-readable label may be derived from this identity, but
the label is never parsed back into authority.

For a graph batch, report exact member identities and aggregate batch time. Per-member time remains unavailable unless
a collector or controlled isolated measurement directly provides it.

## 5. Work packages

Every packet below has one primary concern, named prerequisites, deliverables, tests, and a stop gate. An agent must
not cross into a later packet merely because a local change appears promising.

### MR0 — Freeze baseline and replay recipe

Hardware: one short local Metal run required; no AMD/eGPU.

Deliverables:

- pin full hashes for tinygrad EXP, BoltBeam, llama.cpp, model, OS/Xcode/xctrace, and target resolution;
- reuse the existing fixed-depth-128 profile and add a repeated/interleaved run plan;
- record cache state, warmups, sample count, order, memory pressure, and thermal invalidation rules;
- reproduce the logical/outer count reconciliation: 726 batched dispatches + 77 direct calls;
- retain the first trace as historical evidence rather than overwriting it.

Gate:

- at least five valid measured token samples per runtime, interleaved or order-randomized;
- model/depth/output identity matches across runtimes;
- median and dispersion are recorded; unstable runs become `INCONCLUSIVE`;
- no performance code changes are included.

### MR1 — Typed graph-admission observability

Hardware: CPU/mock implementation and tests first; one Metal census afterward.

Primary files:

- `tinygrad/engine/jit.py` for the normalized admission interface and batch accounting;
- `tinygrad/runtime/graph/metal.py` for Metal capability reasons;
- EXP observer/export code under `extra/llm_research` only if the generic profile stream cannot carry the census.

Deliverables:

- replace opaque boolean-only reasoning with a typed result that retains a compatibility boolean;
- preserve batching behavior byte-for-byte when observation is disabled;
- emit one reconciled census at JIT construction, including explicit prefix barriers and max-batch boundaries;
- capture graph-constructor failures separately from pre-admission rejection;
- add a synthetic offset-width test without allocating a real 4+ GiB buffer;
- ensure no model-specific or Metal-specific reason code leaks into the generic interface.

Gate:

- all logical calls reconcile exactly into graph members, direct calls, ignored slice nodes, or explicit failures;
- zero `unknown` decisions on the Qwen3-8B trace;
- the exact cause and offsets of all 77 historical-style direct calls are measured, not inferred;
- existing graph/JIT tests pass with observation off and on.

Stop condition: if the direct calls are not primarily offset-width failures, re-scope MR2 around the measured reason.
Do not implement an offset solution by assumption.

### MR2 — Safe Metal replay design decision

Hardware: none for the design record; small Metal prototypes may follow only after review.

Compare at least these designs:

1. **Base-buffer rebasing/segmentation:** bind the same bytes through a representable base allocation plus 32-bit local
   offset. Determine ownership of allocation layout, aliasing, lifetime, update slots, and memory overhead.
2. **Multiple representable resource windows:** split model storage into stable resources whose individual bound offsets
   fit the ICB ABI. Determine whether copies are needed; a solution requiring per-token weight copies is rejected.
3. **Hybrid graph/direct encoding:** retain ICB batches but encode previously rejected programs through ordinary direct
   compute encoders with the same command-buffer/lifetime plan if the API allows it. Measure whether this is genuinely
   more compact than current direct submission.
4. **Backend-limited graph partitioning:** make the current safe split explicit and optimal if the ABI cannot represent
   a better replay. This may be the correct refutation.

For every option record:

- correctness and overflow safety;
- whether model weights are duplicated or copied;
- startup, steady-state, and memory cost;
- interaction with dynamic input rebinding and symbolic launch dimensions;
- M1/M2 versus M3+ ICB resource workaround compatibility;
- generic versus Metal-only ownership;
- upstreamability and maintenance surface;
- testability without the full model.

Gate: one written decision selects a safe design or records a supported no-change verdict. Removing/truncating the
uint32 guard, relying on undefined offset behavior, or creating a Qwen-specific allocator is an automatic rejection.

### MR3 — Implement compact replay behind the existing graph boundary

Prerequisite: MR1 census and MR2 decision.

Hardware: focused Metal validation required.

Deliverables:

- implement the chosen representation in the Metal graph/runtime ownership layer;
- keep generic JIT policy backend-neutral;
- preserve generated program source/binary hashes across control/candidate when the experiment changes only replay;
- correctly update dynamic input buffers and symbolic launch dimensions on repeated calls;
- retain resource lifetimes and the existing M1/M2 pipeline-use safety behavior;
- expose strategy/capability facts through the existing target/provider mechanism;
- add no runtime environment selector as permanent policy; a research-only A/B override may exist in EXP and must be
  removed or moved behind an explicit diagnostic interface at closure.

Gate:

- full graph-admission census reconciles with zero silent overflow and zero unexplained direct call;
- output/logit/token parity passes repeated decode;
- no extra steady-state weight copy and no unbounded allocation growth;
- graph/JIT/Metal focused tests pass;
- the observed outer structure is reported, not forced to equal AMD's six programs.

### MR4 — Isolate the replay contribution

Prerequisite: MR3.

Hardware: serialized local Metal lane.

Run a paired A/B where the only intentional variable is replay representation. Required results:

- five or more valid samples per arm, interleaved/randomized;
- identical model, prompt/depth, compiled program identity set, schedule, outputs, and resident buffers;
- wall, selected GPU union, command buffers, batches, direct calls, dispatches, memory, and spread;
- bootstrap confidence interval or the repository's existing paired significance calculation;
- complete admission census for each arm.

Classification:

- `REPLAY_WIN`: at least 3% median whole-step improvement, confidence excludes zero, and no correctness/memory
  regression;
- `REPLAY_NEUTRAL`: structurally cleaner but less than 3% or confidence overlaps zero;
- `REPLAY_REFUTED`: slower, unstable, incorrect, or materially more memory-hungry;
- `INCONCLUSIVE`: identity, thermal, sample, or collector gate failed.

The 3% boundary is an engineering retention threshold, not a claim that replay should close the llama gap. Retain a
correct generic runtime fix only if its maintenance value and measured behavior justify it; otherwise keep the
refutation and proceed using the safe control.

### MR5 — Semantic role identity through the generated program path

Hardware: CPU tests first; one Metal trace.

Deliverables:

- inventory existing `Metadata`, model facts, route-plan, attachment, and observer fields before adding anything;
- define one stable role vocabulary for decode projections, attention, normalization/elementwise, embedding, and
  lm_head without encoding Qwen layer names as runtime policy;
- preserve layer index only as trace metadata, not candidate applicability unless explicitly required;
- propagate role/shape/quant/dtype identity through schedule/JIT/program/graph members;
- map the 24 graph batches and all direct calls to known roles or an explicit generic category;
- add a route census that distinguishes ordinary generated tinygrad schedules from an installed selected plan;
- never infer duration by evenly splitting a graph batch.

Gate:

- every material decode program has stable semantic or generic identity;
- identities remain stable across replay strategies and repeated runs;
- no duplicate role map exists in BoltBeam, BubbleBeam, Metal labels, and tinygrad LLM code;
- production modules do not import research code.

### MR6 — AMD optimization-concept transfer matrix

Hardware: none. This is a git-history/code-analysis packet, not a port.

Audit the historical AMD winners and route fixes at minimum:

- Q4_K generated G3/lane-partition GEMV;
- generated/cooperative Q6_K paths used by decode roles;
- Q4_K attention q/o shape routes;
- flash-decode threshold and implementation regimes;
- `attn_k` and `attn_v` route coverage, including the later route miss;
- larger-shape coverage that changed 14B/32B performance;
- generated G=5 K-only attention where applicable.

For each row publish:

| Field | Meaning |
| --- | --- |
| semantic mechanism | e.g. packed read, cooperative reduction, grouping, fused attention work |
| winning workload roles/shapes | evidence-backed applicability, not guessed generalization |
| compiler primitive required | reduction, vector load, local/shared memory, matrix primitive, etc. |
| AMD-specific implementation | ISA, wave, wait, occupancy, raw kernel, or route details that must not transfer |
| portable descriptor | target-neutral axes and constraints that can enter candidate v2 |
| Metal capability mapping | observed/supported/unknown; never inherit AMD defaults |
| ordinary fallback | current tinygrad-generated control |
| evidence and reopen condition | commit/artifact and what Metal measurement is needed |

Gate: each historical improvement is classified as portable concept, compiler work, backend-specific implementation,
or irrelevant to the first Metal workload. No AMD route id, gfx1100 manifest, raw ISA, wave32 assumption, or hardcoded
shape becomes Metal policy.

### MR7 — Rank the Metal work by measured role cost

Prerequisites: MR4 and MR5; use MR6 for candidate concepts.

Hardware: local Metal.

Deliverables:

- measure isolated exact roles using the existing provider where aggregate graph timing cannot attribute members;
- reconcile isolated identities with whole-model route census;
- estimate upper bounds: eliminate 100% of a role's isolated cost and show the maximum possible whole-token gain;
- classify memory-bound, compute-bound, latency/dispatch-sensitive, or unknown using scope-compatible evidence only;
- rank no more than the first two role families for the initial search campaign.

Gate: a role enters search only when its whole-model importance and exact route identity are demonstrated. Command-
buffer label frequency alone is insufficient. If no role can close at least 5% of whole-step time, stop and revisit the
trace/roofline rather than launch a broad search.

### MR8 — Portable candidate population

Prerequisites: MR6 transfer matrix and MR7 ranking.

Hardware: none for schema/expansion tests.

Deliverables:

- define bounded, target-neutral axes for the selected semantic role(s);
- derive legal values from workload, live target/compiler facts, and supported tinygrad scheduling vocabulary;
- let BubbleBeam propose dimensions and FutureSight reject/prioritize statically;
- let BoltBeam instantiate/hash the exact finite population and include the ordinary tinygrad control;
- represent coupled constraints explicitly rather than generating invalid Cartesian combinations;
- pin budget, timeout, correctness tolerances, shape regime, and compiler revision;
- preserve unsupported/rejected candidates in the result.

Gate:

- deterministic population and hashes at pinned revisions;
- no hand-written MSL, raw MTLB, route-local UOp emitter, online autotune, or process-global environment identity;
- dtype/quant/layout/lanes/accumulator remain separate;
- AMD candidate v1 compatibility and existing Metal provider tests remain green;
- FutureSight output cannot promote a candidate.

### MR9 — Finite isolated Metal search

Prerequisite: MR8.

Hardware: serialized local Metal lane.

Deliverables:

- execute `describe/admit/compile/check/measure` for the complete population;
- compile once, then run exact-shape resident-buffer timing with warmups and raw samples;
- record source/MTLB/plan hashes, launch facts, correctness, and all terminal failures;
- randomize/interleave viable candidates and control where practical;
- repeat finalists to detect selection instability;
- emit a winner or explicit no-win/refutation from BoltBeam.

Gate:

- population is complete; partial measurement cannot select;
- deterministic packed fixtures and real-role output agree with the generic semantic oracle;
- winner beats the isolated generic control under the existing policy;
- a winner remains diagnostic until MR10/MR11.

### MR10 — Exact generated-plan attachment and route census

Prerequisite: MR9 winner. Skip this packet on refutation.

Hardware: Metal smoke required.

Deliverables:

- export immutable selected plan/provenance through the existing generated catalog/route-plan architecture;
- bind only exact target/model-hash/phase/role/shape/quant/dtype/layout/compiler-compatible matches;
- fail closed to ordinary Metal for every mismatch;
- trace requested candidate, admitted candidate, program/binary identity, binding count, and fallback reason;
- prohibit research imports and duplicate selection logic in `tinygrad/**`;
- retain one explicit rollback through the existing plan-selection boundary.

Gate:

- exact route fires for all expected and only expected invocations;
- complete route census has no missing/unexpected bindings;
- all mismatch and stale-artifact tests fall back safely;
- default behavior remains unchanged until whole-model policy admits the candidate.

### MR11 — Matched whole-model A/B and roofline

Prerequisite: MR10.

Hardware: serialized local Metal lane.

The authoritative evaluation is Qwen3-8B Q4_K_M fixed-depth decode, starting at depth 128 and adding depth 512 before
any promotion. Use the same model hash, prompt/token contract, cache state, memory residency, and compiler/runtime
revision in paired generic/candidate arms.

Required evidence:

- at least seven valid samples per arm, interleaved/randomized;
- tok/s, whole-step wall, selected GPU union, outer grouping, dispatch count, and raw samples;
- token/logit correctness and repeated-run stability;
- exact candidate/route census and binary identity;
- compile/load time kept outside steady-state timing but reported;
- peak/current allocation and working-set changes;
- actual workload bytes where known, with modeled/unknown bytes labeled honestly;
- matched llama.cpp control refreshed separately, not interleaved into tinygrad candidate identity;
- no use of raw advertised bandwidth or an unrelated stream proxy as achieved model bandwidth.

Promotion threshold for a searched plan:

- at least 3% median whole-model improvement over generic EXP at both required contexts;
- confidence excludes zero and dispersion does not materially worsen;
- no correctness, route, memory, compile-cache, or fallback regression;
- result survives one clean-process replay from the exported plan;
- BoltBeam policy returns `promote`.

Project-level performance target:

- reach at least 90% of the refreshed llama.cpp throughput at depth 128 **or** publish a measured residual-gap
  decomposition that identifies the unavailable compiler/runtime primitive and a precise reopen condition;
- do not sacrifice the depth-512 result to improve depth 128;
- do not claim parity from an isolated candidate or modeled roofline.

### MR12 — Regression, pruning, and promotion decision

Prerequisites: MR4 for a retained replay change; MR11 for a retained selected plan.

Hardware: local Metal; AMD hardware only in a later explicit recertification packet if promotion is proposed.

Deliverables:

- run focused JIT/graph/Metal/LLM/provider/search tests and the full available suite;
- audit for duplicate target facts, graph policies, role maps, route selectors, schemas, and temporary env toggles;
- remove diagnostic-only adapters, dead commands, scratch profiles, and redundant fixtures;
- preserve compact evidence and replay commands; exclude full xctrace packages and model binaries from git;
- update the earlier Metal compatibility closure without rewriting its historical result;
- issue one verdict per change: keep on EXP, promote later, or remove/refute.

Promotion boundary:

- generic correctness/upstreamable Metal graph work may be proposed EXP -> `dev` -> `master` only as isolated commits
  with backend-neutral tests;
- generated Metal plan data may be proposed only after MR11 and repository purity checks;
- research providers, searches, probes, transfer matrices, and measurement bundles remain EXP/research assets;
- no promotion occurs in this scope without a separate branch-promotion review and required hardware recertification.

### MR13 — Reproducible handoff and closure

Hardware: none after evidence is captured.

Deliverables:

- one start-to-finish README with environment, model hash, commands, expected artifacts, and result interpretation;
- a manifest linking census, replay A/B, transfer matrix, search request/result, route census, whole-model A/B, roofline,
  test results, and policy verdict by content hash;
- a concise current-state update in the tinygrad handoff and BoltBeam benchmark index;
- explicit supported claims, unsupported claims, remaining gap, rollback, and reopen condition;
- clean worktrees at the recorded revisions, excluding documented ignored local traces.

Gate: a new Apple Metal user with the named model can run the documented generic control and, when present, the
generated selected-plan path without reading campaign history or editing source.

## 6. Dependency graph and safe parallelism

```text
MR0 baseline freeze
  -> MR1 graph-admission census
      -> MR2 safe replay design
          -> MR3 replay implementation
              -> MR4 replay A/B -------------------------------+
                                                               |
MR1 census -> MR5 semantic role identity                       |
AMD git history -> MR6 transfer matrix                         |
MR4 + MR5 + MR6 -> MR7 measured role ranking                   |
                    -> MR8 portable population                 |
                        -> MR9 isolated finite search           |
                            -> MR10 exact attachment            |
                                -> MR11 whole-model A/B/roofline|
                                                               |
MR4 retained replay and/or MR11 promoted candidate ------------+
  -> MR12 regression/prune/promotion decision
      -> MR13 reproducible closure
```

Safe non-hardware parallel work after MR0:

- MR1 unit-test/interface work and MR6 historical transfer analysis may proceed independently;
- MR5 role-vocabulary inventory may begin while MR2 is being decided, but it cannot finalize until the census exists;
- MR8 schema fixtures may begin after MR6, but the measured role choice must wait for MR7;
- documentation/test auditing may run alongside hardware packets.

Serialized work:

- only one local Metal measurement/search lane runs at a time;
- MR3 cannot begin before the MR2 safety decision;
- MR9 cannot begin before a frozen MR8 population;
- MR10/MR11 cannot begin without an MR9 isolated winner;
- branch promotion cannot begin inside this scope.

## 7. Test matrix

| Layer | Required tests |
| --- | --- |
| generic graph admission | boolean compatibility, typed reasons, mixed/no-graph/unsupported calls, batch limit/barrier, zero behavior drift with observation off |
| Metal offset safety | boundary at `0xFFFFFFFF`, first invalid byte, argument index, synthetic large slice, no truncation/wrap |
| Metal graph replay | construction, update buffers, symbolic launch dimensions, repeated replay, resource lifetime, M1/M2 safety branch, error propagation |
| accounting | every logical call assigned exactly once; batch members + direct calls reconcile; stable census hash |
| metadata | role/shape/quant/dtype preserved through schedule/JIT/program/graph; missing metadata explicit; labels non-authoritative |
| provider/evidence | capability/version checks, schema/hash validation, dirty/stale/unknown failures, unsupported fields explicit |
| candidate generation | deterministic legal proposals, BoltBeam-owned expansion/hash, coupled constraints, ordinary control included |
| correctness | deterministic quant fixtures, real-role output, finite/repeatable results, whole-model token/logit checks |
| timing | compile exclusion, raw GPU samples, paired order, warmups, thermal invalidation, no evenly divided graph timing |
| routing | exact bind, every mismatch fallback, complete route census, rollback, no research import |
| model | Qwen3-8B fixed depth 128 and 512, resident-memory accounting, clean-process replay |
| regression | existing AMD candidate compatibility in CPU fixtures; Metal/JIT/LLM focused tests; full suite |
| purity | no hand MSL/raw binary/route-local emitter; selected plan provenance and content hashes; ordinary generated fallback |
| docs | commands resolve, artifacts exist, hashes pin identities, one authority per number |

## 8. Risks and explicit mitigations

| Risk | Mitigation/stop rule |
| --- | --- |
| Mistaking 101 buffers for 101 kernels | Always report outer buffers and underlying dispatches separately |
| Assuming all 77 direct calls are offset failures | MR1 requires exact zero-unknown census before design |
| Unsafe uint32 workaround | Never weaken/truncate the guard; MR2 must prove representability |
| Cleaner replay but unchanged device time | MR4 classifies independently; proceed to roles without overstating the result |
| Ranking roles from estimated graph timing | Use aggregate batch facts plus controlled isolated role measurements only |
| Copying AMD-specific policy into Metal | MR6 transfer matrix separates mechanism, portable descriptor, and backend implementation |
| Hardcoding Qwen/model shapes | Applicability uses semantic facts and exact exported guards; generic fallback handles all other shapes |
| Duplicate authority across repos | tinygrad executes, BubbleBeam proposes, FutureSight assesses statically, BoltBeam ranks/promotes |
| Search winner fails whole model | MR10/MR11 are mandatory; isolated winner remains diagnostic |
| Thermal/unified-memory noise | interleaved samples, working-set facts, invalidation rules, clean-process replay |
| Performance change harms AMD | no AMD runtime change in this scope; later promotion requires explicit recertification |
| Scope grows into prefill/MLX | decode Qwen3-8B only until closure; new workloads require a new packet |

## 9. Explicit non-goals

- No AMD eGPU reset, Thunderbolt, PCI, Linux kernel, KFD, or ROCm work.
- No assumption that an AMD optimization implementation is valid on Metal.
- No hand-authored MSL, copied llama.cpp shader, raw precompiled MTLB, or route-local hand UOp hot kernel.
- No removal or truncation of the Metal uint32 ICB offset safety check.
- No model-specific allocator, Metal-only candidate schema, duplicate route manifest, or duplicate target registry.
- No online autotuning during normal model load/inference.
- No prefill optimization, MLX integration, or Metal performance claim beyond the matched Qwen3-8B decode profile.
- No promotion to `dev` or `master` without a later explicit branch-promotion packet.
- No requirement that replay compaction alone beat llama.cpp.
- No claim that Metal is bandwidth-, compute-, kernel-, or launch-bound until scope-compatible evidence says so.

## 10. Completion definitions

### Replay diagnosis complete

- repeated matched baseline exists;
- every call has an admission decision and zero remain `unknown`;
- safe design decision is recorded;
- replay A/B reports correctness, grouping, GPU/wall time, memory, and verdict.

### Search transfer complete

- AMD concepts are classified in the transfer matrix;
- Metal role cost and route identity select a bounded target;
- the target-neutral population completes through the existing BoltBeam/BubbleBeam/FutureSight/tinygrad lifecycle;
- all rows are measured/rejected/blocked explicitly and the result is replayable.

### Performance work complete

One of these outcomes is durable:

1. a generated selected plan passes exact binding, depth-128/depth-512 whole-model gates, rollback, and BoltBeam
   promotion policy; or
2. no plan promotes, and the repository contains a measured refutation plus a specific unavailable primitive or
   evidence-backed reopen condition.

In either case, the generic Metal path remains runnable, experimental machinery stays out of production imports,
temporary seams are pruned, tests pass, and a new user can reproduce the supported result from the documented entry
point.

## 11. Required closure artifacts

Use one dated BoltBeam benchmark bundle and link it from the tinygrad handoff. It must contain:

- `baseline-manifest.json` and raw repeated summaries;
- `graph-admission-census.json`;
- `replay-design-decision.md`;
- `replay-ab.json`;
- `amd-concept-transfer-matrix.json` plus a readable Markdown rendering;
- `role-cost-ranking.json`;
- `search-request.json`, `search-result.json`, and complete population evidence;
- `selected-plan.json` and `route-census.json` when a winner exists;
- `whole-model-ab.json`, `roofline.json`, and refreshed llama control;
- `policy-verdict.json`;
- `test-summary.json`;
- `README.md` with exact replay commands and supported claims.

Full xctrace packages, GGUF files, compiler caches, and temporary binaries stay local/ignored. Compact derived evidence
must retain hashes back to the excluded inputs.
