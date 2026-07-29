# BoltBeam + BubbleBeam/FutureSight Metal compatibility scope

Date: 2026-07-29

Status: approved scope; implementation in progress in local, uncommitted working trees. M0-M4 contracts are
implemented and focused-tested, M5-M6 have bounded Metal smoke evidence only, and M9 has static/raw roofline analysis
plus a non-authoritative streaming proxy. No finite exact-workload search, route binding, or paired whole-model M9
evaluation has run.

Branch boundary: design and provider work begin on tinygrad `exp`. This scope does not authorize AMD/eGPU work,
promotion to `dev`/`master`, or a hand-authored Metal kernel.

## 1. Outcome

Make Apple Metal a real, fail-closed target in the existing machine-search system:

```text
BubbleBeam                     propose legal compiler-owned dimension values
  -> BoltBeam                  instantiate/hash the declared finite candidate population
  -> FutureSight               reject/prioritize canonical candidates using static facts
  -> BoltBeam                  own measured ranking, evidence, ledger, and promotion decision
  -> tinygrad EXP provider     admit, compile, execute, check, and time candidates on Metal
  -> ordinary tinygrad Metal   render MSL, compile MTLB, and execute the selected schedule
```

The first proof workload is Qwen3-8B Q4_K_M decode on the local Apple M4. Compatibility is complete when a finite,
replayable search can finish and report a correct measured population, even if every candidate loses and the generic
Metal control remains selected. Performance promotion is a later outcome and requires a measured whole-model win.

This is machine search under the repository's practical definition: people define the semantic workload, candidate
axes, compiler primitives, and gates; BubbleBeam proposes legal values, BoltBeam performs the authoritative finite
expansion and hashing, the hardware measures candidates, and measured policy selects or rejects the result. No
hand-written MSL or route-local UOp kernel is introduced.

## 2. Current truth

### 2.1 What already works

| Surface | Current state | Reuse decision |
| --- | --- | --- |
| tinygrad Metal runtime | Loads Qwen3-8B Q4_K_M and decodes correctly | Reuse as the sole executor/compiler |
| Metal timing | Eager `MetalProgram(..., wait=True)` returns command-buffer GPU elapsed time | Reuse for isolated candidate timing |
| Metal binaries | `MetalCompiler` emits MTLB bytes | Reuse for immutable binary hashes |
| Metal target facts | Device family, name, working-set budget, allocation size, thread limits, and threadgroup memory are available through Metal | Expose through one provider fact adapter |
| generic quant path | Unsupported AMD routes fall back to the ordinary GGUF dequant + tinygrad graph | Reuse as correctness and performance control |
| tinygrad scheduler vocabulary | `Opt`/`OptOps` already represent local, group, upcast, unroll, coalesce, tensor-core, and related choices | Reuse as candidate plan vocabulary |
| BoltBeam bounded search | Validates a finite population, isolates workers, records blocked/incorrect/measured rows, ranks by an objective, and fails closed on partial measurement | Reused through candidate v2, resolved-target, and explicit execution-regime contracts |
| BoltBeam policy/evidence | Already separates execution from evaluation and requires correctness, route binding, speed, memory fit, and rollback | Reuse as the only promotion authority |
| BubbleBeam/FutureSight | Proposes legal values from supplied workload/target/compiler facts and statically assesses BoltBeam-hashed candidates | Reuse the shared engine and historical wrapper; do not copy it into a Metal fork |

The current local diagnostic baseline is 11.05 tok/s mean for tinygrad EXP and 20.51 tok/s for local llama.cpp Metal on
the same Qwen3-8B Q4_K_M file. Those numbers identify the opportunity; they do not authorize a route or establish the
cause. The recorded tinygrad command sets `--max_context 128` but does not use `--benchmark-context`, so its three
samples start from a one-token seed and are not fixed-depth context-128 authority. The llama.cpp `tg128` observation
also omitted `-d 128`. Both must be recaptured by M9 under the final matched-depth protocol.

### 2.2 What remains incomplete

| Surface | Implemented state | Remaining blocker |
| --- | --- | --- |
| target resolution | `apple_m4_10c` exact descriptor, Darwin autoscan, stable resolved-target hash, and live provider facts are implemented | Backend remains descriptor-only until the measured compatibility loop closes; unknown Apple variants still fail closed |
| candidate/search contract | v1 compatibility plus strict target-neutral v2, finite population, generic heuristic control, and explicit execution regime are implemented | No complete exact-role population has run on hardware |
| BubbleBeam/FutureSight | Input-derived dimension proposals and static assessments keyed by BoltBeam candidate hashes are implemented | Static results remain non-performance evidence, as required |
| provider | One JSON-lines `describe/admit/compile/check/measure` worker and BoltBeam subprocess adapter are implemented | Dirty-tree/pinned replay and a complete exact-workload run remain unproven |
| Metal compile/check | Explicit and heuristic schedules compile; bounded Q4_K/Q6_K nonzero oracle checks pass and MSL/MTLB hashes are emitted | The executed program was the bounded `m=1,n=256,k=256` fixture, not the exact largest role |
| Metal timing | Eager `wait=True` timing and raw-sample output exist | Exact-role candidate/control timing has not run; no candidate timing row is promotion evidence yet |
| Metal graph profile | Aggregate graph events remain available | Evenly divided entry durations remain estimates and cannot rank candidates |
| M9 | Exact GGUF packed-weight/FLOP inventory and raw-bandwidth placement are implemented; one aggregate stream proxy was smoked | No fixed-depth paired whole-model A/B, exact workload traffic attribution, durable streaming artifact, refreshed llama control, or policy verdict exists |

The compatibility job is therefore not flipping `backend_status`. It is completing the exact-workload measurement and
promotion loop without reintroducing the deleted AMD research stack.

## 3. Architectural principles and hard gates

Every implementation packet must pass these gates.

1. **Reuse before addition.** Extend the current target registry, bounded-search controller, evidence types, compiler
   plan vocabulary, and runtime. A new Metal-only controller, ranker, ledger, benchmark schema, or route policy is a
   design failure.
2. **One authority per decision.** BubbleBeam proposes legal dimension values, BoltBeam instantiates the finite
   canonical population, FutureSight statically orders/prunes it, BoltBeam ranks measured results and promotes, and
   tinygrad admits/executes. Static FutureSight score may never become promotion evidence.
3. **Central target resolution.** Registry descriptors plus measured scan facts resolve into one target profile. No
   backend module may maintain a second table for wave width, memory, compiler arch, or collector support.
4. **One provider protocol.** Replace the stale admission-only and measurement-only worker seams with one versioned
   target-neutral protocol. AMD, NVIDIA, and Metal use the same envelope and action model.
5. **One candidate model.** Evolve the existing full-kernel candidate into a target-neutral versioned union. Do not add
   `MetalCandidate`, a Metal search-space JSON dialect, or Metal-specific hashes.
6. **Backend details stay behind adapters.** Generic plans express semantic transforms and tinygrad compiler options.
   Metal owns MSL/MTLB facts; AMD owns ELF/VGPR facts; NVIDIA owns cubin/NCU facts. Missing facts remain explicitly
   unavailable instead of being guessed into common fields.
7. **Orthogonal axes remain separate.** Quant storage, scalar compute dtype, accumulator dtype, value lanes, address
   space, layout, schedule, target identity, and measurement regime must be independent fields. `Q4_K` must not imply
   fp16 accumulation, wave32, a lane map, or Metal.
8. **Generated hot path only.** Search may select ordinary tinygrad graph/scheduler plans. It may not select hand-written
   MSL, raw Metal binaries, or a new route-local Python UOp kernel.
9. **No runtime dependency on research.** Search/provider code stays in `extra/llm_research` on EXP. A selected result
   may enter `tinygrad/**` only as immutable generated plan data consumed by a generic loader.
10. **Fail closed.** Unknown target, incomplete population, stale hash, dirty provider tree, missing correctness,
    estimated graph timing, thermal invalidation, or unmatched route identity produces `BLOCKED`, not a partial winner.
11. **Control remains ordinary tinygrad.** Rollback is the unmodified generic Metal graph. No specialized handwritten
    fallback is added.
12. **Prune as part of completion.** Compatibility is not complete while obsolete worker protocols, false capability
    claims, duplicate target mappings, or dead commands remain.

## 4. Canonical ownership and modules

### 4.1 BoltBeam owns

- target descriptor registry and the resolved-target contract;
- model/workload profiles and the target-neutral candidate schema;
- finite search request, budget, objective, result population, measured ranking, and selected-plan export;
- normalized correctness/performance/source/hardware evidence validation;
- evaluation, ledger, promotion/refutation, rollback requirement, and reports;
- provider capability declarations derived from actual adapters, not optimistic static claims.

### 4.2 BubbleBeam/FutureSight owns

- legal dimension-value proposals from supplied workload facts, live target/compiler facts, and caller-declared axes;
- static legality/feasibility rejection and search-order priority over BoltBeam-expanded canonical candidates;
- deterministic assessment/rejection reports keyed by the candidate hash supplied by BoltBeam;
- no runtime dispatch, final performance rank, policy, or promotion decision.

BoltBeam remains the sole owner of candidate schema, hash, and finite Cartesian expansion. The current
`extra/llm_research/bubblebeam_futuresight.py` must expose reusable dimension-proposal and static-assessment primitives
plus an AMD historical compatibility wrapper. Metal uses those shared primitives plus supplied facts, not copied Q4
lane-map logic or imported AMD route manifests.

### 4.3 tinygrad owns

- device discovery facts that require Metal APIs;
- semantic GGUF Q4_K/Q6_K decode graph construction;
- candidate admission against the actual AST, renderer, compiler, and device limits;
- deterministic MSL generation, MTLB compilation, buffer creation, execution, correctness oracle, and timing;
- source/plan/binary hashes and exact route/candidate trace;
- generic loading of an exported selected schedule if later promoted.

### 4.4 Data flow

Use one content-addressed identity chain:

```text
model hash + role/shape/quant + resolved target hash
  -> search request hash
  -> candidate plan hash
  -> generated MSL hash
  -> MTLB hash
  -> correctness evidence hash + timing evidence hash
  -> complete population hash
  -> selected candidate hash or explicit no-promotion verdict
```

The model path, username, checkout path, and temporary profile path are metadata only and must not enter portable
candidate identity.

## 5. Target model

Keep `apple_metal` as a non-promotable family descriptor. Add an exact target row only after autoscan can prove its
identity; the expected local identity is an Apple M4 variant, but the final id (for example `apple_m4_10c`) must come
from observed hardware, not this document.

Static registry facts may include:

- backend and tinygrad device (`Metal`, `METAL`);
- Apple GPU family/architecture compatibility;
- subgroup/SIMD width when observed or guaranteed for that family;
- supported compiler primitives and collector kinds;
- fields whose status is `hardware`, `measured`, `modeled`, or `unknown`.

Dynamic scan evidence must include:

- device name and registry id;
- Apple/Mac GPU family reported by `supportsFamily`;
- `maxThreadsPerThreadgroup` and `maxThreadgroupMemoryLength`;
- `recommendedMaxWorkingSetSize` and `currentAllocatedSize` as runtime facts, never static VRAM;
- OS version, Metal language version selected by tinygrad, tinygrad commit, and BoltBeam commit;
- compiler/device identity sufficient to invalidate stale MTLB evidence.

Do not encode unified memory as discrete VRAM. Do not publish the current 32 KiB descriptor value without a live or
documented fact. Do not mark the generic family `complete`; only an exact evaluated target may become promotable.

## 6. Canonical candidate contract

Introduce `boltbeam.full_kernel_candidate.v2` as a validated variant of the existing candidate model, with a v1 reader
for historical AMD artifacts. Both versions normalize into one in-memory type.

Required v2 groups:

- `workload`: model/profile hash, phase, role, logical M/N/K, quant storage, scalar input/output/accumulator dtypes, and
  layouts;
- `target`: exact target id, backend, architecture/family, subgroup width, and resolved-target hash;
- `plan`: plan kind plus an ordered compiler transform list, layout transforms, reduction strategy, launch intent, and
  optional primitive request;
- `constraints`: correctness tolerance, maximum threads/threadgroup memory, spill policy when observable, and memory
  budget status;
- `applicability`: exact role/shape/quant/target guards;
- `provenance`: generator and schema revisions; runtime source/binary hashes are returned as evidence, not predicted.

For the first Metal slice, `plan.kind=tinygrad_schedule` and the transform list uses the existing `Opt`/`OptOps`
vocabulary. Backend-specific values are accepted only through a registered capability vocabulary. Examples include a
generic subgroup reduction or `simdgroup_matrix`; AMD `waitcnt` and Metal pipeline properties cannot appear as
untyped top-level fields.

Environment variables such as `MV_BLOCKSIZE`, `MV_THREADS_PER_ROW`, `MV_ROWS_PER_THREAD`, and `MV_UNROLL_MAX` may be
accepted by a CLI compatibility parser, but canonical candidates must bind equivalent values directly to one AST/plan.
Process-global environment state is not candidate identity and must not leak between measurements.

## 7. Provider protocol

Replace the two stale tinygrad worker protocols with one `tinygrad.search_provider.v1` JSON-lines subprocess boundary.
The canonical actions are:

- `describe`: return provider revision, clean/dirty state, resolved target facts, supported plan kinds, quant formats,
  compiler transforms, timing modes, and evidence limitations;
- `admit`: validate candidate identity, workload match, target match, compiler legality, launch/resource limits, and
  model tensor availability without timing;
- `compile`: emit the exact plan, generated MSL hash, MTLB hash, launch geometry, and observable pipeline facts;
- `check`: execute deterministic small and real-shape correctness cases against the generic semantic oracle;
- `measure`: run bounded warmups/samples and return raw observations plus summary, never a promotion decision.

One `admit_and_measure` convenience action may compose these operations but cannot bypass any result group. BoltBeam's
existing process isolation, timeout, complete-population rule, and hash checks remain the controller.

The provider worker lives in EXP research space and imports only stable tinygrad compiler/runtime interfaces. It must
not import an AMD route manifest, Metal-specific search policy, or BoltBeam Python internals. The boundary is JSON so
either repository can evolve independently under explicit schema versions.

## 8. Measurement and evidence rules

### 8.1 Candidate timing

- The promotion/search regime is an explicit request field:
  `{"shape_mode":"exact_workload","warmups":W,"samples":N}`. In this mode the provider executes the candidate's
  exact `m/n/k`; it may not silently clamp or substitute a fixture.
- `bounded_fixture` is a separate compile/correctness development regime with an explicit `fixture_shape`. Its timing
  cannot produce a `MEASURED` exact-workload row, rank a search candidate, or support promotion.
- Compile once outside the measurement samples.
- Use an isolated eager kernel/program execution with `wait=True` and Metal command-buffer GPU timestamps.
- Record warmup count, every raw GPU duration, median, dispersion, launch geometry, and synchronization mode.
- Randomize or deterministically interleave control/candidate order for paired A/B measurements.
- Keep allocations resident and identical across the pair; record working-set facts before and after.
- Reject first-compile time, evenly apportioned `MetalGraph` per-kernel timestamps, unsynchronized host enqueue time,
  non-finite samples, and target/compiler identity drift.

Metal graph events may support aggregate tracing, but their current evenly divided entry durations are explicitly
`estimated` and cannot rank candidates.

Current hardware evidence is bounded-fixture smoke: `m=1,n=256,k=256` Q4_K/Q6_K correctness passed. A candidate whose
metadata named a larger role still executed that bounded fixture unless the worker supplied the exact shape. No exact
role candidate/control timing has run.

### 8.2 Correctness

Each candidate must pass:

1. deterministic small Q4_K/Q6_K block fixtures against a scalar semantic reference;
2. representative real model role/shape output against the ordinary generic tinygrad graph under declared tolerances;
3. repeated-run determinism/finite-output checks;
4. whole-model smoke after binding, including expected route identity and no hidden AMD/custom route;
5. token/logit stability appropriate to the repository's existing benchmark gate.

Correctness and speed use separate artifacts and hashes. A faster incorrect row remains in the population as
`REJECTED_INCORRECT` and can never be selected.

### 8.3 Source and resource evidence

Common evidence fields are source hash, plan hash, binary hash, global/local size, thread count, static threadgroup
memory when observable, and compiler/device identities. Metal does not fabricate VGPR/SGPR or AMD ELF facts. The
shared evidence schema records unsupported fields with reason and provider capability.

### 8.4 Whole-model authority

Candidate microtiming answers whether a schedule improves its kernel. Promotion requires the normal Qwen3-8B path at
a fixed decode depth of 128 tokens with a same-session generic Metal A/B and route trace. `--max_context 128` alone
does not establish that depth; the harness must prefill to the declared depth and leave capacity for measured decode
samples. llama.cpp is an external control and gap tracker, not the promotion baseline for a tinygrad candidate.

The existing 11.05/20.51 tok/s diagnostics must be refreshed under the final protocol. Compatibility does not require
matching llama.cpp. A performance claim requires at least:

- a complete finite search population;
- a correct selected candidate;
- paired whole-model samples after warmup;
- a material win outside the run's noise band (use the existing policy threshold, or 5% if no stricter centralized
  threshold exists);
- no memory, compile, or correctness regression;
- a one-switch rollback to the ordinary generic Metal route.

### 8.5 Decode roofline deliverable

Reuse BoltBeam's existing constant-free roofline math and evidence/report surfaces. Do not add a Metal roofline engine.
The Metal provider supplies target/model measurements; BoltBeam owns the derived placement and report.

The report must keep three differently scoped quantities separately labeled:

1. **Raw advertised bandwidth roof:** the target's advertised bandwidth and any independently sourced/observed compute
   peak. Apple documents the base 10-core M4 Mac mini at 120 GB/s memory bandwidth. This is a raw weight-read ceiling,
   not a measured sustained result. An absent compute peak remains `unknown` until an independent measurement exists.
2. **Aggregate read+write streaming proxy:** the provider's current ordinary Tensor probe performs one device read and
   one device write and reports `(read_bytes + write_bytes) / GPU_time`. It must publish source bytes, aggregate traffic,
   raw samples, storage mode, and missing cache/thermal facts. It is not a weight-read-only workload roof and cannot be
   relabeled `practical_streaming` or used as a promotion denominator until cache exclusion and traffic comparability
   are proven.
3. **Exact workload placement:** exact-role and whole-model measurements joined to GGUF-derived packed weight bytes,
   semantic FLOPs, and measured activation/KV/intermediate traffic where available. Weight-equivalent GB/s must remain
   distinct from observed physical traffic. Percentages may be reported only against a scope-compatible roof/proxy.

The current diagnostic, derived directly from the selected GGUF tensor table, is useful for sizing but is not the final
M9 authority:

| Quantity | Diagnostic value |
| --- | ---: |
| active packed weight lower bound per decoded token | 4,671,768,832 bytes |
| dense matrix work lower bound per token | 15,136,194,560 FLOPs |
| lower-bound arithmetic intensity | 3.240 FLOP/byte |
| raw M4 bandwidth | 120 GB/s |
| raw bandwidth floor / ceiling | 38.93 ms/token / 25.69 tok/s |
| tinygrad EXP at 11.05 tok/s | 51.62 GB/s weight-equivalent, 43.0% of raw advertised roof, 0.167 semantic TFLOP/s |
| llama.cpp at 20.51 tok/s | 95.82 GB/s weight-equivalent, 79.8% of raw advertised roof, 0.310 semantic TFLOP/s |

The byte lower bound streams every matrix tensor once, reads only one row from `token_embd.weight`, and excludes
unmeasured reload, activation, KV-cache, allocator, and cache effects. The FLOP lower bound counts two operations per
matrix MAC and excludes attention/nonlinear work. Therefore percentages are diagnostic placement, not proof of
physical DRAM traffic. The final report must never infer that a 43% placement proves bandwidth is the root cause.

The current aggregate proxy is one direct, non-durable 256 MiB-source/256 MiB-destination smoke: one warmup, three
samples at 6.096/6.178/6.424 ms, median 86.89 aggregate GB/s. Cache capacity/residency and thermal state were not
proven. This result is useful for validating the provider measurement path only; a scope-compatible measured
weight-read roof remains unknown, and no exact-role or paired whole-model workload placement has run.

## 9. First search workload

### 9.1 Scope order

Start with decode, not prefill:

- current tinygrad decode is 11.05 tok/s versus llama.cpp 20.51 tok/s;
- Q4_K/Q6_K weight streaming and fused dequantized matvec are the likely dominant schedule surface;
- decode permits isolated GEMV role measurement without first solving Metal tensor-core prefill;
- the current BoltBeam full-kernel descriptor is an AMD fp16 prefill GEMM and is not reusable as a Metal decode row.

### 9.2 Workload derivation

Profile the real selected GGUF and derive role/shape/quant rows from its tensor inventory; do not hand-copy a role
shape. The current exact inventory identifies `ffn_gate_up` (gate plus up, 72 tensors) as the largest packed-weight
role contribution: 2,038,431,744 bytes/token (43.63% of the packed-weight lower bound) and 47.88% of the counted dense
matrix FLOPs. Cover the remaining Q4_K roles and Q6_K `lm_head` only after this first role family.

The first exact execution row in that largest decode role family is:

```text
m=1, n=12288, k=4096
A[1,4096] @ W[12288,4096]^T -> C[1,12288]
```

Here `m=1` is batch-one decode, `n=12288` is the output-row dimension, and `k=4096` is the reduction/input dimension.
Its request must use `shape_mode=exact_workload`; `m=1,n=256,k=256` remains only a bounded provider smoke fixture.
Candidate metadata carrying the exact role does not count unless compile, correctness, and timing evidence record the
same executed shape.

### 9.3 Initial bounded axes

BubbleBeam should propose legal values from target/compiler facts for BoltBeam to expand, initially including:

- group/reduction width;
- local threads/block size;
- output rows per thread;
- packed inner-period unroll;
- legal vector/coalesced load width and alignment;
- reduction placement and subgroup strategy;
- generic control/heuristic schedule as an explicit population member.

BoltBeam performs the sole finite Cartesian expansion. FutureSight may reject non-divisible, over-threaded,
over-memory, unsupported-transform, or statically dominated rows and determine measurement order. BoltBeam must
retain the declared finite population and the reason for every unmeasured rejection. Search dimensions must be data,
not nested `if target == Metal` policy.

Prefill and `simdgroup_matrix` become a second workload after the decode loop is proven. They must reuse the same
candidate/provider/evidence contracts with different legal dimensions.

## 10. Implementation packets

Current status refers to the local uncommitted working trees, not a landed compatibility claim:

| Packet | Current status |
| --- | --- |
| M0 | Implemented/focused-tested: versioned contracts, compatibility manifest, and drift/fail-closed tests exist |
| M1 | Implemented/focused-tested: exact M4 registry/autoscan/resolved-target plus live provider facts; target remains descriptor-only pending the measured loop |
| M2 | Implemented/focused-tested: strict v2, unchanged v1 identity, finite expansion/control, and execution-regime validation |
| M3 | Implemented/focused-tested: modular input-driven proposals and static assessment; BoltBeam remains schema/hash/expansion authority |
| M4 | Implemented/focused-tested: one JSON-lines worker and one BoltBeam subprocess adapter cover all five actions |
| M5 | Partial hardware smoke: bounded generated Metal programs compile for explicit/heuristic schedules and emit source/MTLB/plan hashes; exact `1x12288x4096` compilation is unrun |
| M6 | Partial hardware smoke: bounded `1x256x256` Q4_K/Q6_K oracle checks pass; exact-role candidate/control correctness and timing are unrun |
| M7 | Not run: no finite exact-workload population result |
| M8 | Not run: no selected plan is bound; generic fallback remains the only runtime behavior |
| M9 | Partial analysis only: exact GGUF inventory/raw advertised roof and a non-authoritative aggregate read+write proxy exist; no paired whole-model or policy verdict |
| M10 | Pending after durable artifacts and landed implementation |

### M0 — freeze contracts and drift tests

Owner: both repositories, no GPU required.

- Add fixture tests that expose the current false Metal capability, missing runtime mapping, AMD-only event filter,
  AMD-specific v1 schema, and missing worker paths.
- Define the resolved-target, candidate v2, provider, and evidence schema documents before implementation.
- Add a cross-repository compatibility manifest containing required schema versions and minimum commits.

Gate: tests fail for the identified gaps and pass only through public contracts; no implementation is duplicated in
fixtures.

### M1 — central target discovery and capabilities

Owner: BoltBeam registry/autoscan plus one tinygrad Metal fact adapter; local Metal required for final fixture capture.

- Add generic probe registration to autoscan and a Darwin/Metal probe.
- Add `tinygrad_device=METAL` to the family descriptor.
- Resolve an exact Apple target from observed facts; leave unknown Apple variants descriptor-only.
- Replace hardcoded collector support arrays with capability resolution from registered adapters.
- Separate static, dynamic, modeled, and unknown hardware facts.

Gate: an autoscan fixture selects the exact registered M4 target, an explicit user target is preserved, unknown Apple
hardware fails closed, and no Metal value is duplicated between registry and runner plan.

### M2 — candidate v2 and v1 compatibility

Owner: BoltBeam.

- Add the target-neutral candidate variant and canonical hash.
- Adapt historical v1 AMD rows into the common in-memory representation without changing their stored bytes/hash.
- Move backend vocabulary validation behind target capability adapters.
- Make finite Cartesian expansion and full-kernel search operate on the common interface.

Gate: existing AMD full-kernel tests pass, Metal schedule candidates validate, AMD-only fields cannot enter a Metal
plan, and two semantically identical canonical candidates hash identically.

### M3 — modular BubbleBeam/FutureSight generation

Owner: tinygrad EXP research.

- Extract target-neutral dimension-proposal, static-assessment, and rejection types from the current Q4 script.
- Keep the historical Q4 lane-map wrapper behavior stable.
- Add a tinygrad schedule vocabulary adapter and live target-fact consumer that filters caller-declared paths/values
  without expanding their Cartesian product.
- Pass legal dimension mappings to BoltBeam's canonical candidate instantiator, then emit a deterministic static
  ordering/rejection report keyed only by BoltBeam-supplied candidate hashes.

Gate: no imports from route manifests in the shared engine; AMD historical tests remain stable; proposals are derived
entirely from inputs; BoltBeam remains the only candidate schema/hash/finite-expansion authority; static ranking cannot
emit a promotion verdict.

### M4 — one tinygrad provider worker

Owner: tinygrad EXP research plus BoltBeam subprocess adapter; no GPU required for protocol tests.

- Implement the common actions and structured error taxonomy.
- Replace `_ANCHOR_WORKLOAD` with the request workload.
- Replace removed `extra/qk` command paths with the canonical worker/LLM entrypoint.
- Remove or adapt the stale admission-only adapter after all callers migrate.
- Ensure dirty-tree, revision, timeout, identity, unsupported-plan, and hardware-absent failures are explicit.

Gate: CPU/mock tests exercise every action and failure; no target-specific worker executable exists.

### M5 — Metal compilation and evidence adapter

Owner: tinygrad Metal runtime/provider; local Metal required.

- Bind a candidate plan to one captured semantic AST using compiler-owned `Opt` transforms and candidate cache context.
- Emit generated MSL/MTLB hashes and exact launch/pipeline facts.
- Add a backend-neutral program evidence event if needed; Metal must not be forced through the AMD ELF decoder.
- Keep scalar dtype, lanes, storage quant, and schedule plan orthogonal per the dtype migration.

Gate: control and at least two distinct legal plans compile; hashes are stable on repeated clean runs; invalid plans
fail admission before execution; no handwritten MSL/UOp hot kernel is introduced.

### M6 — Metal correctness and isolated timing

Owner: tinygrad provider plus BoltBeam evidence validator; local Metal required.

- Implement Q4_K then Q6_K semantic fixtures.
- Execute the first exact role at `m=1,n=12288,k=4096` with resident identical buffers; bounded fixtures remain
  compile/correctness smoke only.
- Measure exact eager command-buffer GPU time with raw samples.
- Mark MetalGraph entry timing non-authoritative in capability/evidence output.
- Generalize `tinygrad_profile_events` device filtering and resource extraction into backend adapters.

Gate: correct rows become `MEASURED`, incorrect rows become `REJECTED_INCORRECT`, unsupported resource fields remain
missing with reasons, and no AMD bandwidth/resource default appears in Metal evidence.

### M7 — finite Qwen3-8B decode search

Owner: all four components; local Metal required.

- Derive exact model roles from the GGUF hash/inventory.
- Freeze a bounded candidate population and include the generic control.
- Run compile/check/measure for every row under isolation and deadline.
- Preserve blocked/rejected rows and require population completeness.
- Export the selected plan or an explicit no-win/refutation result.

Gate: replaying the request at pinned commits reproduces candidate identities and classifications; result is
`COMPLETE`, not merely partially measured.

### M8 — generated route binding and rollback

Owner: tinygrad generic route-plan loader; local Metal required.

- If M7 finds a winner, export immutable selected plan/provenance through the existing route-plan architecture.
- Bind only exact target/model/role/shape/quant matches.
- Trace candidate/plan/binary identity at runtime.
- Keep ordinary generic Metal as one-switch rollback and behavior for every unmatched case.
- Do not reuse or relax gfx1100 route eligibility.

Gate: exact match fires; every mismatched field falls back; rollback restores generic output; production modules do not
import research code.

### M9 — whole-model evaluation and BoltBeam policy

Owner: BoltBeam evaluation; local Metal required.

- Run paired generic/candidate Qwen3-8B decode with final warmup/sample protocol.
- Record memory, compile-cache, correctness, route-binding, and stability evidence.
- Retain the 120 GB/s advertised raw roof, capture the aggregate read+write stream probe under its own proxy label,
  and derive exact-workload placement with existing BoltBeam math only when the traffic scopes are comparable.
- Report per-role work/bytes and gap attribution without inventing unsupported hardware counters; leave practical
  weight-read bandwidth unknown until a scope-compatible measurement exists.
- Refresh the local llama.cpp Metal control separately.
- Promote only through centralized policy; otherwise retain a measured refutation/reopen condition.

Gate: BoltBeam emits one of `promote`, `refute`, `defer`, or `inconclusive` with complete evidence and rollback. A
microkernel win alone cannot promote.

### M10 — documentation, pruning, and closure

Owner: both repositories, no GPU required after artifacts exist.

- Add the exact replay/search/evaluation commands and artifact map.
- Correct capability docs so Metal support means the achieved compatibility level.
- Remove dead worker paths, duplicate target maps, redundant schemas/adapters, and temporary probes.
- Add contract tests preventing stale paths and optimistic capability declarations from returning.
- Update this document with commits, evidence hashes, result, and remaining performance gap.

Gate: clean tests, clean worktrees, no duplicate authority found by `rg` audit, and a new user can run the documented
search/control flow from the two pinned checkouts.

## 11. Dependency graph and safe parallelism

```text
M0
├── M1 target facts
├── M2 candidate v2
└── M4 provider protocol shell
     ├── M3 generator (needs M2 vocabulary)
     └── M5 Metal compile (needs M1 + M2 + M4)
          -> M6 correctness/timing
             -> M7 finite search
                -> M8 binding, only if a winner exists
                   -> M9 whole-model policy
                      -> M10 prune/close
```

M1, M2, and the protocol-only portion of M4 can run in parallel. All local Metal execution is one serialized hardware
lane. M8 is conditional: a complete no-win search skips runtime binding and proceeds to a refutation in M9.

## 12. Test matrix

| Layer | Required coverage |
| --- | --- |
| Registry | family vs exact target, unknown Apple variant, fact provenance, no static unified-memory claim |
| Autoscan | Darwin fixture, multiple/absent devices, explicit target preservation, registered/unregistered result |
| Runner plan | `DEV=METAL`, no AMD `ARCH`, canonical worker command, portable paths |
| Candidate schema | v1 AMD compatibility, v2 canonicalization, orthogonal dtype/quant/layout, target vocabulary rejection |
| Generation | deterministic BubbleBeam value proposals/static assessments, BoltBeam-owned population/hash/budget, no static promotion output |
| Provider protocol | every action, malformed/stale/hash mismatch, dirty tree, timeout, no hardware, unsupported plan |
| Metal compile | distinct plans, source/MTLB hashes, launch limits, compile-cache isolation |
| Correctness | Q4_K blocks, Q6_K blocks, ragged/guarded shape, real role output, repeated finite results |
| Timing | eager GPU timestamps, warmup exclusion, raw samples, graph estimate rejection, paired ordering |
| Evidence | target/compiler binding, unsupported resources explicit, no AMD defaults, content hashes |
| Search | complete/partial/incorrect/blocked populations, deterministic tie break, control candidate present |
| Runtime | exact bind, all mismatch fallbacks, route trace, rollback, no research import |
| Model | Qwen3-8B load/decode at fixed depth 128 with sufficient capacity, memory admission, output stability, generic/candidate A/B |
| Policy | no promotion without correctness/speed/memory/route/rollback; refutation carries reopen condition |
| Documentation | commands resolve, files exist, commits/hashes pinned, figures have one authority |

## 13. Explicit non-goals

- No AMD eGPU reset, PCI, Thunderbolt, Linux, KFD, or kernel recovery work.
- No change to AMD route selection or promotion while proving Metal compatibility.
- No handwritten Metal kernel, MSL source, precompiled MTLB, or copied llama.cpp Metal kernel.
- No assertion that the current bottleneck is bandwidth, dequantization, launch count, or graph capture before evidence.
- No MLX integration or MLX performance claim.
- No prefill `simdgroup_matrix` search until the decode loop is complete.
- No promotion to tinygrad `dev` or `master` in this scope.
- No requirement that Metal beat llama.cpp for compatibility to be complete.

## 14. Completion definitions

### Compatibility complete

- exact Apple target discovered and resolved;
- one canonical provider protocol works on Metal;
- target-neutral candidates are generated, admitted, compiled, checked, and timed;
- a finite real Qwen3-8B decode population completes;
- BoltBeam emits a replayable measured result and honest policy verdict;
- ordinary Metal control and rollback remain available;
- stale/duplicate seams are removed and documentation is runnable.

### Performance route complete

All compatibility conditions plus a selected candidate passes exact route binding, paired whole-model improvement,
memory/correctness/stability gates, and BoltBeam promotion. If no candidate wins, compatibility is still complete and
the output is a measured refutation with a precise reopen condition.

### Promotion beyond EXP

Not authorized by this scope. A later promotion packet must prove the generated-plan closure, public runtime boundary,
branch-specific tests, and no regression to the AMD machine-search path after the pending physical AMD dtype gate.
