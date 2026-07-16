# Memory-Adaptive Prefill Generalization Scope

Date: 2026-07-15
Status: implementation scope
Primary architecture: `docs/prefill-memory-fit-architecture-20260715.md`

## Objective

The user explicitly selects the model to load. Replace model-name/profile-driven execution-route selection with a
fact-derived feasibility planner plus target-local machine
search over:

```text
FULL_RESIDENT_OVERLAY
BOUNDED_PACKED_TILES
DIRECT_PACKED_FALLBACK
REFUSE
```

Feasibility must be reproducible from the selected model's loaded tensor inventory, requested workload, automatically
detected device memory, and target
kernel capabilities. Selection must be reproducible from measured candidate evidence on the actual target. `8B`,
`14B`, model filenames, and benchmark profile IDs may identify evidence fixtures but may not affect semantic admission
or search-cache identity.

## Required runtime flow

```text
GGUF metadata + requested context/prefill + device facts
                         |
                         v
               exact residency accounting
                         |
                         v
          complete role/quant/shape inventory
                         |
                         v
       strategy capability and coverage matching
                         |
                         v
                 safe feasible set
                         |
                         v
       correctness/resource gates + machine search
                         |
                         v
              immutable selected policy
                         |
             +-----------+-----------+
             |           |           |
        model load   prefill route   KV/context
             |           |           |
             +-----------+-----------+
                         |
                         v
            runtime census + measured peak
```

No downstream component may independently reinterpret the model name, environment profile, or parameter count to
choose a different semantic strategy. Safety admission and performance selection are separate: search cannot bypass a
memory/correctness gate, and the safety planner cannot claim one feasible route is fastest without measurement.

## Terms

- **Facts:** values derived from GGUF tensor metadata, transformer configuration, requested workload, allocator/device
  query, or final compiled target capability.
- **Capability:** a structural contract proving that a route accepts a phase, quant format, shape/tail class, dtype,
  target, and resource bound.
- **Profile:** a reproducible benchmark/evidence label. It is never a runtime compatibility predicate.
- **Candidate:** one generated kernel/schedule with a canonical identity and explicit capability contract.
- **Coverage:** proof that every required tensor invocation has exactly one selected correct route or an explicit
  fallback.
- **Peak residency:** simultaneous device allocations at the phase peak, not file size or parameter count alone.
- **Feasible set:** complete strategies/candidates that pass memory, semantic coverage, correctness, target resource,
  and GPU-health gates.
- **Machine policy:** the measured selection from the feasible set for a canonical hardware/model/workload signature.

The autoscanner does not choose among models. It scans the explicitly selected model, detects the GPU and current VRAM
budget, enumerates safe execution routes for that combination, and chooses the fastest valid route.

## Scope boundaries

In scope:

- dense GGUF prefill weight representations and their coexistence with packed decode weights;
- context/KV memory because it participates in the same residency decision;
- full-overlay, bounded packed-tile, direct-packed fallback, and refusal branches;
- Q4_K and Q6_K mixed inventories;
- runtime admission, route binding, manifests, candidate identities, tests, and evidence;
- removal of model/profile identifiers from semantic route selection;
- structural target capability matching for the existing AMD/gfx1100 implementation.

Not required for the first completion gate:

- CPU or disk offload;
- distributed or multi-GPU partitioning;
- speculative decoding or batch sizes greater than the existing runtime supports;
- proving a single tile geometry optimal on every GPU;
- replacing the batch-1 decode GEMV/MMVQ implementation.

Decode must remain correct and packed-weight-backed. Decode hardcodes discovered by this work are recorded separately
and must not be silently broadened without evidence.

## Current authority and debt

Already fact-derived and integrated:

- selected GGUF tensor payloads, per-allocation alignment, and covered FP16 element counts;
- model geometry used to estimate KV bytes;
- one unconditional production scan for GPU capabilities, total/free VRAM, and allocator granularity;
- a live occupied-byte safety reserve instead of a fixed production VRAM ratio/tier;
- arithmetic context admission and the unified feasible-strategy plan;
- Q4_K/Q6_K tensor facts and role/shape inventory;
- structural Q4 G3 and Q6 generated decode predicates;
- profile-free exact runtime inventory/policy identities and cache validation;
- logical-vs-physical M planning and exact route census contracts.
- a model-only production search entry: device identity/facts, VRAM reserve, candidate enumeration, and execution seam
  are internal runtime authorities rather than caller inputs;
- non-square attention Q/O inventory geometry and explicit fixed-route coverage for tied LM-head projections.

Still incomplete:

- candidate-local physical M=512 is the only currently validated overlay/direct prefill M;
- complete measured allocation fields are not yet persisted from guarded search into accelerated cached policies;
- physical base-allocation ownership and final schedule memory-lifetime manifests are not yet joined into pre-dispatch
  admission and runtime reconciliation;
- bounded packed MMQ is not production-complete across Q4_K/Q6_K roles and tails;
- Q4_K full-kernel emission is blocked at the current real compiler/resource boundary;
- historical route IDs and evidence notes retain model-size labels as compatibility/provenance only; the selector audit
  verifies that they do not gate behavior.

## Canonical data contracts

### 1. Memory facts

The planner input must represent, in bytes:

- device total and free memory at the defined probe point;
- allocator/safety reserve policy;
- packed tensor allocations including alignment and duplicated storage;
- optional dense overlay allocations by tensor/role;
- KV data and scale allocations from batch, layer count, KV heads, head/value dimensions, context, and dtype;
- persistent rope/cache/runtime allocations;
- peak prefill activation, output, LDS-independent workspace, and compiler/runtime scratch;
- any candidate-specific global workspace;
- uncertainty or unavailable facts, with a conservative outcome.

Every byte term needs a name, provenance, formula, and inclusion lifetime. The sum must expose both estimated peak and
the admitted budget.

### 2. Workload inventory

Each required routed invocation must carry:

```text
tensor identity
phase and role
quant format and physical block ABI
M, N, K and tail class
input/output/accumulator dtype
call count
packed bytes and optional overlay bytes
bias/epilogue requirements
```

Inventory is read from model facts and actual runtime semantics. A family quant label such as `Q4_K_M` is not enough.

### 3. Target capability

The target descriptor must expose:

```text
backend and architecture
wave size
workgroup/thread limits
LDS budget and allocation granularity
supported integer/floating contraction forms
operand signedness and accumulator forms
barrier/cross-lane primitives
register/scratch evidence limits
supported packed producers and physical ABIs
```

A source-pinned gfx1100 128x128x256 MMQ geometry is a valid candidate fact, not a universal semantic restriction.

### 4. Route capability

Every route must publish a structural predicate and resource/workspace contract over:

```text
phase, quant, M/N/K, tails, dtypes, target, bias/epilogue, batch/microbatch
```

The result is one of `SUPPORTED`, `UNSUPPORTED(reason)`, or `REQUIRES_FALLBACK(reason)`. Profile IDs are forbidden
inputs.

### 5. Immutable plan

The selected plan must include:

```text
strategy
memory facts and arithmetic
requested/admitted context and KV representation
feasible route/candidate sets for every inventory row
machine-selected route/candidate for every inventory row
fallback rows and reasons
canonical identities
expected peak and safety reserve
search objective, measurements, cache key, and compiler/runtime revision
decision provenance
```

The model loader, prefill runtime, and evidence census consume the same plan object.

## Decision semantics

1. If packed weights plus minimum runtime/KV state do not fit, select `REFUSE` unless an explicit offload policy exists.
2. Add `FULL_RESIDENT_OVERLAY` to the feasible set only when its complete simultaneous peak fits and every required
   dense-overlay route is supported.
3. Add `BOUNDED_PACKED_TILES` only when every required invocation has bounded packed coverage and its global workspace
   fits.
4. Add `DIRECT_PACKED_FALLBACK` only when every uncovered invocation has an explicit correct fallback.
5. If the feasible set is empty, select `REFUSE` with all uncovered inventory rows and memory reasons.
6. Correctness/resource/GPU-health gate every feasible candidate before timing.
7. Machine-search the surviving complete strategies on the actual target and select by the declared objective.
8. No branch may allocate a hidden full dequantization or silently reinterpret the plan.
9. Explicit user overrides may restrict the search set but cannot bypass safety or semantic capability checks.

## Machine-search contract

Search inputs:

- target/backend/architecture, wave size, LDS/register/resource limits, and compiler/runtime revision;
- automatically detected total/free VRAM and the selected runtime safety reserve;
- measured target characteristics where useful: global bandwidth, launch latency, supported contraction throughput,
  and LDS behavior;
- complete tensor/role/quant/shape/call-count and content identity for the user-selected model;
- requested prefill lengths, context/KV representation, and execution mode;
- all structurally feasible full-overlay, bounded-tile, direct-packed, fusion, tile, workgroup, staging, and pipeline
  candidates;
- an explicit objective. The current primary objective is steady-state end-to-end tok/s.

Search protocol:

1. Generate candidates from logical grammars and target facts; do not enumerate model names.
2. Reject candidates that exceed memory or final compiled resource limits.
3. Run isolated full-output correctness and guarded GPU-health checks.
4. Measure representative role kernels to prune dominated candidates.
5. Measure complete candidate policies end to end; kernel-only winners do not automatically promote.
6. Store raw samples, route census, allocation peak, program identities, and environment provenance.
7. Select the fastest statistically credible complete policy; retain rollback and invalidation metadata.
8. Cache by canonical facts and revision. Invalidate when any material input, candidate, compiler, runtime, model
   inventory, context objective, or device fact changes.

Search output:

```text
canonical search key
feasible and rejected candidates with reasons
correctness/resource/health evidence
raw timing samples and statistic
selected complete policy
confidence/noise decision
fallback and invalidation policy
```

Autoscanning may run at install/load time or consume a previously cached exact-fact result. It must fail safely to a
correct feasible baseline when interrupted; it must never silently reuse a policy from a merely similar profile.

## Implementation phases

### P0 — Freeze current behavior and evidence

Deliverables:

- unit fixtures for current fitting and non-fitting examples;
- retained token/correctness references for both branches;
- current runtime route census and allocation estimates;
- explicit inventory of existing profile/name selectors.

Acceptance:

- tests distinguish behavioral authority from historical benchmark labels;
- no existing user changes are reverted;
- all later changes can be compared against frozen outputs.

### P1 — Exact memory-facts ledger

Deliverables:

- typed byte-term records with provenance and lifetime;
- exact overlay bytes from actual covered tensors;
- packed allocation bytes rather than parameter-count guesses;
- KV bytes for fp16/int8 plus scale metadata;
- peak workspace hooks per strategy/candidate;
- safety reserve and unknown-fact policy.

Acceptance:

- sum is deterministic and serializable;
- alignment/duplicate allocations are represented;
- unknown free memory never enables a risky overlay automatically;
- unit fixtures hand-check every byte term.

### P2 — Unified strategy planner

Deliverables:

- immutable strategy enum and plan/input/result types;
- one pure decision function;
- explicit refusal/fallback reasons;
- boundary tests immediately below, at, and above each fit threshold;
- same-model/different-memory and same-memory/different-name tests.

Acceptance:

- no model name, path, size label, or profile is an input;
- changing requested context can legitimately change the selected strategy;
- explicit overrides fail safely when impossible;
- plan serialization is stable enough for evidence.

### P3 — Model-loading and context integration

Deliverables:

- construct the plan once from GGUF/device/workload facts;
- merge overlay and context/KV admission into one arithmetic authority;
- remove the separate fixed overlay GB gate from semantic selection;
- pass the immutable plan to model construction and runtime;
- preserve explicit diagnostic overrides without allowing unsafe execution.

Acceptance:

- realized allocations match the selected strategy;
- a non-overlay plan never creates `_pf16_w` buffers;
- context admission uses the same weight residency selected by the plan;
- failure occurs before large allocations.

### P4 — Target and route capability registry

Deliverables:

- typed target capability descriptor for current gfx1100 facts;
- route capability records for full-overlay, bounded Q4, bounded Q6, and direct fallbacks;
- structural compatibility matcher;
- resource/workspace contract consumed by P1/P2.

Acceptance:

- profile identifiers are absent from compatibility predicates;
- unsupported dtypes/tails/targets return reasons rather than falling through;
- exact candidate geometry remains in the candidate descriptor;
- final compiled resource evidence can be reconciled with admitted limits.

### P4A — Autoscanner and machine-policy selection

Deliverables:

- canonical hardware/model/workload/search-revision key without names or size labels;
- candidate enumeration from the feasibility and capability registries;
- guarded correctness/resource/health executor;
- role-level pruning followed by end-to-end policy timing;
- noise-aware selection and exact-key policy cache;
- safe interruption/no-result fallback to a correct feasible baseline;
- raw evidence artifact and runtime policy loader.

Acceptance:

- copying/renaming the selected model preserves the key; changing its content/tensor facts changes it;
- changing GPU capabilities, compiler/runtime revision, context objective, or candidate inventory invalidates it;
- an unsafe or incorrect fast candidate cannot win;
- the selected policy is based on end-to-end tok/s on the actual target;
- cached and freshly searched policies produce identical route census and outputs;
- no known model profile is required to run the search.

### P5 — Profile-free runtime binding

Deliverables:

- remove `COOPERATIVE_Q4K_PROFILE` from semantic selection;
- bind candidates by inventory and capability identity;
- replace filename/profile lookup in production paths with GGUF/config facts;
- retain profiles only in harness/evidence APIs;
- remove size labels from new route IDs and add compatibility aliases where required.

Acceptance:

- renamed/copy-identical model files bind identically;
- synthetic models with matching facts bind without a known profile;
- evidence from the wrong shape/target cannot authorize a route;
- the runtime census reports candidate identity and structural binding facts.

### P6 — Full-overlay branch generalization

Deliverables:

- supported microbatch/candidate set represented as capability data;
- correct prompt remainder behavior for every admitted microbatch;
- schedule lookup by canonical shape/target identity;
- no fixed assumption that 512 is semantically required.

Acceptance:

- unsupported microbatches fail/choose fallback before capture;
- at least two prompt-size/remainder classes are tested;
- current fitting fixture retains output parity and no material performance regression.

### P7 — Bounded packed-tile coverage

Deliverables:

- production-complete outer-K Q4_K/Q8_1 kernel from the exact source-pinned substrate;
- Q6_K bounded route with a separate physical decoder/grammar;
- runtime grids, addresses, IDs, channel/sample offsets, and tails from logical facts;
- bounded LDS/global workspace evidence;
- role coverage for attention, FFN, and required output head behavior;
- explicit direct-packed fallback for any deliberately uncovered row.

Acceptance:

- no full `[N,K]` dequantized allocation;
- all required rows have exactly one route or named fallback;
- full-output numerical checks pass for edge and production shapes;
- final gfx1100 code object has no spills/scratch/resource violation under the promotion gate;
- whole-model prefill route census proves the emitted kernel is actually used.

### P8 — Fallback and partial-coverage semantics

Deliverables:

- deterministic per-row fallback map;
- no implicit ordinary graph fallback for a plan claiming complete bounded coverage;
- optional hybrid overlay/packed strategy deferred unless explicitly represented and measured;
- clear refusal diagnostics listing uncovered rows.

Acceptance:

- deleting one capability produces a predictable fallback/refusal test result;
- no route claim exceeds its evidence scope;
- fallback correctness and memory bounds are independently tested.

### P9 — Manifests and evidence-only profiles

Deliverables:

- candidate manifests keyed by canonical workload/capability identity;
- profile IDs retained only as benchmark provenance;
- model-path inference confined to CLI/harness convenience;
- migration/aliases for retained artifacts;
- schema validation rejecting profile-only semantic admission.

Acceptance:

- runtime can load and route an unknown filename from facts;
- manifests cannot authorize a mismatched shape or target;
- retained benchmark fixtures remain reproducible.

### P10 — Decode coexistence audit

Deliverables:

- proof that packed weights remain resident for decode under every prefill strategy;
- no decode graph observes overlay-only state;
- separate record of decode shape constants (`Hq/Hkv/Hd`, KV dtype, route IDs) classified as capabilities or debt;
- fixed-live-context decode regression matrix.

Acceptance:

- batch-1 outputs match across prefill strategies;
- current packed Q4/Q6 decode route census remains intact;
- no claimed decode improvement is inferred from prefill work.

### P11 — Boundary and end-to-end validation

Required matrix:

- same model at multiple synthetic/admitted VRAM budgets;
- same model at multiple requested contexts;
- different filenames with identical tensor facts;
- different quant mixes with similar parameter counts;
- full-overlay, bounded, direct fallback, and refusal branches;
- prompt lengths below, equal to, and above admitted microbatch;
- Q4/Q6 mixed role inventories and edge tails;
- cold load, warm prefill, decode after prefill, and repeated requests.

Required evidence:

- full token/output parity;
- measured peak allocation versus planned peak;
- route census with canonical candidate identities;
- final ISA/resource evidence for generated kernels;
- GPU health before/after isolated execution;
- whole-prefill tok/s and batch-1 decode tok/s;
- llama comparison under the same model, prompt, context, KV, clocks, warmups, and repetitions.

Acceptance:

- no OOM at or below the admitted boundary;
- deterministic refusal above it;
- no model-name-dependent decision;
- bounded branch materially improves the direct-packed baseline before promotion;
- fitting branch is not regressed beyond the declared promotion threshold.

### P12 — Cleanup and closeout

Deliverables:

- remove superseded duplicate policy functions and semantic profile constants;
- update architecture/current-state docs;
- record rejected schedules separately from runtime policy;
- commit only reviewed, tested changes without absorbing unrelated dirty-tree edits.

Completion means all four strategy outcomes are represented, every runtime choice is fact/capability-derived, both
fitting and non-fitting fixtures execute correctly, the non-fitting bounded route is production-routed, and the full
boundary/evidence matrix passes.

## Workstream ownership

The implementation can proceed in parallel only with disjoint write ownership:

1. **Memory plan foundation:** planner/policy types and pure boundary tests.
2. **Route capability binding:** prefill route compatibility and profile-free candidate admission.
3. **Manifest/profile separation:** evidence schema and harness-only model profiles.
4. **Machine-policy engine:** canonical key, candidate execution, evidence cache, and selection.
5. **Main-thread integration:** `model.py`, context admission, conflict resolution, and end-to-end assembly.
6. **Bounded kernel completion:** Q4/Q6 emitters and their isolated correctness/resource gates, split further by quant.
7. **Independent validation:** cross-boundary tests, route census, allocation measurement, and benchmark protocol.

Workers must preserve the dirty worktree, avoid reverting other edits, and report every file changed. Integration is
dependency ordered: P1/P2/P4 foundations, then P3/P5/P6, then P7/P8/P9, then P10/P11/P12.

## Completion checklist

- [x] One immutable, serializable memory/strategy plan.
- [x] No model name/path/size/profile in semantic strategy selection.
- [x] Exact selected-GGUF packed byte ledger and conservative unknown-memory behavior.
- [x] Unified overlay/context/KV admission.
- [x] Structural target and route capability contracts.
- [x] Exact-fact autoscanner and fail-closed machine-policy cache contract.
- [x] Correctness/resource/health/route-census-gated end-to-end selection contract.
- [x] Profile-free runtime candidate binding.
- [x] Candidate-local logical/physical remainder admission.
- [ ] Production-complete bounded Q4_K route.
- [x] Production-complete bounded Q6_K route or explicit per-row fallback.
- [ ] No hidden full dequantization on non-overlay branches.
- [x] Evidence-only profiles/manifests (selector audit: zero gating findings).
- [ ] Decode packed-weight coexistence and fixed-context regression proof.
- [ ] Boundary, correctness, resource, GPU-health, route-census, and performance matrices pass.
- [x] Superseded model/profile/VRAM-tier selectors removed and docs updated.
