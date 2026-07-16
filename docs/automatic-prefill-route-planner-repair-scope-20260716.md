# Automatic Prefill Route Planner Repair Scope

Date: 2026-07-16
Status: validated implementation scope
Parent architecture: `docs/prefill-memory-fit-architecture-20260715.md`
Parent generalization scope: `docs/memory-adaptive-prefill-generalization-scope-20260715.md`

## Implemented 8B bridge and validation

The retained promoted candidate is now projected onto the selected GGUF's exact runtime inventory and the live scanned
target. Candidate target requirements come from the admitted artifact itself; neither a model/profile name nor a
standalone Graph-GEMM Boolean selects execution. The loader admits the complete FP16 representation with the shared
memory planner and installs exact role/M/N/K bindings only after that admission succeeds. The final-token-only LM head
remains a fixed M=1 invocation outside the pp512 candidate set.

A first live run exposed and repaired an independent execution regression: semantic ownership wrapped each already
resident FP16 weight, and the generated GEMM boundary called `contiguous()` on that wrapper. This allocated another
32/96 MiB dense matrix per layer until OOM. Candidate operand preparation now preserves the concrete semantic alias;
lazy operands still materialize normally.

Live Qwen3 8B Q4_K_M validation on gfx1100:

```text
legacy route/profile flags unset: FULL_RESIDENT_OVERLAY selected
exact graph bindings:           252 / 252 covered linears
candidate route census:         4 / 4 role-shape entries, no missing or unexpected rows
pp512 synced smoke:             145.9 ms, 3509 tok/s
max_context=4096 planned peak:  19,792,535,552 bytes
max_context=4096 budget:        25,322,094,592 bytes
focused tests:                  161 passed
```

The non-fitting/general bounded branch remains separate work under the completion phases below; this result closes the
8B full-overlay planner-to-executor gap.

The follow-up selector cleanup removed the obsolete `PREFILL_GRAPH_GEMM` production/harness path and the dead module
globals `PREFILL_V2`, `PREFILL_CONCRETE_KV`, `PREFILL_REMAINDER_FIX`, `PREFILL_WORKLOAD_REUSE`, and
`PREFILL_TC_ATTN`. Benchmark attribution now serializes the loaded candidate registry for diagnostic tooling instead
of synthesizing a route-selection environment. Offline candidate tools still accept explicit candidate artifact JSON
or paths; those inputs describe what to inspect and never select ordinary model execution.

Post-cleanup live checks retained the intended split without route flags: 8B selected the generated full-overlay route
at 3503 tok/s pp512, while 14B selected the direct-packed route at 357 tok/s pp512. The expanded focused regression
set passed 243 tests.

## Decision

Remove `PREFILL_GRAPH_GEMM` and its Boolean descendants as execution authority. The selected model path remains a
user input. Prefill execution becomes the output of one fact-derived planner that:

1. scans the selected model, requested workload, and current device;
2. enumerates exact compatible candidate policies;
3. rejects candidates whose complete peak residency cannot fit;
4. correctness/resource/health-gates the remaining candidates;
5. measures complete policies on the target;
6. returns one immutable execution plan; and
7. binds and executes that plan without reinterpreting flags.

Memory fit is a hard admission boundary. It is not a performance ranking. If full overlay and packed routes both fit,
the planner must measure them. If only packed routes fit, the planner must select the fastest complete packed policy.

## Validation performed before scoping

The design was tested on the live AMD gfx1100 RX 7900 GRE using the actual Qwen3 8B and 14B Q4_K_M GGUF tensor
inventories, a 4096-token context reservation, the repository's allocation formulas, and a live device scan.

Live facts:

```text
total VRAM                  25,753,026,560 bytes
live free VRAM              25,483,874,304 bytes
admitted budget             25,214,722,048 bytes
allocator granularity                    4,096 bytes
```

Planner results:

| Selected fixture | Candidate | Predicted peak | Result |
|---|---|---:|---|
| Qwen3 8B Q4_K_M | direct packed | 8,587,046,912 | admitted |
| Qwen3 8B Q4_K_M | conservative bounded staging proxy | 9,831,706,624 | admitted |
| Qwen3 8B Q4_K_M | full FP16 overlay | 23,723,241,472 | admitted |
| Qwen3 14B Q4_K_M | direct packed | 13,946,183,680 | admitted |
| Qwen3 14B Q4_K_M | conservative bounded staging proxy | 15,502,008,320 | admitted |
| Qwen3 14B Q4_K_M | full FP16 overlay | 41,926,123,520 | rejected by 16,711,401,472 bytes |

The bounded proxy deliberately overstates packed-MMQ residency: it permits one complete FP16 execution matrix to be
staged at a time and retains the existing conservative graph-materialization bound. It is a feasibility canary, not a
production memory contract. A proper packed MMQ candidate should stage bounded Q4/Q8 tiles and require substantially
less temporary memory.

The existing 14B-oriented bounded Q4_K/Q8_1 AMD cooperative tile was then executed on the live GPU at
16x16x256:

```text
correctness max_abs          0.0001220703125
correctness tolerance        0.001
blockers                     none
candidate median             19.965 ms
direct-packed comparator      8.971 ms
```

The bounded execution mechanism is numerically viable and memory-safe, but the current small research candidate is
not performance-promotable. An automatic planner would correctly admit it and retain direct packed today. This result
validates the architecture while preserving the bounded-kernel performance work as an explicit completion item.

Focused validation: 49 planner, catalog, integration, and bounded-kernel tests passed. The three reported warnings are
pre-existing pytest configuration warnings for unavailable timeout options.

## Required invariant

```text
selected ExecutionPlan
        |
        +-- strategy and exact route bindings
        +-- memory plan and admission evidence
        +-- candidate and inventory identities
        +-- workload/tail mapping
        +-- measured selection evidence
        |
        v
executor performs exactly that plan
```

The following are forbidden:

- an environment variable enabling an execution route;
- a model name, profile, parameter count, or named VRAM tier deciding compatibility;
- target family alone implying that Graph GEMM is active;
- runtime code reconstructing a route decision from a Boolean;
- a hidden full dequantization on a policy admitted as bounded or direct packed;
- a benchmark reporting Graph GEMM from requested environment rather than observed bindings;
- search or debug controls bypassing memory, correctness, or resource admission.

User/debug controls may restrict the candidate set for diagnosis. They may not add an inadmissible candidate or become
the selected policy.

## Current authority leaks to remove

### Runtime policy

- `tinygrad/llm/prefill_policy.py` derives `prefill_graph_gemm` and `prefill_tc_attn` from AMD/gfx1100 target matching.
  Target matching is a valid candidate compatibility input; it is not proof that a candidate is bound, fits, or won.
- `TransformerConfig.prefill_graph_gemm` stores a route Boolean next to the actual policy.
- `model.py` copies that Boolean to every covered linear as `_prefill_graph_gemm`.
- `_pf16` and `route_prefill_linear` branch on the Boolean instead of dispatching an attached route binding.

### Candidate loading

- `extra/qk/prefill_graph_gemm_route.py` still interprets `PREFILL_GRAPH_GEMM` and can load the promoted candidate
  environment implicitly.
- the existing promoted FP16 candidate artifact uses identities from an older canonical vocabulary;
- the model runtime expects a `graph_gemm` policy section, but the current memory-adaptive controller never produces
  one;
- normal production load does not activate the adapter bridge used by the isolated measurement worker.

### Inventory mismatch

- the selected-model inventory contains individual packed source tensors;
- the promoted Graph-GEMM artifact describes grouped post-dequant FP16 execution workloads;
- current exact policy rows do not define a canonical source-to-execution projection between those domains;
- an output projection can appear as physical M=512 in the general prefill inventory even though generation consumes
  only the final prefill token for LM head. Phase and call semantics must be explicit rather than inherited from the
  prefill microbatch.

### Tooling and attribution

- research guards, route manifests, harness profiles, and tests still use `PREFILL_GRAPH_GEMM` as a selector;
- benchmark environment may retain the spelling for historical provenance, but observed runtime attribution must be
  read exclusively from the loaded plan and route census;
- rollback must be a candidate/policy relationship, not `PREFILL_GRAPH_GEMM=0`.

## Target contracts

### Execution inventory

Add a canonical execution inventory distinct from packed storage inventory. Every execution row must include:

```text
execution invocation identity
source packed tensor identities
phase and role
logical and physical M, N, K
quant source ABI
execution input/weight/output/accumulator dtypes
call count and remainder mapping
bias/epilogue semantics
weight preparation and lifetime
activation preparation and lifetime
fallback route identity
```

The projection must be deterministic, lossless, and independently validated. Grouping identical execution shapes is
allowed only if every source tensor remains linked and call counts remain exact.

### Candidate capability

Every candidate publishes:

```text
candidate and compiled-program identity
supported execution-inventory predicate
target requirements
tile/workgroup/staging/writeback vocabulary
tail and remainder coverage
global workspace and persistent sidecars
LDS, VGPR, SGPR, scratch, and spill evidence
correctness and health evidence identity
```

An unavailable or stale field makes that candidate ineligible. Capability matching never reads a profile name.

### Memory plan

For each complete candidate policy, calculate peak live bytes from:

```text
aligned packed backing allocations
+ persistent representation sidecars
+ optional resident dense overlays
+ KV data and scales for admitted context
+ persistent runtime/RoPE state
+ peak activation and output allocations
+ candidate global workspace
+ compiler/runtime allocations
+ live occupied-byte reserve
```

Candidate-local LDS/register usage is a resource gate, not device-global VRAM residency. Unknown global byte terms fail
closed. Planned and measured peaks must use the same allocation ownership vocabulary.

### Selected execution plan

The immutable result contains:

```text
strategy and candidate-policy identity
exact per-invocation route bindings
source and execution inventory identities
device and workload identities
memory terms, budget, predicted peak, and admission reasons
kernel compile/resource artifacts
search objective, raw samples, and selected statistic
fallback policy
cache and invalidation identity
```

Binding presence is the execution authority. A diagnostic property such as `uses_graph_gemm` may be derived from the
bindings, but it cannot be stored as an independently mutable selector.

## Implementation sequence and commit boundaries

### R0 — Freeze the tested boundary

Deliverables:

- preserve the live test arithmetic as deterministic fixtures around the fit boundary;
- add same-model synthetic budgets immediately below, at, and above each predicted peak;
- retain the bounded GPU correctness artifact and direct comparator result;
- record the current 8B/14B decode and prefill baselines for regression comparison.

Gate: tests prove that changing names does nothing, while changing exact memory/workload facts can change feasibility.

Commit: test/evidence only.

### R1 — Replace Boolean policy vocabulary

Deliverables:

- introduce typed `ExecutionPlan`, `RouteBinding`, and candidate rejection records;
- remove `prefill_graph_gemm` as a field accepted by immutable runtime policy;
- expose derived diagnostics from bound route families;
- make strategy restrictions an explicit planner input used only to remove candidates.

Gate: there is no test in which setting a Boolean makes Graph GEMM execute.

Commit: contracts and pure tests; no route behavior change.

### R2 — Build source-to-execution inventory projection

Deliverables:

- map every packed source tensor to its exact execution workload(s);
- represent FP16 overlay, bounded Q4/Q8, bounded Q6, direct packed, and fixed LM-head semantics;
- correct physical M and call count for final-token-only LM head;
- identify role grouping without losing source identity or quant mix;
- canonicalize and hash the projection.

Gate: 8B and 14B inventories have complete, unique source and execution coverage; deliberate omission, ambiguity, or
wrong M fails.

Commit: inventory schema/projection and fixtures.

### R3 — Requalify the existing full-overlay candidate

Deliverables:

- regenerate the four promoted 8B FP16 workload identities under the current canonical vocabulary;
- bind current compile, correctness, resource, and timing artifacts to those identities;
- represent overlay preparation and residency explicitly;
- reject the old stale identities with a precise diagnostic.

Gate: all exact supported execution rows bind; mismatched shape, target, dtype, or stale identity does not.

Commit: artifact identity migration and admission tests.

### R4 — Enumerate real planner candidates

Deliverables:

- add full-overlay Graph GEMM candidate policies to the memory-adaptive catalog;
- retain direct packed as the complete baseline;
- enumerate bounded candidates only for their proven role/quant/shape coverage;
- produce a `graph_gemm` binding section from selected inventory and candidate artifacts;
- include complete memory and resource terms per candidate.

Gate: actual 8B facts expose full overlay plus packed alternatives; actual 14B facts reject full overlay before any
overlay allocation; no target-only default is present.

Commit: catalog and planner integration.

### R5 — Connect production loading to the selected plan

Deliverables:

- activate the explicit policy adapter in the production load path without import side effects;
- consume exact cache records or launch the guarded search according to the declared cold-load policy;
- attach `RouteBinding` objects directly to runtime linears/invocations;
- make `_pf16` dispatch the attached binding;
- reject accelerated plans with missing allocation evidence.

Gate: production and isolated measurement loads consume the same plan schema and produce the same route census.

Commit: load/binder seam.

### R6 — Delete flag authority end to end

Deliverables:

- remove `PREFILL_GRAPH_GEMM` from production model, route, manifest, guard, and benchmark-selection paths;
- remove `_prefill_graph_gemm` and `TransformerConfig.prefill_graph_gemm`;
- replace Boolean route arguments with exact binding dispatch;
- convert rollback to a selected baseline candidate identity;
- retain historical environment spellings only in artifact migration tooling when required.

Gate: repository audit finds no production read of `PREFILL_GRAPH_GEMM`; adding it to the environment cannot change a
runtime census.

Commit: production cleanup, then separate tooling/test cleanup if needed.

### R7 — Recover and validate the fitting full-overlay path

Deliverables:

- automatically bind all promoted full-overlay roles for the fitting fixture;
- verify no hidden profile/model-name requirement;
- validate prompt chunks and remainders across multiple lengths;
- compare whole-prefill throughput with the retained promoted result;
- prove decode remains packed and unchanged after prefill.

Gate: Graph GEMM appears because exact bindings won, not because a flag was set; generated candidate census is
complete; output parity, memory peak, GPU health, prefill threshold, and decode non-regression pass.

Commit: fitting-path recovery and evidence.

### R8 — Complete the non-fitting bounded candidate family

Deliverables:

- Q4_K/Q8_1 cooperative M/N/K tiling with outer-grid execution, full K accumulation, and exact output ownership;
- explicit DS4 activation preparation, cache/reuse lifetime, and preparation cost;
- independent Q6_K packed-tile candidate or explicit direct-packed per-row fallback;
- attention and FFN role coverage, M/N/K tails, mixed-quant policy composition, and final-token LM head handling;
- no complete `[N,K]` FP16 materialization;
- generated code/resource evidence and same-session role timing against direct packed and llama oracle structure.

Gate: bounded candidate is complete only when every selected invocation has one exact route or declared fallback and
the whole policy remains under the admitted peak. The current 16x16x256 pass is a seed proof, not completion.

Commits: separate logical vocabulary, Q4 lowering, Q6/fallback, ownership/tails, and evidence commits.

### R9 — Machine search, cache, and failure behavior

Deliverables:

- baseline-first guarded candidate execution;
- role-level pruning followed by complete end-to-end tok/s selection;
- noise-aware repeated measurements;
- exact-key cache over model content/inventories, device, workload class, candidates, compiler, and runtime;
- deterministic cache invalidation and safe interruption fallback;
- rejection records for OOM prediction, incomplete coverage, correctness, resources, health, and timing noise.

Gate: the faster complete policy wins only after hard gates; stale caches and interrupted searches cannot promote an
accelerated candidate; a cache hit reproduces output and census.

Commit: search policy, then cache/invalidation.

### R10 — End-to-end matrix and closeout

Required matrix:

- both retained GGUF fixtures and renamed copies;
- multiple live/synthetic memory budgets around each boundary;
- context lengths 512, 1024, 2048, and 4096, plus one boundary/tail case;
- cold scan, exact-cache reload, and stale-cache rejection;
- full overlay, bounded/hybrid, direct packed, and refusal outcomes;
- Q4_K, Q6_K, mixed role policy, and LM-head semantics;
- prefill correctness and tok/s, then fixed-depth decode correctness and tok/s;
- pinned-clock performance evidence and unpinned diagnostic repeats;
- planned versus measured peak and pre/post GPU health.

Gate: all completion criteria below pass and obsolete selectors/contracts are removed.

Commits: validation artifacts, documentation, then dead-code cleanup.

## Parallel workstreams

Parallel work is safe only with disjoint ownership:

1. contracts and Boolean-removal audit;
2. packed-source to execution-inventory projection;
3. full-overlay artifact requalification;
4. planner catalog and memory ledger;
5. Q4 bounded kernel and ownership;
6. Q6 candidate or explicit fallback;
7. search/cache engine;
8. independent correctness, memory, route-census, and benchmark validation.

`model.py`, `prefill_policy.py`, and the final runtime binder remain integration-owner files. Workers should not make
overlapping edits there. Each commit must be independently testable, preserve the direct-packed rollback, and avoid
mixing generated evidence, runtime behavior, and unrelated cleanup.

## Definition of complete

- [ ] The user selects the model; GPU capabilities, VRAM, and allocation facts are scanned automatically.
- [ ] No production flag, model name, profile, parameter count, or VRAM tier selects a prefill route.
- [ ] One immutable plan owns strategy, exact bindings, memory admission, and selection evidence.
- [ ] Every packed source tensor maps to explicit phase-correct execution invocations.
- [ ] Full-overlay Graph GEMM is a normal candidate with current exact identities and complete memory facts.
- [ ] The fitting fixture automatically recovers its promoted generated prefill route and performance band.
- [ ] The non-fitting fixture rejects full overlay before allocation and never OOMs from route selection.
- [ ] A bounded packed policy has complete Q4/Q6 coverage or exact declared fallbacks and no hidden full dequantization.
- [ ] Bounded promotion requires a statistically credible whole-prefill tok/s win over direct packed.
- [ ] Planned and measured allocation peaks reconcile within the declared ownership contract.
- [ ] Cache reload reproduces route census and outputs; stale/partial cache fails closed.
- [ ] Decode remains packed, correct, and within the declared non-regression threshold.
- [ ] Benchmarks attribute observed bindings rather than requested environment.
- [ ] The boundary/correctness/resource/health/performance matrix passes.
- [ ] The selector audit reports zero production authority leaks.

Until every item passes, direct packed remains the safe baseline for any inventory lacking a complete faster admitted
policy. The planner architecture is validated; the optimized non-fitting policy is not yet complete.
