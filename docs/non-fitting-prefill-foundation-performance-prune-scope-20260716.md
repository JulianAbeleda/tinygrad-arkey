# Non-fitting prefill: foundation, llama-surpass, autoscan, and prune scope

Date: 2026-07-16  
Status: implementation scope; no route promotion authorized  
Applies to: prefill policies that cannot retain a complete dense FP16 weight overlay inside the admitted device-memory budget

## Decision and ordering authority

This document is the ordering authority for the non-fitting prefill continuation. It narrows earlier scopes where they
would otherwise connect automatic policy search before the bounded route has demonstrated a useful end-to-end result.

The required order is:

```text
reuse and ownership consolidation
  -> canonical bridge repair
  -> bounded-route foundation
  -> complete manually selected policy
  -> correctness/resource/memory proof
  -> reachability-backed repository prune
  -> whole-prefill tok/s optimization
  -> statistically credible llama.cpp surpass
  -> autoscan integration
  -> final maintainability and regression closeout
```

Autoscan is explicitly deferred. The user continues to select the model, and the non-fitting route remains an explicit
candidate/research policy until the complete route surpasses llama.cpp under the benchmark contract in this document.
Before that gate, automatic production selection must retain the proven safe baseline.

Repository pruning is also ordered. Small consolidation repairs needed to establish a single foundation happen first.
Broad deletion and organizational cleanup begin immediately after the retained route, evidence, and rollback boundaries
are frozen, before the performance loop adds more variants. This avoids preserving the wrong abstraction, deleting
evidence still needed to diagnose the route, or optimizing on top of transitional scaffolding.

## Objectives

1. Reuse the repository's existing candidate, inventory, execution, evidence, memory, and policy infrastructure.
2. Establish one model-independent bounded packed-tile architecture for any selected model whose dense overlay does not
   fit; Qwen3-14B on the current GPU is the proving fixture, not a routing condition.
3. Produce a complete mixed-quant policy: cooperative bounded Q4_K where proven, an independently proven Q6_K route or
   exact direct-packed fallback, and the fixed final-token LM head behavior.
4. Optimize measured synchronized whole-prefill token throughput until the complete policy credibly surpasses the
   matched llama.cpp policy.
5. Only after that result, connect live GPU/model scanning, candidate measurement, caching, and automatic selection.
6. Once the foundation and promotion artifacts are frozen, remove dead, superseded, duplicate, and script-bound code so
   the retained path has clear ownership and is maintainable.

## Explicit non-goals

- Do not change the fitting/full-overlay architecture while completing this path.
- Do not optimize decode as part of this scope. Decode is a correctness and non-regression gate only.
- Do not make `8B`, `14B`, a model/profile name, a filename, a target string alone, or a named VRAM tier a route rule.
- Do not introduce another candidate schema, identity function, inventory authority, registry, benchmark harness,
  guarded executor, evidence format, or whole-model policy type.
- Do not enable autoscan because a kernel microbenchmark, one role, or one prompt length wins.
- Do not treat a modeled Boltbeam ceiling as measured throughput.
- Do not perform broad file splitting merely to reduce the largest-file count.
- Do not combine Q4_K and Q6_K physical decoding into a false common grammar.
- Do not restore the removed broad `harness_contract` or create a parallel throughput methodology.
- Do not delete a research artifact that is still the only authority for a negative result, ABI fact, or oracle mapping;
  preserve the evidence first, then delete executable scaffolding if it is no longer required.

## Current state

### Proven substrate

- The direct-packed non-fitting route works end to end, is memory safe, and remains the rollback/default baseline.
- The selected-model inventory, live device scan, memory ledger, and immutable selected-plan concepts exist.
- Full-kernel candidate payloads, semantic candidate identity, target capability, schedule admission, and packed operand
  transforms exist in `extra/qk/runtime_specs.py`.
- Exact production packed-source inventory exists in `tinygrad/llm/model.py`.
- Workload projection and candidate inventory infrastructure exists in `extra/qk/prefill/workload_inventory.py`.
- Execution, correctness, guard, timing, artifact, and dispatch contracts exist under `extra/qk/prefill/`.
- Isolated guarded execution, GPU health checking, final-binary identity, ISA/resource evidence, owner coverage, and
  staging evidence exist.
- Whole-model memory-adaptive candidate and policy composition exists in
  `extra/qk/memory_adaptive_candidate_catalog.py` and its controller.
- The Q4_K/Q8_1 numeric oracle, logical MMQ vocabulary, llama ownership oracle, and bounded cooperative proof kernels
  exist.
- Independent Q6_K vocabulary and a direct-packed fallback exist.

### Known foundation defect

`extra/qk/prefill/current_prefill_execution_adapter.py` currently expects exactly one final `PROGRAM` whose
`candidate_context.canonical_identity` matches the admitted candidate. The final program loses that identity in the
current compile path, producing zero identity-bound programs. Two related AMD compile fixtures also have recorded
resource/VGPR failures. These are foundation failures, not reasons to create a second adapter.

The canonical logical vocabulary, Q6 grammar, Q4/Q8 oracle, and memory-adaptive catalog/controller currently pass their
focused host tests. The final-program bridge must be repaired before a new bounded candidate is integrated.

### Maintainability baseline

A broad filename-based inventory of Python files under `extra/qk` containing `mmq`, `q4k`, `q6k`, `prefill`, or
`memory_adaptive` currently contains 149 files and 25,916 physical lines. It includes 53 executable CLI/main modules.
The number is a navigation baseline, not a deletion quota: the final audit must classify actual reachability and
ownership rather than deleting by filename.

The clearest current ownership problems are:

- `mmq_q4k_q8_atom.py` combines numeric adapters, lifecycle logic, physical mapping, nine kernel families, source
  hashes, launch wrappers, and a probe in 1,006 lines.
- `mmq_bounded_harness.py` combines reusable fixtures, activation preparation, backend dispatch, timing, evidence, and
  CLI behavior in 807 lines; many callers import its private fixture functions.
- `memory_adaptive_tinygrad_seam.py` combines transport/model scanning, candidate enumeration, artifact reconciliation,
  whole-model execution, timing, and CLI behavior in 815 lines.
- `prefill_int8_wmma_spec.py` combines format decode, candidate descriptions, multiple Tensor emitters, scheduler
  integration, and owner proof.
- `mmq_machine_search.py` combines historical source anchors, candidate registry behavior, evidence aggregation,
  search reporting, Boltbeam trace construction, and CLI behavior.
- Import cycles exist between the DS4 emitter and Q4 atom, the fused Q4 module and WMMA spec, and the memory-adaptive
  controller and tinygrad seam.
- Multiple candidate identity functions, MMQ adapters, route registries, profile-derived role contracts, and historical
  invocation scripts overlap the newer canonical infrastructure.

## Canonical ownership model

### Physical format ownership

- `tinygrad/llm/qk_layout.py` and `extra/qk/layout.py` own canonical Q4_K, Q6_K, and Q8_1 physical constants.
- One Q4 grammar owns Q4 block interpretation and Q4/Q8 correction semantics.
- One independent Q6 grammar owns Q6 block interpretation.
- The numeric reference remains independent from generated kernels and runtime selection.

Q4 and Q6 may share exact outer contracts—shape, target, launch, evidence, and policy composition—but not physical
decoding details that only look similar.

### Semantic and candidate ownership

- `mmq_logical_vocabulary.py` owns model-independent logical axes, output ownership, staging, synchronization, and
  candidate semantics.
- `runtime_specs.py` is the sole authority for kernel candidate payloads, canonical identity, target capability,
  schedule, packed operand transforms, and admission.
- Format-specific operation semantics are nested candidate payload data. They do not define a competing top-level
  candidate identity.
- Memory-adaptive `CandidateSpec` owns the separate identity of a complete whole-model policy.

There must be exactly two intentional identity levels:

```text
kernel candidate identity -> exact compiled operation and schedule
whole-policy identity      -> complete invocation-to-candidate/fallback mapping plus memory plan
```

### Inventory ownership

- The selected production GGUF inventory is the only authority for source tensor identity, quant format, role, and
  dimensions.
- The grouped prefill workload inventory is a deterministic projection of that inventory and retains every parent
  source/invocation identity and exact call count.
- Model profiles may identify reproducible fixtures and artifacts only. They cannot create semantic role/shape facts
  that disagree with or replace the loaded inventory.

### Execution and evidence ownership

- `execution_bridge_contracts.py` owns execution request/result contracts.
- `guarded_execution.py` and `isolated_guarded_executor.py` own safe compile/dispatch isolation and health behavior.
- `candidate_inventory_execution.py` owns exact inventory-to-candidate joining and ordered execution.
- Compile identity, final ISA, resources, staging, and owner-coverage modules each retain their narrow evidence jobs.
- `extra/qk/bench.py` remains the only whole-model throughput entry. Prefill authority remains
  `prefill_whole_synced.py`; decode authority remains `decode_runtime_overhead.py`.
- Local kernel timing loops may remain local when their synchronization or sampling semantics differ.

### Runtime and policy ownership

- The selected execution plan is immutable and is the sole production route authority.
- Runtime linears/invocations consume exact bindings; they do not infer a route from flags, model identity, or memory
  class.
- `generated_candidates.py` is the executable candidate registry.
- `route_manifest.py` owns historical status, provenance, and promotion records only.
- The memory-adaptive catalog/controller owns complete-policy feasibility and, only after the surpass gate, automatic
  ranking and cache behavior.

## Target non-fitting architecture

The retained complete model remains in its packed Q4_K/Q6_K representation. Prompt activations use an explicit Q8_1
representation where required. Each workgroup owns a bounded output tile and iterates through bounded K epochs:

```text
packed selected model + prompt activation
  -> exact invocation binding
  -> bounded output tile (M x N)
  -> for each K epoch
       cooperative packed weight load/decode
       cooperative Q8 activation load
       bounded LDS/register stage
       uniform synchronization
       integer WMMA/grouped dot plus FP32 correction
       accumulate into the unique output owner
       reuse/overwrite stage for the next epoch
  -> final output write
```

Required invariants:

- no complete `[N,K]` FP16 weight materialization;
- no hidden whole-model dense overlay;
- bounded global workspace and bounded per-workgroup LDS/register residency;
- exact K accumulation across all epochs;
- one final owner per output, including M/N tails;
- uniform barriers and no out-of-range cooperative load;
- explicit activation preparation, cache identity, lifetime, and measured cost;
- exact candidate identity survives admission, compilation, child reconstruction, dispatch, and evidence;
- every selected invocation has one proven candidate or one explicit declared fallback;
- the mixed policy fits the admitted memory peak before allocation.

The first complete policy may use:

- bounded cooperative Q4_K/Q8_1 for the proven Q4 role/shape rows;
- existing direct-packed Q6_K for Q6 rows until an independent cooperative Q6 candidate wins;
- fixed final-token LM head semantics;
- direct-packed rollback for every accelerated binding.

This is a legitimate complete policy. Q6 cooperative implementation is not required merely for visual symmetry; it is
required only if evidence shows it is needed to surpass llama.cpp or improve the final selected policy.

## Benchmark and llama-surpass contract

### Metric authority

The project objective is synchronized whole-prefill tokens per second. Kernel TFLOP/s, role timing, ISA counts,
roofline projections, and Boltbeam traces are attribution tools only.

The tinygrad authority is:

```text
extra/qk/bench.py --prefill
  -> extra/qk/prefill_whole_synced.py
  -> synchronized production model call
```

The llama authority is the pinned local `llama-bench` wrapper in `extra/llm/llama_bench.py`, using the same GGUF and
GPU. The command must make `-ngl`, prompt length, batch size, microbatch size, repetitions, and output format explicit.
If `-n 0` is not accepted as a stable prompt-row mode by the installed binary, retain `-n 128` and select only the
independent prompt row; record that choice in the artifact.

### Required workload matrix

Primary proving fixture:

- selected model: exact Qwen3-14B Q4_K_M GGUF content identity;
- target: current AMD gfx1100 device identity and scanned resource facts;
- whole prompt lengths: 512, 1,024, 2,048, and 4,096 tokens;
- tinygrad chunk size: 512 with exact whole-length accumulation;
- start-position diagnostics: 0, 512, 1,024, 2,048, and 3,584;
- same context/KV semantics, synchronization boundary, model bytes, and clock policy;
- pinned-clock authority runs, with unpinned runs labeled diagnostic;
- llama and tinygrad run sequentially, never concurrently resident.

Before declaring surpass, collect at least three alternating sessions:

```text
session 1: llama -> tinygrad
session 2: tinygrad -> llama
session 3: llama -> tinygrad
```

Retain raw samples, medians, spread, binary/build revision, GPU identity, clocks, thermals if available, route census,
memory peak, and pre/post GPU health. A fresh matched llama artifact is mandatory. The historical frozen pp512 result
of 1,889.41 tok/s at llama.cpp commit `ac4cddeb0` remains a continuity check, not the sole promotion authority.

### Surpass definition

Autoscan remains prohibited until all hard gates pass and the complete candidate policy satisfies:

1. At each declared whole-prompt length, the median tinygrad/llama tok/s ratio is greater than 1.00.
2. The paired/session bootstrap 95% lower confidence bound for the aggregate ratio is greater than 1.00.
3. The aggregate geometric-mean ratio is at least 1.05, preserving the existing beyond-parity target and avoiding a
   noise-sized promotion.
4. No individual context is below 0.98 in any accepted authority session.
5. The pp512 result is at least the fresh llama median and remains consistent with the historical comparator after
   accounting for build/clock differences.
6. The route census proves that the intended bounded/hybrid policy—not direct fallback or an oracle wrapper—produced
   the result.
7. Correctness, memory, resource, health, and decode non-regression gates pass in the same candidate revision.

If a route beats llama at one context but not the matrix, it is useful optimization evidence but does not unlock
autoscan. If measurement noise prevents a credible conclusion, gather more sessions; do not weaken the threshold.

## Phased implementation scope

### F0 — Freeze behavior, evidence, and ownership baseline

Deliverables:

- record current commit, worktree status, device scan, exact model content/inventory identity, compiler/runtime revision,
  direct-packed route census, planned/measured memory, and prefill/decode baselines;
- preserve the historical llama comparator and collect the exact commands needed for a fresh comparator;
- capture the current focused test result, including expected final-program/VGPR failures;
- generate an import graph and classify current cycles;
- inventory all bounded-path modules, CLIs, private cross-module imports, route/registry entries, dynamic imports,
  subprocess entry points, test-only users, and artifact-only users;
- define the retained direct-packed rollback identity.

Gate: every later behavior, performance, and deletion can be compared with an immutable baseline.

Commit boundary: evidence/tests/documentation only.

### F1 — Repair the canonical final-program bridge

Deliverables:

- trace candidate context from `FullKernelCandidateSetEntry` through admission, warmstart scheduling, final `PROGRAM`,
  child reconstruction, and executable evidence;
- preserve one canonical identity without route-name reconstruction or adapter-local hashing;
- resolve the recorded resource/VGPR failures through the owning lowering/resource contract or retain exact fail-closed
  diagnostics where a candidate is genuinely inadmissible;
- add negative tests for stale identity, multiple programs, lost context, child binary mismatch, and target mismatch.

Gate:

- exactly one final program is identity-bound for every admitted canary;
- compile evidence and executed binary use that identity;
- all canonical bridge tests pass;
- no second adapter or compatibility identity authority is introduced.

Commit boundary: bridge repair and focused tests only.

### F2 — Consolidate inventory and identity authority

Deliverables:

- make selected-model invocation inventory the source authority;
- make grouped workload inventory a lossless projection that retains parent invocation/source identities;
- represent Q4_K, Q6_K, direct packed, bounded Q4/Q8, optional bounded Q6, and fixed LM-head execution semantics;
- migrate retained candidates to `runtime_specs` canonical identity;
- retain one whole-policy identity in the memory-adaptive catalog;
- classify old MMQ identity helpers and adapters as compatibility-only, migrate callers, then mark for deletion;
- prohibit model/profile strings in semantic eligibility and emitter code.

Gate:

- every source tensor maps uniquely to phase-correct execution rows;
- candidate and policy identities are deterministic and profile-name independent;
- rename/copy of the same model content does not alter eligibility;
- deliberate missing, duplicate, stale, wrong-quant, or wrong-shape rows fail closed.

Commit boundaries:

1. inventory projection and tests;
2. kernel identity migration;
3. whole-policy identity migration;
4. compatibility diagnostics.

### F3 — Decouple reusable logic before extending the kernel

Deliverables:

- move Q4/Q8 fixture generation and packed operand construction out of `mmq_bounded_harness.py` into a format-owned,
  import-safe fixture/operand module;
- turn the bounded harness into backend registration, execution orchestration, and thin CLI serialization;
- separate `mmq_q4k_q8_atom.py` into explicit owners for physical/operand adaptation, retained kernel emission, launch
  wrappers, and probes; retain refuted kernel families only when an evidence or regression user exists;
- separate semantic WMMA specification from Tensor emission and scheduler/owner proof;
- separate model scan/transport, artifact reconciliation, whole-model worker, timing worker, and CLI responsibilities in
  the memory-adaptive seam;
- separate static historical/oracle data from active candidate search and reporting;
- break the three known import cycles with one-way descriptor -> emitter -> executor dependencies;
- replace cross-module private imports with narrow public contracts only where the exact semantics are shared.

Gate:

- zero import cycles in the retained bounded path;
- zero cross-module imports of private symbols among retained production/reusable modules;
- CLI modules do not own reusable fixtures, kernel semantics, candidate identity, evidence extraction, or policy logic;
- focused behavior and emitted-source/binary fixtures remain unchanged unless a deliberate repair commit says otherwise.

Commit rule: one ownership boundary per commit. Mechanical moves, behavior changes, evidence updates, and deletions are
never mixed in one commit.

### F4 — Bind the logical bounded Q4 route to canonical infrastructure

Deliverables:

- express Q4_K/Q8_1 operation semantics through the logical vocabulary;
- lower exact M/N/K tile geometry, workgroup/wave mapping, stage storage, K epochs, barriers, correction algebra, output
  ownership, and tails from candidate data;
- package the emitted operation as one `runtime_specs` full-kernel candidate;
- use existing admission, guarded execution, correctness, health, final ISA/resource, staging, and owner-coverage paths;
- eliminate the `qwen3-14b` profile string from DS4 candidate construction and any exact role-shape constants from
  semantic modules;
- retain the numeric oracle as an independent comparator and llama source as an oracle only, never an executable
  production backend.

Progression:

1. smallest bounded tile full-output correctness;
2. target workgroup mapping and resource legality;
3. multi-epoch K accumulation;
4. full 128x128-style cooperative ownership where target evidence supports it;
5. M/N edge tiles and K boundary behavior;
6. exact 14B Q4 role/shape inventory;
7. same machinery on a non-default supported shape.

Gate: every Q4 invocation selected by the policy has exact correctness, immutable-input/output guards, unique owner
coverage, uniform synchronization, no hidden dense materialization, final-binary identity, resource evidence, and a
healthy real-GPU dispatch.

Commit boundaries: vocabulary/descriptor, lowering, K lifecycle, ownership/tails, then evidence. Do not land one giant
kernel-and-integration commit.

### F5 — Complete mixed-quant policy coverage

Deliverables:

- derive exact Q4_K/Q6_K role coverage from production inventory;
- independently validate Q6 direct-packed fallback under the same execution/policy contracts;
- compare a cooperative Q6 candidate only if it has a concrete performance rationale;
- preserve fixed final-token LM-head behavior and exact call count;
- define one complete invocation-to-route map with explicit rollback for every accelerated binding;
- compute complete predicted peak residency including activation packing/cache, workspace, KV/runtime, compiler/runtime
  reserve, and packed backing allocation.

Gate:

- every controlled invocation has exactly one binding or declared fallback;
- no Q4 evidence is reused as Q6 proof;
- no selected route allocates a full dense overlay;
- the complete policy is admitted before allocation and measured peak reconciles with the declared ownership ledger.

Commit boundaries: Q6/fallback qualification, LM-head/call semantics, policy composition, then memory reconciliation.

### F6 — Manual end-to-end integration and foundation proof

The complete candidate remains explicitly selected for validation. Autoscan and automatic production promotion remain
disabled.

Deliverables:

- attach exact route bindings through the existing selected-plan/runtime seam;
- run mixed-route full-output/model-level correctness and route census;
- run prompt lengths and tail cases across the required matrix;
- prove direct-packed rollback is one policy change and still works;
- run decode correctness and fixed-depth throughput non-regression;
- reconcile role timings with synchronized whole-prefill wall without claiming projections as measurements;
- freeze one foundation artifact containing inventory, policy, candidate, binary, memory, correctness, resource, census,
  and benchmark identities.

Foundation gate:

- the route is complete, correct, memory safe, deterministic, independently evidenced, and manually executable end to
  end;
- no route flag/model name/VRAM tier selects semantic behavior;
- direct packed remains the ordinary automatic default;
- failure or interruption returns to direct packed without a partial accelerated policy.

This gate proves the architecture. It does not authorize autoscan and does not require a llama win yet.

### F7 — Post-foundation repository prune

This phase begins immediately after the F6 foundation artifact is frozen. Safe mechanical decoupling from F3 is not
delayed, but broad deletion waits for this point. Active candidate variants needed by the upcoming performance loop are
classified and retained explicitly; dead, duplicate, superseded, and already-refuted scaffolding does not survive merely
because performance work is still ahead.

#### Reachability audit

For every candidate module/script, record:

```text
classification: production | reusable library | test fixture | active research | historical evidence | dead
importers and dynamic importers
CLI/subprocess callers
registry and route-manifest references
artifact/schema readers and writers
tests that prove unique behavior
replacement owner, if any
disposition: retain | decouple | merge | preserve-as-data | delete
```

Documentation mention alone does not make executable code live. Conversely, zero static imports does not prove death
when a module is reached by dynamic import, subprocess path, registry string, or artifact migration.

#### Initial consolidation candidates

Retain and centralize:

- physical layout constants;
- Q4/Q8 and Q6 format grammars as separate owners;
- canonical numeric references;
- logical MMQ vocabulary;
- `runtime_specs` candidate/admission contracts;
- production inventory and grouped projection;
- execution bridge, guarded executor, evidence, memory ledger, and policy catalog;
- canonical whole-model benchmark authorities.

Decouple:

- fixtures and operand packing from bounded harnesses;
- retained emitted kernels from launch/probe wrappers;
- semantic specs from Tensor/scheduler emitters;
- model/device transport from whole-model workers and CLIs;
- active search from historical source anchors and report rendering;
- production runtime API from research-only route wrappers.

Likely retire after verified migration:

- the unused `generated_route_registry.py` scaffold in favor of `generated_candidates.py`;
- the old fail-closed `mmq_atom_boundary.py` stub;
- transitional `mmq_role_adapter.py` and overlapping candidate adapters/identity helpers;
- static profile-derived Q4 role contracts superseded by exact production inventory;
- historical `mmq_invocation_v1` through `v7` executable scaffolding after their unique findings are preserved as data
  or concise documentation;
- refuted atom variants with no regression/evidence consumer;
- research wrappers exported through `tinygrad/llm/route_ops.py` that have no production owner;
- obsolete flags, compatibility aliases, and artifact migrations after their supported migration window closes.

Each item is a candidate, not a preapproved deletion. The reachability record and replacement tests decide.

#### Deletion proof

A module or symbol may be deleted only when:

1. production, dynamic, subprocess, registry, manifest, and test reachability are audited;
2. unique semantic behavior is absent or covered by the retained owner;
3. required negative/performance/oracle evidence is preserved independently of the executable scaffolding;
4. artifact compatibility is migrated or deliberately version-rejected with a clear diagnostic;
5. focused and broad relevant tests pass before and after deletion;
6. route census, output, memory plan, and benchmark authority are unchanged unless the deletion intentionally removes a
   refuted route;
7. `git grep`/`rg`, import probes, and compileall show no dangling references.

#### Maintainability gates

- zero retained bounded-path import cycles;
- zero cross-module imports of private symbols in production/reusable bounded modules;
- one kernel candidate identity authority and one whole-policy identity authority;
- one executable generated-candidate registry;
- one production source-inventory authority and one lossless grouped projection;
- no model/profile/size/VRAM-tier constants in semantic eligibility or emitters;
- no reusable kernel, fixture, evidence, or policy logic owned solely by a CLI script;
- every retained module has one stated responsibility and an independently testable boundary;
- every retained file above 500 lines receives an explicit cohesion review; size alone does not force a split;
- the broad bounded-path Python file/LOC inventory decreases materially from the frozen baseline, with deletions and
  deduplication reported separately from file moves;
- no authored LOC is hidden in generated markers or moved outside accounting to claim reduction;
- active route/search registries contain no refuted, stub, duplicate, or unreachable candidates;
- compatibility aliases have owners and expiry conditions; indefinite aliases fail closeout;
- `python3 sz.py` retains useful headroom below its repository budget;
- core tinygrad modules do not eagerly import research-only `extra.qk` code.

Commit sequence:

1. reachability/classification artifact;
2. public-contract migrations;
3. cycle removal;
4. dead executable/script deletion in small independently tested batches;
5. registry/manifest cleanup;
6. docs/artifact archival cleanup;
7. final LOC/import/reachability report.

Gate: the retained foundation and rollback pass unchanged on a materially smaller, acyclic, single-owner active surface.
The performance phase starts from this pruned baseline.

### F8 — Performance attribution and llama-surpass loop

Deliverables:

- collect same-session per-role timing including activation preparation/cache miss/cache hit and launch overhead;
- use Boltbeam traces and roofline analysis to classify global bytes, LDS traffic, conversion, WMMA utilization,
  synchronization, occupancy/resource cliffs, writeback, launches, and non-GEMM residual;
- compare executed ISA/resource artifacts with the llama oracle structure without assuming identical backend lowering;
- rank controlled candidate changes through hard correctness/resource gates before timing;
- remeasure the complete policy after every accepted role-level improvement;
- run the fresh alternating llama/tinygrad protocol when the complete route approaches parity;
- continue until the surpass definition is met or a measured architectural ceiling is documented.

Optimization order:

1. remove repeated packed representation work;
2. increase cooperative weight/activation reuse;
3. reduce scalar unpack/conversion and register lifetime;
4. escape occupancy cliffs and spills;
5. reduce synchronization and launch count without weakening ownership;
6. optimize Q6 only if its measured policy share blocks the target;
7. optimize residual non-GEMM work only after dominant role targets are achieved.

Stop rules apply to individual variants, not the project. Refuted variants retain compact evidence and are removed from
active registries/search spaces.

Gate: all benchmark and surpass requirements in this document pass on one immutable candidate revision.

Commit rule: each accepted optimization is its own correctness-tested commit; generated evidence follows in a separate
commit when large or run-volatile.

### F9 — Conditional autoscan and machine selection

This phase is unauthorized until F8 passes. If F8 does not pass, F9 is not implemented.

Deliverables after authorization:

- scan selected model content/inventory, current GPU capabilities, live memory, allocator granularity, and requested
  context/workload;
- enumerate only complete, correctness/resource-qualified policies;
- reject dense overlay before allocation when complete peak cannot fit;
- retain direct packed as baseline and measure feasible complete alternatives;
- select by synchronized end-to-end tok/s, not kernel timing;
- cache by model content/inventory, device/resource facts, workload, candidate identities, compiler/runtime revision,
  and benchmark protocol;
- reject stale, partial, interrupted, unhealthy, or mismatched cache records;
- bind exactly the selected immutable plan without reinterpreting environment flags;
- expose diagnostics that describe observed selection but cannot alter it.

Gate:

- cold scan selects the proven faster policy;
- exact cache reload reproduces output and route census;
- stale cache fails closed;
- synthetic memory boundaries choose full overlay, bounded policy, direct packed, or refusal from facts only;
- removing/renaming a model profile does not alter selection;
- no production route-selection flag remains.

Commit boundaries: scanner facts, candidate enumeration, measurement/ranking, cache/invalidation, runtime binding, then
selector cleanup.

### F10 — Final validation and closeout

Required matrix:

- exact 14B proving fixture and renamed-copy identity test;
- at least one additional supported inventory/shape without a model-name source change;
- context/prompt lengths 512, 1,024, 2,048, and 4,096 plus M/N/K tail cases;
- direct packed, bounded/hybrid, and refusal outcomes under synthetic memory budgets;
- Q4_K, Q6_K, mixed policy, and fixed LM-head semantics;
- cold explicit candidate execution and rollback;
- if F9 is authorized: cold autoscan, cache hit, stale cache, interrupted search, and health failure;
- full-output/model correctness, immutable inputs, output guards, GPU health, exact route census, final binary/resources,
  planned/measured memory, and synchronized prefill tok/s;
- fixed-depth decode correctness and declared throughput non-regression;
- pinned-clock authority and labeled unpinned diagnostics;
- fresh llama comparison and raw sample retention;
- import graph, private-import audit, dynamic reachability, compileall, focused tests, broad relevant tests, and `sz.py`.

Closeout artifacts:

- architecture and ownership map;
- canonical inventory and complete selected-policy artifact;
- correctness/compile/resource/staging/owner/memory evidence bundle;
- raw tinygrad and llama benchmark sessions plus statistical decision;
- autoscan authorization decision (`authorized` or `not authorized`);
- route census and rollback proof;
- prune classification/deletion ledger;
- before/after file, LOC, import-cycle, private-import, registry, and test report;
- lessons ledger for refuted routes and deleted historical scaffolding.

## Parallel work boundaries

Parallel work is allowed only after F0 and with disjoint ownership:

1. final-program identity bridge;
2. production-to-execution inventory projection;
3. Q4 logical descriptor/physical grammar;
4. Q4 lowering and owner/tail evidence;
5. Q6 fallback qualification;
6. memory/policy composition;
7. benchmark/comparator artifact collection;
8. read-only reachability and prune classification.

Integration-owner files—especially `tinygrad/llm/model.py`, runtime plan/binding code, `runtime_specs.py`, and the final
candidate registry—must have one active owner at a time. Parallel workers may propose patches or evidence but must not
land overlapping integration edits.

The post-proof prune can parallelize only by disjoint module families after the reachability ledger is frozen. No worker
deletes a shared contract or registry entry while another worker is migrating its callers.

## Commit principles

- Every commit is independently testable and has one semantic purpose.
- Contract introduction, caller migration, compatibility removal, and deletion are separate commits.
- Mechanical moves do not include behavior changes.
- Kernel correctness, performance optimization, production binding, benchmark artifacts, and cleanup do not share one
  giant commit.
- Generated/run-volatile evidence is separated from source changes unless a tiny deterministic fixture is required by
  the source test.
- Direct-packed rollback remains usable at every commit.
- A failed optimization is reverted or quarantined with evidence; it is not left on the active path.
- Push only complete tested commits; do not accumulate the full program into one final commit.

## Global stop and rollback rules

- Any correctness, output ownership, barrier-uniformity, packed ABI, binary identity, resource, memory, timeout, or GPU
  health failure blocks that candidate before timing.
- Predicted OOM blocks allocation; no experimental route may probe OOM as selection logic.
- If the complete candidate does not beat direct packed, it remains research-only regardless of kernel microbenchmarks.
- If the complete candidate does not satisfy the llama-surpass contract, autoscan remains disabled.
- If autoscan fails or has stale/incomplete evidence, direct packed is selected.
- If pruning changes output, route census, memory ownership, benchmark methodology, or performance outside tolerance,
  restore the last retained owner and split the cleanup into a smaller slice.
- Difficulty, compiler limitations, or noisy counters do not justify weakening hard correctness or promotion gates.

## Definition of complete

### Foundation complete

- [ ] Canonical final-program identity survives compile and execution.
- [ ] Production inventory is the sole model-fact authority and grouped inventory is lossless.
- [ ] Exactly one kernel identity and one whole-policy identity remain.
- [ ] Reusable logic has explicit owners and retained bounded-path import cycles/private imports are removed.
- [ ] Complete Q4/Q6/LM-head invocation coverage exists with exact fallbacks.
- [ ] No selected route performs hidden full dequantization or exceeds admitted memory.
- [ ] Correctness, ownership, synchronization, binary, resource, staging, health, memory, and route-census gates pass.
- [ ] The complete policy executes manually end to end with direct-packed rollback.

### Performance and autoscan complete

- [ ] Fresh matched llama artifacts exist for the declared context matrix.
- [ ] The complete policy satisfies the statistical llama-surpass definition.
- [ ] Decode remains correct and within its declared non-regression band.
- [ ] Only then, autoscan is authorized and implemented.
- [ ] Cold scan, exact cache hit, stale cache, interrupted search, and fallback behavior pass.
- [ ] User model selection remains explicit; hardware/memory/workload facts select only among proven policies.

### Prune and maintainability complete

- [ ] Every bounded-path module/script is classified by reachability and ownership.
- [ ] Superseded registries, adapters, stubs, flags, aliases, and historical executable scaffolding are migrated or
      deleted with evidence preserved.
- [ ] Active registries contain only reachable, qualified candidates.
- [ ] No production/reusable cross-module private imports or bounded-path cycles remain.
- [ ] CLI scripts are thin and reusable functions have library owners.
- [ ] Before/after LOC and module counts show real deletion/deduplication rather than relocation.
- [ ] Focused tests, broad relevant tests, compileall, import probes, route census, benchmark authority, and `sz.py` pass.
- [ ] The final architecture, benchmark decision, rollback, and prune ledger are documented.

Until the foundation gate passes, the bounded route is a research candidate. Until the llama-surpass gate passes,
autoscan is not authorized. Until the retained route and evidence are frozen, broad repository pruning is not
authorized. Direct packed remains the safe non-fitting baseline throughout.
