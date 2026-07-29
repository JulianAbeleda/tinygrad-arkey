# Route authority centralization and reuse scope

Date: 2026-07-29

Branches: `exp` is the research, search, benchmark, qualification, and historical-evidence owner. `master` is the lean production runtime and promoted immutable-asset owner. Master must never import EXP.

## Outcome

Inference selects and executes each promoted route from one production-owned descriptor or immutable artifact. EXP executes those same production implementations for qualification. Search provenance, campaign evidence, rejected candidates, and explicit parity oracles remain in EXP and cannot dispatch in production.

This is a behavior-preserving authority cleanup. It does not create a new kernel, rerun machine search, change selected geometries, or require an eGPU until final recertification.

## Non-negotiable invariants

1. Master imports no `extra.llm_research` module.
2. A selected route's identity, exact applicability, and selected configuration have one production authority.
3. Aggregate catalogs reference route descriptors; they do not restate configuration or coverage values.
4. EXP qualification imports production executors. A retained EXP implementation is named and documented as an oracle and cannot be selected by production.
5. Unsupported shapes fail closed to the current generic tinygrad behavior. Existing fail-loud paths remain fail-loud.
6. Canonical candidate identity and candidate-set identity are implemented once.
7. Packed-WMMA coverage and compatibility geometry views are derived from `PACKED_WMMA_ROUTES`.
8. Role families use one normalization authority. Runtime linears prefer the role resolved from model facts and only use a compatibility fallback where unavoidable.
9. Dispatcher dependencies are one-way; route implementations do not import private dispatcher helpers.
10. Search provenance is not inferred from runtime structure. Historical or blocked provenance remains honest and distinct from selected runtime configuration.
11. No hand-authored fallback or oracle is promoted onto master by this cleanup.
12. No performance claim changes without new AMD evidence.

## Authority model

There are four distinct responsibilities. They must not be represented by competing selectors.

- Production selection authority: master descriptor, exact route key, and actual dispatcher.
- Production execution authority: master executor/emitter invoked by inference.
- Qualification authority: EXP runner invoking the production executor and checking correctness, identity, attribution, and performance.
- Provenance authority: EXP promotion bundle/evidence record describing how a selected immutable result was obtained.

The EXP route manifest is an evidence index. It is not a production selector. A legacy EXP implementation is an oracle. It is not a production executor.

## Work graph

### R0 — Characterization boundary

Owner files: master tests only.

Capture current behavior before rewiring:

- Q4_K and Q6_K decode candidate binding for supported and unsupported structural shapes.
- G4 and G5 flash-decode selected configuration and rejection behavior.
- Six packed-WMMA exact keys, geometries, identities, and fail-closed behavior.
- Former `direct_packed` compatibility mode mapping to ordinary fp16 graph behavior.
- Promoted prefill candidate-set identity and exact admissions.
- Production codegen/UOp identity where existing frozen hashes are available.

Acceptance:

- Tests call the production selectors, not a second policy table.
- No GPU execution is required.
- Frozen values are only used for behavioral characterization, not introduced as another runtime authority.

Depends on: nothing. Blocks R1-R6.

### R1 — Delete the unused global shadow selector

Owner files:

- `tinygrad/llm/production_route_policy.py`
- `tinygrad/llm/production_route_interface.py`
- `test/unit/test_production_route_interface.py`
- references discovered by repository-wide search

Action:

- Delete the two unused modules and their self-consistency test.
- Replace any useful trace expectation with tests against actual dispatch/observation.
- Do not replace them with a new global selector.

Acceptance:

- No production or test import remains.
- Repository search finds no stale path reference.
- Characterization tests still pass.

Depends on: R0.

### R2 — Shared route facts and role vocabulary

Owner files:

- `tinygrad/llm/model_facts.py`
- `tinygrad/llm/model_route_plan.py`
- `tinygrad/llm/prefill_routes.py`
- `tinygrad/llm/model.py`
- focused unit tests

Action:

- Make role-family normalization a public master-owned function beside the role vocabulary.
- Reuse it in model route planning and compatibility name resolution.
- Preserve metadata/shape validation in `QwenDenseRoleResolver`; normalization must not weaken it.
- Move neutral quant/role/workload-key helpers out of the dispatcher so packed-WMMA does not import private dispatcher helpers.
- Break `prefill_routes <-> packed_wmma_prefill`.

Acceptance:

- One role-family mapping table/function.
- No import cycle between dispatcher and packed route.
- Exact six packed-WMMA admissions unchanged.
- Existing model facts and route-plan tests pass.

Depends on: R0. Can run parallel with R1 and R3 if file ownership does not overlap R3.

### R3 — Route-local selected configuration reuse

Owner files:

- `tinygrad/llm/flash_decode_attention.py`
- `tinygrad/llm/decode_routes.py`
- `tinygrad/llm/packed_wmma_prefill.py`
- focused unit tests

Action:

- Use `FlashDecodeRouteConfig` as the selected G4/G5 descriptor consumed by decode binding; remove the parallel `_FlashDecodeCandidate` configuration copy.
- Keep candidate binding behavior and public compatibility aliases required by `model.py`.
- Derive `PACKED_WMMA_GEOM` from `PACKED_WMMA_ROUTES`, or remove the compatibility view if there are no callers.
- Ensure all packed exact-key lookups derive from the route rows.

Acceptance:

- G4/G5 route IDs and all selected parameters are defined once.
- Packed coverage and geometry equality tests pass.
- Decode and packed executor identities/codegen remain unchanged.
- Unsupported behavior remains identical.

Depends on: R0. Coordinate `packed_wmma_prefill.py` ownership with R2.

### R4 — Candidate artifact schema and identity

Owner files:

- `tinygrad/llm/prefill_candidate_runtime.py`
- `tinygrad/llm/prefill_graph_gemm.py`
- master candidate artifact tests
- EXP `runtime_specs.py` and route-manifest identity helpers only after master API is stable

Action:

- Make the master artifact module the sole canonical semantic JSON and candidate-set identity implementation.
- Make the production executor call that public implementation.
- Add a neutral decode/validation API usable from EXP.
- In EXP, import master identity primitives. Retain additional research-only schema validation only where it adds information.
- Delete exact duplicate identity implementations.

Acceptance:

- Promoted identity remains `candidate_set:sha256:2783d3ebb084e465d733cc161aa31485ac3f8ec45ff5c3aa4c1790795e852847`.
- One master implementation produces that identity.
- EXP manifest/promotion validation imports it rather than reimplementing it.
- Exact admission behavior is unchanged.

Depends on: R0. Can run parallel with R1-R3.

### R5 — Production-owned observation

Owner files:

- `tinygrad/llm/prefill_route_observer.py`
- `tinygrad/llm/prefill_graph_gemm.py`
- `tinygrad/llm/fused_attention.py`
- `tinygrad/llm/bench.py` and `tinygrad/llm/cli.py` only where needed
- EXP benchmark census callers

Action:

- Define one production-owned route-event/census boundary.
- Preserve route-specific evidence as event fields instead of independent ContextVars.
- Make EXP whole-model qualification use the production observer.
- Remove EXP's duplicate graph census after parity tests.

Acceptance:

- Selected/executed/fallback attribution comes from actual runtime execution.
- EXP census sees production graph-prefill and attention events.
- Caller-supplied route labels cannot be mistaken for proven execution.

Depends on: R4 for graph candidate identity. Can follow R2/R3.

### R6 — EXP qualification rewiring and oracle isolation

Owner files: EXP only.

Families:

- Packed-WMMA canary/evidence and promotion gate.
- Flash decode timing, numerics, lowering, and fingerprint tools.
- Q4/Q6 decode execution adapters and codegen checks.
- Graph-prefill whole-model benchmark and pure-search guard.

Action:

- Run master production owners in every qualification path.
- Use master `set_packed_wmma_canary_verifier` rather than gating the old EXP implementation.
- Retain old EXP implementations only when an explicit parity oracle is still needed.
- Rename or document retained modules as `*_oracle`; ensure no production import or default benchmark path reaches them.
- Delete them once production-vs-oracle parity is covered by immutable hashes/fixtures.

Acceptance:

- Static import audit shows default EXP qualification paths import master production owners.
- Any legacy owner has an explicit oracle-only caller list.
- Production and oracle hashes/parity remain equal where applicable.
- No EXP runner records a route it did not observe executing.

Depends on: stable R3-R5 interfaces.

### R7 — Manifest and promotion-bundle boundary

Owner files: EXP route manifest, generated request/provenance assets, promotion validation, and docs.

Action:

- Reduce the route manifest to evidence/provenance/rollback/search-status facts.
- Generate or validate runtime identity, exact applicability, and selected configuration against master descriptors or an immutable promotion bundle.
- Define one promotion bundle containing route identity, target, exact workload key/config, artifact identity, and evidence references.
- Do not add an empty generated catalog to master.
- Preserve `BLOCKED`/`UNPROVEN` records and historical-selected-plan wording.

Acceptance:

- Stale packed-WMMA attribution is corrected.
- Manifest validation fails when evidence references a nonexistent or mismatched production route.
- No manifest helper can independently select a production route.
- Search provenance vocabulary remains unchanged unless backed by evidence.

Depends on: R3, R4, R6.

### R8 — Memory decision consolidation

Owner files:

- `tinygrad/llm/admission.py`
- `tinygrad/llm/memory_ledger.py`
- `tinygrad/llm/prefill_memory_plan.py`
- associated tests

Action:

- Keep semantic memory annotations, physical allocation ownership, and exact allocation ledgers distinct.
- Make research/context strategy projections views over the exact production budget and allocation facts.
- Remove parallel unknown-budget and route-feasibility decisions from the compatibility planner.
- Preserve the public admission result and all fail-closed behavior.

Acceptance:

- One exact `MemoryBudget`/allocation decision authority.
- Compatibility strategy output is derived and cannot override exact admission.
- Unknown memory remains fail-closed.
- Context-capacity tests and exact-ledger tests pass.

Depends on: none logically, but execute after R1-R7 to keep review and rollback isolated. This is a separate commit series.

### R9 — Prune and repository-wide closure audit

Action:

- Delete obsolete modules, tests, compatibility maps, and stale path references made unnecessary by R1-R8.
- Remove generated `__pycache__` only if tracked; do not modify unrelated user files.
- Run AST/import/table audits for duplicate identities, route tables, role maps, censuses, and policy selectors.
- Verify master surface remains lean and EXP retains research evidence.

Acceptance:

- Master contains no EXP import.
- No dead shadow selector remains.
- No duplicate packed coverage/geometry table remains.
- No duplicate candidate identity implementation remains.
- No duplicated graph census remains on default qualification paths.
- No stale attribution names deleted files as production owners.
- Worktrees are clean after ordered commits.

Depends on: R1-R8.

## Parallel execution waves

- Wave A: R0 only. Establish the safety boundary.
- Wave B: R1, R2/R3 coordinated, and R4 in parallel with disjoint file ownership.
- Wave C: R5 and family-specific portions of R6 after the corresponding production API is stable.
- Wave D: R7.
- Wave E: R8 as an isolated series.
- Wave F: R9, full CPU validation, then AMD recertification request.

Agents share the filesystem. Two agents must not edit the same file concurrently. The orchestrator owns integration, cross-worktree comparisons, plan status, commits, and pushes.

## Validation ladder

1. Syntax/import validation and focused unit tests.
2. Production selector characterization.
3. Candidate identity and artifact validation.
4. Static import-direction audit.
5. Codegen/UOp hash parity where GPU-free.
6. Full available master CPU suite and EXP qualification-unit suite.
7. Clean-clone README/CLI smoke test.
8. Final AMD run: route binding, token parity, 8B smoke, 14B pp512/1024/2048/4096 comparison. Performance evidence is refreshed only here.

Failure at levels 1-6 blocks commits. Missing AMD access does not block landing a behavior-preserving organizational series, but it must remain explicitly pending and no benchmark claim may be changed.

## Commit boundaries

1. `exp: scope route authority centralization`
2. `master: characterize promoted route authority`
3. `master: deduplicate selected route descriptors`
4. `master: centralize candidate identity and route observation`
5. `exp: qualify production route owners`
6. `exp: reduce manifest to evidence authority`
7. `master: consolidate memory admission authority`
8. `repo: prune duplicate route authorities`

Each commit must be independently testable. Do not combine search/provenance reclassification with runtime refactoring.

## Completion definition

Completion means R0-R9 acceptance criteria pass, master and EXP are pushed to their intended branches, and the only remaining action is a separately documented AMD recertification run if hardware was intentionally excluded. It does not mean inventing missing candidate spaces or upgrading historical provenance claims.
