# LLM route/runtime module consolidation scope

Date: 2026-07-29

Status: implementation-ready scope for low-effort agents. Implementation starts on `exp`; no production behavior,
kernel, generated artifact, route geometry, benchmark claim, or hardware state is changed by this scope.

## 1. Outcome

Make the promoted LLM runtime readable as four explicit phases:

```text
selected GGUF + immutable device/model facts
  -> static model route plan
  -> memory and target admission
  -> exact invocation dispatch
  -> promoted executor or ordinary tinygrad fallback
```

The cleanup removes misleading policy-module boundaries, extracts two cohesive orchestration helpers, and leaves each
decision with one obvious owner. It must preserve every selected identity, environment compatibility mode, dispatch
order, fallback, memory decision, JIT boundary, observer event, CLI response, and generated program.

This is organization work, not a new routing architecture. Static model planning and invocation-time admission remain
separate because they consume different facts. Prefill and decode admission remain separate because their input domains
and failure contracts differ.

## 2. Baseline

At scope time, `exp` is clean at `4e19f91db` and matches `origin/exp`.

Current files in scope:

| File | Lines | Current responsibility |
|---|---:|---|
| `route_selection.py` | 43 | Shared route lifecycle values and environment-mode parser. |
| `route_policy.py` | 203 | Three unrelated jobs: serialized QK policy validation, generated-policy lookup, and flash-decode mode/admission. |
| `model_route_plan.py` | 60 | Static Q4/Q6 primitive plan derived from selected model facts. |
| `prefill_policy.py` | 96 | Immutable prefill policy envelope, proof gates, runtime diagnostics, concrete-KV validation. |
| `promoted_prefill_policy.py` | 77 | Exact promoted candidate-set binding to inventory and target. |
| `admission.py` | 243 | Device/memory/context/exact selected-GGUF admission. |
| `prefill_candidate_runtime.py` | 228 | Generated candidate artifact decode, identity, validation, and typed registry. |
| `prefill_routes.py` | 224 | Invocation-time prefill dispatch and ordinary tinygrad fallback. |
| `decode_routes.py` | 189 | Invocation-time Q4/Q6/flash binding and promoted execution. |
| `prefill_route_observer.py` | 77 | Attachment value types mixed with context-local execution observation. |
| `model.py` | 1470 | Model graph plus load order, admission orchestration, attachment installation, JIT routing. |
| `cli.py` | 678 | Tokenizer, registry, runtime state, HTTP service, benchmark/interactive UI. |

The production static decode path is already sound:

```text
GGUF metadata -> model_facts -> model_route_plan -> qk_primitives -> decode_routes
```

The problem is packaging and naming, not a proven duplicate production selector. In particular:

- `model_route_plan.py` is the static primitive authority today.
- `route_policy.py`'s serialized policy loader is a compatibility/qualification surface, not the normal model loader.
- `prefill_candidate_runtime.py` already owns the promoted artifact identity and exact registry.
- `prefill_routes.py` and `decode_routes.py` correctly own different invocation-time contracts.
- `memory_adaptive_authority.py` and `prefill_memory_plan.py` are intentional boundaries and are not consolidation
  targets in this pass.

## 3. Non-negotiable invariants

1. Candidate IDs, route IDs, candidate-set identity, compiler context identity, and selected geometry remain byte-for-byte
   unchanged.
2. `tinygrad/llm/generated/prefill_wmma_lds_dbuf_candidate_set.json` is not edited.
3. Q4/Q6 primitive `parts`, opts, roles, storage modes, kernel modes, and installed module paths remain unchanged.
4. Decode environment behavior remains exact: `TINYGRAD_DECODE_ROUTE`, legacy `FLASH_DECODE`, aliases, invalid-value
   errors, `T == 1`, symbolic start-position handling, and threshold behavior do not drift.
5. Prefill environment behavior remains exact: `TINYGRAD_PREFILL_ROUTE`, legacy packed-WMMA flag, aliases, defaults,
   forced modes, and errors do not drift.
6. Prefill dispatch order remains: scoped research override -> exact packed-WMMA candidate -> exact fp16 Graph-GEMM
   attachment -> ordinary tinygrad linear.
7. Unsupported/unattached/declined routes retain their existing generic fallback or fail-loud behavior. No admission is
   broadened.
8. Static planning never reads environment state or imports execution/research owners.
9. Exact memory admission remains the production safety authority. The strategy planner cannot override it; unknown
   memory remains fail closed.
10. Promoted prefill structural admission remains distinct from VRAM feasibility. `model.py` still installs the promoted
    graph policy only after `FULL_RESIDENT_OVERLAY` is feasible.
11. Same JIT choice/cache keys, prefill scope lifetime, concrete-KV lazy compilation, warmstart order, ring behavior,
    generic control behavior, and state-reset semantics.
12. Packed-WMMA canary/warmstart still occurs before fp16-overlay realization.
13. Observer events occur only inside the active scope and only report actual selected/executed paths. Event identity,
    count, and order remain stable.
14. `KernelProgram` remains the promoted execution/provenance boundary. Production imports no research/oracle executor.
15. CLI registry resolution, locking, cancellation, single-model lifecycle, error schema/status, `/runtime/*`, `/v1/*`,
    benchmark output, and selected-identity reporting remain stable.
16. No master import from `extra.llm_research`, no GPU/eGPU access, and no benchmark claim update.

## 4. Target ownership map

| Target | Final ownership |
|---|---|
| `route_selection.py` | Small shared vocabulary only: `RouteLifecycle`, `RouteCandidatePolicy`, `parse_route_mode`. |
| `model_route_plan.py` | Static model facts -> primitive plan, plus retained serialized-policy compatibility validation/query API. No dynamic execution imports. |
| `admission.py` | Load/runtime device, memory, context, proof, immutable-policy, and concrete-prefill admissibility. |
| `prefill_candidate_runtime.py` | Candidate artifact decode/identity/registry plus exact promoted inventory/target binding. |
| `decode_routes.py` | Decode candidates, exact bind/execute, and decode mode/context admission. |
| `prefill_routes.py` | Exact invocation-time prefill dispatch and fallback only. |
| `prefill_attachments.py` (new) | Immutable attachment/binding types and the load-time installer that validates inventory/policy correspondence. It cannot select policy. |
| `prefill_route_observer.py` | Context-local telemetry only; no attachment construction, policy, or dispatch. |
| `kernel_program.py` | Generic typed promoted-program transport/provenance boundary, unchanged. |
| `runtime_state.py` (new) | Tokenizer, built-in/runtime registry, `RuntimeFault`, and process-wide model lifecycle/warmup/status/cache state. |
| `cli.py` | Argument parsing, HTTP handlers/response formatting, benchmark presentation, interactive loop, and `main`. |
| `model.py` | Transformer/model graph and load/JIT orchestration; consumes owners above rather than reimplementing them. |

Files removed after callers reach their target owners:

- `tinygrad/llm/route_policy.py`
- `tinygrad/llm/prefill_policy.py`
- `tinygrad/llm/promoted_prefill_policy.py`

Net module count is reduced by one even after adding the two explicit attachment/runtime-state owners.

## 5. Exact symbol disposition

### 5.1 `route_policy.py`

Move unchanged to `decode_routes.py`:

- `decode_route_mode`
- `should_use_flash_decode`

Move the two-level dictionary lookup into `qk_primitives.py` as private `_qk_generated_policy_entry`:

- `qk_generated_policy_entry`
- `_qk_generated_policy_entry`

Move the retained serialized-policy compatibility section into `model_route_plan.py`:

- route-kind/spec constants and caller-supplied manifest helpers
- `RoutePolicyRow`
- `ValidatedRoutePolicy`
- `load_qk_route_policy`
- `qk_route_policy_selected`
- `qk_route_policy_selects_q4k_g3`
- `qk_route_policy_selects_q6k_generated`
- validation helpers and existing underscore compatibility aliases

The loader remains caller-supplied-manifest only. It must not become the production model-load selector, consult
runtime candidate singletons, or import `extra`.

Delete `route_policy.py` only after `rg` finds no active imports.

### 5.2 `prefill_policy.py`

Move unchanged into a named prefill-policy section of `admission.py`:

- `_CONCRETE_PREFILL_VALIDATED_M`
- `_EXECUTING_STRATEGIES`
- `_TC_ATTN_TARGET_REQUIREMENTS`
- `_SHARED_ATTENTION_PROOF_FIELDS`
- `_requirements_met`
- `shared_attention_proven_eligible`
- `bounded_packed_projection_proven_eligible`
- `select_prefill_runtime_policy`
- `immutable_prefill_policy`
- `prefill_policy_strategy`
- `prefill_policy_uses_overlay`
- `prefill_concrete_kv_auto_decision`
- `prefill_v2_validate_ubatch`

This is a cohesion move: all functions admit/freeze the load/runtime policy consumed by model orchestration. Do not fold
`prefill_memory_plan.py` or `memory_adaptive_authority.py` into it.

Delete `prefill_policy.py` only after active production, test, and research imports move to `admission.py`.

### 5.3 `promoted_prefill_policy.py`

Move unchanged into `prefill_candidate_runtime.py`, after registry construction:

- `automatic_promoted_prefill_graph_policy`

Export it from `prefill_candidate_runtime.__all__`. It remains structural-only and must return the identical policy
dictionary and identities. Delete the wrapper module after callers move.

### 5.4 Attachment/observation boundary

Move from `prefill_route_observer.py` to new `prefill_attachments.py`:

- `PrefillRouteAttachment`
- `PrefillDirectPackedBinding`

Move from `model.py` to `prefill_attachments.py`:

- `_attach_selected_prefill_inventory`, renamed `attach_selected_prefill_inventory`
- its attachment-only validation/mutation helpers, if any are proven exclusive

The moved installer must require the `direct_packed_policy` argument that `model.py` already supplies. It may not fall
back to calling `direct_packed_prefill_policy` itself, because that would create an attachment -> executor/policy edge.
Characterization must prove that the required argument is identical to the current call-site value.

`prefill_route_observer.py` may temporarily re-export the moved types for compatibility, but production callers must
use the new owner before the compatibility re-export is removed. The installer receives an already-selected immutable
policy; it cannot choose one.

Keep in `prefill_route_observer.py`:

- `PrefillRouteExecution`
- observation/scope ContextVars and context managers
- `notify_prefill_route`
- `notify_prefill_route_execution`
- `prefill_route_scope_active`

### 5.5 CLI/runtime-state boundary

Move from `cli.py` to new `runtime_state.py`:

- `SimpleTokenizer`
- built-in model registry and registry loading
- `_quant_from_name`
- `_device_target`
- `DEFAULT_REGISTRY_PATH`
- `RuntimeFault`
- `RuntimeState`, including warmup, load/adopt/unload, status, metrics, cache, cancellation-related state

Keep or re-export `SimpleTokenizer`, `models`, `_quant_from_name`, `_device_target`, `RuntimeFault`, `RuntimeState`,
`build_registry`, and `DEFAULT_REGISTRY_PATH` from `cli.py` during this pass so existing callers and the active runtime
boundary audit do not break. New production imports should use `runtime_state.py`.

Keep in `cli.py`:

- HTTP status mapping and request/response formatting
- `Handler` and server classes
- benchmark/interactive presentation
- argument parsing and `main`

The extraction must not redesign endpoints or move model semantics into the service layer.

### 5.6 `model.py`

Only perform integration edits in this pass:

- import prefill policy helpers from `admission.py`;
- import promoted candidate policy from `prefill_candidate_runtime.py`;
- import flash admission from `decode_routes.py`;
- call `attach_selected_prefill_inventory` from the new owner;
- import attachment types only where actually required;
- remove the local attachment installer after parity tests pass.

Retain model graph definitions, `Transformer.from_gguf` ordering, model inventory derivation, memory-controller/cache result
validation, forward/logits semantics, JIT selection, warmstart construction, and realization ordering. A larger
`from_gguf` rewrite is explicitly out of scope.

## 6. Dependency target and prohibited edges

Arrows below mean importer -> imported dependency:

```text
model_route_plan -> model_facts
decode_routes -> route_selection
prefill_routes -> route_selection, prefill_attachments, prefill_route_observer, admission
prefill_route_observer -> route_selection
admission -> device_facts, memory_ledger, prefill_memory_plan
prefill_candidate_runtime -> codegen.opt.kernel_pipeline, generated candidate artifact
prefill_attachments -> route_selection (attachment lifecycle type); selected policy/facts are caller-supplied values
qk_primitives -> model_route_plan, decode_routes
model -> admission, model_route_plan, qk_primitives, decode_routes,
         prefill_candidate_runtime, prefill_attachments, prefill_routes/observer
cli -> runtime_state -> model
```

Prohibited:

- `model_route_plan -> model`, `qk_primitives`, `decode_routes`, or research;
- `admission -> model`, route executors, CLI, or research;
- `prefill_candidate_runtime -> model`, route executor, CLI, or research;
- `prefill_attachments -> model.py` concrete classes or route selection;
- observer -> attachment installer, policy, or dispatcher;
- route executor -> `model.py`;
- `runtime_state -> cli.py`;
- any `tinygrad/llm` production module -> `extra.llm_research`.

## 7. Dependency-ordered work graph

### C0 — Freeze behavior contracts

Owner: new/owned tests only. No production edits.

Capture:

- static Q4/Q6 plan entries and legacy serialized-policy queries;
- decode environment aliases/defaults/errors and context threshold;
- promoted candidate-set identity and exact policy dictionary;
- prefill immutable-policy/proof behavior;
- prefill dispatch order and generic fallback;
- attachment installer output attributes and all-or-nothing validation;
- observer scope event sequence;
- runtime registry/status/error/lifecycle JSON;
- current import boundary and generated artifact hash.

Tests must exercise public behavior rather than freeze source layout or duplicate production tables.

Gate: characterization tests pass on the pre-refactor tree.

### C1 — Split `route_policy.py` responsibilities

Primary owner files:

- `model_route_plan.py`
- `qk_primitives.py`
- `decode_routes.py`
- focused route/model tests

Copy the exact symbols to target owners and update owned tests. During parallel work, leave the old module and model
imports intact; the integration packet removes it after equivalence is proven.

Preserve compatibility aliases `_load_qk_route_policy`, `_supported_qk_route_ids`, and `_qk_route_policy_*` at the new
owner until active test/research closure is complete.

Gate: old and new entry points return equal values for characterization fixtures;
`test_qk_route_purity.py`, `test_mmq_atom_boundary.py`, route-policy-sensitive `test_model_route_plan.py` cases, and the
decode/model-plan suites pass.

### C2 — Consolidate prefill policy owners

Primary owner files:

- `admission.py`
- `prefill_candidate_runtime.py`
- focused admission/candidate tests

Copy policy functions without semantic edits. Leave old wrapper modules for the integration packet.

The active research caller `extra/llm_research/prefill/prefill_flash_e2e_parity.py` must eventually import
`immutable_prefill_policy` from `admission.py`. Defer its caller edit, the `prefill_routes.py` lazy import of
`bounded_packed_projection_proven_eligible`, and the cross-owned `test_shared_prefill_policy.py` import edits to serial
C5 so C2 and C3 do not collide.

Gate: policy mappings compare equal after JSON normalization; candidate identities remain exact; memory/admission suites
pass.

### C3 — Extract immutable attachment installation

Primary owner files:

- new `prefill_attachments.py`
- `prefill_route_observer.py`
- new/focused attachment and observer tests

Establish one attachment type identity, provide temporary observer re-exports, and add the installer without changing
model call sites yet.

Gate: exact attachment objects/fields and failure behavior match; observer event sequence is unchanged.

### C4 — Extract runtime service state

Primary owner files:

- new `runtime_state.py`
- `cli.py`
- runtime/CLI boundary tests

Move code, do not redesign it. Preserve CLI compatibility re-exports and HTTP contracts.

Gate: `--help`, registry fixtures, RuntimeState status/lifecycle, error schema/status, mocked endpoint responses, and
`test_tinygrad_llm_cli_boundary.py` match. Import and run the active `extra/audit/tinygrad_runtime_boundary_audit.py`
contract against the compatibility re-exports.

### C5 — Production integration and obsolete-module deletion

Serial integration owner files:

- `model.py`
- `prefill_routes.py`
- `qk_primitives.py` import cleanup if still needed
- active research imports that call moved production APIs
- deletion of three obsolete modules

Switch production imports/calls, remove duplicated old definitions, then delete:

- `route_policy.py`
- `prefill_policy.py`
- `promoted_prefill_policy.py`

Named active caller/source closure includes:

- `extra/llm_research/decode/decode_runtime_overhead.py`
- `extra/llm_research/prefill/prefill_flash_e2e_parity.py`
- `extra/llm_research/prefill/prefill_softmax_reduce_fuse_promotion_gate.py`
- `extra/llm_research/decode/decode_codegen_identity_check.py`
- `extra/audit/pure_machine_search_default_path_census.py`
- `test_shared_prefill_policy.py`

Final `prefill_routes.py` must import `PrefillRouteAttachment` and `PrefillDirectPackedBinding` from
`prefill_attachments.py`, never from the observer. It must import the bounded packed proof gate from `admission.py`.

Gate: repository search finds no active import or source assertion for deleted modules. Dated historical evidence may
retain textual paths; current README/API maps and executable sources may not.

### C6 — Test and documentation ownership cleanup

Owner files:

- affected unit tests
- `tinygrad/llm/README.md`
- `docs/README.md`
- current organization/boundary manifests and active research comments

Rename module-layout tests to behavior/owner tests where necessary. Do not delete semantic coverage merely because a
file disappeared. Do not rewrite dated historical documents or large evidence JSON unless it acts as a current source
assertion.

Gate: target map in README matches imports and `__all__`; no current doc names a deleted module as authority.

### C7 — EXP closure validation

Run all focused gates, import-cycle/static-boundary checks, `git diff --check`, and the broadest established CPU-only EXP
suite. EXP's historical tree is not globally green due archived/hardware tests, so record exact commands and distinguish
pre-existing/environmental failures from changed-surface failures. No GPU access is authorized.

### C8 — Destination-based promotion

Promote reviewed production commits, never merge branches wholesale:

1. `exp`: implementation and broad research/qualification import closure.
2. `dev`: cherry-pick production commits plus qualification-only path updates; rerun CPU qualification gates.
3. `master`: cherry-pick only production runtime, maintained tests, and current README map; run the complete master unit
   suite and LOC ratchet.

No AMD recertification is required because emitted programs and route facts cannot change. If any candidate identity,
program fingerprint, geometry, or dispatch trace changes, stop: that is behavior work outside this scope.

## 8. Low-effort agent orchestration

Maximum concurrency is three child agents plus the orchestrator. Agents edit only assigned primary files, use
`apply_patch`, never run broad formatters, never touch hardware, and do not commit or push unless explicitly assigned.

Wave 1, parallel after C0:

| Agent | Packet | Exclusive primary ownership |
|---|---|---|
| A | C1 static/decode policy split | `model_route_plan.py`, `qk_primitives.py`, `decode_routes.py`, owned route tests |
| B | C2 prefill-policy consolidation | `admission.py`, `prefill_candidate_runtime.py`, owned policy/candidate tests |
| C | C3 attachment extraction | `prefill_attachments.py`, `prefill_route_observer.py`, owned attachment tests |

Wave 2:

| Agent | Packet | Ownership |
|---|---|---|
| D | C4 runtime-state extraction | `runtime_state.py`, `cli.py`, runtime-state/CLI tests |
| E | Read-only review | compare Wave-1 APIs against C0 invariants and report drift/cycles |
| F | Read-only test inventory | verify moved/deleted-module references and required gate coverage |

Wave 3, serial:

- Agent G owns C5 integration (`model.py`, `prefill_routes.py`, deletions, active caller imports).
- Orchestrator reviews every diff, resolves only narrow conflicts, and commits bounded checkpoints.
- Agent H owns C6 current-doc/test-path cleanup after final APIs settle.

Wave 4, parallel read-only validation:

- one agent runs focused route/plan/decode gates;
- one runs prefill/admission/attachment/candidate gates;
- one runs runtime-state/CLI/static-boundary gates;
- orchestrator runs broad EXP validation and reviews the complete diff.

Promotion is serial and orchestrator-owned. No agent may merge `exp` into another branch wholesale.

## 9. Required test gates

Route/static/decode:

- `test_model_facts.py`
- `test_model_route_plan.py`
- `test_qk_route_purity.py`
- `test_mmq_atom_boundary.py`
- `test_llm_route_selection.py`
- `test_llm_decode_routes.py`
- `test_llm_decode_correctness.py`
- `test_route_admission_consistency.py`
- `test_pure_search_guard_boundary.py`

Prefill/admission/candidate:

- `test_shared_prefill_policy.py`
- `test_promoted_prefill_candidate_runtime.py`
- `test_memory_adaptive_route_manifest.py`
- `test_memory_adaptive_model_integration.py`
- `test_memory_adaptive_exact_ledger.py`
- `test_prefill_memory_plan_integration.py`
- `test_llm_context_admission.py`
- `test_prefill_generic_fallback.py`
- `test_prefill_route_memory_semantics.py`
- `test_prefill_graph_gemm_route.py`
- `test_prefill_graph_gemm_runtime.py`
- `test_current_prefill_execution_adapter.py`
- `test_prefill_attention_production_ownership.py`

Execution/transport/static closure:

- `test_llm_kernel_program.py`
- `test_llm_kernel_program_boundary.py`
- `test_llm_generated_runtime.py`
- production-import boundary and source-ownership tests
- `test_tinygrad_llm_cli_boundary.py`

New characterization coverage:

- attachment installer equivalence and atomic failure;
- observer scope event ordering;
- runtime registry overlay/override/error behavior;
- RuntimeState adopt/load/unload/status/cache/lock behavior with mocked model/device operations;
- CLI handler response/error parity without opening a real server or loading a device.

Static commands:

```sh
rg -n "tinygrad\.llm\.(route_policy|prefill_policy|promoted_prefill_policy)" tinygrad test extra scratchpad
rg -n "(route_policy|prefill_policy|promoted_prefill_policy)\.py" tinygrad test extra scratchpad README.md tinygrad/llm/README.md docs/README.md
rg -n "extra\.llm_research" tinygrad/llm
git diff --check
```

The first two searches cover direct imports, dynamic imports, and active filename/source assertions. Dated historical
documents and evidence JSON are exempt unless they are listed as a current authority or executable source assertion.

Run a CPU-only import smoke for `tinygrad.llm.cli`, `tinygrad.llm.generate`, `tinygrad.llm.runtime_state`,
`tinygrad.llm.model_route_plan`, `tinygrad.llm.admission`, `tinygrad.llm.prefill_candidate_runtime`,
`tinygrad.llm.prefill_attachments`, `tinygrad.llm.prefill_routes`, `tinygrad.llm.decode_routes`,
`extra.llm.bench.model_e2e_bench`, and `extra.audit.tinygrad_runtime_boundary_audit`.

Use the repository Python entry point (`uv run --with pytest python -m pytest ...` or an already-provisioned compatible
environment). Tests must not create or access an AMD device.

## 10. Commit and rollback plan

Keep independently revertible commits:

1. `[test] characterize LLM route module contracts`
2. `[repo] separate static and invocation route ownership`
3. `[repo] consolidate prefill admission and candidate policy`
4. `[repo] separate prefill attachment ownership`
5. `[repo] extract LLM runtime service state`
6. `[repo] remove obsolete LLM policy modules`
7. `[docs] align LLM runtime module map`

If the repository commit policy requires a different allowed prefix, use the nearest allowed semantic prefix without
combining packets. Rollback is a normal revert of the first failing packet; never reset or discard unrelated work.

## 11. Risk register

| Risk | Failure signal | Prevention/rollback |
|---|---|---|
| Attachment class identity splits during re-export | `isinstance` route checks decline valid attachments | Define classes once in `prefill_attachments`; observer re-exports the same objects; run attachment/dispatch tests before removing old imports. |
| Import cycle introduced by consolidation | import smoke fails or module is partially initialized | Enforce the dependency graph; target owners never import `model.py` or `cli.py`; rollback the responsible packet. |
| Copy changes canonical identity or policy envelope | candidate-set/policy fixture differs | Move functions verbatim first, compare normalized outputs, and stop before integration on any difference. |
| Static and invocation selection become coupled | static plan starts reading environment/device/tensor state | Source audit plus route-admission-consistency tests; reject the patch rather than adding a generic dispatcher. |
| Old module deleted before active caller migrates | import/source assertion failure in EXP tools | C5 named-caller ledger and broad active-source scans before deletion. |
| CLI extraction breaks reflective compatibility | runtime boundary audit or old import path fails | Transitional re-exports plus direct import smoke; update active callers only in a separate reviewed step. |
| Model load ordering drifts | warmstart/canary/overlay tests or source-order gate changes | Integration-only edits in `model.py`; do not restructure `from_gguf`; revert C5 if ordering changes. |
| EXP-only paths leak into promoted commits | cherry-pick conflict or master imports research | Destination-based commits and master static boundary gate; never merge branch wholesale. |
| Mechanical move is used to introduce behavior work | kernel/artifact/geometry/fingerprint diff | Stop and split that work into a separately authorized scope; this scope cannot absorb it. |

Expected structural delta: three production modules removed, two explicit owner modules added, code largely moved rather
than rewritten, `model.py` reduced only by attachment installation, and `cli.py` reduced by runtime-state extraction.
There is no target LOC reduction beyond eliminating wrappers/duplicate imports; clarity and single ownership are the
acceptance criteria.

## 12. Completion definition

Complete means all of the following are true:

- the three obsolete policy modules are deleted;
- `route_selection.py` remains vocabulary-only;
- static planning, admission, attachment, invocation dispatch, execution, and observation each have one named owner;
- `model.py` no longer defines attachment installation or imports obsolete policy modules;
- `cli.py` no longer owns tokenizer/registry/runtime lifecycle state;
- no active source imports or asserts paths for deleted modules;
- candidate/artifact/program identities and route behavior are unchanged;
- focused EXP gates pass without hardware;
- production commits promote cleanly through `dev` to `master`;
- master full unit suite, zero-xfail policy, static import boundary, README commands, and LOC ratchet pass;
- all three promoted worktrees are clean and synchronized only after the user authorizes pushing.

## 13. Explicitly out of scope

- kernel, UOp, renderer, scheduler, or generated-code changes;
- route geometry, search-space, candidate ranking, or artifact regeneration;
- memory algorithm or strategy redesign;
- changing ordinary tinygrad fallback semantics;
- endpoint/API redesign;
- a broad `Transformer.from_gguf` rewrite;
- historical-document rewriting;
- GPU/eGPU reset, execution, performance recertification, or benchmark publication.
