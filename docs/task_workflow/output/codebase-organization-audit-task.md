# Codebase Organization Audit Task

Status: completed and landed on 2026-07-26. Phase 1 is recorded in
`docs/task_workflow/output/codebase-organization-audit-report.md`; the reviewed A1-A11 follow-up actions also landed.

Current maintenance note (2026-07-28): the checker reports one later drift,
`extra/qk/decode/capture_prefill_compile.py`, which has no manifest record. Its ownership must be decided separately;
it was not classified merely to make this cleanup pass.

## Repository

`/home/ubuntu/tinygrad-arkey`

## Governing references

Read these before designing or changing anything:

- `/home/ubuntu/knowledge_base/principles/codebase-organization-principles.md`
- `/home/ubuntu/tinygrad-arkey/structure/Development/coding-principles.md`
- `/home/ubuntu/tinygrad-arkey/sz.py`
- `/home/ubuntu/tinygrad-arkey/extra/audit/pure_machine_search_default_path_census.py`
- `/home/ubuntu/tinygrad-arkey/docs/authored-core-loc-consolidation-scope-20260716.md`

Inspect the current tree before designing anything. Treat dated documents and generated census output as historical evidence, not current truth.

## Objective

Create a deterministic, machine-enforced organizational and workflow census for the tinygrad repository. The first complete audit scope is `extra/qk`; later phases may extend the proven audit model to the rest of the repository. The audit must find opportunities to decouple responsibilities, centralize duplicated authority, modularize reusable execution, prune obsolete LOC, and promote durable production-worthy capabilities from research areas into the main `tinygrad/` package.

The audit must answer:

1. What is every authored source file for?
2. Which domain owns it?
3. Is it production, default-path, fallback, research, evidence, test, tooling, or obsolete?
4. Where does authoritative knowledge live?
5. Which files duplicate authority?
6. Which dependency directions violate declared boundaries?
7. Which large files contain multiple unrelated responsibilities?
8. Which research or probe files no longer have a live consumer or unresolved question?
9. How much LOC belongs to each domain, role, and disposition?
10. What can plausibly be consolidated or deleted without touching generated files or artifacts?
11. Which workflows repeat setup, input normalization, lowering, execution, validation, measurement, or reporting machinery?
12. Which repeated functions or scripts should become shared reusable assets?
13. Which modules mix policy, authority, execution, evidence, and presentation and should be decoupled?
14. Which rules are duplicated and should be centralized under one owner?
15. Which `extra/` implementations are proven enough to be promoted into `tinygrad/`?
16. Which experimental files have completed their purpose and can be pruned after preserving their verdict?

This is an organization and promotion audit, not a style checker and not an automatic refactor. It must produce an evidence-backed migration and pruning plan. It must not move or delete production code during the audit task.

## Primary outcomes

Classify findings into these action types:

- `decouple`: separate independent policy, authority, execution, transport, evidence, or presentation concerns.
- `centralize`: move duplicated knowledge or rules to one explicit source of truth.
- `modularize`: extract a coherent reusable execution asset behind a smaller interface.
- `reuse`: replace cloned workflow machinery with an existing or proposed shared asset.
- `promote`: move a proven durable implementation from `extra/`, `bench/`, or another research surface into its proper `tinygrad/` owner.
- `prune`: delete obsolete, superseded, refuted, duplicated, or completed experimental LOC after preserving its durable conclusion.
- `retain`: keep code where it is because locality, narrative coherence, or differing semantics outweigh apparent similarity.
- `investigate`: defer action because ownership, runtime relevance, or evidence is not established.

The audit must distinguish moving LOC from reducing LOC. Moving a file into `tinygrad/` is not a LOC improvement. Consolidation counts only when duplicated knowledge or execution is removed. Pruning counts only when the removed capability is obsolete, replaced, or intentionally retired.

## Execution scope

This task must complete a deep audit of `extra/qk`, not a shallow classification of the entire repository.

Phase 1 coverage:

- Fully classify authored files under `extra/qk`.
- Trace every shipped/default-path workflow that enters or depends on `extra/qk`.
- Include external `tinygrad/`, `bench/`, `test/`, and documentation files only when needed to establish a consumer, owner, boundary, authority artifact, or replacement.
- Report cross-boundary dependencies without requiring full semantic classification of every external file.

Later coverage, explicitly out of scope for this task:

- Full `tinygrad/llm` organizational census
- Full scheduler and codegen census
- Full AMD renderer and runtime census
- Full repository benchmark and test census

Do not claim whole-repository completeness. The deliverable must state exact Phase 1 coverage and unresolved boundaries.

## Principles

Apply these principles explicitly:

- Minimize what a reader must hold in their head.
- Optimize for locality, discoverability, modifiability, and comprehensibility.
- Prefer deep modules with small interfaces.
- Centralize authoritative knowledge.
- Modularize execution without scattering authority.
- Do not merge similar-looking code unless it represents identical knowledge.
- Do not reward lower LOC when it increases indirection.
- Flag premature abstraction and shallow-wrapper proliferation.
- Organize around domain meaning rather than implementation file type.
- Preserve sequential algorithms when splitting would scatter their narrative.
- Treat generated code, production code, research code, tests, and evidence as different roles.
- Every warning must state which principle it threatens and the concrete evidence.
- Do not assign subjective numeric quality scores.
- Audit complete workflows rather than judging files only in isolation.
- Promote semantic owners and reusable primitives, not benchmark harnesses or campaign scaffolding.
- Preserve useful evidence while removing the executable machinery that no longer earns maintenance cost.
- Treat version control as the archive for completed one-off scripts; do not keep dead executable code in the active tree merely because it might someday be useful again.

## Deliverables

Create:

- `extra/audit/codebase_organization_audit.py`
- `extra/audit/codebase_organization_manifest.json`
- `bench/codebase-organization-audit/latest.json`
- `bench/codebase-organization-audit/summary.md`
- `bench/codebase-organization-audit/workflow_table.json`
- `bench/codebase-organization-audit/action_candidates.json`
- `test/unit/test_codebase_organization_audit.py`

The JSON manifest is the human-authored source of organizational intent. It must support both explicit file records and declarative group rules.

Use explicit file records for:

- Production and default-path files
- Authority owners
- Cross-boundary adapters
- Promotion, consolidation, pruning, and investigation candidates
- Files whose role differs from their directory convention

Use declarative group rules for ordinary tests, benchmarks, evidence captures, diagnostics, and clearly uniform campaign files. A tracked authored file must be covered by either one explicit record or one unambiguous group rule.

## Manifest schema

Every covered authored source file must have:

- `path`
- `domain`
- `purpose`
- `role`
- `status`
- `default_path`
- `authority_keys`
- `public_surface`
- `allowed_dependency_domains`
- `disposition`
- `evidence`
- `workflow_ids`
- `responsibilities`
- `reusable_assets`
- `duplication_group`
- `promotion_target`
- `retention_criterion`

Use constrained values where appropriate.

Suggested `role` values:

- `authority`
- `execution`
- `adapter`
- `integration`
- `test`
- `benchmark`
- `research`
- `evidence`
- `diagnostic`
- `maintenance`

Suggested `status` values:

- `production`
- `promoted_default`
- `fallback`
- `active_research`
- `retained_reference`
- `refuted`
- `superseded`
- `deletion_candidate`
- `active_regression`
- `unresolved_reproducer`
- `historical_one_off`

Suggested `disposition` values:

- `keep`
- `consolidate`
- `move`
- `replace`
- `archive`
- `delete`
- `investigate`

Suggested `action` values:

- `decouple`
- `centralize`
- `modularize`
- `reuse`
- `promote`
- `prune`
- `retain`
- `investigate`

`purpose` must be one concrete sentence describing the knowledge or behavior owned by that file. Reject placeholders such as `utilities`, `helpers`, `miscellaneous`, or a restatement of the filename.

Group rules must provide the same organizational fields at the group level and explain why the grouped files have one uniform purpose. Conflicting or overlapping rules are hard errors. A group rule must not hide a production, default-path, authority, or action-candidate file.

## Division of responsibility

The audit script may make only objective, reproducible determinations:

- Tracked authored and generated file discovery
- Token-bearing LOC
- Manifest coverage and schema validation
- Imports and dependency edges
- Strongly connected components
- Inbound and outbound reference counts
- Declared boundary violations
- Duplicate exclusive authority declarations
- References from workflows, manifests, tests, commands, and documents
- Gross, replacement, and net LOC arithmetic

Claude must make and document semantic judgments:

- Whether two implementations encode the same knowledge
- Whether responsibilities are genuinely independent
- Whether a wrapper is shallow
- Whether an abstraction reduces cognitive load
- Whether code is a reusable primitive or campaign-specific machinery
- Whether evidence is strong enough for promotion or pruning

Do not encode subjective judgments as filename heuristics or numeric organization scores. Every semantic finding must cite inspected code, consumers, contracts, and evidence.

## Workflow model

Inventory workflows as first-class records rather than inferring organization only from files. Each workflow record must contain:

- `workflow_id`
- `domain`
- `purpose`
- `entry_points`
- `phases`
- `inputs`
- `outputs`
- `authority_files`
- `execution_files`
- `evidence_files`
- `shared_assets`
- `default_path`
- `retention_criterion`

Trace at least these workflow families where present:

- Model loading and admission
- Decode route selection and execution
- Prefill route selection and execution
- Quantized kernel specification and lowering
- Scheduler and codegen lowering
- AMD rendering, compilation, queue submission, and synchronization
- Search, promotion, rollback, and route-manifest updates
- Benchmark setup, execution, measurement, and report generation
- Correctness, parity, resource, and performance gates
- Fault reproduction, capture, correlation, and reporting

For every workflow, identify repeated phases and distinguish shared knowledge from merely similar control flow.

## Reusable asset analysis

Find repeated machinery for:

- CLI and environment parsing
- Device and model discovery
- Shape and route specification
- Allocation and execution setup
- Kernel compilation and dispatch
- Timing and synchronization
- Numeric parity and tolerance reporting
- Artifact schema and serialization
- Promotion gates and verdict records
- Fault and resource evidence capture

For each candidate reusable asset, report:

- Existing implementations
- The knowledge they duplicate
- Semantic differences that must remain explicit
- Proposed owning module
- Proposed minimal interface
- Call sites that would migrate
- LOC removed versus LOC introduced
- Tests required to prove equivalent behavior

Do not recommend a shared helper merely because code looks similar. Recommend centralization only when the implementations encode the same rule or contract.

## Promotion gates

Code deserves promotion into `tinygrad/` only when all applicable conditions hold:

- It participates in a shipped or default production path, or owns a general compiler/runtime invariant.
- Its behavior has an explicit stable contract.
- Correctness is covered at the boundary where it can fail.
- Performance claims, when relevant, are supported by current representative evidence.
- It does not depend on benchmark harnesses, research-only imports, campaign state, or evidence files.
- Environment flags and rollback paths are either part of an intentional public policy or removed during promotion.
- The proposed destination is the domain owner, not merely a convenient directory.
- Promotion simplifies the caller-facing surface or removes a hidden second system.
- Existing copies can be removed or reduced after migration.
- The code is useful beyond one frozen experiment, shape, model, or artifact.

Every promotion candidate must include:

- Current path
- Proposed `tinygrad/` destination
- Owning domain and invariant
- Current runtime consumers
- Required dependency cleanup
- Required tests
- LOC moved
- LOC deleted through consolidation
- Remaining rollback or compatibility obligations
- Evidence supporting promotion
- Evidence still missing

Promotion recommendations must be `ready`, `blocked`, or `not justified`. Do not recommend promotion solely because a file is used by the current default path.

Promotion should normally distill a durable invariant or primitive rather than transplant an entire research file. The preferred shape is:

1. Identify the core-owned invariant or capability.
2. Define the smallest stable interface at the owning `tinygrad/` boundary.
3. Move or reimplement only the durable production mechanism.
4. Move boundary tests that defend the contract.
5. Remove research harnesses, evidence coupling, campaign state, obsolete flags, and duplicate implementations.

If the only proposal is to relocate a file without simplifying ownership or deleting research structure, classify it as `not justified`.

## Pruning gates

Code is a high-confidence prune candidate only when:

- It is superseded, refuted, duplicated, or associated with a completed experiment.
- No production or retained workflow imports it.
- It is not a manually invoked operational entry point without an explicit replacement.
- Its durable conclusion is preserved in a manifest, ledger, test, or concise document.
- Any rollback or compatibility obligation has expired or moved to an explicit owner.
- Removing it does not erase the only reproducer for an unresolved bug.
- References, commands, and generated indexes can be updated in the same cleanup.

For each prune candidate report gross LOC removed, replacement LOC added, and net authored LOC reduction.

## Test and one-off script pruning policy

A test or research script is not valuable merely because it once produced evidence. GitHub and Git history provide cheap recovery for deleted scripts. The active repository should retain tests and tools that defend current behavior, reproduce unresolved failures, specify a current contract, or support a current workflow.

Classify every test-like or executable research file into one of these roles:

- `active_regression`: collected or intentionally invoked and protects a current contract.
- `active_validation`: current promotion, parity, performance, resource, or integration gate.
- `unresolved_reproducer`: smallest known reproducer for an open correctness, performance, runtime, or hardware issue.
- `operational_tool`: manually invoked current maintenance or diagnostic entry point.
- `retained_reference`: not routinely executed, but uniquely specifies a current algorithm, format, or compatibility obligation.
- `historical_one_off`: completed experiment whose durable result exists elsewhere.
- `superseded`: replaced by a stronger test, workflow, or generalized implementation.
- `orphan_unknown`: no current consumer or recorded purpose can be established.

High-confidence deletion candidates satisfy all applicable conditions:

- No production import or runtime registry references the script.
- No CI configuration or current test collector executes it.
- No current workflow, command document, manifest, subprocess call, or promotion gate names it.
- It is not the only reproducer for an unresolved issue.
- Its experiment completed, was refuted, or was superseded.
- Its durable conclusion is already recorded in a current ledger, manifest, regression test, or concise document.
- Any still-relevant invariant is defended by a smaller or stronger replacement test.
- Recovery from repository history is sufficient if historical implementation detail is needed later.

Treat absence of Python imports as only one fact. Standalone scripts may be invoked through shell commands, CI, documentation, manifests, subprocesses, or operator practice.

Use these deletion-confidence classes:

- `delete_ready`: purpose completed or superseded, no live consumer, no unique unresolved evidence, and recovery information is recorded.
- `delete_after_verdict_capture`: executable machinery is obsolete but its conclusion must first be preserved concisely.
- `retain_active`: current regression, validation, workflow, or operational tool.
- `retain_reproducer`: unique or materially useful reproducer for unresolved work.
- `investigate_owner`: relevance cannot be established honestly.

Do not create an in-repository archive directory for removed scripts. That moves clutter instead of pruning it. For every `delete_ready` recommendation, record:

- Current path
- One-sentence former purpose
- Last known workflow or campaign
- Replacement test, manifest, ledger, or document when one exists
- Current audited commit identity
- Recovery instruction using repository history
- LOC removed

Repository recoverability lowers the retention threshold for inactive standalone scripts. It does not authorize deletion of current regression tests, unresolved reproducers, retained specifications, or operational tools.

## Coverage

Discover tracked authored source under:

- `tinygrad/`
- `extra/`
- `bench/`
- `test/`
- Root-level Python scripts

Include Python and project-owned JavaScript.

Exclude:

- Files carrying `@generated` or `autogenerated` markers
- `tinygrad/runtime/autogen/`
- `docs/artifacts/`
- Generated captures and binary evidence
- Vendored assets
- Virtual environments
- Build outputs
- Untracked local files

Generated files should be counted and reported separately, not entered into the authored-purpose manifest. Phase 1 requires complete manifest coverage only for `extra/qk`; external files are dependency evidence unless explicitly promoted into the audit scope.

Use `sz.py` token-bearing LOC rather than physical newline counts.

## Hard errors

- An authored source file has no manifest entry.
- A manifest path does not exist.
- Duplicate manifest entries exist.
- Purpose, domain, role, status, or disposition is missing.
- A default-path file is classified as `refuted`, `superseded`, or `deletion_candidate`.
- Production `tinygrad` code imports `extra.audit` or research-only modules.
- Multiple files claim the same exclusive authority key.
- A declared dependency-domain rule is violated.
- A generated file is incorrectly counted as authored.
- An authored `extra/qk` file is covered by neither an explicit record nor a group rule.
- An explicit record and group rule ambiguously cover the same file.
- A production, default-path, authority, or action-candidate file is hidden inside a group rule.
- A `ready` promotion candidate depends on research-only or evidence modules.
- A prune candidate still has a live production or retained-workflow consumer.
- Two workflows encode the same authority rule under different owners without an explicit reason.
- A `delete_ready` script is still collected by the test runner, invoked by CI, named by a live manifest, or documented as a current command.
- A unique unresolved reproducer is classified as historical, superseded, or deletion-ready.

## Review warnings

- A file exceeds a configurable LOC threshold.
- A file declares multiple unrelated authority keys.
- A large file's stated purpose contains multiple independent responsibilities.
- A module has high fan-out or high fan-in.
- Import cycles cross declared domains.
- Human review concludes that a shallow wrapper adds an interface without hiding meaningful complexity.
- A research or evidence file has no current manifest route, test, document, or authority artifact referring to it.
- Several scripts in one domain repeat entry-point, argument, setup, or record-shaping logic.
- Human review concludes that similar files encode duplicated knowledge.
- A fallback or rollback path has no stated retention criterion.
- A deletion candidate is still imported or referenced.
- A file's declared role conflicts with its location or dependencies.
- A workflow repeats setup or reporting mechanics already owned by another workflow.
- An `extra/` module is on the default production path without an explicit promotion or retention decision.
- Promotion would move code without deleting duplication or simplifying an ownership boundary.
- Experimental machinery remains after its verdict has been recorded and no retention criterion exists.
- A standalone test script has no current caller, contract, unresolved issue, retained evidence role, or named owner.
- Multiple historical scripts preserve the same verdict after a stronger replacement gate exists.

Do not make warnings fail `--check` unless their invariant is objective.

## Import graph

Use Python AST parsing to construct internal imports without importing project modules.

Report:

- Inbound dependency count
- Outbound dependency count
- Strongly connected components
- Cross-domain edges
- Forbidden edges
- Unreferenced entry points
- Production-to-research dependencies
- Workflow-to-file and file-to-workflow relationships
- Promotion candidates with research-only dependency chains
- Prune candidates with hidden inbound references
- Test collection, CI, shell-command, manifest, subprocess, and documentation references

Handle relative imports correctly.

## Commands and modes

Support:

```bash
python3 extra/audit/codebase_organization_audit.py
python3 extra/audit/codebase_organization_audit.py --check
python3 extra/audit/codebase_organization_audit.py --scope extra/qk
python3 extra/audit/codebase_organization_audit.py --scope tinygrad/llm
python3 extra/audit/codebase_organization_audit.py --top 30
```

Normal mode writes JSON and Markdown reports.

`--check` must be deterministic, must not rewrite the manifest, and must return nonzero only for hard errors.

`--scope` must restrict reporting while still detecting dependency edges crossing the selected boundary.

## Report requirements

The Markdown summary must include:

- Authored and generated LOC totals
- LOC grouped by domain, role, status, and disposition
- Manifest coverage
- Default-path source footprint
- Largest authored files
- Duplicate-authority findings
- Cross-domain dependency violations
- Import cycles
- Research and evidence files without a live owner
- Consolidation candidates
- Deletion candidates that are genuinely unreferenced
- Test and one-off script inventory by active, unresolved, operational, retained-reference, historical, superseded, and unknown status
- `delete_ready` and `delete_after_verdict_capture` tables with recovery information
- Workflow inventory and repeated workflow phases
- Decoupling candidates
- Centralization candidates
- Modularization and reusable-asset candidates
- Promotion candidates ranked as `ready`, `blocked`, or `not justified`
- Gross, replacement, and net LOC impact for every proposed action
- A recommended sequence that preserves behavior and keeps commits independently reviewable
- Files requiring human classification
- Exact evidence for every finding
- Exact audit scope and coverage limitations
- A separation between machine-derived facts and Claude's semantic findings

Do not describe a file as dead merely because it has no imports. Entry points, scripts, tests, generators, and manually invoked tools may legitimately have no inbound imports.

## Bootstrap strategy

Fully classify `extra/qk` because it currently contains roughly 17.7K authored LOC across 123 files while only a small subset is directly named by the default-route census. Identify complete workflows within it before labeling individual files as reusable, promotable, or removable.

Do not expand into a full audit of `tinygrad/llm`, scheduler/codegen, AMD renderer/runtime, `bench/`, or `test/` during this task. Record those areas as later phases with the concrete boundary information learned from `extra/qk`.

Use the current route manifest and route census as evidence, but do not trust their attribution blindly. The current default-path census has reported an attribution-blocked state. Surface inconsistencies instead of copying them into the new manifest.

Do not auto-generate semantic purposes from filenames. Inspect each file enough to write an accurate one-sentence purpose. Mechanical path-based defaults are acceptable only for obvious generated, test, or artifact classifications.

Produce a proposed campaign sequence after the census. Limit the detailed action set to up to ten high-confidence candidates; return fewer when the evidence does not justify ten. Place lower-confidence findings in an investigation backlog.

1. Remove high-confidence obsolete probes and completed campaign machinery.
2. Centralize duplicated schemas, route rules, evidence records, and workflow setup.
3. Extract reusable assets only after proving the repeated semantics are identical.
4. Decouple policy and authority from execution where they are mixed.
5. Promote production-worthy primitives into their proper `tinygrad/` owners.
6. Remove compatibility copies and update callers.
7. Re-run the organizational audit and report net authored LOC reduction.

Prefer deleting obsolete scripts over moving them to an archive folder. Preserve durable conclusions, current contracts, and unresolved reproducers; let repository history preserve obsolete implementations.

Each proposed campaign step must be independently testable and must not mix NFC movement with behavior changes.

## Success criteria

The task is complete when:

- Every tracked authored `extra/qk` file is covered explicitly or by one unambiguous group rule.
- Every default-path entry into `extra/qk` is traced to its callers and durable authority.
- Machine-derived facts and human semantic findings are clearly separated.
- Up to ten high-confidence actions have concrete evidence, ownership, tests, and net LOC effects; no minimum action quota is imposed.
- Promotion recommendations identify the minimal durable primitive rather than merely a file destination.
- Prune recommendations preserve required verdicts and unresolved reproducers.
- Every test-like `extra/qk` file is classified as active, unresolved, operational, retained reference, historical, superseded, or unknown.
- Every `delete_ready` candidate includes enough repository identity and replacement context to recover or understand it later.
- Unknowns are labeled honestly rather than forced into action categories.
- The audit implementation remains a small deterministic checker rather than a new static-analysis framework.
- No production behavior, file placement, or runtime path changes in this task.

## Tests

Add temporary-tree tests with positive controls proving that the audit detects:

- An unmanifested authored file
- A stale manifest path
- Duplicated authority
- A forbidden production-to-research import
- A cross-domain import cycle
- A generated file incorrectly classified as authored
- A default-path deletion candidate
- An unreferenced but explicitly retained CLI entry point that must not be called dead
- A false promotion candidate that depends on research-only modules
- A valid promotion candidate with a stable contract and production consumer
- A prune candidate with a hidden workflow reference
- Two workflows duplicating the same declared authority
- Similar-looking implementations whose distinct authority keys correctly prevent consolidation
- A standalone script referenced only through a documentation command
- A test collected dynamically rather than imported directly
- A completed one-off script correctly classified as `delete_ready`
- A unique unresolved reproducer correctly retained

Also prove deterministic JSON and Markdown output.

## Constraints

- Do not modify production behavior.
- Do not delete or move files in this task.
- Do not perform the proposed consolidation or promotion in this task.
- Do not reclassify code merely to make the audit pass.
- Do not use Git history as proof of runtime relevance.
- Git history may be used to record recoverability and provenance after current-tree consumers and obligations have been audited.
- Do not introduce third-party dependencies.
- Keep the auditor independent from tinygrad runtime imports.
- Reuse `sz.py` accounting rules instead of implementing a conflicting LOC definition.
- Keep the implementation substantially smaller and simpler than the code surface it audits.
- Separate this audit commit from any later cleanup commit.
- Do not count movement between directories as LOC reduction.
- Do not promote code to make the authored budget appear smaller or to bypass `sz.py` accounting.
- Do not convert campaign-specific scripts into permanent framework layers without multiple durable consumers.
- Do not retain obsolete scripts solely because deletion feels irreversible; repository history is the recovery mechanism.
- Do not create an archive directory containing scripts selected for deletion.
- Keep the audit implementation limited to discovery, LOC accounting, manifest validation, import/reference analysis, boundary checks, and deterministic reporting.
- Do not add automated heuristics for abstraction quality, responsibility coherence, or semantic duplication.
- Record the audited commit identity in the report so concurrent changes cannot silently invalidate conclusions.
- Do not begin hot-path moves or deletion while the GPU-fault attribution experiment depends on the current schedule or allocation layout.

## Required final report

Report:

- Files created
- Coverage achieved
- Hard errors found
- Highest-confidence consolidation or deletion candidates
- Highest-confidence decoupling, centralization, modularization, and reusable-asset candidates
- Promotion candidates classified as `ready`, `blocked`, or `not justified`
- Projected gross and net LOC effects
- Recommended cleanup and promotion sequence
- Areas that could not be classified honestly
- Exact commands used
- Assumptions that still require human approval

The audit must not algorithmically decide that organization is good or bad. The manifest records human organizational intent; the script detects objective drift and produces evidence for human review.
