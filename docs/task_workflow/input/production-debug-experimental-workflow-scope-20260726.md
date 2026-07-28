# Production, Debug, and Experimental Workflow Scope

Date: 2026-07-26
Updated: 2026-07-28

Repository: `/Users/julianabeleda/env/tinygrad-arkey`

Status: in progress. The persistent `master`, `dev`, and `exp` branches/worktrees were established on 2026-07-28.
The production-boundary inventory, test-script placement, enforcement, and pilot selective promotion remain open.

## 1. Objective

Establish three durable branch roles, each in its own worktree:

- `master`: clean production.
- `dev`: debugging, reproduction, qualification, and production-candidate integration.
- `exp`: experimental implementations, risky probes, parameter searches, and disposable scripts.

The production branch must contain the shipped hot path plus only the tests, authorities, manifests, and operating
documentation required to prove and maintain that path. Debug tooling that is useful but not required to defend
production stays on `dev`. Unproven or disposable work stays on `exp`. Movement toward production is selective:
files do not enter a cleaner tier merely because a whole branch was merged.

Target pipeline:

```text
exp experiment -> dev debug/qualification -> master production
       ^                    |
       +-- rejected/rework -+
```

This task is about repository workflow and enforcement. It is not permission for an unbounded code rewrite or bulk deletion.

## 2. Motivation

Multiple agents recently shared the production worktree while one process ran a long GPU benchmark. A branch checkout,
edits to `tinygrad/uop/ops.py`, a stash cycle, and unlocked GPU work changed the source and device state underneath
benchmark subprocesses. The results lost code-identity and GPU-ownership authority.

The incident exposed two separate requirements:

- Git branches alone do not isolate files. Concurrent work requires separate worktree directories.
- Worktrees do not isolate the GPU. Every GPU command still requires `/tmp/gpu-bench.lock`.

The repository also contains production paths, active integration candidates, reusable debug authorities, one-off
probes, historical experiments, and raw artifacts in overlapping locations. A production/debug/experimental boundary
should make placement obvious, promotion deliberate, and cleanup routine.

## 3. Non-goals

- Do not rewrite Git history.
- Do not force-push `master`, `dev`, or `exp`.
- Do not create a second permanent laboratory repository in this task.
- Do not delete an active agent's worktree, branch, stash, or uncommitted files.
- Do not classify all tests or benchmark authorities as non-production merely because they are not imported at runtime.
- Do not merge `dev` wholesale into `master` or `exp` wholesale into `dev`; both source branches intentionally own
  files that the cleaner destination must not receive.
- Do not use this workflow migration to perform unrelated lowering, scheduler, renderer, decode, or prefill refactors.
- Do not run a broad GPU benchmark campaign merely to validate branch mechanics.

## 4. Definitions

### 4.1 Production

`master` is production. It must remain deployable, benchmarkable, and recoverable at every commit.

Production includes:

- Shipped `tinygrad/` runtime and compiler paths.
- `extra/` code with a real production consumer or a canonical production authority.
- Regression tests for shipped behavior and previously observed faults.
- Canonical correctness and performance benchmark entry points.
- Route, capability, provenance, and ownership manifests used to select or audit production behavior.
- Compact operating documents and durable findings ledgers needed to avoid repeating unsafe work.
- Fail-loud guards for configurations that are known unsafe and have no supported fallback.

Production excludes:

- One-off debug probes after their result has been promoted.
- Failed or abandoned implementation candidates.
- Duplicate benchmark wrappers around an existing authority.
- Raw benchmark dumps with no ledger owner or retention reason.
- Active scratch plans, temporary handoffs, and in-progress task state after a task closes.
- Research implementations with no production importer, promotion candidate, or active bounded task.

### 4.2 Debug and qualification

`dev` is the durable debugging and qualification branch. It contains reusable reproduction scripts, diagnostics,
debug-only tests, benchmark harnesses, and production candidates that are ready for structured validation.

`dev` may permanently own useful debugging tools that do not belong in production. It is not a raw artifact archive:
stale reproductions and superseded wrappers are removed after their conclusions are banked. Production code promoted
from `dev` carries only the minimum regression tests and authorities needed on `master`; debug-only assets remain.

### 4.3 Experimental

`exp` is the durable experimental branch and has its own worktree. It contains unproven implementations, risky GPU
probes, parameter sweeps, one-off test scripts, and candidates whose production value or safety is not established.
It is based on `dev`, so experiments can use the debug and qualification tools without copying them into production.

The maintained surfaces are subtractive: `exp` contains production + debug + experimental assets, `dev` contains
production + debug assets, and `master` contains production assets only. Confirmed dead assets belong in none of them.

`exp` is a laboratory, not an archive. Failed work remains recoverable from Git history; unique conclusions move to a
compact ledger, and dead scripts are deleted. Concurrent experimental tasks use dedicated branches and worktrees such
as `experiment/decode-ctx128` or `experiment/prefill-search`, normally forked from `exp` and selectively folded back.

Unproven work stays on `exp` or an `experiment/*` branch. It does not enter `dev` merely because it compiles.

### 4.4 Test and script placement

| Asset | Owning branch | Rule |
|---|---|---|
| Regression test required to defend shipped behavior | `master` | Must accompany the production behavior it protects. |
| Canonical production correctness or performance authority | `master` | Keep the smallest stable entry point and its ownership record. |
| Reusable reproducer, diagnostic, trace tool, or debug-only test | `dev` | Retain while it has a named debugging contract and owner. |
| Production candidate under qualification | `dev` | Promote only its bounded production diff and required regression evidence. |
| Unproven prototype, risky probe, sweep, or one-off script | `exp` | Delete after rejection or promote to `dev` after evidence is banked. |
| Raw benchmark output | External artifact store | Keep only compact, attributable summaries and required baselines in Git. |

## 5. Required topology

| Purpose | Branch | Worktree |
|---|---|---|
| Production | `master` | `/Users/julianabeleda/env/tinygrad-arkey` |
| Debug and qualification | `dev` | `/Users/julianabeleda/env/tinygrad-arkey-dev` |
| Experimental | `exp` | `/Users/julianabeleda/env/tinygrad-arkey-exp` |
| Concurrent experiment | `experiment/*` from `exp` | `/Users/julianabeleda/worktrees/<task-name>` |
| Emergency production repair | `hotfix/*` from `master` | dedicated worktree |

The directory names may be adjusted if occupied, but `master`, `dev`, and `exp` must never share a directory.

## 6. Invariants

1. No experiment, debug session, stash cycle, or branch checkout occurs in the production worktree.
2. `master` advances only through a reviewed selective promotion or hotfix.
3. Ordinary experiments begin on `exp` or an `experiment/*` worktree, never on `master`.
4. `dev` receives experimental work only after it has a named debugging or qualification purpose.
5. Neither `dev` nor `exp` is merged wholesale toward production; promotion uses a reviewed bounded diff based on the destination branch.
6. Production runtime never imports an asset owned only by `dev` or `exp`; `dev` never depends on an asset that exists only on `exp`.
7. Every GPU command from every worktree acquires `/tmp/gpu-bench.lock`.
8. Every published benchmark records commit, branch, worktree, dirty state, command, model identity, route identity, artifact path, power state, and lock ownership.
9. A task-specific probe is deleted after its evidence is promoted unless it becomes an owned reusable diagnostic on `dev`.
10. Recovery from Git history is acceptable for retired experiments; dead code does not remain in any tier solely to make recovery convenient.
11. Production tests and authorities may remain even when they are not runtime-imported, provided they defend a shipped path and have a named contract.
12. After the tiers diverge, production behavior changes are synchronized downward as reviewed commits or bounded
    patches. A production cleanup deletion is not propagated into a branch that owns the deleted debug or experimental
    asset. Whole-branch equality is not a goal in either direction.

## 7. Promotion contract

A candidate may move from `exp` to `dev` only when it has:

- A named production problem or capability target.
- A reachable consumer or an explicit integration point.
- Focused correctness evidence with a positive control.
- No known destructive filesystem or GPU behavior.
- An owner, bounded scope, and removal condition for temporary code.
- Task-specific probes separated from reusable implementation code.
- A bounded commit or destination-based patch that excludes unrelated experimental files.

A candidate may move from `dev` to `master` only when it has:

- Production-path reachability demonstrated, not inferred from file names.
- Correctness evidence on every admitted model, shape, backend, and route it claims.
- Performance evidence for every performance claim, collected through the canonical authority.
- Route and kernel identity evidence proving the intended implementation ran.
- Fault evidence or a bounded fault-free campaign when the change concerns GPU safety.
- A supported rollback, or an explicit fail-loud guard when rollback is unsafe.
- Regression coverage for the defect or policy boundary.
- Updated route, provenance, ownership, and current-state records.
- No dead probes, stale benchmark wrappers, duplicate implementation, or abandoned compatibility path.
- A clean production-boundary audit and repository size-budget result.
- A clean `dev` worktree and a destination-based promotion diff containing no debug-only or unrelated files.

## 8. Promotion mechanics

1. Experimental work remains on `exp` or `experiment/*` until its focused gates pass.
2. A bounded commit or patch is applied to a candidate branch based on `dev`; `exp` is never merged wholesale.
3. Reusable debugging tools stay on `dev`, while dead probes and raw artifacts are removed after evidence is banked.
4. Qualification authorities run on the exact `dev` commit and worktree containing the candidate.
5. A production promotion branch is created from current `master` and receives only the implementation, required
   regression tests, required authorities, and durable production records.
6. Production gates run against that exact promotion commit; debug-only files are absent from its diff.
7. A reviewed PR or bounded merge advances `master` from the promotion branch.
8. The promoted commit receives a durable findings or current-state update.
9. The new production commits are selectively synchronized downward into `dev` and then `exp`; tier-placement
   deletions are excluded where the destination branch owns the asset.
10. Temporary promotion and `experiment/*` branches/worktrees are removed after their retained work reaches its owner.

Because `dev` permanently owns debug assets, do not use branch equality with `master` as a goal and do not merge
`dev` wholesale. Construct a bounded promotion branch from `master` containing only approved production files and
commits. Apply the same rule between `exp` and `dev`.

## 9. Hotfix mechanics

1. Create `hotfix/<name>` from the current production commit in a dedicated worktree.
2. Reproduce the production defect with the smallest authoritative test.
3. Implement and validate only the bounded repair.
4. Promote directly to `master` after review.
5. Merge or cherry-pick the production hotfix down into `dev`, then `exp`, immediately.
6. Delete the hotfix branch and worktree after all three durable branches contain the repair.

## 10. Execution phases

### Phase 0: Safety census

Record without modifying state:

- Current `master` and remote commit.
- Existing branches, worktrees, stashes, dirty files, and active agents.
- Active GPU processes, lock ownership, and power profile.
- Unpushed commits and branches that another actor owns.
- Current branch-protection and CI behavior.

Stop if any planned path is already owned or if an unexpected process is using the GPU without the lock. Do not clean another actor's state.

Deliverable: `docs/task_workflow/output/production-debug-experimental-workflow-phase0-20260726.md`.

### Phase 1: Establish branch and worktree topology

- Create `dev` from the selected clean production commit, then create `exp` from `dev`.
- Create `/Users/julianabeleda/env/tinygrad-arkey-dev` with `git worktree add`.
- Create `/Users/julianabeleda/env/tinygrad-arkey-exp` with `git worktree add`.
- Add a concise operator document with canonical commands for entering each durable worktree and for creating and
  retiring concurrent `experiment/*` worktrees.
- Document that `git checkout` in the production worktree is forbidden for ordinary debug and experimental work.
- Add a positive-control command that prints branch, commit, worktree path, and dirty state before a benchmark.

Do not move existing active task branches during this phase.

### Phase 2: Production-boundary inventory

Classify authored files using importer reachability, manifest ownership, authority ownership, and active-task evidence.

Required categories:

- `production_runtime`
- `production_authority`
- `production_regression_test`
- `production_manifest_or_ledger`
- `debug_tool_or_reproducer`
- `debug_qualification_test`
- `production_candidate_under_qualification`
- `experimental_candidate`
- `experimental_probe`
- `historical_or_dead`
- `unresolved`

Every non-runtime production asset must state the shipped behavior it defends. Every deletion candidate must state its last importer, last use, recovery commit, and whether its conclusion was banked.

Do not infer purpose solely from location under `extra/`, `test/`, `docs/`, or `bench/`.

### Phase 3: Add enforceable policy

Create or extend a repository audit that fails when:

- Production runtime imports a debug-only module.
- Production runtime or authority imports an experimental module.
- A `dev` debug tool imports an asset owned only by `exp`.
- A production route points to a missing authority or manifest entry.
- A task probe is classified as a production dependency.
- A promoted candidate lacks its required correctness or route-identity record.
- A production benchmark dispatch target is missing.
- A retained non-runtime asset has no owner or defended behavior.
- A branch-specific workflow file attempts to make `dev` or `exp` content a runtime dependency of `master`.

The audit must use explicit manifest data and importer evidence. It must not use file-name heuristics as final authority.

Add focused unit tests for each failure mode.

### Phase 4: CI and repository protection

Define and, where credentials allow, configure:

- No direct pushes to `master` except an operator-approved emergency.
- Required production-boundary audit on promotion PRs.
- Required focused unit and authority smoke tests.
- A path-sensitive check that identifies changes to shipped hot paths.
- A promotion checklist in the PR template.
- A rule that records the exact promoted commit and authority artifacts.
- A path/ownership check that rejects debug-only or experimental assets from a `master` promotion diff.

If GitHub branch protection cannot be changed from the environment, produce exact operator steps and mark the external action pending. Do not claim protection is enabled without a positive check.

### Phase 5: Initial cleanup and placement

Use the Phase 2 inventory to perform a bounded first cleanup:

- Delete confirmed dead task probes after banking unique conclusions.
- Fold redundant benchmark output into the canonical ledger.
- Promote genuinely reusable utilities into an owned stable location.
- Keep production regression tests and canonical authorities on `master`.
- Retain reusable debug scripts, reproducers, traces, and qualification harnesses on `dev` with explicit owners.
- Move unproven prototypes, sweeps, risky probes, and one-off scripts to `exp` or `experiment/*` rather than leaving
  them on `master` or allowing them to accumulate on `dev`.
- Remove completed `docs/task_workflow/input` and `in_progress` files from production after their output is banked.

No item in `unresolved` may be deleted in this phase.

### Phase 6: Pilot promotion

Use one small, low-risk change to exercise the complete pipeline:

```text
exp -> dev -> qualification gates -> bounded production branch -> master
```

The pilot must prove:

- Filesystem isolation among production, debug, and experimental work.
- GPU lock ownership if the pilot uses the GPU.
- Exact commit identity in artifacts.
- Required CI and audit behavior.
- Absence of debug-only and experimental files from the production diff.
- Clean temporary branch/worktree retirement.
- Selective downward synchronization of the new production commits into `dev` and `exp` after promotion.

Do not choose a lowering refactor, GPU fault repair, or performance-sensitive kernel as the pilot.

### Phase 7: Close bootstrap state

- Publish the final workflow document and production-boundary manifest.
- Record the pilot evidence and any external protection action still pending.
- Delete temporary migration probes and duplicate scope documents.
- Remove this input document from production after its durable output is accepted.
- Record baseline LOC and authored-file counts for production, debug assets, experimental assets, and retired assets.

## 11. Required deliverables

- `docs/production-debug-experimental-workflow.md`
- A machine-readable production-boundary ownership manifest.
- Explicit debug and experimental ownership classes in that manifest.
- A production-boundary audit with focused unit tests.
- A promotion checklist or PR template.
- A Phase 0 safety census.
- A Phase 2 classification report with evidence per file or owned group.
- A cleanup ledger listing deleted, retained, promoted, and unresolved assets.
- Pilot promotion evidence.
- Final LOC and file-count comparison.
- Exact operator instructions for any branch protection that could not be automated.

Names may follow existing repository conventions, but there must be one canonical document and one canonical machine-readable ownership source.

## 12. Evidence rules

- Empty output is not evidence. Every audit and benchmark needs a positive control.
- A clean Git status does not prove commit identity; record the commit explicitly.
- A branch name does not prove filesystem isolation; record the worktree path.
- A worktree does not prove GPU isolation; record lock ownership.
- Unit-test parity does not prove a deleted runtime path was unreachable; use importer and dispatch evidence.
- Git recoverability does not justify deleting an unresolved asset.
- A benchmark result without route identity cannot promote a route.
- A document stating that a guard exists does not prove the guard executes.

## 13. Stop conditions

Stop and request operator review if:

- Creating `dev` or `exp` would overwrite or strand an existing branch.
- An existing worktree is owned by an active actor.
- `master` contains unpushed commits whose ownership is unclear.
- A proposed deletion has a production importer or unresolved route-manifest reference.
- A proposed `exp` to `dev` change contains unrelated experimental work.
- A proposed `dev` to `master` change contains debug-only or unrelated files.
- Branch protection requires credentials or settings unavailable to the agent.
- A benchmark would run without the GPU lock or against a mutable source directory.
- The migration requires history rewriting or a force push.

## 14. Completion criteria

The task is complete only when:

- `master` is documented and enforced as the production branch.
- `dev` exists in a separate worktree and is documented as the durable debug and qualification branch.
- `exp` exists in a separate worktree and is documented as the experimental branch.
- Concurrent `experiment/*` worktree creation and retirement are reproducible.
- The promotion and hotfix paths have both documented commands and enforced gates.
- Production imports no debug-only or experimental asset, and `dev` imports no `exp`-only asset.
- Every retained non-runtime production asset has an owner and defended behavior.
- Every retained `dev` script has a named debug/qualification contract; dead and one-off scripts reside only in Git
  history after their conclusions are banked.
- Confirmed dead probes are removed and recoverable from recorded commits.
- One low-risk experiment completes the selective `exp` to `dev` to `master` promotion pipeline.
- The production worktree remains unchanged during debug and experimental work.
- The pilot proves that debug-only and experimental files do not appear in the production promotion diff.
- GPU evidence records exclusive lock ownership.
- The promoted production behavior is selectively synchronized downward into `dev` and `exp` without deleting
  branch-owned debug or experimental assets.
- The final production-boundary and LOC reports are retained.
- This bootstrap input is retired after its durable output is accepted.

## 15. Expected outcome

The repository should end with three distinct working surfaces: a small, stable, and provable `master`; a durable
`dev` environment for debugging, reproduction, and qualification; and an `exp` environment for disposable experiments.
Useful work moves forward through selective gates, while production fixes synchronize downward. Failed work leaves
durable conclusions and Git history, not permanent dead scripts in any active branch.

## 16. Execution revision: exhaustive production reorganization

Date authorized: 2026-07-28

Active phase: scope update followed by repository census and classification from the `exp` worktree. No pruning is
authorized until the classification ledger in R7 is complete and every removal has a destination or recovery record.

### 16.1 Reorganization objective

Make `master` a minimal production repository without weakening the proof of shipped behavior:

- Fork-authored code executed by the production runtime belongs under `tinygrad/`, not behind an `extra/` adapter.
- Reusable reproducers, diagnostics, qualification harnesses, debug-only tests, and detailed investigation notes
  belong on `dev`.
- Unproven implementations, risky probes, sweeps, scratch scripts, and disposable artifacts belong on `exp`.
- Dead or superseded assets are deleted after their unique conclusion and recovery commit are recorded.
- Production keeps only regression tests, canonical authorities, compact current-state records, operating instructions,
  runtime fixtures, and manifests required to build, run, validate, or recover the shipped path.

The pruning direction is mandatory:

```text
exp (production + debug + experimental)
  -> dev (production + debug)
    -> master (production)
```

Work is audited and made coherent on `exp` first. `dev` is then produced by removing experimental-only assets, and
`master` is produced last by removing debug-only assets. This avoids using the most-pruned branch as the source of a
richer branch and makes every subtraction reviewable.

This applies to every current fork-added or fork-modified asset, not only `extra/qk`. Unchanged upstream files are not
automatically deletion candidates; an upstream boundary is changed only when the Arkey production path depends on it
or the file has an explicit fork-owned disposition.

### 16.2 Measured baseline

Baseline production commit before the census: `5b2439eac`.

Topology at the baseline:

| Branch | Worktree | State |
|---|---|---|
| `master` | `/Users/julianabeleda/env/tinygrad-arkey` | clean production baseline |
| `dev` | `/Users/julianabeleda/env/tinygrad-arkey-dev` | clean, equal to baseline |
| `exp` | `/Users/julianabeleda/env/tinygrad-arkey-exp` | clean, equal to baseline |

Fork comparison base: upstream common ancestor `6e1b61f16` (2026-06-10). Current files added since that base:

| Area | Added files |
|---|---:|
| `tinygrad/` | 97 |
| `extra/` | 161 |
| `test/` | 162 |
| `docs/` | 405 |
| `bench/` | 43 |
| `structure/` | 27 |
| `scratchpad/` plus root scratch scripts | 10 |

Current files modified from that base include 116 under `tinygrad/`, 19 under `extra/`, 2 under `test/`, and 3 under
`docs/`. Deleted upstream files are tracked separately as sync history and are not counted as current assets.

The existing organization audit covers only `extra/qk`: 95 authored files and 13,516 token-bearing LOC. It has 94
explicit records and one hard drift, `extra/qk/decode/capture_prefill_compile.py`. The audit conservatively reports 55
files reachable through `tinygrad` boundary wrappers, but wrapper reachability is not proof of default-path execution.
R2 must resolve actual consumers before promotion or removal.

### 16.3 Required classification record

The machine-readable inventory must give every in-scope file or explicitly uniform group:

- Current path and owning branch: `master`, `dev`, `exp`, or `delete`.
- Category: runtime, production authority, production regression, operating record, debug tool, debug test,
  experimental candidate, one-off probe, raw artifact, generated file, vendored/upstream, or unresolved.
- Production consumer or CLI entry point, including lazy/dynamic imports and shell/document references.
- Default-path status and supported backend/model/route contract.
- Disposition: retain, promote to `tinygrad`, move to `dev`, move to `exp`, consolidate, or delete.
- Destination path when promoted, plus import and test migration requirements.
- Retention criterion and owner for every non-runtime file retained on `master` or `dev`.
- Last use, banked conclusion, and recovery commit for every deletion.
- Evidence confidence and an explicit unresolved flag; unresolved files cannot be removed.

File names and directory location are discovery signals, never final classification evidence.

### 16.4 Work ledger

| ID | Status | Work item | Completion evidence |
|---|---|---|---|
| R0 | completed | Establish and push the three clean worktrees | `master`, `dev`, and `exp` all at `5b2439eac` before divergence |
| R1 | in progress | Census every current fork-added and fork-modified file from `exp` | Inventory counts reconcile with Git and `sz.py`; no unclassified path |
| R2 | pending | Resolve the production import/dispatch closure | Actual call sites distinguish live runtime dependencies from optional wrappers and test-only seams |
| R3 | pending | Classify all tests | Production regressions stay; debug tests move to `dev`; experimental and obsolete tests move to `exp` or delete |
| R4 | pending | Classify all docs | Master keeps current operating/contract records; detailed investigations move to `dev`; scratch and stale prompts leave master |
| R5 | pending | Classify `bench/`, `docs/artifacts/`, and generated evidence | Runtime fixtures and canonical baselines stay; raw or replay evidence moves off master |
| R6 | pending | Classify non-QK `extra/`, root scratch files, and `scratchpad/` | `audit`, `hardware`, `llm`, `remote`, `tools`, and `usbgpu` receive explicit owners |
| R7 | pending | Freeze the classification and cleanup ledgers | Human-readable report and machine-readable inventory reviewed with zero unresolved removals |
| R8 | pending | Reorganize the broad `exp` surface | Production, debug, and live experimental assets are coherent; confirmed dead assets are removed |
| R9 | pending | Promote live production code from `extra/` on `exp` | Production runtime has no fork-owned default-path implementation in `extra/`; focused tests pass |
| R10 | pending | Derive `dev` from the audited `exp` result | Experimental-only assets are absent; production and owned debug assets remain |
| R11 | pending | Derive `master` from the qualified `dev` result | Debug-only assets are absent; the diff contains only ledger-authorized production placement changes |
| R12 | pending | Verify correctness, boundaries, budget, and links on all tiers | Tier-specific audits, tests, import checks, doc links, and `sz.py` pass |
| R13 | pending | Publish and close the migration | Clean pushed branches, final counts, recovery map, and scope moved to output |

Status values are `pending`, `in progress`, `blocked`, and `completed`. A row changes only when its stated completion
evidence exists; partial exploration does not count as completion.

### 16.5 Required outputs

- `docs/task_workflow/output/production-reorganization-inventory-20260728.json`
- `docs/task_workflow/output/production-reorganization-report-20260728.md`
- `docs/task_workflow/output/production-reorganization-cleanup-ledger-20260728.json`
- An expanded production-boundary manifest and audit covering all fork-owned surfaces, not only `extra/qk`.
- Before/after tracked-file, authored-LOC, test, document, and artifact counts per branch.
- A recovery map from every deleted path to its last retaining commit or archive tag.

### 16.6 Execution order and branch safety

1. Move active audit execution to the `exp` worktree and complete R1-R7 there without deleting unresolved assets.
2. On `exp`, promote required runtime code into `tinygrad/`, organize live experimental/debug assets, and delete only
   confirmed dead assets. Run the broadest-tier gates.
3. Apply the shared production/debug commits to `dev`, then prune every `exp`-only asset listed in the frozen ledger.
   Run the debug/qualification gates on the exact `dev` commit.
4. Apply the shared production commits to `master`, then prune every `dev`-only asset listed in the frozen ledger.
   Run the production gates on the exact `master` commit.
5. Never merge a richer tier wholesale into a cleaner tier. Use reviewed bounded commits or destination-based patches
   so `exp`-only files cannot enter `dev` and debug-only files cannot enter `master`.
6. Do not propagate cleaner-tier deletion commits backward into a richer tier that owns the removed files.
7. Run R12, push all three clean branches, record their identities and counts, then complete R13.

### 16.7 Reorganization completion gates

The repository reorganization is complete only when:

- Every fork-added and fork-modified current file has one recorded owner and disposition.
- `exp` equals the maintained production + debug + experimental set, `dev` equals production + debug, and `master`
  equals production; confirmed dead assets exist in none of the three branch tips.
- No unresolved file was moved or deleted.
- No fork-authored production runtime path depends on an implementation owned by `extra/`, `dev`, or `exp`.
- Every `extra/` asset retained on `master` is an explicit canonical production authority, runtime fixture, or unchanged
  upstream optional surface with a documented exception.
- Master contains no root scratch scripts, raw benchmark dumps, stale execution prompts, completed handoffs, or
  one-off diagnostic probes.
- Every test retained on master names the shipped behavior or policy boundary it protects.
- Every document retained on master is current, canonical, and linked from a maintained index or owner.
- The expanded organization audit passes with complete manifest coverage and no hard errors.
- Production tests and supported CPU/Metal/AMD smoke gates pass at the exact clean commit, subject to available hardware.
- `sz.py` passes and the final report quantifies the reduction rather than asserting cleanliness qualitatively.

### 16.8 Low-effort agent execution protocol

Preferred worker profile: Luna with low reasoning effort. If the active agent runtime does not expose a Luna model
override, use the available low-effort worker model and record that substitution in the handoff. Model availability
must not change the evidence contract or authorize weaker classification.

Agents work read-only from `/Users/julianabeleda/env/tinygrad-arkey-exp` at the exact recorded `exp` commit. During the
census they may run searches, parsers, tests that do not require exclusive hardware, and Git history queries. They may
not edit, move, delete, commit, push, install, run GPU workloads, change branches, or modify generated audit outputs.

Each agent returns findings to the root agent rather than creating a competing inventory file. Every finding uses:

```text
path | owner_branch | category | disposition | destination | consumer_or_reference |
retention_or_recovery | confidence | unresolved_reason
```

Allowed `owner_branch` values are `master`, `dev`, `exp`, and `delete`. Allowed dispositions are `retain`, `promote`,
`move`, `consolidate`, `delete`, and `unresolved`. An agent must use `unresolved` rather than infer intent from a name,
directory, age, lack of imports, or a failed test.

#### Packet A: runtime and tooling boundary

Owned paths:

- Current fork-added or fork-modified `tinygrad/**` and `extra/**` files.
- Root executable scripts and `scratchpad/**` only when needed to resolve a runtime/tooling reference.

Required questions:

- Which `extra` modules are actually executed by production rather than merely exposed by a lazy wrapper?
- Which live production modules must move under `tinygrad`, and to what domain owner?
- Which diagnostics and hardware tools belong on `dev`, and which unproven probes belong on `exp`?
- Which assets are superseded or dead with a banked conclusion and recoverable commit?
- What owns `extra/qk/decode/capture_prefill_compile.py`, the current audit coverage drift?

Explicit seams to resolve include `tinygrad/llm/route_ops.py`, `tinygrad/codegen/experimental.py`,
`tinygrad/llm/__main__.py`, `extra/qk`, `extra/llm`, `extra/audit`, `extra/hardware`, `extra/remote`,
`extra/gpu_fault_analysis`, `extra/tools`, and `extra/usbgpu`.

#### Packet B: tests, benchmarks, and evidence

Owned paths:

- Current fork-added or fork-modified `test/**` files.
- `bench/**`.
- `docs/artifacts/**`.

Required questions:

- Which tests defend a shipped runtime behavior or production policy boundary?
- Which tests are debug reproducers, qualification harnesses, experimental candidates, or tests of retired code?
- Which benchmark files are canonical inputs/baselines versus raw campaign output?
- Which artifacts are runtime fixtures or unique promotion evidence versus replay/debug output?
- Which retained evidence has a current consumer, owner, and bounded retention rule?

Every proposed production test retention must name the shipped behavior and implementation it protects. Every artifact
deletion must identify a compact replacement record or state that no unique conclusion exists.

#### Packet C: documents and repository surface

Owned paths:

- `docs/**` excluding `docs/artifacts/**`.
- `structure/**`, root scratch scripts, `.claude/**`, and `scratchpad/**`.

Required questions:

- Which documents are current production operating instructions, contracts, or compact findings ledgers?
- Which are completed handoffs, stale execution prompts, duplicated scopes, detailed debug investigations, or scratch?
- Which documents must move to `dev`, which unfinished experimental records belong on `exp`, and which are delete-ready?
- Which retained master documents are reachable from a maintained index or explicit owner?
- Which path references would break under each proposed move or deletion?

Task-workflow inputs remain on master only while genuinely open. Completed scopes move to output, and detailed closed
investigations still move to `dev` unless production operation requires them.

#### Root reconciliation

The root agent owns paths not delegated, resolves cross-packet conflicts, and produces the R7 inventory and cleanup
ledger. It must verify:

- Packet path sets cover every current fork-added and fork-modified file exactly once, with documented exclusions.
- A production consumer overrides a lower-tier proposal until the consumer is promoted or removed safely.
- Test and document dispositions follow the final runtime disposition, not an earlier guess.
- Counts reconcile with Git, `sz.py`, and the expanded audit.
- No removal begins while any cross-packet dependency or unresolved owner remains.

Agent handoffs are evidence inputs, not authority. Only the reconciled, committed R7 ledgers authorize R8-R11 edits.

### 16.9 Agent census checkpoint

Checkpoint date: 2026-07-28. Worker commit: `b57362ca6` on `exp`.

Luna was not exposed by the active runtime, so all packets used the documented available low-effort worker fallback.
Every worker remained read-only and reported a clean worktree; no GPU, generated-output, branch, commit, or push action
occurred inside a packet.

| Packet | Status | Paths covered | Principal result |
|---|---|---:|---|
| A: runtime/tooling | completed | 393 | 213 `tinygrad` and 180 `extra` paths; resolved live calls behind the three production seams and classified the missing QK audit file |
| B: tests/evidence | completed | 380 | 164 tests, 43 bench files, and 173 artifact files; 72 tests and 102 staged/frozen artifacts remain dependent on Packet A dispositions |
| C: docs/surface | completed | 276 | Document and repository-surface groups classified; maintained-index drift and broken retained links identified |

The packet sets cover 1,049 assigned paths. Root-owned repository metadata, path-set reconciliation, and all cross-packet
dependencies still require R1/R7 work. High-confidence raw evidence and scratch deletion candidates were identified,
but **zero removals are authorized** until the canonical inventory records their compact replacement and recovery path.

### 16.10 First production-closure checkpoint

Checkpoint date: 2026-07-28. The first bounded runtime slice is complete on `exp` and is eligible for selective
propagation after its commit is verified:

| Source | Destination | Production consumers | Boundary change |
|---|---|---|---|
| `extra/qk/prefill/flash_prefill_attention_spec.py` | `tinygrad/schedule/wmma/flash_prefill.py` | `tinygrad/llm/fused_attention.py:162`; `tinygrad/codegen/opt/postrange.py:602` | Both callers import the core descriptor directly; the former `route_ops.py` and `codegen/experimental.py` lazy shims were removed |

The descriptor contents were preserved as a rename, active path references and route-manifest authority metadata were
updated, and the generated organization census plus pure-machine-search census were refreshed. CPU verification
completed for `test/unit/test_tinygrad_boundary.py`, descriptor validation/JSON, and Python compilation of the moved
module and callers. A broader semantic/residency batch had 32 passes, 8 skips, and 12 CPU semantic/cycle or barrier
assertion failures; it produced no missing-module or import failure attributable to this move. The scoped organization
audit now reports `ORG_R1_PASS_CENSUS_PINNED` with 94 explicit records, zero hard errors, and 67 warnings after the
compile reproducer received an evidence-based dev diagnostic record; its deletion remains blocked until its conclusion
is banked.

This checkpoint does not authorize broader pruning. The remaining R1-R7 ledgers, generated evidence ownership, and
cross-packet conflicts must be reconciled before deleting or tier-pruning additional assets.

### 16.11 Second production-closure checkpoint

The second bounded runtime slice is complete on `exp` and is eligible for selective propagation after its commit is
verified:

| Source | Destination | Production consumer | Boundary change |
|---|---|---|---|
| `extra/qk/codegen_recurrence_unroll.py` | `tinygrad/codegen/late/recurrence.py` | `tinygrad/codegen/__init__.py:89-91` when `SCHED_UNROLL>1` on AMD | The compiler imports and calls the core owner directly; the recurrence forwarding shim was removed from `codegen/experimental.py` |

The transform body was preserved as a byte-identical rename. The organization manifest and lowering findings were
updated, and `SCHED_UNROLL_DEBUG` was explicitly classified as a non-cache-affecting diagnostic gate. The focused
CPU suite (`test/unit/test_codegen_recurrence_unroll.py`, flash-prefill/boundary/gate/cache/audit tests) passed 61
tests. Coverage includes identity and fail-closed cases, canonical AFTER-chain carry reconstruction, nested range
and private reinitialization-register duplication, and AMD dispatch. No AMD execution or GPU numeric result is claimed.

This slice also does not authorize broader pruning. Subsequent codegen, quant, decode, route-policy, and packed-WMMA
slices still require their own tests and evidence closures.

### 16.12 Third production-closure checkpoint

The register-store devectorizer is now centralized without merging away its distinct behavior:

| Source | Destination | Production consumer | Boundary change |
|---|---|---|---|
| `extra/qk/reg_store_devec.py` | `tinygrad/codegen/late/reg_store.py` | `tinygrad/codegen/__init__.py:264` after `pm_distinct_reg_store_devec` on AMD/coalesced-load paths | The separate `pm_reg_store_devec` rule was ported into the core owner; the extra module and `codegen/experimental.py` forwarding shim were removed |

The duplicate-pointer residual behavior is preserved as a separate matcher, and malformed targets/value-width
mismatches fail closed. Seven focused tests and the existing reg-store/coalesced-load regression set pass. The
organization manifest now records the core owner. The complete matrix and acceptance gates are recorded in
`docs/task_workflow/output/reg-store-devec-test-scope-20260728.md`.

The lowering baseline authority still requires `llvm-readelf`, which is unavailable in this environment, and the
checked-in lowering fingerprint differs from the current snapshot. Those authority checks remain a verification gap;
they do not indicate a matcher failure.

The open authority work is tracked as the input task
`docs/task_workflow/input/reg-store-devec-amd-authority-verification-scope-20260728.md`. It remains an input until the
clean-commit baseline and fingerprint checks are rerun with the required LLVM tooling.

### 16.13 Fourth production-closure checkpoint

The narrow AMD `fdot2` primitive hook is now centralized:

| Source | Destination | Production consumer | Boundary change |
|---|---|---|---|
| `extra/qk/fdot2_lowering.py` | `tinygrad/codegen/late/fdot2.py` | `tinygrad/codegen/__init__.py` at the two graph-rewrite hooks and the post-linearization hook; `tinygrad/codegen/opt/gemm_consumer.py` | Core imports are direct; all three `codegen/experimental.py` fdot2 forwarding shims and the old extra module were removed |

The matcher remains default-off and AMD-only through `V_DOT2_LOWERING`. Its exact scalar-f32 output contract, optional
accumulator ordering, fail-closed controls, and linearized dependency replacement are pinned by
`test/unit/test_fdot2_lowering.py`; the focused fdot2/GEMM suite passes 15 tests. This is primitive exposure only and
does not claim AMD execution, ISA availability, or a speedup. The lowering baseline/fingerprint authority follow-up
remains open under `docs/task_workflow/input/reg-store-devec-amd-authority-verification-scope-20260728.md`.

### 16.14 Fifth production-closure checkpoint

The opt-in latency-aware list scheduler is now centralized:

| Source | Destination | Production consumer | Boundary change |
|---|---|---|---|
| `extra/qk/codegen_list_scheduler.py` | `tinygrad/codegen/late/list_scheduler.py` | `tinygrad/codegen/late/linearizer.py` under `SCHED_LIST`, plus its structural-boundary probes | Core imports are direct; the `list_schedule` and `structural_ops` forwarding shims were removed from `codegen/experimental.py` |

The scheduler still only reorders within straight-line blocks, preserves topological dependencies and structural
boundaries, and remains default-off. Four focused CPU tests cover latency-shadow ordering, structural boundaries,
the structural inventory, and old-boundary absence. No performance or AMD execution claim is made.

### 16.15 Sixth production-closure checkpoint

The AMD warp primitive and auto-lowering pair is now consolidated in one core owner:

| Source | Destination | Production consumer | Boundary change |
|---|---|---|---|
| `extra/qk/amd_warp_reduce.py` and `extra/qk/warp_reduce_lowering.py` | `tinygrad/codegen/late/warp_reduce.py` | `tinygrad/codegen/__init__.py` under `WARP_REDUCE_LOWERING`, plus existing decode/GEMV/MMQ emitters | Core owns the hand-built shuffle/reduction API and matcher; the experimental `warp_reduce_pm` shim and both old extra modules were removed |

The matcher still lowers only scalar float ADD/MAX reductions over power-of-two WARP/GROUP_REDUCE axes, and remains
default-off. Four CPU structural/boundary tests pass. AMD execution, ISA output, and performance remain outside this
slice's claim.

### 16.16 Seventh production-closure checkpoint

The duplicated Q4/Q6 option parser is now centralized without moving their kernel builders:

| Source | Destination | Production consumer | Boundary change |
|---|---|---|---|
| `extra/qk/quant/q4_k_gemv_primitive.py:parse_opt`, `extra/qk/quant/q6_k_gemv_primitive.py:parse_opt` | `tinygrad/codegen/opt.parse_opt` | `tinygrad/llm/prefill_routes.py`, `tinygrad/llm/qk_primitives.py` | `tinygrad/llm/route_ops.py` parse shims removed; quant kernel builders and mixed layout tooling remain in `extra` for later route-specific slices |

Seven CPU parser/boundary tests pass. Full Q4/Q6 builder promotion is intentionally deferred because builder callers,
Q6 coverage, and layout ownership are not yet closed.
