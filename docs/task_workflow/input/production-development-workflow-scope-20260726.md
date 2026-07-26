# Production and Development Workflow Scope

Date: 2026-07-26

Repository: `/home/ubuntu/tinygrad-arkey`

Status: input

## 1. Objective

Make `master` the production branch and establish a separate `dev` integration branch in its own worktree.

The production branch must contain the shipped hot path plus the minimum tests, authorities, manifests, and operating documentation required to prove and maintain that path. Experimental work must happen in isolated feature worktrees, pass through `dev`, and reach `master` only after it satisfies an explicit promotion contract.

Target pipeline:

```text
feature worktree -> dev integration -> master production
```

This task is about repository workflow and enforcement. It is not permission for an unbounded code rewrite or bulk deletion.

## 2. Motivation

Multiple agents recently shared `/home/ubuntu/tinygrad-arkey` while one process ran a long GPU benchmark. A branch checkout, edits to `tinygrad/uop/ops.py`, a stash cycle, and unlocked GPU work changed the source and device state underneath benchmark subprocesses. The results lost code-identity and GPU-ownership authority.

The incident exposed two separate requirements:

- Git branches alone do not isolate files. Concurrent work requires separate worktree directories.
- Worktrees do not isolate the GPU. Every GPU command still requires `/tmp/gpu-bench.lock`.

The repository also contains production paths, active integration candidates, reusable authorities, one-off probes, historical experiments, and raw artifacts in overlapping locations. A production/development boundary should make promotion deliberate and cleanup routine.

## 3. Non-goals

- Do not rewrite Git history.
- Do not force-push `master` or `dev`.
- Do not create a second permanent laboratory repository in this task.
- Do not delete an active agent's worktree, branch, stash, or uncommitted files.
- Do not classify all tests or benchmark authorities as non-production merely because they are not imported at runtime.
- Do not merge a development branch containing known-unready work into production.
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

### 4.2 Development integration

`dev` is the integration branch. It contains candidates that are intended for the next production promotion and are ready for cross-feature integration testing.

`dev` is not a permanent archive. Rejected work remains recoverable from Git history, while durable conclusions move into a compact ledger.

### 4.3 Feature work

Each active task uses one branch and one dedicated worktree. Suggested names:

```text
feature/decode-ctx128
feature/lowering-refactor
feature/prefill-search
hotfix/<bounded-production-fix>
```

Unproven work stays on its feature branch. It does not enter `dev` merely because it compiles.

## 5. Required topology

| Purpose | Branch | Worktree |
|---|---|---|
| Production | `master` | `/home/ubuntu/tinygrad-arkey` |
| Integration | `dev` | `/home/ubuntu/tinygrad-arkey-dev` |
| Feature | `feature/*` | `/home/ubuntu/worktrees/<task-name>` |
| Emergency production repair | `hotfix/*` from `master` | dedicated worktree |

The directory names may be adjusted if occupied, but production and development must never share a directory.

## 6. Invariants

1. No experiment, refactor, stash cycle, or branch checkout occurs in the production worktree.
2. `master` advances only through a reviewed promotion or hotfix.
3. Feature branches fork from current `dev` unless the work is an emergency hotfix or a production-only documentation correction.
4. `dev` receives only candidates intended for the next promotion window.
5. A failed candidate is reverted or removed from `dev` before `dev` can promote.
6. Every GPU command from every worktree acquires `/tmp/gpu-bench.lock`.
7. Every published benchmark records commit, worktree, dirty state, command, model identity, route identity, artifact path, power state, and lock ownership.
8. A task-specific probe is deleted after its evidence is promoted unless it becomes an owned reusable diagnostic.
9. Recovery from Git history is acceptable for retired experiments; dead code does not remain in production solely to make recovery convenient.
10. Production tests and authorities may remain even when they are not runtime-imported, provided they defend a shipped path and have a named contract.

## 7. Promotion contract

A candidate may enter `dev` only when it has:

- A named production problem or capability target.
- A reachable consumer or an explicit integration point.
- Focused correctness evidence with a positive control.
- No known destructive filesystem or GPU behavior.
- An owner, bounded scope, and removal condition for temporary code.
- Task-specific probes separated from reusable implementation code.

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
- A clean `dev` worktree and a promotion diff containing no unrelated candidate.

## 8. Promotion mechanics

1. Feature work remains isolated until its focused gates pass.
2. The feature branch is reviewed and merged into `dev`.
3. Integration authorities run on the exact `dev` commit proposed for production.
4. Dead probes and temporary artifacts are removed on `dev` before promotion.
5. A promotion PR or bounded merge advances `master` from the reviewed `dev` state.
6. The promoted commit receives a durable findings or current-state update.
7. The feature worktree and branch are removed after the production commit is pushed.
8. `dev` is synchronized to the new production tip before accepting the next candidate batch.

Do not merge `dev` wholesale when it contains unrelated or unready work. Revert the unready work first, or construct a bounded promotion branch from `master` containing only approved commits.

## 9. Hotfix mechanics

1. Create `hotfix/<name>` from the current production commit in a dedicated worktree.
2. Reproduce the production defect with the smallest authoritative test.
3. Implement and validate only the bounded repair.
4. Promote directly to `master` after review.
5. Merge or cherry-pick the production hotfix back into `dev` immediately.
6. Delete the hotfix branch and worktree after both branches contain the repair.

## 10. Execution phases

### Phase 0: Safety census

Record without modifying state:

- Current `master` and remote commit.
- Existing branches, worktrees, stashes, dirty files, and active agents.
- Active GPU processes, lock ownership, and power profile.
- Unpushed commits and branches that another actor owns.
- Current branch-protection and CI behavior.

Stop if any planned path is already owned or if an unexpected process is using the GPU without the lock. Do not clean another actor's state.

Deliverable: `docs/task_workflow/output/production-development-workflow-phase0-20260726.md`.

### Phase 1: Establish branch and worktree topology

- Create `dev` from the selected clean production commit.
- Create `/home/ubuntu/tinygrad-arkey-dev` with `git worktree add`.
- Add a concise operator document with canonical commands for creating and retiring feature worktrees.
- Document that `git checkout` in the production worktree is forbidden for ordinary development.
- Add a positive-control command that prints branch, commit, worktree path, and dirty state before a benchmark.

Do not move existing active feature branches during this phase.

### Phase 2: Production-boundary inventory

Classify authored files using importer reachability, manifest ownership, authority ownership, and active-task evidence.

Required categories:

- `production_runtime`
- `production_authority`
- `production_regression_test`
- `production_manifest_or_ledger`
- `development_candidate`
- `task_probe`
- `historical_or_dead`
- `unresolved`

Every non-runtime production asset must state the shipped behavior it defends. Every deletion candidate must state its last importer, last use, recovery commit, and whether its conclusion was banked.

Do not infer purpose solely from location under `extra/`, `test/`, `docs/`, or `bench/`.

### Phase 3: Add enforceable policy

Create or extend a repository audit that fails when:

- Production runtime imports a development-only module.
- A production route points to a missing authority or manifest entry.
- A task probe is classified as a production dependency.
- A promoted candidate lacks its required correctness or route-identity record.
- A production benchmark dispatch target is missing.
- A retained non-runtime asset has no owner or defended behavior.
- A branch-specific workflow file attempts to make `dev` content a runtime dependency of `master`.

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

If GitHub branch protection cannot be changed from the environment, produce exact operator steps and mark the external action pending. Do not claim protection is enabled without a positive check.

### Phase 5: Initial cleanup and placement

Use the Phase 2 inventory to perform a bounded first cleanup:

- Delete confirmed dead task probes after banking unique conclusions.
- Fold redundant benchmark output into the canonical ledger.
- Promote genuinely reusable utilities into an owned stable location.
- Keep production regression tests and canonical authorities on `master`.
- Move active work into feature branches rather than leaving it dirty on `master`.
- Remove completed `docs/task_workflow/input` and `in_progress` files from production after their output is banked.

No item in `unresolved` may be deleted in this phase.

### Phase 6: Pilot promotion

Use one small, low-risk change to exercise the complete pipeline:

```text
feature worktree -> dev -> integration gates -> master -> branch/worktree cleanup
```

The pilot must prove:

- Filesystem isolation between production and development.
- GPU lock ownership if the pilot uses the GPU.
- Exact commit identity in artifacts.
- Required CI and audit behavior.
- Clean feature-branch retirement.
- Synchronization of `dev` after production promotion.

Do not choose a lowering refactor, GPU fault repair, or performance-sensitive kernel as the pilot.

### Phase 7: Close bootstrap state

- Publish the final workflow document and production-boundary manifest.
- Record the pilot evidence and any external protection action still pending.
- Delete temporary migration probes and duplicate scope documents.
- Remove this input document from production after its durable output is accepted.
- Record baseline LOC and authored-file counts for production, authorities, development candidates, and retired assets.

## 11. Required deliverables

- `docs/production-development-workflow.md`
- A machine-readable production-boundary ownership manifest.
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

- Creating `dev` would overwrite or strand an existing branch.
- An existing worktree is owned by an active actor.
- `master` contains unpushed commits whose ownership is unclear.
- A proposed deletion has a production importer or unresolved route-manifest reference.
- `dev` contains unrelated candidates at promotion time.
- Branch protection requires credentials or settings unavailable to the agent.
- A benchmark would run without the GPU lock or against a mutable source directory.
- The migration requires history rewriting or a force push.

## 14. Completion criteria

The task is complete only when:

- `master` is documented and enforced as the production branch.
- `dev` exists in a separate worktree and is documented as integration-only.
- Feature worktree creation and retirement are reproducible.
- The promotion and hotfix paths have both documented commands and enforced gates.
- Production imports no development-only asset.
- Every retained non-runtime production asset has an owner and defended behavior.
- Confirmed dead probes are removed and recoverable from recorded commits.
- One low-risk feature completes the full promotion pipeline.
- The production worktree remains unchanged during development work.
- GPU evidence records exclusive lock ownership.
- The final production-boundary and LOC reports are retained.
- This bootstrap input is retired after its durable output is accepted.

## 15. Expected outcome

The repository should end with a small, stable production surface that is still provable, a bounded integration branch that contains only promotion candidates, and disposable feature worktrees for active investigation. Successful work moves forward through explicit gates. Failed work leaves durable conclusions and Git history, not permanent dead scripts in production.
