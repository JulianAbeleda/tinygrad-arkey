# Production, Debug, and Experimental Workflow Scope

Date: 2026-07-26
Updated: 2026-07-28

Repository: `/Users/julianabeleda/env/tinygrad-arkey`

Status: open; `master` is production, but the persistent `dev` and `exp` branches/worktrees and selective promotion
pipeline have not been established

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
12. After a production promotion or hotfix, `master` is synchronized downward into `dev`, then the updated `dev` is
    synchronized into `exp`, without pulling debug or experimental assets upward.

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
9. The new `master` tip is synchronized downward into `dev` and then `exp`.
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
- Downward synchronization of the new `master` tip into `dev` and `exp` after promotion.

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
- The promoted `master` commit is synchronized downward into `dev` and `exp`.
- The final production-boundary and LOC reports are retained.
- This bootstrap input is retired after its durable output is accepted.

## 15. Expected outcome

The repository should end with three distinct working surfaces: a small, stable, and provable `master`; a durable
`dev` environment for debugging, reproduction, and qualification; and an `exp` environment for disposable experiments.
Useful work moves forward through selective gates, while production fixes synchronize downward. Failed work leaves
durable conclusions and Git history, not permanent dead scripts in any active branch.
