# Research worktree cleanup scope (2026-07-27)

## Objective

Reduce the tinygrad development surface to the production master checkout while preserving a recoverable record of
all research branch tips and all dirty worktree contents. Cleanup is complete when master remains usable, one pushed
archive ref reaches every removed committed tip, dirty state has been committed before removal, and the disposition
of every unregistered directory is recorded.

## Non-goals

- Do not change production code during cleanup.
- Do not reinterpret or promote abandoned experiments.
- Do not run GPU benchmarks.
- Do not delete repositories unrelated to `/home/ubuntu/tinygrad-arkey`.
- Do not claim that archived research is merged into master.

## Keep set

- `/home/ubuntu/tinygrad-arkey` on `master`.
- `origin/master`.
- The untracked hardware-counter task input, after content review and normal documentation commit.
- One remote archive ref: `archive/research-worktrees-20260727`.
- This scope, moved to `docs/task_workflow/output/` with the final ledger.

## Inputs

- All refs under `refs/heads/` except `master`.
- All registered worktrees reported by `git worktree list --porcelain`.
- Detached worktree tips, which are not protected by a local branch.
- Dirty and untracked state in registered worktrees.
- Unregistered directories under `/home/ubuntu/worktrees/` that look campaign-related.

## Preservation protocol

1. Review the master-only untracked task before tracking it.
2. For each dirty registered worktree, create one clearly named archival commit containing tracked modifications and
   untracked files. These commits are preservation records, not promotion claims.
3. Build one archive commit whose parents include master, every research branch tip, and every detached worktree tip.
   Its tree is master; parent reachability, not a synthetic file merge, is the preservation mechanism.
4. Push the archive commit to `origin/archive/research-worktrees-20260727`.
5. Prove every branch and detached tip is an ancestor of the pushed archive ref.
6. Only after that proof, remove registered worktrees and delete their local branches.
7. Delete remote feature branches only after their exact tips are reachable from the pushed archive ref.

## Dirty-worktree contract

The following worktrees were dirty at scope creation and must not be force-removed:

- `feature/14b-decode-ctx128-and-depth-decay`: modified G5 decode files plus KFD scope and qgroup timing utility.
- `fix/14b-short-prefill-vector-store`: modified launch/PMC runtime files plus untracked decode probes.
- `luna-tinygrad-worker-smoke`: modified benchmark logs/status artifacts.

Each must become clean through an archival commit. `git reset`, checkout-based reversion, and forced worktree removal
are prohibited.

## Unregistered-directory contract

- `/home/ubuntu/worktrees/tinygrad-g5-qgroup-parity` may be removed only if its campaign-owned source deliverables are
  present on master and any remaining unique files are either obsolete probes or preserved separately.
- `/home/ubuntu/worktrees/llama-fattn64-ablation` is not a tinygrad worktree and remains untouched unless a separate
  llama.cpp cleanup scope owns it.
- Unknown or unexpectedly large unique content stops deletion of that directory; it does not block cleanup of proven
  registered tinygrad worktrees.

## Deletion protocol

- Use `git worktree remove` for registered worktrees after they are clean and archived.
- Use normal filesystem deletion only for an unregistered directory that passes the contract above.
- Delete local research branches only after archive reachability succeeds.
- Delete the two existing remote feature branches only after archive reachability succeeds.
- Run `git worktree prune` after registered removals.
- Never delete `master` or rewrite its history.

## Completion evidence

The output ledger must record:

- Archive branch name and commit SHA.
- Pushed remote confirmation.
- Archived branch and detached-tip count.
- Archival commits created for dirty worktrees.
- Removed worktree paths and deleted local/remote branches.
- Retained directories and the reason each was retained.
- Final master status, including any intentionally retained untracked paths.

## Stop conditions

- A dirty worktree cannot be committed normally.
- Archive push fails.
- Any removed-tip candidate is not an ancestor of the pushed archive ref.
- Unexpected source changes appear in master.
- An unregistered directory contains unique content whose ownership cannot be established.

# Execution result

## Verdict

Cleanup completed under the preservation-first contract. Production remains `/home/ubuntu/tinygrad-arkey` on master.
All removed committed work is reachable from a pushed archive branch; no dirty registered worktree was force-removed.

## Archive authority

- Branch: `archive/research-worktrees-20260727`
- Commit: `4c562900627cbca3aae86ef3e7fd099f8b448016`
- Remote: `origin/archive/research-worktrees-20260727`
- Unique parent tips checked: 34, including master, every research branch tip, and both detached worktree tips.
- Reachability proof: every captured tip passed `git merge-base --is-ancestor <tip> <archive>` and the remote SHA matched the local archive SHA.
- The archive tree includes `docs/archive/research-worktrees-20260727.md`, mapping archived branch names to exact commits.

## Dirty state preserved

| former branch | archival commit | preserved content |
|---|---|---|
| `feature/14b-decode-ctx128-and-depth-decay` | `4d4600c72` | G5 decode/QG work, KFD observability scope, qgroup timing utility |
| `luna-tinygrad-worker-smoke` | `1476f5fb4` | worker smoke logs and status artifacts |
| `fix/14b-short-prefill-vector-store` | `c9c19090b` | launch observer changes, PMC/resource probes, privileged launcher |

These commits are archive evidence, not production promotion claims.

## Removed

- 33 registered non-master worktrees.
- 31 local research branches.
- Remote branches `feature/14b-decode-ctx128-and-depth-decay` and `fix/14b-short-prefill-vector-store`.
- Detached historical checkout `/home/ubuntu/tinygrad-0716`; its exact tip remains an archive parent.
- Superseded orphan `/home/ubuntu/worktrees/tinygrad-g5-qgroup-parity` (7.7 GB, of which 7.5 GB was `.venv`).

The orphan's production code was already promoted in master commit `8deca39bb`. Its sole candidate-only document was an
obsolete pre-width4 negative result superseded by
`docs/task_workflow/output/14b-decode-attention-isa-cadence-scope-20260727.md`, so it was intentionally discarded.

## Retained

- `/home/ubuntu/tinygrad-arkey` and `master`.
- `archive/research-worktrees-20260727`, locally and on GitHub.
- `docs/task_workflow/input/14b-decode-hardware-counter-integration-scope-20260726.md`, now tracked as a valid open task.
- `/home/ubuntu/worktrees/llama-fattn64-ablation` (1.2 GB): separate llama.cpp repository, not owned by this scope.
- BoltBeam worktrees: separate repository, not owned by this scope.

## Recovery

To recover any removed branch, read `docs/archive/research-worktrees-20260727.md` from the archive branch and create a
new branch at the recorded SHA. Do not merge the archive branch into master: its extra parents exist only to retain
research history.
