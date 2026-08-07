# Agent attribution provenance record

Date: 2026-08-06
Branch: `nvidia-bringup-20260731`
Status: **provenance record. Documents the commit-attribution conventions used across
this campaign and normalizes them going forward. No history rewritten; no commit hashes
changed; every hash cited elsewhere in `docs/` remains valid.**

## 1. Why this record exists

Two coding agents contributed to this repository, and until 2026-08-06 they used
**incompatible attribution conventions**. The result is that standard git tooling
(`git shortlog`, `git log --author`, host contributor graphs) reports one agent and is
blind to the other, understating total agent contribution by roughly 2x and misattributing
the accountable author on a subset of commits.

This record states what actually happened so the log is decodable without it.

## 2. Measured state as of 2026-08-06

Counts are over all branches, `git log --all`, at HEAD `afbe8fbfa`.

### 2.1 Claude - attributed via `Co-Authored-By` trailer

Author field carries the human (`Julian Abeleda`); the agent and its model version are
recorded in a trailer. **Invisible to `git shortlog` and `git log --author`.**

| Trailer identity | Commits | First | Last |
| --- | ---: | --- | --- |
| `Claude Opus 4.8 (1M context)` | 1176 | 2026-06-14 | 2026-07-24 |
| `Claude Opus 4.8` | 72 | 2026-06-20 | 2026-07-05 |
| `Claude Opus 5 (1M context)` | 204 | 2026-07-24 | 2026-08-02 |
| `Claude Fable 5` | 31 | 2026-06-10 | 2026-07-04 |
| **Total** | **1512** | **2026-06-10** | **2026-08-02** |

### 2.2 Codex - attributed via `author` field

Author field carries the agent (`Codex <codex@local>`); no trailer. **Counted by
`git shortlog`; the accountable human does not appear on these commits.**

| Author identity | Commits | First | Last |
| --- | ---: | --- | --- |
| `Codex <codex@local>` | 678 | 2026-07-23 | 2026-08-06 |

### 2.3 Consequence for the author field

`Julian Abeleda` carries 3729 commits (2026-05-18 -> 2026-08-01). That figure is **not**
hand-authored volume: approximately 1512 of those commits carry a Claude trailer. The
author field on this repository means "accountable human," not "typed by hand," and should
not be read as a productivity measure.

## 3. Timeline

```text
2026-05-18  campaign begins (AMD)
2026-06-10  Claude Fable 5 first co-authored commit
2026-06-14  Claude Opus 4.8 becomes primary
2026-07-23  Codex enters, claiming the author field
2026-07-24  Claude Opus 4.8 -> Opus 5 handover
2026-07-31  nvidia-bringup-20260731 branched
2026-08-02  Claude last co-authored commit
2026-08-06  convention normalized (this record)
```

## 4. Normalized convention (effective 2026-08-06)

1. **`author` is always the accountable human**: `Julian Abeleda
   <julian.abeleda@gmail.com>`. Repo-local `user.name` / `user.email` set accordingly on
   this date; the prior value was `Codex <codex@local>`.
2. **Every contributing agent emits a `Co-Authored-By` trailer**, including its model
   version where the agent exposes one:
   - `Co-Authored-By: Claude <model> <noreply@anthropic.com>`
   - `Co-Authored-By: Codex <codex@local>`
3. Trailers are additive. A commit with contributions from both agents carries both.

## 5. What was deliberately NOT done

**No history rewrite.** `git filter-repo` on the 678 Codex-authored commits would change
every hash from 2026-07-23 forward. This repository's methodology cites commit hashes as
authority across `docs/` (for example `6f1abd047`, `3f0784e7b`, `3fa55377c`, `2794d6772`),
so rewriting would invalidate the provenance chain the citations exist to provide. The
inconsistency is dated and documented here instead; that is strictly cheaper and loses no
information.

**No `.mailmap`.** A mailmap would remap the `Codex` author identity for display, which
collapses it into the human rather than preserving it as a distinct contributor. That is
the opposite of the intent: both agents should remain attributable.

## 6. How to count contribution correctly

`git shortlog -sn` is wrong on this repository. Use:

```sh
# Codex (author field)
git log --all --author="Codex" --oneline | wc -l

# Claude, all model versions (trailer)
git log --all --grep="Co-Authored-By: Claude" --oneline | wc -l

# Claude, by model version
git log --all --grep="Co-Authored-By: Claude Opus 5" --oneline | wc -l
```

## 7. Note for later analysis

The trailers give dated, model-versioned provenance. Booked recovery per commit is
therefore separable by model era (Opus 4.8: 2026-06-14 -> 2026-07-24; Opus 5: 2026-07-24 ->
2026-08-02; Codex solo: 2026-08-02 -> present). This has not been analyzed and is recorded
only as an available measurement, not a claim.
