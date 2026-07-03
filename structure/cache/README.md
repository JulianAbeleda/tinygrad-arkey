# cache

Optional token-saving context layer for agents working in a repo.

Use this folder when a project is large enough that agents repeatedly spend time rediscovering the same stable facts.

Read this folder after `structure/INDEX.md` and `structure/Purpose/` when the task requires development context but does not yet require broad source inspection.

## Files

- `module-notes/` - optional deeper notes loaded only when a task needs that area

The old `repo-cache.md` / `repo-map.md` files were removed in the 2026-07-03 maintainability cleanup because they
duplicated stale repo maps. Use `../../README.md`, `../../docs/README.md`, and BoltBeam ledgers for current
navigation.

## Rules

- Keep this layer concise enough to read at the start of development work.
- Record stable repo facts, not transient session notes.
- Prefer pointers to owning files over duplicated explanations.
- Keep module notes one-topic-per-file.
- Do not store secrets, copied chats, runtime memory, local logs, or user-private session state.
- Update cache files when command routing, public behavior, architecture boundaries, data contracts, verification commands, or important ownership maps change.

## Portability

This folder should be safe to copy between projects.

Replace placeholder content with project-specific facts, but keep the role stable:

- `module-notes/` answers "what deeper context should be loaded only for this task?"
