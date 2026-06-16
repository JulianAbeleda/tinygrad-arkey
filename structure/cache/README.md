# cache

Optional token-saving context layer for agents working in a repo.

Use this folder when a project is large enough that agents repeatedly spend time rediscovering the same stable facts.

Read this folder after `structure/INDEX.md` and `structure/Purpose/` when the task requires development context but does not yet require broad source inspection.

## Files

- `repo-cache.md` - compact human-maintained project briefing
- `repo-map.md` - compact file and ownership map
- `module-notes/` - optional deeper notes loaded only when a task needs that area

## Rules

- Keep this layer concise enough to read at the start of development work.
- Record stable repo facts, not transient session notes.
- Prefer pointers to owning files over duplicated explanations.
- Keep module notes one-topic-per-file.
- Do not store secrets, copied chats, runtime memory, local logs, or user-private session state.
- Update cache files when command routing, public behavior, architecture boundaries, data contracts, verification commands, or important ownership maps change.

## Portability

This folder should be safe to copy between projects.

Replace placeholder content with project-specific facts, but keep the file roles stable:

- `repo-cache.md` answers "what should an agent know before reading source?"
- `repo-map.md` answers "which files own which behavior?"
- `module-notes/` answers "what deeper context should be loaded only for this task?"

