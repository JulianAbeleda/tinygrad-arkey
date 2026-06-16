# Repo Cache

Last updated: YYYY-MM-DD
Analyzed source: `<folders/files analyzed>`
Scope: stable implementation context for token-saving orientation.

## Project Shape

Describe the project in 3-8 bullets.

Include:

- primary user-facing surfaces
- primary runtime or build system
- important boundaries
- what this repo is not

## Core Facts

Record stable facts an agent usually needs before editing.

Examples:

- language and package manager
- important commands
- primary entrypoints
- state or storage locations
- environment variables by name only
- external services by name only
- generated files or ignored folders

Do not include secret values.

## Main Flow

Summarize the normal execution flow.

Prefer file pointers:

1. `<entrypoint>` parses input.
2. `<module>` routes the request.
3. `<module>` performs the core behavior.
4. `<module>` renders or returns output.

## Key Boundaries

List boundaries that protect correctness or safety.

Examples:

- filesystem scope
- network scope
- auth and secret handling
- durable vs ephemeral state
- read/write authority
- public API compatibility

## Verification

List common checks from cheapest to broadest.

Examples:

- `<formatter command>`
- `<typecheck/build command>`
- `<unit test command>`
- `<integration smoke command>`
- `<live or external command, if any>`

Note any commands that require network, credentials, GUI access, or destructive state.

## Current Direction

Summarize the active development direction in a few bullets.

This should help a new agent avoid reopening settled design questions.

## Known Risks

List durable risks or gotchas.

Do not record transient debugging notes.

## Update Rule

Update this cache when changing:

- command routing or public behavior
- architecture boundaries
- core data contracts
- state or storage layout
- provider or external integration behavior
- verification commands
- important file ownership

Do not record transient logs, secrets, copied chats, local runtime memory, or session-specific user content.

