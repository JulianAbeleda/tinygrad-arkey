# Repo Map

Last updated: YYYY-MM-DD
Purpose: compact file ownership map for agents before reading source.

## Root

- `<file>` - `<purpose>`
- `<file>` - `<purpose>`

## Entrypoints

- `<path>`
  - owns: `<behavior>`
  - key symbols: `<functions/classes/types>`

## Core Modules

- `<path>`
  - owns: `<behavior>`
  - key symbols: `<functions/classes/types>`

- `<path>`
  - owns: `<behavior>`
  - key symbols: `<functions/classes/types>`

## Data And State

- `<path>`
  - owns: `<data contract or state behavior>`

## UI Or Interface

- `<path>`
  - owns: `<CLI/API/UI behavior>`

## Tests And Scripts

- `<path>` - `<what it verifies>`
- `<path>` - `<what it verifies>`

## Documentation

- `structure/INDEX.md` - project purpose and first pointers.
- `structure/Purpose/` - LLM role boot layer.
- `structure/Development/` - development principles, roadmap, and implementation guidance.
- `structure/System Guide/` - durable architecture explanation.
- `structure/User_Guide/` - setup and user procedures.
- `structure/Deployment/` - deploy and release procedures.
- `structure/cache/` - local or committed cache layer, depending on project policy.

## Read Strategy

For most development tasks:

1. Read `structure/INDEX.md`.
2. Read `structure/Purpose/`.
3. Read `structure/cache/repo-cache.md`.
4. Read the relevant module note under `structure/cache/module-notes/`.
5. Then read only the owning source file or docs.

