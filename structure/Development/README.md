# Development

Human-facing notes for building the project itself.

This layer should stay model-agnostic.

It exists so project logic, release thinking, implementation rules, and workflow remain legible even if the active model or development assistant changes.

## Use This Folder For

- coding principles
- project-specific coding overrides
- scope and local development environment boundaries
- session handoff context
- release strategy
- roadmap
- security principles
- handoff context
- project-specific scaffolds and conventions

## Rule

Keep `Development/` focused on shaping the repo, not on running the product day to day.

Use `User_Guide/` for operation.
Use `Purpose/` for role alignment.

## Canonical Scaffolds

- `purpose-template.md` — canonical scaffold for `structure/Purpose/README.md`
- `structure-convention.md` — default folder pattern and what each layer is for

## Local Project Files

- `scope.md` — local fork scope, development environment, and boundaries
- `session-handoff.md` — current stop point, validation state, and next actions
- `coding-principles.md` — general coding and commit discipline
- `tinygrad-coding-overrides.md` — tinygrad-specific commit prefixes and local validation rules
- `amd-optimization-checklist.md` — checklist for AMD remote bridge, Q4_K, and Radeon 7900 XTX optimization work
