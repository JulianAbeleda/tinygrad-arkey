# Development

Human-facing notes for building the project itself.

This layer should stay model-agnostic.

It exists so project logic, release thinking, implementation rules, and workflow remain legible even if the active model or development assistant changes.

## Use This Folder For

- coding principles
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

## Key documents (this fork)

- `coding-principles.md` — core engineering principles for the repo.
- `performance-primitive-research-principles.md` — supplement for GPU-perf / quantized-primitive /
  machine-search work (isolated wins mislead → gate on in-model W==D, etc.).
- `tinygrad-coding-overrides.md` — fork-specific overrides on top of upstream tinygrad conventions.
- `roadmap.md`, `release-strategy.md` — direction + release thinking.

For the engineering work itself (decode/prefill/MMVQ), the source of truth is `../../docs/README.md` plus the route
manifest under `../../bench/qk-search-spaces/default_route_manifest.json`. Historical handoffs and audits were removed
from the active tree; use git history for archaeology.
