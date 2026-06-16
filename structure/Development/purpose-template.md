# Purpose Template

Use this template when creating or updating `structure/Purpose/` for a new repo.

The goal is not to write a project biography. The goal is to give LLMs a small, stable boot surface before they touch project-specific systems.

## Required Files

The default `Purpose/` layer should include:

```text
Purpose/
  README.md
  boot-protocol.md
  control-plane.md
  delegation-contract.md
  roles.md
  agents/
```

## Role System

The active role system has two roles only:

- `Development Agent`
- `Audit Agent`

Model profiles are optional and live under `Purpose/agents/`.

Do not hard-code model names into `roles.md`.

## Source Responsibilities

- `README.md` explains the boot layer.
- `boot-protocol.md` defines load order and boot declaration.
- `control-plane.md` defines model-to-role assignment and precedence.
- `delegation-contract.md` defines task packets, handoffs, and done criteria.
- `roles.md` defines the active role contracts.
- `agents/<model>.md` defines optional model defaults.

## Portability Rules

- Keep the layer project-agnostic.
- Use placeholders for project-specific runtime, memory, deployment, or automation systems.
- Do not assume a specific execution system, memory layer, model, or orchestration layer.
- Keep role contracts separate from model profiles.
- Keep assignment authority centralized in `control-plane.md`.

## Reuse Rule

When creating a new project, copy the folder pattern and adapt the project-specific text.

Do not copy personal notes, private operating context, or project-specific runtime contracts into the generic scaffold.
