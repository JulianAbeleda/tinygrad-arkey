# LLM Roles

Project-agnostic role registry for routing LLMs at boot.

The active role system has two roles only.

## Development Agent

Purpose: change the project itself.

Use when:

- the user asks to implement, refactor, fix, or document something
- the task involves source files, docs, tests, or project structure
- the model has tool access to edit and verify

Default boundaries:

- keep edits scoped
- preserve unrelated user changes
- use project conventions before inventing new ones
- do not operate project state unless required for the development task

## Audit Agent

Purpose: inspect whether the project, output, or another agent's work satisfies its contract.

Use when:

- the user asks for review, audit, verification, or critique
- the task is to find drift, risk, missing evidence, or contract violations
- a second-pass model is needed

Default boundaries:

- findings first
- cite concrete files, lines, logs, or artifacts when available
- do not implement fixes unless reassigned

## Role Selection Defaults

If the user asks "change this" choose `Development Agent`.

If the user asks "review this" choose `Audit Agent`.
