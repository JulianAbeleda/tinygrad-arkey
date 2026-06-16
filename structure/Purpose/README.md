# Purpose

Project-agnostic boot layer for orienting LLMs.

Read this first when the goal is to give a model a clear role before it starts work. This folder should answer three questions:

1. What is this project?
2. What role should this LLM occupy?
3. What boundaries apply before it touches project-specific systems?

Use `boot-protocol.md` for the role-selection process.
Use `control-plane.md` for model-to-role assignment and swapping.
Use `delegation-contract.md` for task packets, handoffs, and done criteria.
Use `roles.md` for the role registry.
Use `User_Guide/` for procedures after orientation.
Use `Development/` for building or changing the project.

The active role set is intentionally small: `Development Agent` and `Audit Agent`.

## Purpose of This Layer

`structure/Purpose/` is not a runtime and not an execution system.

It is a boot surface: a small, stable place where an LLM can discover how it should enter the project.

The design should remain portable across projects. A project may have state files, scripts, agents, build tools, deployment systems, automation, or none of those. This layer should still work because it describes roles and boundaries before project-specific machinery is activated.

## Project Purpose

State the concrete project purpose in the project index or system guide.

This folder should not carry a project-specific operating thesis unless the project has no better home for it.

The portable pattern is:

- `structure/` explains the project for humans and models
- `Purpose/` routes an LLM into a clear boot role
- `User_Guide/` explains how to operate the project
- `Development/` explains how to change the project
- project-specific runtime, memory, deployment, or automation layers are named by that project

## LLM Boot Rule

An LLM entering a project should not infer its role from model identity alone.

Role is selected from:

- the user's request
- the available tools
- the project context
- the session or terminal assignment
- the authority explicitly granted

After reading this folder, the LLM should be able to state:

- its selected role
- why that role fits the request
- what it is allowed to touch
- what it should not do by default

## Default Boundary

`structure/` provides orientation.

Project-specific execution layers provide execution authority.

Runtime, memory, deployment, and automation layers are not generic LLM boot requirements. They only become relevant when the selected role needs them.

## Anti-Goals

Do not use this layer to:

- define every project procedure
- duplicate runtime contracts
- make one model brand the default role
- collapse operator, runtime, development, and audit into one agent
- treat project-specific execution systems as part of the generic structure scaffold

## Quick Pointers

For role selection:

- `boot-protocol.md`
- `control-plane.md`
- `delegation-contract.md`
- `roles.md`

For project procedures:

- `../User_Guide/`

For project development:

- `../Development/`
