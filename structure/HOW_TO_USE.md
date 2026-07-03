# How To Use This Structure

Use this file when you have a project and one or more LLMs, but you are not sure what to do first.

The short version:

1. Put the project explanation in `INDEX.md`.
2. Put setup and operating steps in `User_Guide/`.
3. Put build and coding rules in `Development/`.
4. Put durable architecture in `System Guide/`.
5. Put deployment and release steps in `Deployment/`.
6. Use `Purpose/` when you want an LLM to enter the project with a clear role.
7. Use `cache/` when you want compact repo context before broad source inspection.

## What This Is For

This structure keeps project knowledge readable outside any one chat, model, tool, or runtime.

The `Purpose/` layer is for role alignment. It helps answer:

- should this LLM build something?
- should this LLM audit something?
- what is the task scope?
- what should the next handoff look like?

The core loop is:

1. The operator writes a task.
2. A development model does the work.
3. The development model produces a handoff.
4. An audit model reviews the work.
5. The operator accepts, asks for revisions, or sends findings back to development.

## Reader Paths

### Completely New

Start here if you have not used an LLM role system before.

1. Read this file.
2. Read `User_Guide/getting-started.md`.
3. Read `Purpose/examples/dev-audit-loop.md`.
4. Copy the example task packet and replace the placeholders with your task.
5. Send the task packet to the model you want to use for development.
6. Copy the development handoff into the audit model.
7. Use the audit result to accept, revise, or block the work.

You do not need to understand every file in `Purpose/` before starting. The example is enough for a first pass.

### Tech Savvy

Start here if you are comfortable with projects, tools, and LLMs, but want the routing pattern.

1. Read `INDEX.md`.
2. Read `Purpose/README.md`.
3. Use `Purpose/delegation-contract.md` to write the task packet.
4. Use `Purpose/control-plane.md` only if you want stable model-to-role assignments.
5. Use `Purpose/examples/dev-audit-loop.md` as the reference flow.

Your main job is to keep scope, success criteria, and verification explicit.

### Engineers

Start here if you are integrating this into an existing repo or team workflow.

1. Read `Development/structure-convention.md`.
2. Customize `Development/coding-principles.md`.
3. Add focused cache notes under `cache/module-notes/` only when repeated agent work would otherwise rediscover the same stable facts.
4. Add focused files under `cache/module-notes/` for stable, task-specific areas.
5. Fill `Purpose/control-plane.md` with project-appropriate assignments only if persistent routing is needed.
6. Add optional model profiles under `Purpose/agents/`.
7. Keep role contracts in `Purpose/roles.md` model-agnostic.
8. Keep task and handoff formats in `Purpose/delegation-contract.md`.

The main engineering rule is separation of authority: roles, models, assignments, and tasks should remain separate files.

## Key Terms

`Role` means the job the LLM is doing, such as `Development Agent` or `Audit Agent`.

`Model` means the LLM being used. A model can change without changing the role.

`Control plane` means the routing file that says which model is assigned to which role.

`Task packet` means the operator's written assignment: role, task, context, scope, success criteria, verification, and handoff.

`Handoff artifact` means the output one agent gives to the next agent or human so the work can continue without guessing.

`Assignment source` means why the LLM is in the role: direct user instruction, handoff, control-plane assignment, model profile, or default role selection.

## Minimal First Task

Use this shape for a first development task:

```text
Role:
Development Agent

Task:
Make <specific change>.

Context:
<Why the change is needed.>

Scope:
<Files or folders the agent may touch.>

Out of scope:
<What should not be touched.>

Success criteria:
- <observable result>
- <scope is preserved>

Verification:
<test, build, review step, or "not required">

Handoff artifact:
Summarize changed files, verification, risks, and next action.
```

Then send the handoff to an audit model using the example in `Purpose/examples/dev-audit-loop.md`.
