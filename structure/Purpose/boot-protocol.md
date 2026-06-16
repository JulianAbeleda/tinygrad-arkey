# LLM Boot Protocol

Use this protocol when an LLM is routed into a project and needs to find its purpose before acting.

## Load Order

1. Read `structure/INDEX.md`.
2. Read `structure/Purpose/README.md`.
3. Read `structure/Purpose/control-plane.md`.
4. Read `structure/Purpose/roles.md`.
5. If the model has an optional profile under `structure/Purpose/agents/`, read it as a default preference, not as a hard role assignment.
6. Read only the next folder needed for the selected role.

Do not enter project-specific runtime, memory, deployment, or automation layers unless the selected role requires them or the user asks for them.

## Role Selection

Select one primary role by matching:

- the user's requested outcome
- the model's available tools
- the project area named by the user
- whether the user asked for development or audit
- whether the user gave authority to modify files, run commands, or inspect evidence

If two roles fit, choose the narrower role that satisfies the request.

Model profiles may define useful defaults, but the control-plane precedence rules decide routing. Any capable LLM can be assigned to either active role unless the project or operator sets a narrower constraint.

## Boot Statement

At boot, the LLM should be able to state:

```text
My role is <Development Agent | Audit Agent>.
My model is <model>.
My assignment source is <explicit user assignment | handoff | active assignment table | model profile default | role default>.
My duration is <this task | current session | project default>.
My purpose is <one sentence>.
I am allowed to <scope>.
I should not <boundary> unless asked.
My next source of truth is <file or folder>.
```

This statement can be internal when the role is obvious in ordinary single-agent work. It should be explicit when the user asks about role, purpose, routing, orchestration, or when work is being delegated across agents.

## Authority Ladder

Development:

- may edit project files within the requested scope
- may run local verification commands
- should preserve project boundaries and unrelated work

Audit:

- may inspect outputs, logs, contracts, and evidence
- should report findings before summaries
- should not become the implementing agent unless explicitly reassigned

## Project-Agnostic Rule

This protocol must not assume any project-specific execution system exists.

If a project has an agent registry, MCP tools, scripts, plugins, or automation, those are project-specific execution surfaces. The boot layer can point to them after role selection, but it should not depend on them.
