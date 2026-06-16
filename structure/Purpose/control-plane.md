# LLM Control Plane

This is the central routing layer for swapping LLMs between roles.

## Purpose

The control plane answers:

- which roles exist
- which models are available
- which model is currently assigned to each role
- what wins when a task delegation conflicts with a default model profile
- how to swap models without rewriting role contracts

## Core Rule

Roles, models, and assignments are separate.

- `roles.md` defines role contracts.
- `agents/<model>.md` defines optional model profiles.
- this file defines current model-to-role assignments and precedence.
- `delegation-contract.md` defines task and handoff formats.

Do not hard-code model names into role definitions.
Do not define task packet formats here.

## Active Roles

| Role | Contract |
| --- | --- |
| Development Agent | Change project files within scope and verify the change |
| Audit Agent | Review work for correctness, evidence, drift, and contract fit |

## Model Profiles

Model profiles are optional.

A profile may define a model's preferred default role, strengths, boundaries, and boot statement. It does not own the role.

Future models should be added by creating a profile under `agents/`, not by editing `roles.md`.

Model profiles should define:

- default role
- boot statement
- allowed secondary roles
- boundaries
- role escalation rules
- links to relevant project standards

## Active Assignment Table

This table is the current routing map. It can be changed without changing the role contracts.

| Role | Assigned model | Fallback model | Notes |
| --- | --- | --- | --- |
| Development Agent | `<fill per project>` | optional | Implementation, documentation, scoped edits |
| Audit Agent | `<fill per project>` | optional | Review, verification, findings, evidence checks |

The table is the routing control surface. Swap models here for persistent routing changes.

If a role has no assigned model, routing falls through to optional model profile defaults, then to role selection defaults.

Fallback activates only when:

- the assigned model is unavailable
- the assigned model cannot access the required tools
- the operator explicitly invokes the fallback
- the assigned model fails to produce the required handoff artifact

## Precedence

When routing a task, apply this order:

1. Explicit user assignment in the current task.
2. Handoff request from the previous agent.
3. Active assignment table in this file.
4. Optional model profile default.
5. Role selection defaults in `roles.md`.

The highest available layer wins.

Example:

```text
Assign <model> as Development Agent for this task.
```

This overrides that model's default profile for that task.

## Swap Command Pattern

The operator should be able to swap models by writing:

```text
Assign <model> as <Development Agent | Audit Agent> for <duration>.
```

Allowed durations:

- `this task`
- `current session`
- `project default`

A session is one continuous conversation context. It ends when the operator closes it, the context is reset, or the operator assigns a different session-level route.

Examples:

```text
Assign <model-a> as Development Agent for this fix.
Assign <model-b> as Audit Agent for this review.
```

The receiving LLM should confirm:

```text
Role: <Development Agent | Audit Agent>
Model: <model>
Assignment source: explicit user assignment
Duration: <this task | current session | project default>
Purpose: <one sentence>
Scope: <task, files, systems, or artifact>
Boundaries: <what I will not touch or assume>
Done means: <role-specific done criteria>
```

## Mid-Task Swap

If the operator swaps models mid-task, the outgoing or supervising agent should provide a handoff artifact before the incoming model continues.

Minimum mid-task handoff:

```text
Current task:
<What is being done.>

Current state:
<What has already changed, been inspected, or been decided.>

Artifacts:
<Files, diffs, commits, logs, or copied output.>

Remaining work:
<What the incoming model should do next.>

Known risks:
<Unverified assumptions, failed checks, or constraints.>
```

The incoming model should continue from the handoff artifact, not restart from scratch unless the operator asks for a fresh pass.

## Control-Plane Update Rule

If the operator wants a persistent routing change, update the active assignment table.

If the operator wants a one-off routing change, put the assignment in the task delegation packet.

Do not edit:

- `roles.md` for model swaps
- model profiles for one-off assignments
- handoff templates for model swaps
