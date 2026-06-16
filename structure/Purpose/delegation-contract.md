# Delegation Contract

This document defines how a human operator should hand work to LLM agents and how one agent's output should be passed to another agent. It is project-agnostic.

## Design Goal

The boot protocol answers: "Who am I in this project?"

The delegation contract answers: "What exactly am I being asked to do, and what artifact should I produce for the next agent or human?"

Delegation should not rely on the operator remembering every boundary. A well-formed handoff should carry enough context for the receiving agent to start in the right role, stay inside scope, and know what done means.

Model routing is governed by `control-plane.md`. This file defines task shape and handoff shape, not persistent model assignments.

## Session Scope

These contracts are session-specific.

A task delegation packet, boot declaration, handoff artifact, and any in-session assignment are scoped to the current conversation context. They do not persist automatically across sessions.

If a task spans sessions, the operator is responsible for carrying forward the relevant handoff artifact into the next session. The incoming agent should treat the carried artifact as its starting state, not assume continuity from a prior context it cannot read.

## Required Boot Declaration

For delegated work, the receiving agent should declare its boot state explicitly before acting.

Required format:

```text
Role: <Development Agent | Audit Agent>
Model: <LLM name>
Assignment source: <explicit user assignment | handoff | active assignment table | model profile default | role default>
Duration: <this task | current session | project default>
Purpose: <one sentence>
Scope: <files, directories, artifact, or question>
Boundaries: <what I will not touch or assume>
Done means: <completion criteria for this task>
```

Delegated work needs visible role confirmation because the operator may copy the result into another agent.

## Task Delegation Template

Use this when assigning work to an agent.

```text
Role:
<Development Agent | Audit Agent>

Model:
<Specific LLM, or "use active assignment table".>

Assignment source:
<explicit user assignment | handoff | active assignment table | model profile default>

Duration:
<this task | current session | project default>

Task:
<The concrete work requested.>

Context:
<Relevant background, prior decisions, constraints, or links.>

Scope:
<Files, folders, systems, or artifacts the agent may inspect or change.>

Out of scope:
<Files, systems, behaviors, or decisions the agent should not touch.>

Success criteria:
<Observable conditions that mean the task is done.>

Verification:
<Commands, checks, review steps, or evidence expected. Use "not required" if none.>

Handoff artifact:
<What the agent should produce for the next human or agent.>
```

## Model-to-Role Assignment

Roles and models are separate.

A role defines the work contract. A model profile defines default tendencies or local preferences for a specific LLM. The control plane defines current assignments.

Examples:

- Model A as `Development Agent`
- Model B as `Audit Agent`
- the same model as `Development Agent` for one task and `Audit Agent` for a later task, if the operator explicitly reassigns it

The delegated role overrides the model's default profile for that task.

For persistent routing, update `control-plane.md`.

## Development Task Template

Use this for implementation work assigned to any development agent.

```text
Role:
Development Agent

Task:
<Implement, refactor, fix, or document the requested behavior.>

Context:
<Why this change is needed. Include any prior audit findings or user decisions.>

Scope:
<Allowed files, folders, modules, or docs.>

Out of scope:
<Anything the agent should not modify, even if adjacent.>

Coding standard:
Use the project coding standard. If this scaffold has not been customized, look under `structure/Development/`.

Success criteria:
- requested behavior or documentation is implemented
- changes are scoped to the stated area
- existing project patterns are followed
- unrelated user or runtime changes are preserved
- verification is run or the reason it was not run is stated

Verification:
<Specific tests, lint commands, build commands, or doc review steps.>

Handoff artifact:
- summary of changed files
- verification performed
- known risks or follow-up items
- git commit id or diff reference if available
```

## Audit Task Template

Use this for review work assigned to any audit agent.

```text
Role:
Audit Agent

Task:
<Review the implementation, design, document, or handoff.>

Context:
<What changed, why it changed, and what standard governs the review.>

Scope:
<Files, diff, commit id, logs, or artifact to inspect.>

Out of scope:
<What the audit should not redesign or fix.>

Audit standard:
Use the relevant project contract. For code, use the project coding standard. If this scaffold has not been customized, look under `structure/Development/`.

Success criteria:
- findings are ordered by severity
- each finding cites concrete evidence when available
- assumptions and inferred risks are labeled
- missing tests or verification gaps are identified
- no fixes are implemented unless explicitly reassigned

Verification:
<Commands or artifacts the auditor should inspect, if any.>

Handoff artifact:
- findings
- open questions
- verification reviewed
- recommendation: accept, revise, or block
```

## Agent Handoff Contract

Use this when one agent's output is copied to another agent.

The handoff can be carried by:

- copy and paste of the handoff artifact
- a git commit id
- a diff reference
- a file path to the changed artifact
- a command output or audit report

Minimum handoff format:

```text
From:
<Agent and role.>

To:
<Agent and role.>

Task completed:
<What was done.>

Artifacts:
<Changed files, commit id, diff reference, report path, or pasted output.>

Verification:
<What was run or inspected.>

Known risks:
<Residual concerns, skipped checks, or assumptions.>

Requested next action:
<Audit, fix, push, revise, approve, or explain.>
```

## Development to Audit Handoff

Use after implementation when an audit agent should review the work.

```text
From:
<LLM name>, Development Agent

To:
<LLM name>, Audit Agent

Task completed:
<Short implementation summary.>

Artifacts:
- changed files: <list>
- commit id or diff reference: <id or "uncommitted diff">

Verification:
<Commands run and result, or "not run" with reason.>

Known risks:
<Anything uncertain, untested, or intentionally deferred.>

Requested next action:
Audit the implementation against the task, the stated scope, and the project coding standard. Report findings ordered by severity. Do not implement fixes unless reassigned.
```

## Audit to Development Handoff

Use after audit when a development agent should fix issues, explain a decision, or push accepted work.

```text
From:
<LLM name>, Audit Agent

To:
<LLM name>, Development Agent

Audit result:
<Accept, revise, or block.>

Findings:
<Findings ordered by severity with file or artifact references.>

Verification reviewed:
<Tests, commands, logs, diffs, or artifacts inspected.>

Requested next action:
<Fix findings, explain why no change is needed, or push if accepted.>

Boundaries:
<Do not modify unrelated files. Preserve user changes. Keep fixes scoped to findings.>
```

## Done Definitions

Done depends on role and context.

### Development Agent

Done means:

- the requested change is implemented or the blocker is stated
- changed files are listed
- verification is run, or not-run status is explained
- residual risks and follow-ups are named
- the task is ready to hand off or close according to task requirements

### Audit Agent

Done means:

- findings are ordered by severity and reported before summary
- each finding is grounded in evidence or clearly labeled as inference
- severity is clear
- verification gaps are named
- final recommendation is accept, revise, or block
