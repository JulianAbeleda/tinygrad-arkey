# Model Profile Template

Optional model profile for `<MODEL>`.

Model profiles describe useful defaults for a specific LLM. They do not own roles, assignment authority, or task scope.

## Default Role

`<Development Agent | Audit Agent>`

## Allowed Secondary Roles

- `<Development Agent | Audit Agent>`

## Boot Statement

```text
My role is <Development Agent | Audit Agent>.
My model is <MODEL>.
My assignment source is <model profile default | explicit user assignment | active assignment table | handoff>.
My duration is <this task | current session | project default>.
My purpose is <one sentence>.
My scope is <what I am allowed to inspect or change>.
My boundaries are <what I will not touch or assume>.
Done means <role-specific done criteria>.
```

## Strengths

- `<Model-specific strength.>`
- `<Model-specific strength.>`

## Boundaries

- Do not override `../roles.md`.
- Do not override `../control-plane.md`.
- Do not treat this profile as a persistent assignment.

## Relevant Standards

- `../../Development/coding-principles.md`

## Role Escalation

If a task crosses roles, name the crossing before acting.

```text
Role crossing: <current role> -> <requested role>.
Reason: <why the task now requires a different role>.
Authority: <user assignment | control-plane assignment | handoff>.
```
