# Structure

Human-facing layer. Model-agnostic. Not runtime internals.

`structure/` exists so a project can be understood, used, built, and deployed without depending on a specific model, prompt, or orchestration setup.

## First Pointer

If you are new to this structure, start with `HOW_TO_USE.md`.

If the goal is fast role alignment for a model or operator, start with `Purpose/`.

**For this project's actual engineering work** (the AMD-only quantized-decode fork): start with
`../docs/README.md` (the current doc map) and `../bench/qk-search-spaces/default_route_manifest.json` (route state).
Historical handoffs and old audits are available through git history, not active-tree docs.

- `HOW_TO_USE.md` — plain-English workflow and reader paths
- `Purpose/README.md` — project-agnostic boot layer for orienting LLMs
- `Purpose/boot-protocol.md` — model boot order and authority ladder
- `Purpose/control-plane.md` — model-to-role assignment and swapping
- `Purpose/delegation-contract.md` — task packets, handoffs, and done criteria
- `Purpose/roles.md` — two-role registry for Development Agent and Audit Agent
- `Purpose/examples/dev-audit-loop.md` — worked example of the development-to-audit loop

## Default Project Layers

The portable `structure/` defaults are:

- `INDEX.md`
- `HOW_TO_USE.md`
- `Purpose/`
- `User_Guide/`
- `Development/`
- `System Guide/`
- `Deployment/`
- `cache/`

These are the load-bearing human layers.

`cache/` is optional but recommended when agents repeatedly need the same repo orientation. It should stay compact, modular, and free of secrets or transient session state.

## Rule

Keep `structure/` useful even if:

- the active model changes
- the runtime changes
- the deployment target changes

The point is legibility outside the live runtime.

## How To Use This Template

Use this repo as the canonical source when creating or updating project `structure/` folders.

Adapt the project-specific content, but keep the folder pattern and the role of each layer stable unless there is a strong reason to change it.
