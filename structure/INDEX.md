# tinygrad Structure

Human-facing layer for this fork of `tinygrad`. Model-agnostic. Not runtime internals.

`structure/` exists so this fork can be understood, changed, tested, and handed between humans or models without depending on a specific chat, prompt, or orchestration setup.

## Project Snapshot

`tinygrad` is a compact deep learning stack with a Tensor API, autograd, lazy execution, kernel lowering/codegen, runtimes for multiple devices, examples, tests, and docs.

This fork lives at:

- Local checkout: `/Users/julianabeleda/env/tinygrad-arkey`
- GitHub fork: `https://github.com/JulianAbeleda/tinygrad-arkey`
- Upstream parent: `https://github.com/tinygrad/tinygrad`

## First Pointer

If you are new to this structure, start with `HOW_TO_USE.md`.

If the goal is fast role alignment for a model or operator, start with `Purpose/`.

- `HOW_TO_USE.md` — plain-English workflow and reader paths
- `Purpose/README.md` — project-agnostic boot layer for orienting LLMs
- `Purpose/boot-protocol.md` — model boot order and authority ladder
- `Purpose/control-plane.md` — model-to-role assignment and swapping
- `Purpose/delegation-contract.md` — task packets, handoffs, and done criteria
- `Purpose/roles.md` — two-role registry for Development Agent and Audit Agent
- `Purpose/examples/dev-audit-loop.md` — worked example of the development-to-audit loop
- `Development/amd-optimization-checklist.md` — AMD remote bridge and Q4_K optimization checklist

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

## How To Use This Structure

Use this folder as the durable orientation layer for this fork.

Keep project facts here concise. Put source-level details in `cache/`, contributor workflow in `Development/`, user setup in `User_Guide/`, and architecture notes in `System Guide/`.
