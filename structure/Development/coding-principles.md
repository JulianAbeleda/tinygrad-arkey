# Coding Principles

These rules are for writing and shaping project code.

They are not optional style preferences. They are build rules.

## Core Rule

Project code should favor:

- centralization of authority
- modularization of execution
- abstraction for simplicity
- orthogonality for independence

These four principles are meant to coexist.

Centralize what defines the system.

Modularize what carries the system out.

Abstract what should stay simple at the interface.

Keep independent concerns orthogonal so change stays local.

## Centralize

Keep one clear authority point for:

- environment and config access
- schemas and durable data shape
- integration boundaries to external services
- routing rules and system policy
- state definitions that other modules depend on

Do not duplicate load-bearing logic across scripts, modules, or interfaces.

If multiple parts of the system need the same rule, move that rule to one explicit source of truth.

## Human-Facing And Machine-Enforced

When a rule is useful to a human and mechanically enforceable, prefer both.

The human-facing layer should explain the rule in plain language.

The machine-facing layer should check the rule automatically when the enforcement cost is reasonable.

Examples:

- a task can be readable in a task-state file and still carry `owner`, `next-review`, and `criterion`
- a module contract can document a test and the test can be run by the project's test runner
- a commit message rule can live in the coding standard and also be checked by a project-specific commit checker

Do not leave load-bearing rules as decorative policy when a small script can enforce them.

Do not over-enforce rules whose judgment cost is higher than the drift they prevent.

## Modularize

Break execution into bounded parts for:

- modules
- adapters
- API handlers
- UI components
- maintenance routines

Modules should be easy to replace, audit, and test without rewriting the whole system.

Modularity does not mean scattering authority. It means keeping execution surfaces narrow and composable.

## Abstract

Hide implementation complexity behind stable, legible interfaces.

The visible surface of the system should be simpler than the machinery behind it.

Use abstraction to reduce cognitive load, not to create mystery.

Good abstraction means:

- a small number of clear commands or entry points
- shared utilities instead of repeated low-level logic
- stable interfaces around backends, paths, runtime state, and control-plane actions
- implementation detail staying inside the owning module

Bad abstraction means:

- vague wrappers that hide where authority actually lives
- helper layers that duplicate underlying rules
- naming that sounds generic but does not reduce real complexity

## Orthogonalize

Keep distinct concerns independent so one change does not force unrelated changes elsewhere.

Orthogonality protects the system from entanglement.

Good orthogonality means:

- runtime use and meta-development stay separate
- policy and execution do not collapse into one file
- feature activation, backend dispatch, path resolution, and audit logic have distinct boundaries
- changing one backend or one feature does not require rewriting unrelated system pieces

Bad orthogonality means:

- one module carrying multiple unrelated responsibilities
- path, policy, state, and execution logic fused together
- side effects that leak across the system without a clear boundary
- a refactor in one subsystem forcing incidental edits in many others

## Implementation Principles

These principles are extracted from mature systems codebases and should apply across languages.

### Encode Invariants

Make invalid states hard to represent.

Use types, schemas, constructors, state machines, and validation boundaries to express system invariants directly.

Do not rely on comments, naming, or caller discipline for rules the code can encode.

Prefer:

- typed states over stringly-typed status values
- constructors that validate durable objects
- narrow public APIs around sensitive state transitions
- explicit capability or permission values instead of ambient access
- machine-readable contracts when humans and tools both depend on the rule

### Keep Public Surfaces Boring

Hide internal complexity behind stable, ordinary interfaces.

The more complex the implementation, the simpler the caller-facing surface should be.

Prefer:

- small entry points
- obvious names
- predictable return values
- feature flags or configuration that expose real choices, not internal machinery
- documentation that explains tradeoffs without requiring the reader to learn the whole subsystem

### Separate Ergonomics From Semantics

Convenience should not blur what the system means.

Ergonomic helpers are good when they preserve the same failure modes, authority boundaries, and data model as the lower-level API.

Avoid helpers that:

- skip validation
- hide ownership of state
- swallow meaningful errors
- make a policy decision look like a formatting or transport detail
- create a second unofficial way to perform the same operation

### Treat Errors As System Information

Errors should preserve both machine-actionable structure and human-useful context.

Application code may use broad error types when the caller only needs context and exit behavior.

Library, protocol, storage, and integration boundaries should expose errors precise enough for callers to make decisions.

Every error path should answer:

1. What failed?
2. Where did it fail?
3. Is the failure recoverable?
4. Does the caller need a typed distinction?
5. What context would the operator need during debugging?

### Contain Dangerous Power

Unsafe operations, direct system calls, global state, unchecked casts, raw concurrency primitives, and destructive side effects must be isolated behind small reviewed boundaries.

The boundary should document:

- what invariant makes the operation valid
- who is allowed to call it
- what state it may mutate
- what tests, assertions, or runtime checks defend it

Do not spread privileged operations through convenience helpers.

### Design For Replacement

External services, storage backends, model providers, runtimes, and platform-specific integrations should sit behind replaceable adapters.

Replacement does not mean pretending every backend is identical.

It means:

- shared policy lives above the adapter
- backend-specific behavior stays inside the adapter
- capability differences are explicit
- tests can exercise the contract without depending on every real backend

### Test Behavior At The Boundary

Unit tests are not enough for code whose risk lives at integration boundaries.

Use the cheapest test that can catch the real failure mode.

Prefer:

- small unit tests for pure logic
- regression tests for fixed bugs
- integration tests for command, API, storage, and adapter behavior
- property or fuzz tests for parsers, protocol handling, and state transitions
- concurrency stress tests where ordering matters

### Classify Evidence Before Fixing Mechanisms

When a bug crosses compiler, runtime, hardware, benchmark, or policy boundaries, classify what each piece of evidence
is allowed to prove before changing code.

Trust invariants over symptoms. A microgate may prove a local invariant, but it does not prove a full route is safe. A
full-model run may expose a regression, but it is not trustworthy until the harness itself is known to populate the
right state and measure the intended path.

Separate:

- compile failure from numeric failure
- local microgate evidence from integration evidence
- correctness evidence from speed evidence
- route binding from route promotion
- harness failure from system failure
- fix-off bug pins from fix-on success criteria

Then fix the smallest violated invariant. Do not let a passing narrow gate authorize a wider claim, and do not let a
broken harness turn noise into a route or compiler verdict.

### Explain Tradeoffs Close To The Code

When code chooses performance, compatibility, portability, simplicity, or strict correctness over another value, explain that choice near the implementation.

Good comments explain why the shape exists.

Bad comments restate what the code already says.

## Reducing Code The Right Way

When a codebase feels "too big," the instinct to shrink it is correct but the
target is usually wrong. **Line count is not the metric. Knowledge duplication
is.** Optimize for one authoritative source per piece of knowledge and for deep,
legible modules — fewer lines is a side effect of doing that, never the goal.

### Tiny means understandable
For tinygrad-shaped systems, "tiny" does not mean minimizing line count at any
cost. It means keeping the core idea small enough that one capable engineer can
understand, modify, and reason about it end-to-end.

Prefer:

- a small conceptual core over many special cases
- boring public interfaces over clever internal machinery
- replacing old complexity with simpler tested machinery
- contained backend/runtime power instead of ambient global behavior
- table-driven experiments instead of cloned scripts
- tests that make shrinking and replacement safe

Avoid:

- deleting useful tests or backend knowledge just to reduce LOC
- adding framework layers that make the system harder to reason about
- preserving obsolete probes after their verdict is recorded
- letting machine search, runtime flags, or hardware-specific paths become a
  second hidden system

### DRY means knowledge, not lines
Eliminate duplicated *knowledge* — a rule, a schema, a policy — by giving it a
single source of truth. Do **not** eliminate code that merely *looks* alike but
encodes *different* concepts. Two functions with the same shape that read
different inputs or mean different things are not a DRY violation.

### Duplication is cheaper than the wrong abstraction
If you are tempted to merge similar-looking code, first prove it represents the
*same* knowledge. If it does not, leave the duplication — a wrong abstraction is
harder to maintain than the duplication it replaces, because every future
divergence must fight the forced shared shape. Prefer accidental duplication
over a leaky generalization.

### Abstract only what has earned it (Rule of Three / AHA)
Wait until a pattern repeats ~3 times *and is genuinely identical* before
abstracting it. Avoid hasty abstraction; let the right shape emerge. Premature
generalization is itself an anti-pattern.

### If similar things should be merged, fix the inputs first
When near-duplicate code is blocked from merging only because its inputs differ
(different ad-hoc schemas, formats, conventions), the elegant move is to
**unify the inputs into one typed/canonical shape, then** collapse the code into
one parameterized path. Standardize the data, and the duplication becomes real
duplication you can correctly remove.

### Prefer data over code (table-driven)
Replace N near-identical functions/branches with **one parameterized
implementation driven by a declarative table**. A new case becomes a *row of
data*, not a new function or file. This is the durable cure for "every variant
got its own copy."

### Deep modules, simple surfaces
Complexity, not line count, is the enemy. Favor modules whose interface is much
simpler than their implementation. A small number of deep, orthogonal modules
beats many shallow ones.

### No-new-file rule (anti-sprawl)
A new experiment, variant, or case adds a *row to a table* or a *parameter* — not
a new file or a copy-pasted function/`main()`. New behaviors extend the existing
authoritative module. Copy-pasting a builder, a `main()`, or a helper is the
sprawl this section exists to prevent; when you reach for it, stop and extend
instead.

## A performance primitive is an operation PLUS its required memory locality

A *semantic* primitive names a computation: attention, matmul, GEMV, softmax. A *performance* primitive also
controls where the data lives while that computation runs. The two are not the same, and confusing them
silently produces correct-but-slow code.

The flash-prefill arc (Increment 2) is the cautionary tale. We proved the custom-kernel path could express the
math — score-free fused causal attention, exact vs SDPA, JIT-captured. But the kernel left `d` (the output
dim) as a GLOBAL lane, so each of 129 lanes independently re-streamed all of K/V from HBM. Score-free but
**reuse-free**: ~129× redundant reads, ~170–760× slower than SDPA. The math primitive was present; the
**locality primitive** (load a K/V tile into LDS once, reuse it across many lanes, keep online max/sum/acc in
registers, write only compact state) was missing. Reuse-free flash attention is not flash attention.

So a primitive's contract includes its memory hierarchy:
- `attention_with_LDS_tile_reuse`, not `attention`
- `matmul_with_TC_schedule`, not `matmul`
- `GEMV_with_coalesced_quant_loads`, not `GEMV`
- `decode_with_multi_queue_overlap`, not `decode`

Practical consequences:
- **Measure at the hardware boundary.** Wall-clock around `.realize()` measures host dispatch and cache
  hits, not GPU execution — it reported the reuse-free kernel as 2.7× *faster* when it was 300× slower. Use
  the GPU's own per-kernel timing (DEBUG=2 `tm`).
- **Bank correctness separately from performance.** A kernel can pass correctness/expressibility/integration
  gates and still be unshippable on speed. Record that split honestly; don't let "it works" imply "it's fast."
- **Reclassify the missing capability precisely.** When performance fails, name the *primitive* that's
  missing (here: cooperative LDS tile reuse + barrier + register-resident accumulation), not the feature
  (here: "flash attention"). That tells you whether the next arc is `[nn]` model work or `[codegen]`/`[runtime]`
  hardware-locality work — a different, harder boundary.

## Anti-Patterns

Avoid:

- duplicated business logic in multiple files
- hidden config access spread across the repo
- one-off scripts that silently redefine canonical rules
- feature work that bypasses shared system boundaries
- modules that own both policy and every downstream implementation detail
- abstractions that make the surface more confusing instead of simpler
- subsystem coupling that makes independent changes impossible
- convenience APIs that bypass canonical validation
- broad error handling at boundaries where callers need typed decisions
- privileged operations scattered through ordinary business logic
- adapters that quietly redefine shared policy
- tests that only confirm implementation details while missing boundary behavior

## Commit Discipline

Every commit should be a small, self-contained change with a project-appropriate subsystem prefix.

The allowed prefixes are project-specific. Keep them in a local override file when this scaffold is reused across projects.

Use the subsystem that owns the behavior being changed. Documentation-only changes use `[docs]` unless the documentation is part of a subsystem contract, in which case use that subsystem.

Non-functional changes must be marked as NFC:

```text
[runtime] NFC — extract preflight helper
```

Do not mix NFC refactors with behavior changes. If a cleanup enables a functional fix, split it into separate commits.

Examples:

```text
[runtime] fix audit evidence labels
[state] ignore generated runtime manifests
[docs] document commit discipline
```

If the project has a commit-message checker, use that checker as the machine-enforced version of this rule.

Malformed commits:

- missing subsystem prefix
- using NFC for a behavior change
- mixing unrelated subsystems in one commit
- bundling generated runtime files with source changes

## Practical Test

Before merging new code, ask:

1. Where is the single source of truth?
2. What module boundary owns execution?
3. Did this abstraction make the interface simpler?
4. Did this change preserve orthogonality between concerns?
5. Did this change create a duplicate rule?
6. If this integration changes, is there one place to update it?
7. Is the commit message prefixed with the owning subsystem?
8. If this is NFC, is it free of behavior changes?
9. Did the code encode the invariant instead of relying on caller discipline?
10. Are dangerous operations contained behind a small boundary?
11. Does the error shape match what the caller needs to know?
12. Is the behavior tested at the boundary where it can actually fail?
13. Has each piece of evidence been classified by what it can and cannot prove?

If those answers are unclear, the code is not shaped correctly yet.
