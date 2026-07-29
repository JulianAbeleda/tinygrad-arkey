# Tensor UOp program API migration scope

Date: 2026-07-29

Promotion direction: implement and validate on `exp`, promote to `dev`, validate, then promote the production API/runtime/docs subset to `master`.

## Problem

`Tensor.custom_kernel(...)` describes the mechanism as “custom” even though the method says nothing about authorship, tuning, selection, or provenance. In this repository, promoted machine-search programs, research programs, and generic tinygrad mechanism tests can all use the same lazy graph-construction primitive. A new reader can therefore mistake the method name for evidence of a hand-tuned fallback.

The implementation is not an eager kernel launcher. It constructs a lazy opaque UOp program over writable buffer arguments and returns one ordered Tensor per argument. The public name should describe that behavior.

## Outcome

`Tensor.uop_program(...)` becomes the canonical public spelling. It owns the existing implementation and documentation.

`Tensor.custom_kernel(...)` remains a silent compatibility wrapper with the same signature and return contract because current upstream still exposes that alpha API. Repository-owned active runtime paths use `uop_program`; the old spelling remains only for compatibility validation and historical records.

This migration changes no emitter, UOp graph, call ordering, selected route, provenance, ABI, tensor shape, contiguity rule, gradient callback, scheduling behavior, or performance claim.

## Pinned API and semantics

```python
def uop_program(self, *inputs: Tensor, fxn: Callable,
                grad_fxn: Callable | None = None) -> list[Tensor]: ...

def custom_kernel(self, *inputs: Tensor, fxn: Callable,
                  grad_fxn: Callable | None = None) -> list[Tensor]:
  """Compatibility spelling for uop_program."""
  return self.uop_program(*inputs, fxn=fxn, grad_fxn=grad_fxn)
```

Canonical semantics, preserved exactly:

- The receiver is argument slot zero; positional inputs occupy slots one through N.
- Every supplied Tensor is a potentially writable program argument.
- Ordinary arguments become contiguous before placeholder construction.
- Existing `AFTER` nodes are preserved without redundant materialization.
- A `MEMORY_SEMANTIC` value with concrete buffer identity is preserved.
- `fxn` receives matching UOp placeholders and returns the lazy program body.
- `grad_fxn` is forwarded unchanged.
- The result is a `list[Tensor]` with one element per supplied argument, each ordered after the same opaque call.
- Result element zero is conventionally the receiver/output, but the API is not single-output-only.
- Neither spelling realizes or launches the program eagerly.

The new docstring must use “build a lazy opaque UOp program,” not “execute a custom kernel.”

## Compatibility policy

- Keep `Tensor.custom_kernel` silent: no `DeprecationWarning`, logging, or environment gate.
- Keep its exact argument order, keywords, defaults, and return type.
- Do not assign provenance to legacy callers based on the method name.
- Do not publish a removal date while upstream still uses the old spelling.
- Do not rename `UOp.custom_kernel` in this migration. It is the internal upstream substrate and renaming it would create unrelated fork surface.
- Do not add a second implementation. The compatibility method delegates to the canonical Tensor method.

## Inventory baseline

At scope time all worktrees are clean:

- EXP `997710b83`
- dev `b3f839206`
- master `b034db86b`

Direct Python `.custom_kernel(...)` call sites:

- EXP: 25 actual calls across core and generic mechanism tests, plus one prose false-positive.
- dev: 23 actual calls.
- master: four calls: the Tensor wrapper’s internal `UOp.custom_kernel`, two direct internal UOp calls in `tinygrad/nn`, and the LLM typed boundary.

Master has no direct Tensor mechanism test calls. EXP/dev retain broader generic and AMD-only tests.

Current production LLM transport is already centralized in `tinygrad/llm/kernel_program.py`; that boundary is the sole production Tensor caller to migrate.

## Non-goals

- Do not remove or warn on `Tensor.custom_kernel`.
- Do not rename `UOp.custom_kernel`.
- Do not rename Ops, emitters, generated artifacts, kernel symbols, route IDs, or environment variables.
- Do not use this API rename to reclassify provenance.
- Do not rename `custom_kernel_attention` and historical prefill configuration identifiers in the same commit series; those are route/config vocabulary and require a separate compatibility decision.
- Do not mechanically rewrite archived handoffs, historical benchmark records, or frozen JSON evidence.
- Do not alter GPU code or run the eGPU.

## Work packages

### UP0 — Scope and baselines

Owner: orchestrator.

- Record semantics, caller counts, branch heads, compatibility policy, promotion rules, and gates.
- Commit this scope on EXP before implementation.

Acceptance: this document is the first migration commit.

### UP1 — Canonical Tensor API

Files:

- `tinygrad/tensor.py`
- a focused CPU test, preferably `test/unit/test_uop_program.py`

Implementation:

- Move the current Tensor implementation body to `uop_program` unchanged.
- Add the silent `custom_kernel` compatibility wrapper.
- Keep the underlying `UOp.custom_kernel` call unchanged.

Focused tests:

- canonical and compatibility signatures match;
- both names preserve list arity and result ordering;
- both build equivalent lazy `AFTER`/`CALL` structure;
- multiple outputs are retained;
- `fxn` and `grad_fxn` forwarding are unchanged;
- the legacy spelling emits no warning;
- a CPU/PYTHON numeric smoke passes without GPU hardware.

Acceptance: CPU tests pass and `tinygrad/tensor.py` has exactly one implementation body.

### UP2 — Production runtime migration

Files:

- `tinygrad/llm/kernel_program.py`
- `test/unit/test_llm_kernel_program.py`
- `test/unit/test_llm_kernel_program_boundary.py`
- current production README/docs/comments that describe the Tensor API

Requirements:

- The typed LLM boundary calls `output.uop_program(...)` exactly once.
- Single-output promoted/oracle/research helpers retain result-zero behavior.
- Multi-output research execution retains the complete list/sequence.
- Mocks and assertions use the canonical name.
- AST policy requires `uop_program` for production LLM transport and rejects direct legacy Tensor calls in production LLM source.
- Current README wording says `Tensor.uop_program` is a transport, not provenance, and notes the silent upstream compatibility spelling once.

Acceptance: existing route, identity, multi-output, CLI, and static tests pass unchanged in meaning.

### UP3 — Active EXP/dev caller migration

Files: retained generic execution tests and current non-archival documentation/policy assertions only.

- Migrate active repository-owned Tensor calls in EXP/dev tests to `.uop_program(...)`.
- Retain one focused compatibility test using `.custom_kernel(...)`.
- Update assertions that describe the current canonical interface.
- Do not rewrite archived handoffs or immutable evidence solely to remove a historical token.
- Do not touch direct `UOp.custom_kernel` calls in `tinygrad/nn`.

Acceptance: AST search finds no repository-owned active Tensor legacy call outside the compatibility test/wrapper; AMD-only test bodies are syntax-preserving mechanical changes and are not executed on this Mac.

### UP4 — Static naming gates

Production gate:

- `tinygrad/llm` contains no attribute call named `custom_kernel`.
- `tinygrad/llm/kernel_program.py` is the only production LLM attribute caller named `uop_program`.
- The Tensor compatibility wrapper is present and delegates to `uop_program`.
- Direct internal `UOp.custom_kernel` remains allowed.

EXP/dev gate:

- Active source, benchmarks, campaigns, and retained tests do not call the Tensor legacy spelling except the focused compatibility test.
- Archived prose and frozen evidence are outside the static AST gate.

Do not use a broad textual ban because `custom_kernel` remains a valid compatibility API and appears truthfully in history.

### UP5 — EXP validation

Required before promotion:

1. New Tensor API/compatibility tests.
2. LLM kernel-program boundary and static tests.
3. Decode, flash, fused-attention ownership, and UOp identity tests.
4. CPU/PYTHON generic program tests.
5. Syntax checks for mechanically migrated AMD-only tests.
6. Full EXP historical suite, with pre-existing hardware/toolchain failures separated from regressions.
7. `git diff --check` and clean commit boundaries.

### UP6 — Promotion

Commit boundaries:

1. `[docs] scope Tensor UOp program API migration` — EXP only.
2. `[tensor] add canonical UOp program API` — promote to dev and master.
3. `[runtime] use canonical UOp program transport` — promote to dev and master.
4. `[test] migrate active UOp program callers` — promote only files present and appropriate on each branch.
5. `[docs] document canonical UOp program interface` — current docs only; split by branch surface.
6. `[docs] record Tensor UOp program migration result` — EXP only.

Promotion rules:

- EXP validates first.
- Dev receives the canonical API and migrations for its retained qualification/research surface, then validates.
- Master receives the minimal production/API/docs subset only after dev passes.
- Master receives no EXP implementation, campaign, archived handoff, or research artifact.
- Resolve branch conflicts by preserving legitimate retained tests/features while applying identical Tensor semantics.

### UP7 — Publication

EXP:

- Focused canonical/compatibility and architectural gates green.
- Full suite result recorded with environment-dependent failures separated.

Dev:

- Focused API, LLM, retained fallback/research, and static gates green.

Master:

- Full `test/unit` green.
- LLM CLI help and CPU metadata smoke green.
- Production LLM source has no direct legacy Tensor call.
- README names `Tensor.uop_program` as transport, not provenance.

Push EXP, dev, and master only after their gates pass, then verify each local HEAD equals its remote branch.

## Completion definition

The migration is complete when `Tensor.uop_program` is the canonical documented and repository-owned spelling; the old Tensor name is a silent compatibility wrapper only; production LLM transport uses the canonical method behind the typed provenance boundary; active EXP/dev calls and tests are migrated without rewriting history; all focused/master gates pass; the result is documented; and EXP, dev, and master are clean, pushed, and remote-aligned.
