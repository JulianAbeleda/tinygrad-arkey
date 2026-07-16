# Authored core LOC consolidation scope

Date: 2026-07-16

## Objective

Reduce accidental duplication and mixed ownership in the largest authored tinygrad modules without changing public APIs, generated-kernel behavior, scheduling order, device discovery, model selection, or runtime semantics. A file split is useful only when it creates a durable ownership boundary; moving the same code between files is not a LOC improvement by itself.

## Baseline

- Authored budgeted source: 29,995 / 30,000 token-bearing Python/JavaScript lines (`python3 sz.py`).
- Candidate surface: 4,393 physical lines / 3,410 budgeted lines across `llm/{model,admission,qk_primitives,prefill_routes}.py` and `codegen/{opt/{kernel_lds,postrange,packed_weight},late/devectorizer}.py`.
- Focused baseline: 181 passing tests and three known `test_current_prefill_execution_adapter.py` AMD compile failures (one missing candidate identity binding and two spill-free VGPR overflows).

## Definition of complete

The consolidation is complete when all of the following hold:

1. Repeated Q4K/Q6K installation and direct-packed execution mechanics have one shared owner; format-specific layout and decode semantics remain explicit.
2. Admission inputs, context-memory terms, and selected-model inventory are constructed through one canonical path used by exact and compatibility flows.
3. Tensor-core/LDS precontract candidate construction and validation have one codegen owner; `postrange` only supplies scheduling context and applies the result.
4. Repeated late-codegen lane projection, register-store reconstruction, reduction-width, packed serialization, and dequant scaffolding are consolidated where the shared contract is exact.
5. Large diagnostic/UI modules are split only along independently testable application or protocol boundaries. Runtime/ISA hot paths are split only when fixtures prove the move mechanical.
6. Intentional public API aggregators (`tensor.py`, `mixin/__init__.py`) and cohesive algorithms remain intact.
7. Focused tests pass with no new failures; the three AMD baseline failures either remain byte-for-byte equivalent in cause or are fixed deliberately.
8. Import-boundary, size-accounting, compileall, and broad relevant unit suites pass. No production core module eagerly imports research-only `extra.qk` code.
9. `python3 sz.py` passes with useful headroom below 30,000. LOC reduction must come from deletion/consolidation, not generated markers or moving authored code to an unbudgeted directory.
10. The final diff has no debug artifacts, dead compatibility aliases, unexplained hardcoded device/model choices, or unrelated worktree changes.

## Work tracks

### A. Quantized model primitives

Owner: `tinygrad/llm/qk_primitives.py`.

- Share installation traversal, packed-view creation, materialization/alias bookkeeping, registry updates, and budget summaries.
- Keep Q4K/Q6K block layouts, storage dtypes, transforms, and linear execution hooks format-owned.
- Required coverage: both quant formats, alias/materialization ownership, resident and direct-packed modes, route purity.

### B. Direct-packed prefill execution

Owner: `tinygrad/llm/prefill_routes.py`.

- Introduce a minimal format adapter for packed weight metadata and kernel invocation.
- Share transfer, output/scratch allocation, part dispatch, and partial reduction.
- Preserve route selection, strictness, semantic markers, module roles, and direct-output behavior.

### C. Model admission and loading policy

Owners: `tinygrad/llm/admission.py` and policy/loading portions of `tinygrad/llm/model.py`.

- Canonicalize `AdmissionInputs` construction and context-memory accounting.
- Reuse selected-model inventory across exact and compatibility plans.
- Preserve scanned GPU/VRAM facts, explicit user model selection, trained/requested context semantics, and cache-mode decisions.

### D. Tensor-core/LDS candidate construction

Owners: `tinygrad/codegen/opt/kernel_lds.py` and the tensor-core candidate portion of `postrange.py`.

- Put geometry derivation, validation, producer/fragment instantiation, and contract assembly behind one API.
- Keep pass ordering and scheduling policy in `postrange`.
- Preserve candidate identity, warm-start context, packed-weight transform, UOp shape, and resource gates.

### E. Late-codegen and packed-weight helpers

Owners: `tinygrad/codegen/late/devectorizer.py` and `tinygrad/codegen/opt/packed_weight.py`.

- Share lane/index projection only where address and dtype semantics are identical.
- Share vector register-store reconstruction and reduction-width derivation.
- Centralize packed component serialization and dequant expression scaffolding while retaining declarative format records.
- Reject abstractions that add branches to hot rewrite callbacks without deleting more complexity than they introduce.

### F. Safe ownership splits

- `tinygrad/viz/js/index.js`: graph, profiler, rewrite browser, then shared state/UI utilities.
- `tinygrad/viz/serve.py`: HTTP routing, rewrite serialization, profile/SQTT loading, disassembly/CFG support.
- AMD PSP: normal boot/ring protocol versus opt-in diagnostics/experiments; extract repeated system-memory mapping only with hardware-independent tests.
- AMD/NV runtime: move genuinely identical queue/device lifecycle behavior into HCQ support; do not force transport-specific packet or synchronization behavior through a common abstraction.

### G. High-risk follow-up boundaries

- AMD ISA: instruction selection, WMMA/LDS lowering, lifetime/wait transforms, and encoding/facade are distinct responsibilities, but extraction requires stable renderer and ISA fixtures first.
- UOp operations: IR node API, pattern DSL, rewrite engine, and tracing are distinct responsibilities, but imports, caches, generated matchers, and rewrite hot paths make a mechanical split unsafe without a dedicated compatibility gate.
- X86 ISA and NV runtime follow the same rule: split only after behavior fixtures cover encoding/ABI and queue synchronization respectively.

Current audit decision: AMD ISA and UOp extraction are no-go in this cycle. The AMD extraction fixture is already red at the baseline
(three tensor-core programs emit 792/2336/1132 bytes versus stale expectations of 680/1940/772), so it cannot prove a mechanical move.
Before extraction it needs selected-UOp snapshots at every matcher seam, fresh-process imports, interpreted/generated matcher parity, and
exact final-stream hashes. UOp extraction additionally needs a fresh-process matrix covering interning/GC, pickle identity, metadata,
`UPat.location`, matcher caches, tracing substitution, rewrite traversal modes, and the lazy `extra.qk` boundary.

These tracks are considered scoped but not automatically changed: organizational churn without a verified ownership or deduplication gain fails the completion definition.

## Verification gates

1. Per-track focused unit tests and before/after `sz.py` file counts.
2. Cross-track LLM admission, model ownership, prefill route, kernel LDS, packed weight, and devectorizer tests.
3. Current-prefill compile fixtures compared with the recorded three-failure baseline.
4. Size accounting, core/research import-boundary tests, and fresh-process import probes.
5. `python3 -m compileall -q tinygrad`.
6. Broad relevant unit suite, then diff/ownership self-review and final LOC report.
