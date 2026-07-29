# Kernel program execution boundary result

Date: 2026-07-29

Status: complete on `exp`, validated through `dev`, and production-only changes validated on `master`. No AMD device or eGPU action was used.

## Result

Explicit LLM program execution now carries declared lifecycle provenance instead of exposing `Tensor.uop_program` at each route call site:

- promoted production programs use `execute_promoted_program` and accept only machine-search- or tinygrad-scheduler-generated provenance;
- EXP/dev research programs use `execute_research_program`;
- the one three-output EXP producer uses `execute_research_program_outputs`, preserving its complete output tuple;
- the dev-only legacy flash executor is explicitly research-only;
- oracle vocabulary remains available for a future independent fallback, but no current call site was falsely classified as an oracle.

The boundary does not infer provenance from names. It validates explicit route/program IDs and a callable emitter, then delegates once to the unchanged low-level Tensor transport.

## Production routes migrated

- Q4_K G3 decode GEMV
- Q6_K decode GEMV and large-vocabulary reduction
- admitted G4/G5 flash-decode tile and combine programs
- admitted fused-prefill attention program

Flash execution now refuses to label an unadmitted or configuration-mismatched geometry as promoted. The EXP head-dimension sweep constructs the same descriptor directly and executes it as research-only.

No emitter body, UOp graph, route choice, ABI, tensor shape, scheduling option, fallback decision, environment gate, or benchmark claim changed.

## Static boundary

AST-based tests enforce that:

- no direct `.custom_kernel(...)` compatibility call remains under `tinygrad/llm`;
- the only direct `.uop_program(...)` call under `tinygrad/llm` is the transport in `tinygrad/llm/kernel_program.py`;
- production LLM source cannot import or call oracle/research executors;
- EXP/dev runtime, benchmark, qualification, and campaign source cannot call `.custom_kernel(...)` directly.

Tests that directly exercise the generic Tensor mechanism remain allowed. `Tensor.custom_kernel` remains a silent
upstream-compatibility spelling; it does not assign provenance to a caller.

## Commit map

| Purpose | EXP | dev | master |
|---|---|---|---|
| typed boundary | `da9891060` | `60135b83a` | `5a76822a8` |
| promoted hot paths | `f6eba061a` | `b892ac69b` | `326bf6b8a` |
| production static gate | `9582b737c` | `a2559e8e0` | `b034db86b` |
| research classification | `69ebe9728` | `ae5d99687` | not promoted |
| EXP/dev research static gate | `c748db645` | `3e3bd56e2` | not promoted |
| dev-only legacy flash classification | not applicable | `b3f839206` | not promoted |

The exhaustive scope is EXP commit `78fc5c8ea` and remains EXP-only.

## Validation

Focused EXP production gate:

- boundary, decode route/kernel, flash descriptor/identity, and fused-attention ownership: **62 passed**;
- boundary plus current-decode/MMQ/state-phase EXP selection: **65 passed, 7 skipped**, with four current-decode failures reproduced on unchanged dev because this Mac lacks the AMD execution/toolchain path.

Full EXP historical suite:

- **1,624 passed, 27 skipped, 4 xfailed**;
- **68 failed**, all in the existing hardware/toolchain/baseline-sensitive EXP categories (no AMD device, no `/dev/kfd`, missing ROCm LLVM tools/libraries, AMD binary fixture variance, and pre-existing experimental lowering assertions). The refactor-focused suites above are green.

Dev focused gate after its dev-only legacy migration:

- **65 passed**;
- static search reports only `tinygrad/llm/kernel_program.py` as a direct low-level caller.

Master final gate:

- full `test/unit`: **483 passed, 11 skipped, 4 xfailed**, with 8 passing subtests;
- `python -m tinygrad.llm --help`: passed;
- `python -m tinygrad.llm.bench --metadata-only --target CPU`: passed;
- no master import from `extra.llm_research` and no EXP implementation/oracle was promoted.

## Evidence boundary

This was a naming, ownership, and enforcement refactor. AMD/eGPU recertification is intentionally deferred as an evidence refresh because no generated program, selection policy, or execution semantics changed.
