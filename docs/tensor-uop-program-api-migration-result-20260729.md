# Tensor UOp program API migration result

Date: 2026-07-29

Status: complete on EXP, validated through dev, and validated on master. No AMD device or eGPU action was used.

## Result

`Tensor.uop_program(...)` is now the canonical descriptive API for building a lazy opaque UOp program over writable Tensor buffers.

`Tensor.custom_kernel(...)` remains a silent upstream-compatibility wrapper with the same signature and return behavior. It delegates to `uop_program` and emits no warning. Current upstream still publishes the old alpha spelling, so retaining the wrapper avoids unnecessary downstream breakage while removing the misleading name from repository-owned execution paths.

The internal `UOp.custom_kernel` substrate is intentionally unchanged. It is not a provenance label and is not exposed by the LLM hot path.

## Preserved semantics

- receiver/output remains placeholder slot zero;
- positional inputs retain their existing slot order;
- contiguity, `AFTER`, and concrete `MEMORY_SEMANTIC` handling is unchanged;
- emitter and gradient callbacks are forwarded unchanged;
- graph construction remains lazy;
- the return remains one ordered Tensor per supplied argument;
- single-output helpers still return element zero;
- the EXP three-output producer still receives the complete output sequence.

No emitter, UOp graph, selected route, provenance, tensor shape, ABI, scheduling option, fallback, environment variable, or performance claim changed.

## Runtime ownership

The LLM typed boundary now calls `output.uop_program(...)` exactly once. Production route call sites still read:

- `execute_promoted_program(...)` for admitted machine-generated programs;
- `execute_oracle_program(...)` for explicitly classified oracles outside master;
- `execute_research_program(...)` or its multi-output variant for research outside master.

The transport name therefore describes mechanism, while the typed boundary describes provenance.

## Static enforcement

AST gates prove that:

- `tinygrad/llm` contains no call to the legacy Tensor spelling;
- `tinygrad/llm/kernel_program.py` is the sole LLM caller of `.uop_program(...)`;
- active repository-owned Tensor tests use the canonical spelling;
- the only remaining `.custom_kernel(...)` call syntax is in `tinygrad/tensor.py` and `tinygrad/nn/__init__.py`, where the receiver is the unchanged internal UOp substrate;
- the legacy Tensor compatibility path is exercised through a focused no-warning equivalence test.

Historical handoffs and frozen evidence were not rewritten merely to erase an accurate old API name.

## Commit map

| Purpose | EXP | dev | master |
|---|---|---|---|
| canonical Tensor API | `c6fcce5ac` | `dedd0e2fa` | `784b1de27` |
| typed runtime transport | `94c1e3789` | `95ca81a28` | `6c979c7de` |
| active caller migration | `ea842531b` | `06ca41929` | not needed; master has no retained direct Tensor test callers |
| canonical spelling gate | `d5d09bebe` | `0626b9282` | `5e0484912` |
| transport mocks | `35d3e9445` | `5728ed7c1` | `45c68c88f` |
| current documentation | `e7a6dcb4f` | `3d7d01a91` | `04af89469` |

The exhaustive scope is EXP commit `454457863` and remains EXP-only.

## Validation

EXP focused gate:

- API compatibility, typed boundary, static architecture, route/identity, CPU/PYTHON generic program, and current documentation tests: **107 passed**;
- the narrower canonical API test: **4 passed**;
- direct generic packed-weight/Q4 decode tests: **32 passed**;
- syntax checks passed for all mechanically migrated AMD-only test bodies.

Full EXP historical suite:

- **1,629 passed, 27 skipped, 4 xfailed**;
- **68 failed**, exactly the same count and categories as the pre-migration run: unavailable AMD hardware, missing ROCm/LLVM tools or libraries, platform-specific AMD binary fixtures, and existing experimental lowering assertions;
- passing count increased by five because of the new API and static coverage; no new failure appeared.

Dev focused gate:

- **76 passed**;
- static caller scan reports only the three intentional internal UOp substrate sites.

Master final gate:

- full `test/unit`: **488 passed, 11 skipped, 4 xfailed**, with 8 passing subtests;
- `python -m tinygrad.llm --help`: passed;
- `python -m tinygrad.llm.bench --metadata-only --target CPU`: passed;
- README names `Tensor.uop_program` as transport and explicitly separates it from provenance;
- no research implementation or EXP-only document was promoted.

## Completion statement

New code and documentation can now say `Tensor.uop_program` without suggesting that a promoted generated program is a hand-tuned fallback. Existing upstream callers remain source-compatible through the silent legacy wrapper.
