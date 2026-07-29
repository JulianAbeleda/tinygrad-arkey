# Tensor custom-kernel alias removal result

Date: 2026-07-29

Status: complete and validated through master. No GPU or eGPU action was used.

## Decision

The temporary `Tensor.custom_kernel(...)` upstream-compatibility wrapper was removed at the user's direction. This fork now exposes one Tensor-level API for the mechanism: `Tensor.uop_program(...)`.

This is intentionally source-incompatible for external callers that still use the upstream alpha spelling. No repository-owned active caller used the wrapper when it was removed.

The internal `UOp.custom_kernel(...)` substrate remains unchanged. The remaining `custom_kernel_attention` identifiers are historical route/config symbol names, not the removed Tensor method; changing them is outside this narrow API removal.

## Commit map

| Purpose | EXP | dev | master |
|---|---|---|---|
| remove Tensor alias and update gates | `d68171ef6` | `5f4a47112` | `7a7dfa965` |
| remove compatibility documentation claim | `dc01d0144` | `b8c0f305a` | `4f0411b21` |

## Validation

- EXP focused API/LLM/route gate: **101 passed**.
- Dev focused API/LLM/route gate: **70 passed**.
- Master full `test/unit`: **487 passed, 11 skipped, 4 xfailed**, with 8 passing subtests.
- Master LLM CLI help: passed.
- Master CPU metadata smoke: passed.
- `hasattr(Tensor, "custom_kernel")` is false by test.
- AST policy finds no active Tensor legacy caller.
- Static source search finds only the internal UOp method and unrelated historical route/config identifiers.

No emitter, UOp graph, route, provenance, ABI, tensor shape, scheduling behavior, or performance path changed.
