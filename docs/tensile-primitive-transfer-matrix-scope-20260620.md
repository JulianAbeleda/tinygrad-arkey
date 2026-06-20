# Tensile Primitive Transfer Matrix Scope

Date: 2026-06-20

Artifact:
`bench/qk-tensile-primitive-transfer/scope.json`

Command:

```bash
python3 extra/qk_tensile_primitive_transfer_matrix_scope.py
```

Verdict:
`PASS_TENSILE_PRIMITIVE_TRANSFER_MATRIX_SCOPED`

## Purpose

Stop treating Tensile as one opaque primitive.

Tensile must be decomposed into transfer rows. A future experiment is valid only if it names the row it is proving,
the local artifact it uses, and the pass/fail criterion that would let the row transfer to prefill or decode.

## Online Sources Used

- Tensile kernel parameters:
  `https://rocm.docs.amd.com/projects/Tensile/en/docs-7.1.1/src/conceptual/kernel-parameters.html`
- Tensile benchmark protocol:
  `https://rocm.docs.amd.com/projects/Tensile/en/docs-7.1.1/src/conceptual/benchmarking.html`
- Tensile nomenclature:
  `https://rocm.docs.amd.com/projects/Tensile/en/docs-7.0.2/src/reference/nomenclature.html`
- ROCm/Tensile repository status:
  `https://github.com/ROCm/Tensile`

These sources justify the split: Tensile is a benchmark-driven GEMM/tensor-contraction generator, and its kernel
parameters are independent performance knobs, not a single transferable mechanism.

## Local Sources Used

- `bench/qk-tensile-extraction/shape_matrix.json`
- `bench/qk-tensile-extraction/codegen_oracle.json`
- `bench/amd-broad-backend-roadmap/bb5a10_p8_timing_authority_reconciliation_result.json`
- `bench/qk-decode-native-tooling/readiness.json`
- `docs/prefill-tensile-DEFINITIVE-source-of-truth-20260619.md`

## Transfer Matrix

| Primitive row | Prefill status | Decode transfer status | Minimum pass |
|---|---|---|---|
| Problem form: dense GEMM / tensor contraction | Proven direct | Mostly blocked | Decode candidate must preserve q8 format and lifecycle placement while showing timing-grade movement. |
| Macro tile / workgroup / thread tile | Matched as static feature | Conditional | q8 tile family must move native-to-oracle timing by `>=30us`, or the row closes. |
| Matrix instruction / WMMA | Proven shared for dense fp16 | Low direct transfer | Keep only if fused dequant-to-WMMA or equivalent accumulation is correct and timing-grade. |
| Global read vectorization / coalescing | Proven in Tensile | Conditional | Same-binary q8 ablation with packed-block vector loads must show `>=30us` movement. |
| LDS staging layout | Descriptive, not sufficient | Blocked unless paired | No standalone LDS work. It must be paired with K-loop overlap, waits, and resource policy. |
| Software-pipelined K-loop | Primary open prefill transfer candidate | Conditional shared scheduler capability | Complete same-harness bridge, then test one overlap candidate against the valid baseline. |
| Wait/barrier schedule | Part of pipeline candidate | Conditional shared scheduler capability | Only valid with a dataflow that needs those waits; standalone wait tuning is closed. |
| Spill-free accumulator/resource policy | Primary open prefill transfer candidate | Conditional | Candidate must report scratch/private `0` and acceptable VGPR/occupancy before timing promotion. |
| Library logic / solution selection | Proven external artifact route | Policy-blocked or artifact project | Either accept external artifact policy, or define a native selection/tuning project with fallback and quality gates. |
| Timing and launch authority | Open same-harness bridge | Required for any transfer | Time captured authority and current candidates under one common synchronized or device-timestamp harness. |

## Stop Rules

- Do not run another P8 kernel variant unless it names a matrix row.
- Do not reopen standalone LDS. Our current staged-LDS candidate is correct and slow; LDS is only meaningful with
  overlapped movement and resource control.
- Do not transfer Tensile to decode by analogy. Decode needs q8 role-joined evidence and timing movement.
- Do not compare mixed-kernel or mixed-harness TFLOPS rows.
- Do not treat the external Tensile `.co` route and native tinygrad codegen transfer as the same project.

## Phases

| Phase | Name | Gate |
|---|---|---|
| PTM-0 | Freeze matrix and sources | Complete by this scope. |
| PTM-1 | Same-harness authority bridge | Captured `43.026 TFLOPS` authority kernel and current P8 candidates timed under one common harness. |
| PTM-2 | Prefill transfer decision | Choose exactly one native row: K-loop overlap, resource policy, or timing/launch correction. |
| PTM-3 | Decode applicability gate | q8 row needs `>=30us` same-binary movement, W==D quality, and role-joined gate/up evidence. |
| PTM-4 | Native or artifact policy | Decide native backend project versus external artifact dependency with fallback and provenance. |

## Next

Run PTM-1: same-harness authority bridge.

That is the next smallest proof because it resolves whether the `43.026 TFLOPS` captured authority row and current
`18-21 TFLOPS` P8 rows are separated by real kernel quality or by timing/kernel identity mismatch.
