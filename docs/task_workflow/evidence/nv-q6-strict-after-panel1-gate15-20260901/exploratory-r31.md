# NV Q6 true-late Q8 panel-1 Gate 7 decision (2026-08-31)

## Decision

`REJECT_TRUE_LATE_Q8_PANEL1_PERFORMANCE`

The only candidate change is a dependency-constrained panel-1 preload after the penultimate half-0 p-group. Packed trusted-FP16 arithmetic, 170-owner one-body ownership, combined initial publication, all-partials output, and the frozen fixup remain unchanged.

## Binary gate

- Candidate cubin: `51edb7a2406f81c565702597b6f1fb5f934ea659194a0a7b84186da5489e96cc`
- Frozen fixup cubin: `483de2ee3eed3597932a8632f9892377ce054e77bfe34c2420fe5a5d54ff5514`
- Panel-1 first load/store: `0x9780` / `0xa980`
- Panel-1 span: `288` instructions
- Instructions/registers/stack/LDL-STL: `5136` / `255` / `0 B` / `0/0`

## Correctness

- Partials/final uint32 exact: `True` / `True`
- Trusted maximum/mean error: `0.00067138671875` / `2.1467494661919773e-05`
- Trusted failing elements: `0`

## Locked R31

| Arm | Main median | Fixup median | Total median |
|---|---:|---:|---:|
| Anchor | 229.824 us | 24.608 us | 254.432 us |
| True late | 228.384 us | 24.896 us | 253.216 us |

Main candidate-anchor paired median/wins: `-1.312 us`, `27/31`. Total: `-1.760 us`, `23/31`.

GPU lock acquired: `True`. Evidence: `docs/task_workflow/evidence/nv-q6-strict-after-panel1-gate15-20260901/exploratory-r31.json`
