# NV Q6 true-late Q8 panel-1 Gate 7 decision (2026-08-31)

## Decision

`PROMOTE_TRUE_LATE_Q8_PANEL1`

The only candidate change is a dependency-constrained panel-1 preload after the penultimate half-0 p-group. Packed trusted-FP16 arithmetic, 170-owner one-body ownership, combined initial publication, all-partials output, and the frozen fixup remain unchanged.

## Binary gate

- Candidate cubin: `33dff84bd0fa8464a241fd8c3b8efb23bba2f226dbdb191a4ddfd69cfc0fb78b`
- Frozen fixup cubin: `483de2ee3eed3597932a8632f9892377ce054e77bfe34c2420fe5a5d54ff5514`
- Panel-1 first load/store: `0x97f0` / `0xaa20`
- Panel-1 span: `291` instructions
- Instructions/registers/stack/LDL-STL: `5144` / `255` / `0 B` / `0/0`

## Correctness

- Partials/final uint32 exact: `True` / `True`
- Trusted maximum/mean error: `0.00067138671875` / `2.1467494661919773e-05`
- Trusted failing elements: `0`

## Locked R31

| Arm | Main median | Fixup median | Total median |
|---|---:|---:|---:|
| Anchor | 232.096 us | 24.704 us | 256.800 us |
| True late | 227.232 us | 24.960 us | 252.032 us |

Main candidate-anchor paired median/wins: `-4.704 us`, `31/31`. Total: `-4.736 us`, `31/31`.

GPU lock acquired: `True`. Evidence: `docs/task_workflow/evidence/nv-q6-ordered-region-bridge-gate15-20260902/result.json`
