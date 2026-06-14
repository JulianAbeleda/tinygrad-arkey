# QK Three-Way Load Microbench

Decision: `wide_load_not_sufficient`

Compares the current schedulable v1 Q4_K partial kernel, the schedulable
`vector_load` kernel, and the opaque `tile_custom` wide-load kernel. This is
a diagnostic artifact only: no runtime integration or full decode follows
from this result.

Method note: v1 and `vector_load` are the apples-to-apples comparison. Both
use the schedulable primitive path and `LOCAL:0:32`. `tile_custom` is an
opaque no-LOCAL control, so it can show construction feasibility but cannot
by itself prove load-width performance.

## Summary

- tensors: `2`
- meaningful gain threshold: `5.00%`
- tie band: `3.00%`
- run full decode next: `False`
- next allowed gate: `stop_wide_load_only_branch`

## Tensor Results

| tensor | decision | v1 GB/s | vector GB/s | tile GB/s | vector vs v1 | tile vs v1 | vector status | tile status |
|---|---|---:|---:|---:|---:|---:|---|---|
| `blk.0.attn_output.weight` | `wide_load_not_sufficient` | 165.53 | 133.14 | 32.89 | -19.57% | -80.13% | `pass` | `pass` |
| `blk.0.ffn_gate.weight` | `wide_load_not_sufficient` | 384.25 | 341.65 | 36.92 | -11.09% | -90.39% | `pass` | `pass` |

## Interpretation

The schedulable vector path does not beat v1 meaningfully. Do not chase load width alone; diagnose instruction mix or downstream dot/dequant cost.
