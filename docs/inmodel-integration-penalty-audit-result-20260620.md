# In-Model Integration Penalty Audit Result - 2026-06-20

Verdict: `PASS_INMODEL_INTEGRATION_AUDIT_AMDAHL_LEDGER`

Run:

```bash
DEV=AMD PREFILL_V2=1 PROFILE=1 PYTHONPATH=. python3 extra/qk_inmodel_integration_penalty_audit_probe.py
```

Output:

```text
bench/qk-inmodel-integration-penalty/inmodel_integration_penalty_audit_result.json
```

## Result

The authority profile is `symbolic_start_pos_bound_0`, matching the banked `qk_prefill_v2_measure.py` path.

| row | value |
|---|---:|
| banked PREFILL_V2 wall | `183ms / 512` |
| profiled HCQ graph span | `179.94ms` |
| PROFILE-inflated host wall | `462.62ms` |
| graph kernels | `321` |

The graph span, not the PROFILE host wall, is the timing authority for this audit.

## Role Split

| component | share | GPU time |
|---|---:|---:|
| FFN/projection matmul | `71.15%` | `128.03ms` |
| attention/KV | `25.21%` | `45.36ms` |
| lm_head | `1.67%` | `3.00ms` |
| elementwise glue | `1.64%` | `2.96ms` |
| other | `0.33%` | `0.59ms` |

## Amdahl Ledger

| hypothetical improvement | full prefill speedup |
|---|---:|
| matmul `1.10x` | `1.069x` |
| matmul `1.25x` | `1.166x` |
| matmul `45 -> 78.6 TFLOPS` equivalent | `1.437x` |
| attention `1.25x` | `1.053x` |
| remove all non-matmul | `1.406x` |

## Decision

Matmul is still the dominant in-model bucket, but a small isolated GEMM win does not move production much:
a `10%` matmul-only gain projects only `~6.9%` full-prefill speedup. Further prefill GEMM work needs either a
large in-model matmul improvement or graph/integration work that preserves the isolated gain inside the full
PREFILL_V2 lifecycle.

Decode inherits the same rule: q8 producer/consumer rows must be lifecycle-proven, not promoted from isolated
component speed.
