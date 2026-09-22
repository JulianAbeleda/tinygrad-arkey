# NVIDIA pp512 composed unroll-4 + Q4-V result (2026-08-29)

## Decision

**PASS, retain default-off.** The two independently qualified wins compose on
the same compiler gate/up + K + Q/O safe-cut graph.

Both arms used the compiler route (`--arm candidate`); the control omitted only
`--q4-v`. This is required because `--arm control` disables the compiler hooks
and is not a matched control for this experiment.

## Exact gates

- Candidate: 198 Q8 producers and 198 compiler mains (72 gate/up, 36 K, 72
  Q/O, 18 Q4-V).
- Control: 180 Q8 producers and 180 compiler mains (72 gate/up, 36 K, 72
  Q/O).
- Candidate has 198 canonical unique weights and 54 remaining FP16 overlays;
  control has 180 canonical unique weights and 72 overlays.
- Both arms report zero copies, fixups, and partial workspace, finite output,
  matching token 198, full-logit allclose, and exact deep20 stage/KV replay.

## Matched synchronized R9

| arm | minimum ms | median ms | tok/s (minimum) |
|---|---:|---:|---:|
| unroll-4 + Q4-V candidate | 67.153915 | 67.235719 | 7624.276 |
| unroll-4 no-Q4-V control | 69.165843 | 69.315714 | 7402.498 |

Q4-V recovers **2.011928 ms minimum** and **2.079995 ms median**, or **+221.778
tok/s / +2.996%** at the minimum-wall throughput. Candidate/control logits
have max absolute difference `0.10494757`, mean absolute difference
`0.013181783`, with token 198 equal.

Evidence: `docs/task_workflow/evidence/nv-prefill-composed-unroll4-q4v-20260829/`.
