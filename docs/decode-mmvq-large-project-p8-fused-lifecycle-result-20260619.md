# Decode MMVQ large project P8 fused lifecycle result - 2026-06-19

Purpose: execute `decode-mmvq-large-project-p8-fused-lifecycle-scope-20260619.md`.

Artifacts:

- `extra/qk_decode_mmvq_p8_fused_lifecycle_decision.py`
- `bench/qk-decode-mmvq-large-project/p8_fused_lifecycle_decision.json`

## Verdict

**P8_COMPLETE_ARTIFACT_YES_NATIVE_PROJECT_LEVEL**.

The fused q8+MMVQ primitive is real and build-worthy, but not as imported llama Q4 routing and not as a bounded current
tinygrad-native edit.

## P8a - Lower-Bound Model

P8a passes.

| item | value |
|---|---:|
| current tinygrad `ffn_gate/up` baseline | `168.54 us` |
| required for `1.10x` local win | `153.22 us` |
| imported consumer math from P6, gate+up | `50.53 us` |
| q8 producer from P5 | `6.30 us` |
| additive lower bound | `56.83 us` |

The lower bound is far below the gate. The primitive is worth building if the lifecycle can be fused enough to avoid
the separate-launch overhead that killed P7e.

## P8b - Current Native Expressibility

P8b says current native routes are not enough:

| route | measured | gate result |
|---|---:|---|
| COMGR fused gate/up lifecycle | `177.72 us` | FAIL |
| AMD DSL full gate/up consumer only | `166.65 us` | FAIL |
| graph artifact route | PASS | artifact/stub only |
| DSL capability map | `FAIL_A1_NO_BOUNDED_FEATURE` | project-level |

tinygrad can express multi-output stubs and graph-swapped artifact nodes, but the mature native schedule is not available
as a bounded UOp/codegen edit.

## P8c - Handwritten Prototype

P8c passes for the artifact route:

| route | value |
|---|---:|
| modeled q8 lifecycle | `107.64 us` |
| hipcc/LLD artifact lifecycle | `115.24 us` |
| speedup vs P7e baseline | `1.46x` |
| graph route | PASS |

This proves the primitive can clear the local gate when the consumer schedule is mature enough and the gate/up consumer
is fused.

## P8d - Decision

Decision:

- imported llama Q4 route: **closed** as a local timing win;
- fused q8/MMVQ artifact route: **feasible research flag**;
- native tinygrad route: **project-level renderer/scheduler transfer**.

Whole-decode W==D evidence for the q8 route:

| ctx | baseline tok/s | q8 route tok/s | speedup |
|---:|---:|---:|---:|
| 128 | `79.5` | `84.5` | `1.063x` |
| 512 | `73.0` | `77.4` | `1.060x` |
| 1024 | `71.3` | `75.4` | `1.058x` |
| 4096 | `65.1` | `68.4` | `1.051x` |

## Consequence

Do not continue imported Q4 artifact routing. It is correct and useful as an oracle, but it loses after lifecycle costs.

If decode work continues, there are only two coherent paths:

1. Ship or keep the q8 fused artifact route as a research flag, with its known quality/policy gates.
2. Fund native renderer/scheduler transfer against the artifact oracle. That is a compiler/backend project, not a
   bounded primitive patch.
