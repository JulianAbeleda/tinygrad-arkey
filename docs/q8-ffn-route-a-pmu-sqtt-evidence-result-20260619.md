# q8 FFN Route A PMU/SQTT evidence result (2026-06-19)

Executed the post-A1 evidence gate from `q8-ffn-route-a-scheduler-codegen-result-20260619.md`.

Verdict: **NO_A2_REOPEN**.

Route A A2 remains closed for q8 decode. The native tinygrad path still has no bounded compiler/codegen feature with a
credible `>=30us` movement.

## Artifacts

Probe:

- `extra/q8_ffn_route_a_pmu_sqtt_evidence.py`

Outputs:

- `bench/q8-ffn-amd-scheduler-project/pmu_sqtt_evidence.json`
- `bench/q8-ffn-amd-scheduler-project/pmu_sqtt_pmc_q8_gateup_full.json`
- `bench/q8-ffn-amd-scheduler-project/pmu_sqtt_sqtt_q8_gateup_full.json`

## What ran

The probe ran the real-GGUF q8 fused gate/up AMD DSL/ASM consumer under tinygrad's built-in AMD profiling hooks:

- `PROFILE=1 PMC=1`
- `PROFILE=1 SQTT=1`, with a small SQTT buffer and single traced shader-engine mask

This avoids HIP/ROCm attachment. It exercises the same HCQ/KFD path as the q8 research route.

## Result

| item | result |
|---|---:|
| PMC profile runnable | yes |
| SQTT capture runnable | yes |
| PMC events | `2` |
| SQTT events | `12` |
| SQTT blob bytes | `1,775,712` |
| SQTT decode usable | no |
| A2 reopen | no |

The captured program is the intended kernel:

- `q8_b2b_fullrow_reduce`

The PMC event schedule includes:

- `SQ_BUSY_CYCLES`
- `SQ_INSTS_VALU`
- `SQ_INSTS_SALU`
- `SQC_LDS_IDX_ACTIVE`
- `SQC_LDS_BANK_CONFLICT`
- `GRBM_GUI_ACTIVE`
- `GL2C_HIT`
- `GL2C_MISS`

SQTT capture produced non-empty instruction-trace blobs, but the local decoder failed on every instruction-trace blob:

```text
ValueError('unknown cdna format word=0xf4080100')
```

So the state is precise:

- tinygrad HCQ-level PMU/SQTT collection is available for this q8 path;
- the current local SQTT decoder is not a usable instruction-timeline attribution oracle here;
- PMC/SQTT therefore does not identify a bounded feature that clears the A2 `>=30us` gate.

## Interpretation

This does **not** mean hardware feedback is impossible. It means the available hardware feedback does not currently
change the Route A decision.

A1 already named the concrete static differences:

- vector/coalesced load shape;
- `s_clause` / `s_delay_alu` scheduling markers;
- wait/reduction details;
- descriptor/local-id/runtime encoding;
- register/live-range scheduler differences.

But A1 could not assign any one feature a credible `>=30us` movement. This PMU/SQTT pass tried to strengthen that
claim with runtime evidence. It succeeded at capture, but not at decoded attribution. Without decoded stall/timeline
evidence, reopening A2 would violate the project rule: do not fund a deeper build unless the failed layer and movement
budget are named.

## Decision

Do not start Route A A2 for q8 decode.

Route A remains a project-level AMD scheduler/codegen roadmap item:

- latency-aware instruction scheduling;
- register/live-range scheduling;
- semantic placement of scheduler annotations;
- load-width/coalescing as part of a whole scheduler;
- robust SQTT decode/counter attribution for HCQ kernels.

Route B remains the practical q8 research route:

- hipcc/LLD artifact imported through HCQ;
- graph-safe;
- `115.24us` isolated lifecycle;
- `1.051-1.063x` W==D decode;
- default off and policy-bound.

The only legitimate way to reopen Route A for q8 decode is new attribution that names a bounded `>=30us` feature or a
funded broader AMD scheduler/codegen project that is justified beyond this single primitive.
