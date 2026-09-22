# Fused RMSNorm to shared-Q8 provider qualification record

Date: 2026-08-05
Verdict: **provider boundary PASS; full-model promotion NO-GO**

## Compiler blocker and fix

The initial fused provider failed final NV program verification because the
metadata global-store address did not mention the local lane axis. GPU-dim
lowering therefore inserted an implicit lane-zero `Invalid` index. The store
already carried its own lane-zero gate, so the late gate mover could not
consume the nested implicit gate.

The emitter now makes lane ownership explicit in both places: active lane zero
selects the exact Q8_1 metadata index, inactive lanes select safe in-bounds
index zero, and the store remains lane-zero gated. No shared codegen rule was
changed. A hermetic final-NV rewrite regression passes under `SPEC=1` for the
admitted fp32 and fp16 contracts.

## Isolated numerical boundary

`scratchpad/nv_fused_rmsnorm_q8_provider_gate.py` compares the fused provider
against the exact two-program construction it replaces: the qualified
ordinary RMSNorm emitter followed by the ordinary llama-Q8_1 provider.

Native NV results for normal, all-zero, and dynamic-range deterministic
inputs were bitwise equal: every case had zero mismatches across all 1152
packed `uint32` outputs, including int8 packets and fp16 `d|sum` metadata.
Machine-readable authority:
`docs/task_workflow/output/nv-fused-rmsnorm-q8-provider-gate-20260805.json`.

## Fresh full-model qualification

The fresh-process d512/count8 comparison leased only block 1. Census observed
exactly one `rmsnorm_q8_1_llama_provider_4096` and exactly three shared-Q8
consumers. Full logits were finite; all eight generated tokens and every
argmax matched g0.

The requested promotion gate nevertheless failed:

| field | g0 | fused block 1 |
| --- | ---: | ---: |
| programs | 876 | 876 |
| fused providers | 0 | 1 |
| shared-Q8 consumers | 0 | 3 |
| max absolute logit delta | - | 0.0133566856 |

The numerical threshold was `<=0.01`, and topology required a strict program
count decrease. Both failed. The exact program multiset replaced six programs
with six: Q6's external reduction disappeared, but boundary/generic program
changes consumed that count saving. The route was not timed, scaled, or
promoted. Full authority:
`docs/task_workflow/output/nv-shared-q8-fused-qualification-20260805.json`.

## Consequence

The fused provider itself is no longer a correctness or compiler blocker. The
remaining route questions are consumer approximation and topology ownership.
A subsequent scope must either fuse/group consumers enough to produce a real
program-count reduction or justify a wall-only gate explicitly; it must not
silently relax the existing `0.01` numerical contract.
