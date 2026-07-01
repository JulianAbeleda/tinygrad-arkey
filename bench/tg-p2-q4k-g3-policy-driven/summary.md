# TG-P2: Q4_K G3 Policy-Driven Selection

Verdict: **TG_P2_PASS_Q4K_G3_POLICY_DRIVEN**

Selector authority for the generated Q4_K G3 decode GEMV moved from the model-side env default
(BubbleBeam / DECODE_Q4K_G3_ANYSHAPE) to the BoltBeam route policy (`boltbeam.route_policy.v1`,
`selected_route=decode_q4k_g3_generated`). This is a selector-authority migration: output and speed
are preserved; owned warp stays one rollback flag away (`BUBBLEBEAM_FUTURESIGHT=0`).

## What changed
- **BoltBeam** emits `decode_q4k_g3_generated` selected rows **per weight tensor** from profile role/quant/shape
  facts (structural eligibility `(cols//256)%4==0 and rows%32==0`), not once per attention shape. New candidate
  `decode_q4k_g3_generated` (`machine_authored_generated`, default-on) + a data-driven `shape_rule` on the
  candidate manifest. No model-name hardcode.
- **tinygrad** consumes those rows: `_SUPPORTED_QK_ROUTE_IDS` gains `decode_q4k_g3_generated`;
  `_load_qk_route_policy` validates G3 params/shape; `_qk_route_policy_selects_q4k_g3(out,in)` binds per tensor;
  `Q4KPrimitiveLinear.__call__` authorizes G3 when the policy selects it and **fails loud in strict mode** on any
  silent fallback (`TG_P2_BLOCKED_HIDDEN_FALLBACK`).

## Gates
| gate | result |
|---|---|
| BoltBeam policy tests | PASS (per-tensor eligible G3; Q6_K + ineligible-shape negative; suite 246 passed) |
| tinygrad unit tests | PASS (8 passed) |
| route-bound smoke | PASS (policy selects G3 with `DECODE_Q4K_G3_ANYSHAPE=0` -> G3 kernel fires) |
| rollback smoke | PASS (`BUBBLEBEAM_FUTURESIGHT=0` -> owned warp) |
| strict hidden-fallback | PASS (ineligible selected tensor raises) |
| census | PASS (`PMS_R0_PASS_CENSUS_PINNED`, no new purity debt) |
| no-policy default | PASS (byte-identical output) |

## Token/W==D basis
existing_g3_authority (bench/amd-isa-backend-g3-weight-promotion/latest.json: token_match + route_clean all ctx, lag -0.13..+0.41% ctx512-4096) + fresh route-bound byte-identity (route_bound.json)

## Remaining purity debt (unchanged, out of scope for TG-P2)
- decode_q6k_coop_shipped (TG-P3)
- decode_attention_owned_two_kernel (TG-P5)
- prefill_pipe_role_selective_default (TG-P4)
