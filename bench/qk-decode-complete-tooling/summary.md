# Decode complete tooling summary

Verdict: `COMPLETE_TOOLING_PASS_WITH_EXPLICIT_GAPS`.

## What is complete

- Schema and inventory exist for role identity, ATT body attribution, timing authority, reduce/glue Amdahl, and llama comparison.
- Q4 `attn_q/o` has full in-model ATT body attribution from commit `3aa7bb04a`.
- Q6 `ffn_down` and `lm_head` have ATT body attribution through `q6_surface_fallback`; runtime/cache identity proves the same programs are used in-model.
- llama launch and runtime rows are joined into the same artifact family.
- Timing policy is explicit: ATT is not a timer; same-process interleaved role A/B and W==D ctx sweeps are the promotion authorities.

## Role coverage

| role | capture | verdict |
|---|---|---|
| `attn_q/o` | `inmodel_activation` | `PASS_INMODEL_ROLE_JOIN_NATIVE_Q4K_COOP` |
| `ffn_down` | `q6_surface_fallback` | `PASS_INMODEL_ROLE_JOIN_NATIVE_COOP` |
| `lm_head` | `q6_surface_fallback` | `PASS_INMODEL_ROLE_JOIN_NATIVE_COOP` |
| `ffn_gate/up` | `runtime_identity_only` | `PASS_RUNTIME_IDENTITY_ATT_MISSING` |
| `attn_k/v` | `runtime_identity_only` | `PASS_RUNTIME_IDENTITY_ATT_MISSING` |

## Reduce/glue decision

`NO_REDUCE_GLUE_BUILD_GATE`. The currently priced stage-2 tax is real, but it does not clear the build gate as a standalone direct-output/reduce-fusion route.

## Timing decision

The imported llama Q4 route lost role-local timing for both `attn_output` and `ffn_gate/up`. The fused q8 artifact route remains the only measured decode speed route: min speedup `1.0506912442396314`, median speedup `1.05888873945705`, dNLL `0.0028866150416475556`.

## Remaining gaps

- Fresh ATT role-join for `ffn_gate/up` is still missing; runtime identity exists, but body attribution is not captured for that exact high-share role.
- Full-model Q6 activation capture is still blocked by the 4.68 GB AMD allocation issue; surface equivalence is acceptable for visibility, not timing promotion.
- No reliable per-kernel graph timing authority exists; final changes still need W==D ctx sweep.

## Final tooling consequence

The tooling is now complete enough to prevent the wrong build: do not fund reduce/glue fusion from packet visibility alone. The next decode implementation choice remains either the already measured q8 research flag or a project-level native scheduler/renderer effort.
