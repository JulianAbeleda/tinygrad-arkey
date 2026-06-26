# Decode generated fused PV tile scope

## Goal

Expose a generated/searchable decode-attention PV tile that has the same physical economics as the owned decode tile, without using the owned precompiled kernel.

Target verdict:

`GENERATED_FUSED_PV_TILE_ROUTE_CLEAN_AND_WD_COMPETITIVE`

The current generated split x-lane route is correctness-clean but performance-refuted. The wall audit verdict is:

`PV_WALL_CONFIRMED__GENERATED_GLOBAL_COLUMN_SCALAR_CODEGEN`

That means existing generated attention does not fail because of route flags, host sync, dispatch count, or online-state recurrence. It fails because the generated PV stage is represented as global output-column work, not as a fused tile lifecycle.

## Current wall

Current generated split route:

`score -> max -> flash_xlane_pv_from_m -> gmax -> den -> combine`

The refuted PV program is:

`flash_xlane_pv_from_m_32_128`

Its structural issue:

| Property | Current generated PV | Needed fused PV tile |
|---|---|---|
| Output column `d` | `AxisType.GLOBAL` | local/cooperative ownership |
| Token loop | x-lane reduced | tile-local, register/LDS backed |
| V reuse | materialized per output column | reused inside tile |
| K/V staging | no fused LDS tile | explicit LDS/register tile lifecycle |
| Dot primitive | scalar generated multiply chain | packed half dot lowering, ideally `v_dot2` |
| Searchability | route selectable, but wrong representation | tile shape and primitive choices searchable |

Owned oracle facts already confirmed by `bench/qk-isa-primitive-audit/owned_decode_attention.json`:

| Primitive | Owned route has it |
|---|---:|
| vector dot | yes |
| LDS | yes |
| cross-lane | yes |
| vector global load | yes |
| spills | no |

## Definition of the generated fused PV tile

A candidate can enter W==D only if it satisfies this minimum shape:

1. The candidate owns one KV split tile, not one scalar output column.
2. The candidate stages K/V or equivalent tile-local data for reuse.
3. The candidate computes score and online softmax state inside the tile lifecycle.
4. The candidate accumulates PV inside the same tile lifecycle.
5. The candidate writes partial PV plus metadata once per `(query head, split)` or equivalent compact tile output.
6. The candidate uses local/cooperative `d` ownership, not `d = AxisType.GLOBAL` as the primary PV axis.
7. The candidate has a route identity separate from the refuted `flash_xlane_pv_from_m_*` route.

## Gates

| Gate | Requirement | Failure verdict |
|---|---|---|
| P0 scope gate | repo has scope + executable blocker gate | `FUSED_PV_TILE_SCOPE_INCOMPLETE` |
| P1 structural gate | generated fused tile builder exists and has local/cooperative `d` ownership | `FUSED_PV_TILE_BLOCKED__NO_GENERATED_TILE_BUILDER` |
| P2 standalone numeric gate | candidate output matches NumPy reference on fixed seeded tensors | `FUSED_PV_TILE_FAIL__NUMERIC` |
| P3 route gate | candidate fires in decode graph, owned tile absent | `FUSED_PV_TILE_FAIL__ROUTE` |
| P4 materialization gate | no `E_49152` regression | `FUSED_PV_TILE_FAIL__MATERIALIZATION` |
| P5 W==D gate | no catastrophic collapse, promotion band met | `FUSED_PV_TILE_REFUTED__WD` |

## Kill rules

Stop and classify instead of tuning blindly if any of these happen:

| Failure | Classification |
|---|---|
| cannot express local `d` plus token lanes plus GQA group state | codegen representation gap |
| route fires but emits `E_49152` | materialization regression |
| standalone numeric fails | reducer/state semantics bug |
| route clean but W==D remains collapsed | generated tile still lacks owned primitive economics |
| candidate requires precompiled owned binary | not pure generated/search route |

## Implementation sequence

1. `extra/qk_decode_attention_fused_pv_tile_gate.py`

   Purpose: canonical executable gate for the fused generated PV tile project. Initially records the current blocker and refuses to call the route promotable until a real builder exists.

2. Add a generated builder in `extra/qk_flash_decode.py`.

   Proposed name:

   `flash_fused_pv_tile_whole_cache_kernel`

   Required program identity:

   `flash_fused_pv_tile_whole_cache_32_128`

3. Add standalone numeric mode to the gate.

   It should compare candidate PV/meta output against a NumPy reference for fixed `(Hq=32,Hkv=8,Hd=128,L=128)` tensors.

4. Wire model route only after P2 passes.

   Proposed default-off flag:

   `DECODE_ATTN_FUSED_PV_TILE=1`

5. Add candidate to decode eval only after P3/P4 pass.

6. Run W==D only after route-clean and materialization-clean gates pass.

## Promotion rule

Do not promote based on standalone numeric success. Promotion requires:

| Requirement | Source |
|---|---|
| token correctness | route gate |
| generated route fires | route gate |
| owned tile/combine absent | route gate |
| no `E_49152` | materialization gate |
| W==D competitive | decode eval |
| latest artifact committed | bench artifact |

## Expected first verdict

Until `flash_fused_pv_tile_whole_cache_kernel` exists, the canonical verdict should be:

`FUSED_PV_TILE_BLOCKED__NO_GENERATED_TILE_BUILDER`

This is useful: it prevents confusing the refuted split x-lane route with the desired fused tile route.
