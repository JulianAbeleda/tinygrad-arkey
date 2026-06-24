# AMD ATT In-Model Role Join Result

Date: 2026-06-19

Artifacts:

- Scope: `docs/amd-att-inmodel-role-join-scope-20260619.md`
- Probe: `extra/qk_att_inmodel_role_join.py`
- Result: `bench/qk-att-inmodel-role-join/result.json`
- Summary: `bench/qk-att-inmodel-role-join/summary.md`

## Verdict

`PASS_INMODEL_ROLE_JOIN_NATIVE_Q4K_COOP`.

The `blk.0.attn_output` in-model role interval was traced with ATT and joined to the exact HCQ programs launched inside
the interval. It uses the intended native tinygrad Q4_K coop primitive in-model. There is no fallback/dense route and no
runtime-cache identity bug for this role.

## Gates

| Gate | Result |
|---|---:|
| ATT start/stop sync | PASS |
| ATT body packets | PASS |
| HCQ programs captured | PASS |
| decode primitives enabled | PASS |
| native Q4_K coop present | PASS |

## Target

| Field | Value |
|---|---:|
| Role | `blk.0.attn_output` |
| Linear type | `Q4KPrimitiveLinear` |
| Shape | `4096 x 4096` |
| Activation shape | `[1, 1, 4096]` |
| Decode enabled | `true` |
| Parts | `1` |
| Kernel mode | `partial` |

The activation was captured from the actual block-0 attention path, then the role call was warmed once outside ATT and
traced inside an ATT interval.

## Program Join

The role interval launched three HCQ programs:

| Program | Launch | Role |
|---|---:|---|
| `q4k_coop_partial_4096_4096` | global `[256,1,1]`, local `[16,8,1]` | native Q4_K coop |
| `r_32_32_4_8` | global `[32,1,1]`, local `[32,1,1]` | stage-2 reduce/glue |
| `E_32_32_4n1` | global `[32,1,1]`, local `[32,1,1]` | reshape/glue |

ATT trace:

| Metric | Value |
|---|---:|
| Body-like packets | `16,137` |
| `VALUINST` | `13,136` |
| `INST` | `2,094` |
| `WAVESTART` / `WAVEEND` | `182 / 182` |
| Nonzero bytes | `35,161` |

## Meaning

For this role, the in-model path is not secretly using a different compiled kernel, dense fallback, or wrong runtime
cache entry. The runtime/cache-identity hypothesis is closed for `blk.0.attn_output`.

The remaining explanation for this role is the lifecycle around the intended primitive:

- native Q4_K coop partial kernel;
- separate stage-2 reduction;
- glue kernel;
- graph/model scheduling around those kernels.

This matches the earlier decode diagnostic: there is a real stage-2/reduce tax, but this single Q4_K attention-output
surface is not large enough to explain the full model-level `76% standalone -> 44% in-model` collapse by itself.

## Decision

Do not fund a runtime-cache fix for Q4_K `attn_output`; this role already launches the intended program in-model.

If continuing ATT on decode, use the same role-join method on a higher-share Q6_K role:

```text
ffn_down or lm_head
```

Those roles are more likely to explain the model-level residual. If they also show correct program identity, the decode
gap should be treated as scheduler/resource/lifecycle project work rather than a bounded wiring bug.

