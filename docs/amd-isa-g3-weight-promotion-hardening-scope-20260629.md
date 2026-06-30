# AMD ISA G3 Weight Promotion Hardening Scope - 2026-06-29

## Purpose

Promote/harden the generated G3 LaneMap Q4_K GEMV route now that it has proven speed parity with the owned hand-warp
GEMV.

This is the follow-up to:

```text
AMD_ISA_G3_PARITY_PASS_MATCHES_OWNED
```

The objective is not to make a new kernel. The objective is to make the already-proven generated route the clean
search-owned route for the Q4_K weight roles where it matches owned.

## Current Ground Truth

`extra/amd_isa_g3_vs_owned_weight_parity.py` proved:

| ctx | owned | G3 BubbleBeam | delta |
|---:|---:|---:|---:|
| 512 | 103.95 | 103.64 | 0.30% |
| 1024 | 102.27 | 101.85 | 0.41% |
| 2048 | 99.63 | 99.36 | 0.27% |
| 4096 | 94.90 | 94.61 | 0.31% |

Route attribution:

```text
BUBBLEBEAM_FUTURESIGHT=1:
  Q4_K gate/up -> G3 LaneMap
  Q4_K q/o     -> G3 LaneMap
  Q4_K down    -> G3 LaneMap
  owned warp leakage: 0
  lane-partition bridge leakage: 0
  fallback graph leakage: 0
  token_match: true
```

Decision:

```text
Do not start offline Q4_K layout reshuffle.
G3 already matches owned. Promote/harden G3 as the pure speed-equivalent route.
```

## Promotion Target

Roles eligible for G3 promotion:

| role | shape | quant | current proof |
|---|---|---|---|
| FFN gate/up | `4096 -> 12288` | Q4_K | G3 parity proven |
| FFN down | `12288 -> 4096` | Q4_K | G3 parity proven under BubbleBeam |
| attention q/o projection | `4096 -> 4096` | Q4_K | G3 parity proven |

Roles not included:

- Q6_K routes;
- lm_head;
- arbitrary non-Q4_K shapes;
- prefill/batched GEMM;
- decode attention tile.

## New Tool

Add:

```text
extra/amd_isa_g3_weight_promotion_gate.py
```

Artifacts:

```text
bench/amd-isa-backend-g3-weight-promotion/latest.json
bench/amd-isa-backend-g3-weight-promotion/summary.md
bench/amd-isa-backend-g3-weight-promotion/route_counts.json
bench/amd-isa-backend-g3-weight-promotion/search_space_update.json
```

## Required Work

### P0 - Bind Promotion Contract

Define the exact candidate contract:

```text
candidate_id: q4k_gemv_g3_lanemap_generated
status: speed_equivalent_to_owned
search_generation_status: search_generated
eligible_roles:
  - q4k_ffn_gate_up_4096_12288
  - q4k_ffn_down_12288_4096
  - q4k_attn_qo_4096_4096
```

The contract must include:

- required env flags;
- rollback env flags;
- forbidden routes;
- correctness gate;
- promotion threshold;
- shape/quant guards.

### P1 - BubbleBeam Default Selection

Make BubbleBeam/FutureSight select G3 for the eligible Q4_K roles without manual scheduler forcing.

Important distinction:

```text
BUBBLEBEAM_FUTURESIGHT=1
```

is acceptable as the search-selection mode. But the route must not require:

```text
Q4K_GEMV_SCHEDULER=6
```

for normal BubbleBeam selection.

The forced scheduler flag may remain diagnostic only.

### P2 - Route Hardening

For generated-G3 promotion arms, route attribution must prove:

- G3 kernels fire for eligible Q4_K roles;
- owned warp kernels do not fire for those roles;
- lane-partition bridge does not fire;
- fallback graph does not fire for those roles;
- Q6_K and other non-target routes are unchanged and explicitly excluded.

### P3 - W==D Promotion Gate

Measure at:

```text
ctx512
ctx1024
ctx2048
ctx4096
```

Required:

- token match at every context;
- G3 within 5% of owned/default at every context;
- deterministic repeated run or noise/spread recorded;
- route counts recorded.

### P4 - Search Space / Docs Update

Update the active search-space recommendation so the layout reshuffle is no longer the next target while G3 parity holds.

Required output:

```text
bench/amd-isa-backend-g3-weight-promotion/search_space_update.json
```

Fields:

```json
{
  "retire_or_deprioritize": ["offline_q4k_weight_layout_reshuffle"],
  "promote_candidate": "q4k_gemv_g3_lanemap_generated",
  "do_not_search": [
    "generic_scheduler_gemv",
    "tensor_packed_word_restructure",
    "cross_lane_reduce_only"
  ],
  "rollback": {
    "disable_g3": "...",
    "force_owned": "..."
  }
}
```

## Allowed Verdicts

```text
AMD_ISA_G3_PROMOTION_PASS_SPEED_EQUIVALENT
AMD_ISA_G3_PROMOTION_BLOCKED_ROUTE_ATTRIBUTION
AMD_ISA_G3_PROMOTION_BLOCKED_TOKEN_MATCH
AMD_ISA_G3_PROMOTION_BLOCKED_SPEED_REGRESSION
AMD_ISA_G3_PROMOTION_INCONCLUSIVE_NOISE
```

## Success Criteria

Pass only if:

1. token match holds at all contexts;
2. G3 route fires for all eligible Q4_K roles under BubbleBeam selection;
3. owned/bridge/fallback leakage is zero for eligible roles;
4. W==D is within 5% of owned/default at all contexts;
5. rollback path remains available;
6. search-space update records G3 as speed-equivalent and layout reshuffle as deprioritized.

## Non-Goals

- Do not implement a new layout.
- Do not change Q6_K routes.
- Do not change prefill.
- Do not change decode attention tile.
- Do not remove owned kernels from the repository.
- Do not remove rollback flags.
- Do not edit `autogen/**`.

## Claude Prompt

Use this prompt verbatim:

```text
You are working in /home/ubuntu/tinygrad-arkey.

Read and follow:

  docs/amd-isa-g3-weight-promotion-hardening-scope-20260629.md

Context:
The G3-vs-owned parity gate passed:

  AMD_ISA_G3_PARITY_PASS_MATCHES_OWNED

G3 BubbleBeam matches owned within <0.5% at ctx512/1024/2048/4096, with token_match true and clean route attribution:

  Q4_K gate/up -> G3
  Q4_K q/o     -> G3
  Q4_K down    -> G3
  owned warp leakage: 0
  bridge leakage: 0
  fallback leakage: 0

Therefore do not start the offline Q4_K layout reshuffle. Promote/harden G3 as the pure speed-equivalent Q4_K GEMV route.

Task:
Add the promotion gate:

  extra/amd_isa_g3_weight_promotion_gate.py

Artifacts:

  bench/amd-isa-backend-g3-weight-promotion/latest.json
  bench/amd-isa-backend-g3-weight-promotion/summary.md
  bench/amd-isa-backend-g3-weight-promotion/route_counts.json
  bench/amd-isa-backend-g3-weight-promotion/search_space_update.json

Requirements:

  - define candidate q4k_gemv_g3_lanemap_generated
  - eligible roles: Q4_K gate/up, Q4_K down, Q4_K q/o projection
  - BubbleBeam/FutureSight selection should use G3 without requiring Q4K_GEMV_SCHEDULER=6 in normal mode
  - forced scheduler flag may remain diagnostic only
  - token_match true at ctx512/1024/2048/4096
  - route attribution proves G3 fires and owned/bridge/fallback do not leak for eligible roles
  - W==D within 5% of owned/default at all contexts
  - update search_space_update.json to promote G3 and deprioritize offline layout reshuffle while parity holds
  - keep rollback flags

Allowed verdicts:

  AMD_ISA_G3_PROMOTION_PASS_SPEED_EQUIVALENT
  AMD_ISA_G3_PROMOTION_BLOCKED_ROUTE_ATTRIBUTION
  AMD_ISA_G3_PROMOTION_BLOCKED_TOKEN_MATCH
  AMD_ISA_G3_PROMOTION_BLOCKED_SPEED_REGRESSION
  AMD_ISA_G3_PROMOTION_INCONCLUSIVE_NOISE

Do not remove owned kernels. Do not change Q6_K, prefill, or decode attention. Do not edit autogen/**.
```

