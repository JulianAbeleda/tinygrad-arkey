# AMD ISA G3 vs Owned Weight-Path Parity Scope - 2026-06-29

## Purpose

Run the cheap, decisive parity check before committing to the offline Q4_K weight-layout reshuffle project:

```text
BUBBLEBEAM_FUTURESIGHT / generated G3 LaneMap
vs
shipped owned Q4_K warp GEMV
```

The weight-path ceiling audit selected `offline_weight_layout_reshuffle_for_q4k_gemv`, but it also recorded one
critical unresolved question:

```text
Generated-G3 is route/purity-equivalent, but speed parity vs owned was not measured in that pass.
```

Resolve that first.

## Current Ground Truth

From `bench/amd-isa-backend-weight-path-ceiling/latest.json`:

| item | result |
|---|---|
| verdict | `AMD_ISA_WEIGHT_W4_PASS_SEARCH_SPACE_READY` |
| measured achievable bandwidth | `820 GB/s` |
| realistic ceiling | `163 tok/s` |
| current owned/default | about `94-103 tok/s` |
| weight GEMV wall share | about `58%` GPU-compute |
| biggest roles | `ffn_down`, `ffn_gate_up`, `attn_qkvo_proj` |
| selected target | `offline_weight_layout_reshuffle_for_q4k_gemv` |
| caveat | G3 generated route speed parity vs owned is unmeasured |

Prior GEMV facts:

- generic scheduler GEMV is correct but about 2x off owned;
- cross-lane/reduce is not the GEMV bottleneck;
- Tensor-level packed-word restructuring cannot force owned's coalesced thread-map;
- G3 LaneMap is the pure-search-generated route for Q4_K GEMV shapes;
- purity and speed are separate verdicts.

## Question

Does generated G3 already match shipped owned warp GEMV speed for the major Q4_K weight roles?

If yes:

```text
Do not start layout reshuffle yet.
Bind/promote/search over generated G3 as the speed-equivalent pure route.
```

If no:

```text
The layout/representation project is justified.
Proceed to offline Q4_K weight-layout reshuffle.
```

## New Tool

Add:

```text
extra/amd_isa_g3_vs_owned_weight_parity.py
```

Artifacts:

```text
bench/amd-isa-backend-g3-vs-owned-weight-parity/latest.json
bench/amd-isa-backend-g3-vs-owned-weight-parity/summary.md
bench/amd-isa-backend-g3-vs-owned-weight-parity/route_counts.json
bench/amd-isa-backend-g3-vs-owned-weight-parity/per_role.json
```

## Arms

Measure at least these arms:

| arm | env | expected route |
|---|---|---|
| `owned_default` | shipped default, `BUBBLEBEAM_FUTURESIGHT=0`, `Q4K_GEMV_SCHEDULER=0` or unset | owned warp for Q4_K major roles |
| `generated_g3_bubblebeam` | `BUBBLEBEAM_FUTURESIGHT=1`, no manual bridge | generated G3 LaneMap for Q4_K major roles |
| `generated_g3_forced` | `Q4K_GEMV_SCHEDULER=6` where applicable | generated G3 LaneMap, direct forced arm |

Optional diagnostic arms:

| arm | purpose |
|---|---|
| `scheduler_fp16` | confirm old ~2x-off generic scheduler baseline if needed |
| `owned_no_proj` | only if projection route ambiguity affects attribution |

## Contexts

Measure:

```text
ctx = 512, 1024, 2048, 4096
```

Use the same clock-pinned / interleaved W==D discipline used by previous GEMV promotion gates where possible. If
clock-pinning is unavailable, report noise/spread and do not overclaim small differences.

## Required Route Attribution

For each arm and context, record:

| field | meaning |
|---|---|
| `tokens_match` | generated arm tokens match owned |
| `tok_s` | W==D tok/s |
| `route_counts` | counts of owned warp, generated G3, scheduler, bridge, fallback kernels |
| `roles_fired` | gate/up, down, q/o, q/k/v, Q6_K roles |
| `forbidden_routes_seen` | owned in generated arm, lane-partition bridge, fallback graph |
| `per_role_gpu_time` | if available from PROFILE/N4-style attribution |
| `spread_pct` | repeat noise |

Generated arm must be classified as not-pure if it uses:

- `q4k_gemv_warp_kernel` for the Q4_K roles under test;
- `q4k_lane_partition_gemv_kernel`;
- fallback graph kernels for major roles.

## Pass / Fail Thresholds

Use both speed and attribution:

| condition | verdict |
|---|---|
| generated route not clean | `AMD_ISA_G3_PARITY_BLOCKED_ROUTE_ATTRIBUTION` |
| tokens mismatch | `AMD_ISA_G3_PARITY_BLOCKED_TOKEN_MATCH` |
| generated within 5% of owned at all contexts | `AMD_ISA_G3_PARITY_PASS_MATCHES_OWNED` |
| generated within 5% at ctx512/1024 but lags >5% at ctx4096 | `AMD_ISA_G3_PARITY_MIXED_LONGCTX_LAG` |
| generated lags owned by >5% on any major Q4_K role/context | `AMD_ISA_G3_PARITY_FAILS_SPEED_LAYOUT_NEEDED` |
| measurements too noisy | `AMD_ISA_G3_PARITY_INCONCLUSIVE_NOISE` |

## Decision Branch

### If G3 Matches Owned

Do not start offline layout reshuffle yet.

Next scope should be:

```text
generated G3 promotion / search binding hardening
```

Required next outputs:

- update search-space recommendation from layout reshuffle to G3 promotion;
- ensure BubbleBeam chooses G3 without manual flags;
- run W==D and route gates;
- consider `generated_g3` as the pure replacement for owned warp where speed-equivalent.

### If G3 Lags Owned

Proceed to:

```text
offline Q4_K weight-layout reshuffle
```

The parity artifact must identify:

- which roles lag: gate/up, down, q/o, all;
- which contexts lag;
- whether lag is route-specific, role-specific, or global;
- whether generated G3 uses the expected lane map but still cannot match memory layout.

## Non-Goals

- Do not implement the offline layout in this phase.
- Do not change defaults.
- Do not modify G3 codegen unless needed for route attribution instrumentation.
- Do not optimize kernels.
- Do not edit `autogen/**`.
- Do not conflate purity with speed.

## Claude Prompt

Use this prompt verbatim:

```text
You are working in /home/ubuntu/tinygrad-arkey.

Read and follow:

  docs/archive/amd-isa-g3-vs-owned-weight-parity-scope-20260629.md

Context:
The weight-path ceiling audit passed with:

  AMD_ISA_WEIGHT_W4_PASS_SEARCH_SPACE_READY

It selected offline Q4_K weight-layout reshuffle as the next target, but with one explicit caveat:

  Generated-G3 is route/purity-equivalent, but speed parity vs owned was not measured.

Resolve that caveat before starting any layout work.

Task:
Add a cheap, decisive G3-vs-owned parity harness:

  extra/amd_isa_g3_vs_owned_weight_parity.py

Artifacts:

  bench/amd-isa-backend-g3-vs-owned-weight-parity/latest.json
  bench/amd-isa-backend-g3-vs-owned-weight-parity/summary.md
  bench/amd-isa-backend-g3-vs-owned-weight-parity/route_counts.json
  bench/amd-isa-backend-g3-vs-owned-weight-parity/per_role.json

Compare:

  owned_default
  generated_g3_bubblebeam
  generated_g3_forced

Contexts:

  512, 1024, 2048, 4096

Requirements:

  - token_match true
  - route attribution proves generated G3 fired for major Q4_K roles
  - generated arm must not silently use q4k_gemv_warp_kernel, lane-partition bridge, or fallback graph for those roles
  - W==D tok/s measured under interleaved/clock-disciplined conditions where possible
  - route_counts and per_role artifacts written

Allowed verdicts:

  AMD_ISA_G3_PARITY_PASS_MATCHES_OWNED
  AMD_ISA_G3_PARITY_MIXED_LONGCTX_LAG
  AMD_ISA_G3_PARITY_FAILS_SPEED_LAYOUT_NEEDED
  AMD_ISA_G3_PARITY_BLOCKED_ROUTE_ATTRIBUTION
  AMD_ISA_G3_PARITY_BLOCKED_TOKEN_MATCH
  AMD_ISA_G3_PARITY_INCONCLUSIVE_NOISE

Decision:

  If G3 matches owned within 5% at all contexts, do not start layout reshuffle. Recommend generated G3 promotion/search binding hardening.

  If G3 lags owned by >5%, proceed to offline Q4_K weight-layout reshuffle and record which roles/contexts lag.

Do not implement layout reshuffle in this task. This is a measurement and decision gate only.
```

