# AMD ISA Weight Path Real Ceiling Scope - 2026-06-29

## Purpose

Start the next decode-speed track from the real wall:

```text
Q4_K / Q6_K weight-memory path
```

The decode-attention ceiling audit closed the attention track with:

```text
AMD_ISA_ATTENTION_CEILING_PASS_MOVE_TO_NON_ATTENTION
```

The decisive result was that decode is weight-memory-bound, not attention-bound. The next track must therefore start
from the weight-read floor and derive the real headroom in FFN/projection GEMV.

This is an exhaustive scope for the new starting point:

```text
weight floor -> shipped owned/generated GEMV routes -> full decode W==D -> search/codegen/layout levers
```

Do not start by tuning kernels. Start by proving the ceiling and loss stack.

## Current Ground Truth

### Decode attention is not the wall

From the ceiling audit:

| item | value |
|---|---:|
| Qwen3-8B-Q4_K weight-read floor | `5.03 GB / 960 GB/s = 5.24 ms/token = 191 tok/s` |
| realistic bandwidth ceiling estimate | about `153 tok/s` at 80% bandwidth |
| attention KV-read floor @ctx4096 | about `35 us`, less than 1% of weight floor |
| owned route vs weight floor | about `54% / 50%` at ctx512/4096 |
| native attention route vs floor | about `37% / 30%` |

Attention conclusion:

```text
Matching owned attention buys only +10.5% @ctx512 and +2.9% @ctx4096.
Search should move to non-attention FFN/weight path.
```

### Existing GEMV facts

The codebase already has mature Q4_K/Q6_K decode primitive paths in `tinygrad/llm/model.py` and `extra/`.

Important existing routes:

| route | status |
|---|---|
| `Q4K_GEMV_WARP=1` | shipped/default-on for Q4_K FFN gate/up and down when shape allows |
| `Q4K_GEMV_WARP_PROJ=1` | shipped/default-on for Q4_K 4096x4096 q/o projection; measured +1.6% W==D |
| `Q4K_GEMV_SCHEDULER=1` | generic scheduler fp16-logical GEMV; correct but about 2x off owned |
| `Q4K_GEMV_SCHEDULER=2/3/5` | packed/tensor scheduler experiments; correct but not fast enough |
| `Q4K_GEMV_SCHEDULER=6` | generated G3 LaneMap codegen route |
| `BUBBLEBEAM_FUTURESIGHT=1` | now routes Q4_K gate/up, down, and 4096x4096 projections through generated G3 LaneMap in the pure-search result |
| Q6_K primitive routes | active for mixed quant paths such as FFN down, with separate coop/warp experiments |

Prior results to preserve:

| artifact/doc | conclusion |
|---|---|
| `docs/scheduler-gemv-vs-owned-result-20260625.md` | generic scheduler GEMV is about 2x off owned; reduce/cross-lane is not the bottleneck |
| `docs/q4k-scheduler-coalesced-gemv-result-20260625.md` | Tensor-level packed-word restructuring cannot impose owned coalescing/thread-map; the gap is representation/codegen |
| `docs/gemv-pure-search-generated-route-scope.md` | G3 generated LaneMap eventually became pure-search-generated for Q4_K GEMV shapes |
| `bench/qk-proj-gemv-warp/decision.json` | Q4_K projection warp route transfers +1.6% W==D and is default-on |

The historical hard lesson:

```text
The owned GEMV edge is not the reduce.
It is packed-word coalescing + in-register dequant lifecycle + block-group-K/thread-map.
```

## New Tooling Track

Add a new audit suite:

```text
extra/amd_isa_weight_path_ceiling_audit.py
extra/amd_isa_weight_path_route_attribution.py
extra/amd_isa_weight_path_probe_matrix.py
extra/amd_isa_weight_path_search_scope_builder.py
```

Artifacts:

```text
bench/amd-isa-backend-weight-path-ceiling/latest.json
bench/amd-isa-backend-weight-path-ceiling/summary.md
bench/amd-isa-backend-weight-path-ceiling/weight_floor.json
bench/amd-isa-backend-weight-path-ceiling/route_attribution.json
bench/amd-isa-backend-weight-path-ceiling/probe_matrix.json
bench/amd-isa-backend-weight-path-ceiling/search_space_recommendation.json
```

## Phase W0 - Weight Floor And Role Inventory

Goal: establish the mathematical and practical ceiling for decode weight reads.

Required model:

| item | required |
|---|---|
| model profile | model name, quant mix, layer count, hidden dim, FFN dim, vocab/head metadata if needed |
| persistent weight bytes | total persistent bytes for Q4_K/Q6_K decode primitives |
| per-token touched bytes | estimate and measured/metadata-derived bytes read per decode token |
| bandwidth assumptions | measured local bandwidth if available; otherwise conservative `960 GB/s` and 80% realistic ceiling |
| theoretical ceiling | `bytes_per_token / bandwidth` -> tok/s |
| realistic ceiling | apply measured efficiency or 80% assumption |
| current W==D | owned/default route tok/s at ctx512/1024/2048/4096 |

Role inventory must include:

| role | shapes |
|---|---|
| FFN gate/up | usually `4096 x 12288`, two linears or fused gateup |
| FFN down | usually `12288 x 4096` |
| attention q/k/v/o projections | usually `4096 x 4096` |
| mixed Q6_K roles | especially FFN down in Q4_K_M-style profiles |
| lm_head / embeddings if decode path touches them | record or explicitly exclude |

W0 verdicts:

```text
AMD_ISA_WEIGHT_W0_PASS_FLOOR_PINNED
AMD_ISA_WEIGHT_W0_INCONCLUSIVE_MISSING_MODEL_BYTES
AMD_ISA_WEIGHT_W0_INCONCLUSIVE_MISSING_WD
```

## Phase W1 - Route Attribution And Wall Share

Goal: prove which weight kernels dominate full decode wall and whether they are shipped owned, generated G3, scheduler,
or fallback graph kernels.

Add:

```text
extra/amd_isa_weight_path_route_attribution.py
```

Required outputs per context:

| field | meaning |
|---|---|
| `kernel_name` | program name |
| `role` | ffn_gate, ffn_up, ffn_down, attn_q, attn_k, attn_v, attn_o, other |
| `quant` | Q4_K, Q6_K, Q8, fp16, unknown |
| `route_class` | owned_warp, generated_g3, scheduler, coop, fallback_graph, copy/reduce |
| `calls_per_token` | call count |
| `gpu_time_per_token` | measured per-kernel GPU time |
| `wall_share_estimate` | Amdahl-corrected where possible |
| `bytes_estimate` | estimated bytes read |
| `effective_bandwidth` | bytes/time |
| `tokens_match` | if route variation is compared |

Must separate:

- shipped default route;
- BubbleBeam/generated G3 route;
- explicit owned warp route;
- scheduler fallback route;
- any Q6_K route.

W1 verdicts:

```text
AMD_ISA_WEIGHT_W1_PASS_WALL_ATTRIBUTED
AMD_ISA_WEIGHT_W1_BLOCKED_ROUTE_ATTRIBUTION
AMD_ISA_WEIGHT_W1_INCONCLUSIVE_PROFILE_NOISE
```

## Phase W2 - Owned/Generated/Math Gap Decomposition

Goal: decompose the gap for each weight role.

For each role and route, compute:

```text
weight_floor_time
measured_kernel_time
effective_bandwidth_pct
over_floor_ratio
gap_to_owned
gap_to_realistic_floor
full_decode_gain_if_perfect
full_decode_gain_if_match_best_route
```

This phase must answer:

1. Is the shipped owned warp route itself only ~50% of the weight floor?
2. Does generated G3 match owned for all major Q4_K roles, or only route/purity?
3. Is Q6_K now the limiting mixed-quant role?
4. Which role has the highest Amdahl-adjusted headroom?
5. Is the next lever codegen, weight layout, fusion, or quant policy?

W2 verdicts:

```text
AMD_ISA_WEIGHT_W2_PASS_GAP_DECOMPOSED
AMD_ISA_WEIGHT_W2_PASS_OWNED_IS_NEAR_CEILING
AMD_ISA_WEIGHT_W2_PASS_GENERATED_LAGS_OWNED
AMD_ISA_WEIGHT_W2_INCONCLUSIVE_MISSING_ROUTE
```

## Phase W3 - Probe Matrix

Goal: create controlled probes for each plausible weight-path lever.

Add:

```text
extra/amd_isa_weight_path_probe_matrix.py
```

Required probes:

| probe id | lever | purpose |
|---|---|---|
| `WP0_BANDWIDTH_MEASURE` | raw bandwidth | measure local effective read bandwidth for packed-weight-sized streams |
| `WP1_OWNED_WARP_ROLE_SWEEP` | owned warp coverage | verify gate/up, down, q/o projection route and per-role speed |
| `WP2_GENERATED_G3_ROLE_SWEEP` | generated route parity | compare BubbleBeam G3 vs owned for gate/up, down, q/o |
| `WP3_Q6K_ROLE_SWEEP` | mixed quant | measure Q6_K roles and whether Q6_K is now the wall |
| `WP4_GATEUP_FUSION` | horizontal fusion | test whether fused gate/up weight read improves wall or worsens locality |
| `WP5_WEIGHT_LAYOUT_RESHUFFLE_SIM` | Marlin-style layout | estimate or prototype offline layout that makes packed-word lane map natural |
| `WP6_DEQUANT_LIFECYCLE` | in-register dequant | measure redundant dequant / nibble extraction work vs loaded word reuse |
| `WP7_BLOCK_GROUP_K` | thread-map/K split | sweep K-block grouping / rows per workgroup / lane word ownership |
| `WP8_DIRECT_OUT_NO_PARTIALS` | partial reduction overhead | test direct output vs partials+sum where legal |
| `WP9_QUANT_POLICY` | Q4/Q6 role policy | estimate demoting/upgrading selected roles by bytes and quality risk metadata |

Each probe must report:

```text
probe_type: measurement-only | semantic-preserving | semantic-masking | layout/offline | microkernel
token_match_required
route_attribution_required
bytes_moved_delta
kernel_time_delta
W==D_delta
decision: pursue | refuted | inconclusive
```

W3 verdicts:

```text
AMD_ISA_WEIGHT_W3_PASS_PROBE_MATRIX_READY
AMD_ISA_WEIGHT_W3_PASS_LEVER_SELECTED
AMD_ISA_WEIGHT_W3_PASS_ALL_LOCAL_LEVERS_REFUTED
AMD_ISA_WEIGHT_W3_BLOCKED_CORRECTNESS
```

## Phase W4 - Search Space Builder

Goal: produce a search-owned candidate space for the next implementation phase.

Add:

```text
extra/amd_isa_weight_path_search_scope_builder.py
```

Output:

```text
bench/amd-isa-backend-weight-path-ceiling/search_space_recommendation.json
```

Candidate axes must be explicit:

| axis | examples |
|---|---|
| route | owned_warp, generated_g3, scheduler, new_native_isa |
| role | gate/up, down, q/o, q/k/v, Q6_K role |
| weight layout | current GGUF Q4_K, reshuffled lane-major, block-group-K layout |
| dequant lifecycle | per-use, per-word in-register, amortized activation quant, q8 dot |
| thread map | row-per-wg, word-col lanes, block-group-K split |
| output mode | partials+sum, direct_out |
| fusion | gateup fused, qkv fused, none |
| quant policy | keep Q6_K, demote to Q4_K, q8 artifact, mixed |

Each candidate must include:

```text
expected_ceiling
risk
implementation_cost
correctness_gate
route_gate
promotion_gate
do_not_search_reason if refuted
```

W4 verdicts:

```text
AMD_ISA_WEIGHT_W4_PASS_SEARCH_SPACE_READY
AMD_ISA_WEIGHT_W4_BLOCKED_NO_LIVE_LEVER
```

## Decision Rules

The final audit should choose exactly one next implementation target:

| condition | next target |
|---|---|
| generated G3 lags owned by >=5% on major roles | improve generated codegen parity |
| owned route is only 50-60% of floor and dominates wall | optimize owned/native GEMV algorithm/layout |
| Q6_K roles dominate residual wall | Q6_K route/quant policy track |
| direct_out removes meaningful partial/reduce overhead | direct_out route hardening |
| layout reshuffle shows >=10% W==D ceiling | offline weight-layout search |
| no role has >=5% W==D ceiling | stop weight-local tuning and broaden system target |

## Non-Goals

- Do not change defaults.
- Do not implement a new GEMV kernel in this audit.
- Do not re-open decode attention unless the weight audit contradicts the attention ceiling.
- Do not present semantic-masking probes as promotable.
- Do not edit `autogen/**`.
- Do not conflate pure-search purity with speed; both must be measured separately.

## Required Artifacts

Final `latest.json` must include:

```json
{
  "verdict": "...",
  "weight_floor": {},
  "role_inventory": {},
  "route_attribution": {},
  "gap_decomposition": {},
  "probe_matrix": {},
  "search_space_recommendation": {},
  "decision": {
    "next_target": "...",
    "reason": "..."
  }
}
```

`summary.md` must include:

- current full-decode W==D table;
- weight floor / realistic floor table;
- per-role wall-share table;
- owned vs generated vs floor table;
- probe matrix table;
- selected next implementation target;
- explicit refuted levers.

## Claude Prompt

Use this prompt verbatim:

```text
You are working in /home/ubuntu/tinygrad-arkey.

Read and follow:

  docs/amd-isa-weight-path-real-ceiling-scope-20260629.md

Context:
The decode-attention ceiling audit closed the attention track with:

  AMD_ISA_ATTENTION_CEILING_PASS_MOVE_TO_NON_ATTENTION

Decode is weight-memory-bound. The next track starts from the weight-read floor:

  Qwen3-8B-Q4_K: 5.03 GB / 960 GB/s = 5.24 ms/token = 191 tok/s hard floor
  realistic 80% bandwidth ceiling ~= 153 tok/s
  current owned/default ~= 94-103 tok/s

Prior GEMV work already proved:

  - generic scheduler GEMV is correct but about 2x off owned;
  - cross-lane/reduce is not the bottleneck;
  - Tensor-level packed-word restructuring cannot force owned coalesced thread-map;
  - the owned edge is packed-word coalescing + in-register dequant lifecycle + block-group-K/thread-map;
  - G3 generated LaneMap achieved pure-search-generated route coverage for Q4_K GEMV shapes, but speed/floor headroom must now be re-audited from the weight wall.

Task:
Build an audit-only weight-path ceiling and search-scope suite.

Add:

  extra/amd_isa_weight_path_ceiling_audit.py
  extra/amd_isa_weight_path_route_attribution.py
  extra/amd_isa_weight_path_probe_matrix.py
  extra/amd_isa_weight_path_search_scope_builder.py

Artifacts:

  bench/amd-isa-backend-weight-path-ceiling/latest.json
  bench/amd-isa-backend-weight-path-ceiling/summary.md
  bench/amd-isa-backend-weight-path-ceiling/weight_floor.json
  bench/amd-isa-backend-weight-path-ceiling/route_attribution.json
  bench/amd-isa-backend-weight-path-ceiling/probe_matrix.json
  bench/amd-isa-backend-weight-path-ceiling/search_space_recommendation.json

Phases:

  W0: weight floor and role inventory
  W1: route attribution and wall share
  W2: owned/generated/math gap decomposition
  W3: controlled probe matrix
  W4: search-space recommendation

Roles to audit:

  FFN gate/up
  FFN down
  attention q/k/v/o projections
  Q6_K mixed-quant roles
  any fallback graph/copy/reduce kernels that show nontrivial wall share

Routes to distinguish:

  owned_warp
  generated_g3
  scheduler
  coop
  fallback_graph
  Q6_K route

Required probes:

  WP0_BANDWIDTH_MEASURE
  WP1_OWNED_WARP_ROLE_SWEEP
  WP2_GENERATED_G3_ROLE_SWEEP
  WP3_Q6K_ROLE_SWEEP
  WP4_GATEUP_FUSION
  WP5_WEIGHT_LAYOUT_RESHUFFLE_SIM
  WP6_DEQUANT_LIFECYCLE
  WP7_BLOCK_GROUP_K
  WP8_DIRECT_OUT_NO_PARTIALS
  WP9_QUANT_POLICY

Final decision:
Choose exactly one next implementation target, or explicitly stop weight-local tuning if no role has >=5% W==D ceiling.

Constraints:

  - audit only; do not implement a new GEMV kernel
  - do not change defaults
  - do not edit autogen/**
  - do not re-open attention unless the weight audit contradicts the ceiling
  - keep pure-search purity and speed as separate verdicts
  - stop at first hard blocker only if it prevents producing a truthful audit artifact

Final report must include:

  - weight floor
  - realistic ceiling
  - per-role wall share
  - route attribution
  - owned vs generated vs floor
  - probe matrix results
  - selected next implementation target
  - refuted levers
  - exact files changed
```

