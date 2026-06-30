# Pure Machine Search: Remaining Hot-Kernel Scope

Date: 2026-06-30

Status: execution scope for moving the current default path closer to pure machine search. This is a handoff for Claude.
It updates the rough "about four kernels" framing into a phase plan with citations, gates, and non-goals.

## Executive Summary

The default path is much closer to pure machine search than the raw kernel count suggests, but it is not pure yet.

Current state:

| default-path area | current writer | current selector | actual state | pure-search status |
|---|---|---|---|---|
| Decode Q4_K GEMV | generated G3 LaneMap codegen route | BubbleBeam/FutureSight route selection | speed-equivalent to owned for tracked Q4_K decode roles | closest to pure; harden/generalize, do not reopen |
| Decode Q6_K | existing coop route | model route guards | direct/lane-map route was token-correct but W==D-refuted by about 5-6% | keep shipped route; no active pure replacement |
| Decode attention | owned two-kernel tile + combine ships; native generated route exists | model route guards | native route is correct/route-bound but correct-not-fast; attention now low-leverage | reference/fallback plus low-priority research, not max-out target |
| Prefill GEMM | specialized graph-GEMM/pipe route | default route policy; role-selective pipe now default | role-selective pipe is TIER_A through ctx8192 | promote into manifest/search-owned route; then generalize |
| Everything else | tinygrad generated kernels | tinygrad scheduler/model graph | norms, rope, elementwise, generated graph operations | already generated enough for this scope |

The work now is **not** "write four replacement kernels." The work is:

1. Make the successful generated/specialized routes first-class search candidates with manifests, route attribution, rollback, and durable promotion
   records.
2. Remove one-off env/flag selection logic from the active surface.
3. Generalize the search substrate across quant type, shape, model family, and GPU target.
4. Only reopen a hot kernel if a ceiling/attribution audit says it can move whole-model tokens/s.

## Source Citations

Load these before implementing:

| claim | citation |
|---|---|
| Current pure-machine-search narrative and boundaries | `docs/pure-machine-search.md` |
| Roadmap and status table | `docs/pure-machine-search-roadmap.md` |
| Active-work audit and agnostic search scope | `docs/claude-active-work-audit-and-agnostic-search-scope-20260630.md` |
| G3 Q4_K promotion/parity | `bench/amd-isa-backend-g3-weight-promotion/summary.md`, `bench/amd-isa-backend-g3-weight-promotion/latest.json`, `docs/amd-isa-g3-weight-promotion-hardening-scope-20260629.md` |
| Q6_K direct route refutation | `bench/amd-isa-backend-q6k-direct-speed/summary.md`, `bench/amd-isa-backend-q6k-direct-speed/latest.json`, `docs/amd-isa-q6k-direct-route-full-scope-20260629.md` |
| Decode attention two-kernel exhaustion | `docs/decode-two-kernel-problem-audit-result-20260625.md`, `bench/qk-decode-two-kernel-problem-audit-20260625/decision.json` |
| Decode attention ceiling/low leverage | `bench/amd-isa-backend-decode-attention-ceiling/summary.md`, `bench/amd-isa-backend-decode-attention-ceiling/latest.json`, `docs/amd-isa-decode-attention-ceiling-audit-scope-20260629.md` |
| Native attention generated route history | `docs/pure-machine-search-roadmap.md`, `bench/amd-isa-backend-phase-n7/latest.json` |
| Prefill global pipe promotion | `bench/qk-prefill-pipe-promotion/summary.md`, `bench/qk-prefill-pipe-promotion/latest.json` |
| Prefill role-selective promotion | `bench/qk-prefill-pipe-role-selective/summary.md`, `bench/qk-prefill-pipe-role-selective/latest.json`, `extra/qk_prefill_pipe_role_selective.py` |
| Prefill authority harness | `extra/qk_prefill_whole_synced.py` |
| Generic search labels | `docs/generic-low-level-search-goal-scope.md` |
| Current search ledger | `bench/qk-project-search-ledger/ledger.jsonl`, `bench/qk-project-search-ledger/schema.json` |

## Definitions

Use these terms strictly:

```text
generated:
  produced by tinygrad/codegen or a declared codegen route from a manifest.

search-owned:
  candidate exists in a declared search space, has route identity, was evaluated by the authority gate,
  and has a promote/refute/rollback ledger entry.

pure default:
  the default route is generated + search-owned, and the old hand-owned kernel is only reference/fallback.

fallback/reference:
  still present for rollback, correctness comparison, or ceiling audits; not on the normal default path.
```

Do not claim purity from "selected by a flag" alone. Selection is only one part of search ownership.

## Current Performance Actuals

### Decode Q4_K G3

`bench/amd-isa-backend-g3-weight-promotion/summary.md`:

| ctx | owned tok/s | G3 tok/s | result |
|---:|---:|---:|---|
| 512 | 103.79 | 103.93 | speed-equivalent |
| 1024 | 101.98 | 102.04 | speed-equivalent |
| 2048 | 99.56 | 99.74 | speed-equivalent |
| 4096 | 94.83 | 94.44 | speed-equivalent |

Decision: treat Q4_K G3 as the promoted pure/generated route for its tracked decode roles. Keep owned warp as rollback
and reference. Do not start Q4_K layout reshuffle while parity holds.

### Decode Q6_K

`docs/claude-active-work-audit-and-agnostic-search-scope-20260630.md` records the speed gate:

| ctx | shipped coop baseline | direct candidate | delta |
|---:|---:|---:|---:|
| 512 | 103.63 | 97.35 | -6.06% |
| 1024 | 101.68 | 95.76 | -5.82% |
| 2048 | 99.21 | 94.19 | -5.06% |
| 4096 | 94.50 | 89.99 | -4.77% |

Decision: Q6_K direct is refuted/default-off. Do not reopen without a new route premise and a fresh residual audit.

### Decode Attention

The shipped owned route is a two-kernel primitive:

```text
owned_flash_tile_gqa_whole -> owned_flash_combine
```

The native/generated attention route is correct and route-bound, but below owned. Later ceiling audits show attention is
low-leverage for whole decode under the current Qwen3-8B-Q4_K_M/gfx1100 target.

Decision: keep owned attention as the shipped route. The native route is infrastructure/research. Do not spend broad
search on attention unless attention wall-share becomes dominant for a new context/model/quant target.

### Prefill

Global pipe promotion:

| ctx | old default | global pipe | delta |
|---:|---:|---:|---:|
| 512 | 3598 | 4289 | +19.2% |
| 1024 | 3506 | 4095 | +16.8% |
| 2048 | 3253 | 3708 | +14.0% |
| 4096 | 2821 | 3137 | +11.2% |
| 8192 | 2234 | 2423 | +8.5% |

Role-selective pipe, now default:

| ctx | old default | global pipe | role-selective | role-selective vs global |
|---:|---:|---:|---:|---:|
| 512 | 3593 | 4292 | 4434 | +3.3% |
| 1024 | 3492 | 4092 | 4236 | +3.5% |
| 2048 | 3259 | 3708 | 3846 | +3.7% |
| 4096 | 2779 | 3083 | 3192 | +3.5% |
| 8192 | 2266 | 2461 | 2532 | +2.9% |

Decision: role-selective pipe is a promoted default. The next pure-machine-search task is making this route
manifest/search-owned and generalizable, not re-proving the speed result.

## Phase Plan

### PMS-R0: Default-Path Kernel Census

Goal: replace the rough "four kernels" story with a machine-readable census of what actually runs.

Build:

```text
extra/pure_machine_search_default_path_census.py
```

Inputs:

```text
tinygrad/llm/model.py
extra/qk_gemv_g3_codegen_lowering.py
extra/qk_prefill_graph_gemm_route.py
extra/qk_prefill_pipe_role_selective.py
extra/qk_owned_flash_decode_graph_node.py
bench/amd-isa-backend-g3-weight-promotion/latest.json
bench/qk-prefill-pipe-role-selective/latest.json
bench/amd-isa-backend-q6k-direct-speed/latest.json
bench/amd-isa-backend-decode-attention-ceiling/latest.json
```

Outputs:

```text
bench/pure-machine-search-default-path-census/latest.json
bench/pure-machine-search-default-path-census/summary.md
bench/pure-machine-search-default-path-census/default_route_table.json
bench/pure-machine-search-default-path-census/fallback_table.json
```

Required rows:

```text
route_id
workload: decode | prefill
role
quant
shape_guard
current_default
writer: generated | codegen_emitter | owned_asm | tinygrad_generated
selector: BubbleBeam | manifest | env_guard | hardcoded_default | tinygrad_scheduler
authority_artifact
rollback_flag
purity_status
next_action
```

Acceptance:

- The census can answer: which non-tinygrad-generated kernels execute on the default path?
- It separates default-path kernels from fallback/research kernels.
- It does not infer from filenames only; it must cite route attribution or model route guards.

Verdicts:

```text
PMS_R0_PASS_CENSUS_PINNED
PMS_R0_BLOCKED_ROUTE_ATTRIBUTION_MISSING
```

### PMS-R1: Route Manifest

Goal: move route identity out of scattered env-flag logic into a declarative manifest.

Build or extend:

```text
bench/qk-search-spaces/default_route_manifest.json
extra/qk_route_manifest.py
```

Minimum manifest entries:

```text
decode_q4k_g3_generated
decode_q6k_coop_shipped
decode_q6k_direct_refuted
decode_attention_owned_two_kernel
decode_attention_native_correct_not_fast
prefill_pipe_role_selective_default
prefill_pipe_global_rollback
```

Minimum schema:

```json
{
  "route_id": "prefill_pipe_role_selective_default",
  "workload": "prefill",
  "profile_id": "qwen3_8b_q4_k_m_gfx1100",
  "roles": ["attn_qo", "attn_kv", "ffn_down"],
  "excluded_roles": ["ffn_gate_up"],
  "quant": ["Q4_K", "Q6_K", "fp16"],
  "shape_guards": [{"M": 512, "N": "*", "K": "*"}],
  "env": {"PREFILL_GEMM_PIPELINE": "1", "PREFILL_PIPE_ROLE_SELECTIVE": "1"},
  "rollback": {"PREFILL_PIPE_ROLE_SELECTIVE": "0"},
  "authority_gate": "extra/qk_prefill_whole_synced.py",
  "promotion_artifacts": ["bench/qk-prefill-pipe-role-selective/latest.json"],
  "status": "promoted_default",
  "purity_status": "search_selected_specialized_route"
}
```

Acceptance:

- Existing gates can read route IDs from the manifest instead of manually spelling env maps.
- Refuted routes remain in the manifest with `status=refuted`, so the search does not rediscover them.
- Rollback flags are explicit.

Verdicts:

```text
PMS_R1_PASS_ROUTE_MANIFEST_READY
PMS_R1_BLOCKED_FLAG_CONTRACT_AMBIGUOUS
```

### PMS-R2: Table-Driven Evaluator

Goal: one evaluator for correctness, route attribution, speed, ceiling, and ledger update.

Build:

```text
extra/qk_candidate_evaluator.py
```

It should consume:

```text
route_manifest
profile_id
candidate route_id
baseline route_id
contexts
authority type: decode_wd | prefill_whole
threshold policy: tiered
```

It should produce:

```text
bench/qk-candidate-evaluator/<route_id>/latest.json
bench/qk-candidate-evaluator/<route_id>/summary.md
bench/qk-candidate-evaluator/<route_id>/route_attribution.json
bench/qk-candidate-evaluator/<route_id>/ledger_update.json
```

Required checks:

- token/logit correctness;
- route-bound and no hidden fallback;
- default-off/default-on contract;
- W==D or whole-prefill speed across declared contexts;
- noise/spread disclosure;
- tiered threshold classification;
- rollback availability;
- appendable project-search-ledger row.

Acceptance:

- Can re-evaluate `decode_q4k_g3_generated` and reproduce speed-equivalent/pass.
- Can re-evaluate `prefill_pipe_role_selective_default` and reproduce pass.
- Can re-evaluate `decode_q6k_direct_refuted` and preserve refuted status.

Verdicts:

```text
PMS_R2_PASS_EVALUATOR_REPLAYS_KNOWN_DECISIONS
PMS_R2_BLOCKED_AUTHORITY_HARNESS_INCOMPLETE
```

### PMS-R3: Search-Space Generator From Manifest

Goal: BubbleBeam should generate candidates from manifests/profiles, not hardcoded phase scripts.

Build or extend:

```text
extra/qk_pure_search_next_candidate.py
extra/qk_search_space_manifest_check.py
bench/qk-search-spaces/search_profiles.json
```

Search profile dimensions:

```text
workload: decode | prefill
role: ffn_gate_up | ffn_down | attn_qo | attn_kv | lm_head | attention_tile | attention_combine
quant: Q4_K | Q5_K | Q6_K | Q8_0 | fp16
shape: M,N,K,Hq,Hkv,Hd,ctx,max_context
target: AMD_gfx1100 first; extensible GPU descriptor later
route_family: lanemap | coop | graph_gemm_pipe | native_isa_attention | owned_reference
```

Acceptance:

- Candidate generator refuses a candidate if its primitive family is not in the declared profile.
- It emits `NO_UNTRIED_CANDIDATE_TARGETS_A_FAILED_ROW` instead of wandering.
- It carries `do_not_search` rows from current refutations:
  - Q4_K offline layout reshuffle while G3 parity holds;
  - Q6_K direct half-warp route as built;
  - broad decode-attention combine/fusion under current ceiling;
  - scheduler-only/resource-only attention retuning;
  - N1B scalar-address hoist as previously refuted;
  - native-attention register-accum/LDS occupancy as already measured low-leverage.

Verdicts:

```text
PMS_R3_PASS_MANIFEST_DRIVEN_CANDIDATES
PMS_R3_BLOCKED_SEARCH_SPACE_SCHEMA
```

### PMS-R4: Promote Current Successes Into Search Ledger

Goal: make current actuals durable search results.

Append/update `bench/qk-project-search-ledger/ledger.jsonl` rows for:

```text
decode/q4k_g3_generated_speed_equivalent
decode/q6k_direct_refuted
decode/attention_native_correct_not_fast_low_leverage
prefill/pipe_global_promoted
prefill/pipe_role_selective_promoted
```

Each row must include:

```text
candidate_id
profile_id
route_id
workload
primitive_class
owned_or_baseline
correctness
route_identity
authority_benchmark
verdict
rollback
artifact_links
learned_rule
do_not_search_implications
```

Acceptance:

- Ledger is the durable source of "do not reopen" and "promoted" decisions.
- The candidate evaluator can read the ledger and avoid repeated dead paths.

Verdicts:

```text
PMS_R4_PASS_LEDGER_CURRENT
PMS_R4_BLOCKED_LEDGER_SCHEMA_MISMATCH
```

### PMS-R5: Turn G3 Into A Fully Generated/Search-Owned Template

Goal: close the last wording gap around Q4_K G3: it is generated and speed-equivalent, but the lane-map family must be
represented as a reusable template, not only a one-off route.

Scope:

- Do not chase speed.
- Do not rewrite the already-promoted G3 route unless route identity/correctness stays exact.
- Focus on provenance and generalization.

Build:

```text
extra/qk_lanemap_template_audit.py
bench/qk-lanemap-template-audit/latest.json
```

Questions:

1. Which parts of G3 are parameterized by shape/quant/role?
2. Which parts are hardcoded for Q4_K/Qwen3/gfx1100?
3. Which pieces are generated by `extra/qk_gemv_g3_codegen_lowering.py` versus hand-authored topology?
4. What is the minimal generic `LaneMapTemplate` schema?

Acceptance:

- Produce a template schema for lane ownership, packed-word load pattern, dequant body, reduction pattern, output store,
  and shape guards.
- Reconstruct the existing G3 route from the template and prove route/token/speed equivalence.

Verdicts:

```text
PMS_R5_PASS_G3_TEMPLATE_PROVEN
PMS_R5_BLOCKED_TEMPLATE_NOT_LOSSLESS
```

### PMS-R6: Prefill Route As Search-Owned Template

Goal: make the role-selective prefill pipe a search-owned route rather than a promoted manual policy.

Scope:

- Do not change default speed path unless evaluator proves no regression.
- First encode the role policy and tile/pipe choices as a manifest template.
- Then replay the promotion gate.

Build:

```text
extra/qk_prefill_pipe_template_audit.py
bench/qk-prefill-pipe-template-audit/latest.json
```

Template dimensions:

```text
role
M,N,K
tm,tn
pipe enabled
role include/exclude
LDS staging
register pressure estimate
BLAS ceiling ratio
```

Acceptance:

- Role-selective policy is reproduced from role/ceiling facts, not a special-case if-statement.
- The evaluator reproduces `ROLE_SELECTIVE_PASS_BEATS_GLOBAL`.
- Rollback remains `PREFILL_PIPE_ROLE_SELECTIVE=0` or manifest baseline route.

Verdicts:

```text
PMS_R6_PASS_PREFILL_TEMPLATE_PROVEN
PMS_R6_BLOCKED_ROLE_POLICY_NOT_ENCODED
```

### PMS-R7: Attention Reopen Gate, Not Attention Rewrite

Goal: keep attention honest without wasting work.

Build:

```text
extra/qk_attention_reopen_gate.py
bench/qk-attention-reopen-gate/latest.json
```

Inputs:

```text
profile_id
ctx set
model architecture
quant mix
measured attention wall-share
native-vs-owned gap
whole-decode Amdahl estimate
```

Rules:

- If attention perfect-parity gain is below the active threshold, verdict is `DO_NOT_REOPEN_ATTENTION`.
- If a new model/context makes attention wall-share large, emit a new scope for attention only then.
- Never start from "the native route is slower" alone; start from whole-model leverage.

Acceptance:

- Current Qwen3-8B-Q4_K_M/gfx1100 returns `DO_NOT_REOPEN_ATTENTION`.
- The artifact explains what would make attention active again, for example much longer context, different KV layout,
  larger Hq/Hkv/Hd, MoE reducing FFN wall-share, or target GPU differences.

Verdicts:

```text
PMS_R7_PASS_ATTENTION_REOPEN_GATE
PMS_R7_BLOCKED_WALL_SHARE_ATTRIBUTION_MISSING
```

### PMS-R8: Quant/Shape/GPU Generalization

Goal: make the system agnostic enough that future wins are found by profile, not by Qwen/gfx1100 assumptions.

Build on:

```text
docs/claude-active-work-audit-and-agnostic-search-scope-20260630.md
```

Add profile descriptors:

```text
bench/qk-search-spaces/profiles/qwen3_8b_q4_k_m_gfx1100.json
bench/qk-search-spaces/profiles/<model>_<quant>_<gpu>.json
```

Minimum profile fields:

```json
{
  "model": {"family": "qwen3", "params": "8b", "layers": 36, "hidden": 4096, "ffn": 12288, "heads": 32, "kv_heads": 8, "head_dim": 128},
  "quant_mix": {"Q4_K": ["ffn_gate_up", "ffn_down", "attn_qo"], "Q6_K": ["lm_head", "selected_roles"]},
  "gpu": {"vendor": "AMD", "arch": "gfx1100", "wave": 32, "vram_gb": 24, "measured_copy_gbps": 820},
  "authority_contexts": [512, 1024, 2048, 4096, 8192],
  "threshold_policy": "tiered_residual"
}
```

Acceptance:

- Existing Qwen3-8B/gfx1100 routes are regenerated from the profile.
- Adding a new quant/model/GPU starts with census + ceiling + route attribution, not hand-edited flags.

Verdicts:

```text
PMS_R8_PASS_PROFILE_DRIVEN_SEARCH_READY
PMS_R8_BLOCKED_PROFILE_SCHEMA_GAPS
```

## Claude Execution Prompt

Use this prompt for the next agent:

```text
You are in /home/ubuntu/tinygrad-arkey on master. Read:

1. docs/pure-machine-search-remaining-hot-kernels-scope-20260630.md
2. docs/pure-machine-search.md
3. docs/pure-machine-search-roadmap.md
4. docs/claude-active-work-audit-and-agnostic-search-scope-20260630.md

Task: execute PMS-R0 through PMS-R2 only, stopping at the first hard blocker.

Rules:
- Audit first. Do not implement new kernels.
- Do not reopen Q4_K G3, Q6_K direct, or broad decode-attention tuning.
- Build the default-path census, route manifest, and table-driven evaluator enough to replay known decisions:
  G3 pass, Q6_K direct refuted, prefill role-selective pass.
- Every artifact must cite source files and benchmark artifacts.
- Default behavior must remain unchanged.
- Do not delete owned/fallback kernels.
- Commit only if gates pass or a precise blocker artifact is written.

Expected artifacts:
- bench/pure-machine-search-default-path-census/latest.json
- bench/pure-machine-search-default-path-census/summary.md
- bench/qk-search-spaces/default_route_manifest.json
- extra/qk_route_manifest.py
- extra/qk_candidate_evaluator.py
- bench/qk-candidate-evaluator/<route_id>/latest.json for the three replay routes

Acceptable final verdicts:
- PMS_R0_PASS_CENSUS_PINNED + PMS_R1_PASS_ROUTE_MANIFEST_READY + PMS_R2_PASS_EVALUATOR_REPLAYS_KNOWN_DECISIONS
- or the earliest precise PMS_R*_BLOCKED_* verdict with a JSON artifact explaining the missing evidence.
```

## Non-Goals

- Do not rewrite attention.
- Do not rebuild Q6_K direct route without a new measured premise.
- Do not start Metal/NVIDIA portability here.
- Do not claim all kernels are generated just because there are only a few hand-owned hot kernels left.
- Do not delete fallback/owned kernels; pure promotion means fallback/reference, not removal.
- Do not use wall-clock deltas without route attribution and correctness.

## End State

After this scope, the project should have:

```text
current default path -> route manifest -> evaluator -> ledger
```

At that point, "pure machine search" becomes an operational property:

```text
route is pure if the manifest declares it,
the generator can produce/select it,
the evaluator proves it,
and the ledger promotes it with rollback.
```

The next kernel work should then be driven by the manifest/evaluator, not by intuition about the remaining hot kernels.
