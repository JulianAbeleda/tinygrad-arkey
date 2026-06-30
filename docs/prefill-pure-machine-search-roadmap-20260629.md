# Prefill Pure-Machine-Search Roadmap

Date: 2026-06-29

Status: future execution roadmap. This mirrors the decode process: start with theoretical ceilings, then whole-step attribution, then route candidates, then correctness, speed, search binding, and promotion. Do not implement kernels from this document directly; execute phase by phase.

## Executive Summary

Decode and prefill are different problems.

```text
decode:
  mostly GEMV / per-token / weight-memory-bound
  route wins came from Q4_K/Q6_K direct routes and decode attention routing

prefill:
  mostly GEMM / multi-token / compute-and-memory tiled
  route wins come from GEMM tiling, WMMA/MFMA use, LDS staging, K-loop scheduling, and long-context integration
```

Current prefill state, from existing artifacts:

```text
current default graph-GEMM / eightwave-ish baseline:
  ctx512  ~= 3597 tok/s
  ctx1024 ~= 3504 tok/s
  ctx2048 ~= 3248 tok/s
  ctx4096 ~= 2803 tok/s

external BLAS / per-shape ceiling probe:
  ffn_gate_up: hipblasLt ~= 69.8 TFLOPS
  ffn_down:    rocBLAS   ~= 70.9 TFLOPS
  attn_q/o:    rocBLAS   ~= 76.7 TFLOPS
  attn_k/v:    rocBLAS   ~= 51.8 TFLOPS

existing aggressive candidate:
  pipe_tm2_tn2:
    ctx512  4253 tok/s (+19.1%)
    ctx1024 4037 tok/s (+15.9%)
    ctx2048 3659 tok/s (+13.4%)
    ctx4096 3110 tok/s (+11.5%)
  status: promising but not yet authority/promoted
```

The honest roadmap is:

```text
P0: theoretical floor / ceiling
P1: refresh whole-prefill baseline and authority
P2: role-level attribution
P3: candidate search space and refutation ledger
P4: route design for selected candidate
P5: correctness / equivalence gate
P6: speed and long-context gate
P7: BubbleBeam/search binding
P8: promotion / rollback package
P9: quant/shape/target generalization
```

The first executable phase should be P0/P1/P2 audit-only. Do not start by implementing a new GEMM tile.

## Why Prefill Needs A Separate Track

Decode proof does not automatically transfer to prefill.

Decode wins proved:

```text
route attribution discipline
W==D authority
quant-route search
effective bandwidth math
BubbleBeam promotion gates
native AMD ISA/codegen capability
```

They did not prove:

```text
prefill GEMM route parity
prefill attention route parity
prefill long-context stability
prefill shape coverage
prefill pure-machine-search promotion
```

Prefill has different dominant axes:

```text
M = prompt chunk / tokens, often 512
N = output features
K = hidden / FFN dimension
roles = ffn_gate_up, ffn_down, q/o proj, k/v proj, attention QK/PV, norm/rope/copy
```

For prefill, the central question is not "can we read weights faster for one token?" It is:

```text
Can generated/search-owned GEMM/attention routes approach the practical BLAS/Tensile/hand-ASM ceiling across prompt contexts without long-context regression?
```

## Source Artifacts To Treat As Prior Evidence

Read these before running new work:

```text
bench/qk-prefill-external-blas/ceiling.json
bench/qk-prefill-long-context-harness-authority-role-tax/baseline_whole_prefill_by_ctx.json
bench/qk-prefill-long-context-harness-authority-role-tax/per_role_time_tax_by_ctx.json
bench/qk-prefill-long-context-harness-authority-role-tax/graphgemm_vs_tensile_integration_by_role.json
bench/qk-prefill-long-context-harness-authority-role-tax/decision.json
bench/qk-prefill-aggressive-target-proof-20260624/decision.json
bench/qk-prefill-aggressive-target-proof-20260624/whole_prefill_baseline.json
bench/qk-prefill-aggressive-target-proof-20260624/whole_prefill_candidates.json
bench/qk-prefill-long-context-no-regression-audit/baseline_prefill_by_context.json
docs/prefill-long-context-no-regression-audit-result-20260623.md
docs/decode-campaign-final-synthesis-20260623.md
```

Important prior conclusions:

```text
eightwave:
  promoted default in prior prefill lane
  stable small gain, roughly +2-3% short/mid ctx, +1.6-1.9% long ctx

pipe_tm2_tn2:
  large apparent gain, +11-19%
  not yet promoted because authority/long-context/role-transfer proof incomplete

role tax:
  ffn_gate_up and ffn_down dominate
  kv_proj shape mismatch / WG-starvation was previously flagged
  attention/KV and integration overhead need whole-context profiling

external BLAS ceiling:
  per-shape GEMMs can reach roughly 50-77 TFLOPS on 7900 XTX depending shape/library
```

## Core Equations

### 1. GEMM FLOP Count

For each GEMM role:

```text
F_role = 2 * M * N * K
```

where:

```text
M = prompt chunk tokens
N = output feature dimension
K = input feature dimension
```

Examples for M=512:

```text
ffn_gate_up:
  M=512, N=12288, K=4096
  F = 2 * 512 * 12288 * 4096

ffn_down:
  M=512, N=4096, K=12288
  F = 2 * 512 * 4096 * 12288

q/o proj:
  M=512, N=4096, K=4096
  F = 2 * 512 * 4096 * 4096

k/v proj:
  M=512, N=1024, K=4096
  F = 2 * 512 * 1024 * 4096
```

### 2. Per-Role Time Floor

Given measured practical TFLOPS for a role:

```text
T_floor_role = F_role / TFLOPS_role
```

The floor should use measured practical ceilings, not marketing peak.

From `bench/qk-prefill-external-blas/ceiling.json`:

```text
ffn_gate_up practical ceiling ~= 69.8 TFLOPS (hipblasLt)
ffn_down    practical ceiling ~= 70.9 TFLOPS (rocBLAS)
q/o         practical ceiling ~= 76.7 TFLOPS (rocBLAS)
k/v         practical ceiling ~= 51.8 TFLOPS (rocBLAS)
```

### 3. Whole-Prefill Floor

For one chunk:

```text
T_floor_chunk =
  sum(T_floor_gemm_roles)
  + T_floor_attention
  + T_floor_norm_rope_copy
  + T_launch_sync
  + T_graph_overhead
```

For whole prefill over context `C` with chunk size `M`:

```text
num_chunks = ceil(C / M)
T_floor_prefill(C) = sum over chunks j of T_floor_chunk(start_pos_j)
R_floor_prefill(C) = C / T_floor_prefill(C)
```

Attention grows with context; FFN/projection GEMM is mostly chunk-size dependent.

### 4. Amdahl For Candidate Wins

For a candidate that affects fraction `p` of whole-prefill time and removes fraction `r`:

```text
S = 1 / (1 - p * r)
R_new = R0 * S
```

Use the tiered promotion policy:

```text
TIER_A_MAJOR:
  >=5.0% median W==D/prefill improvement

TIER_B_RESIDUAL:
  >=2.0% and <5.0%, only with clean mechanism proof and no protected context regression >1%

TIER_C_EQUIVALENT_CLEANUP:
  -1.0% to +2.0%, only for purity/simplification
```

## Phase P0: Theoretical Ceiling / Floor Audit

Goal: compute a credible prefill ceiling before touching kernels.

Build:

```text
extra/qk_prefill_theoretical_ceiling_audit.py
```

Write:

```text
bench/qk-prefill-theoretical-ceiling/latest.json
bench/qk-prefill-theoretical-ceiling/summary.md
bench/qk-prefill-theoretical-ceiling/role_flops.json
bench/qk-prefill-theoretical-ceiling/roofline_floor.json
bench/qk-prefill-theoretical-ceiling/whole_prefill_floor_by_ctx.json
bench/qk-prefill-theoretical-ceiling/source_artifacts.json
```

Inputs:

```text
bench/qk-prefill-external-blas/ceiling.json
bench/qk-prefill-long-context-harness-authority-role-tax/baseline_whole_prefill_by_ctx.json
bench/qk-prefill-aggressive-target-proof-20260624/whole_prefill_baseline.json
bench/qk-prefill-aggressive-target-proof-20260624/whole_prefill_candidates.json
```

Required outputs:

```text
current_prefill_by_ctx
external_blas_floor_by_role
aggressive_candidate_by_ctx
gap_current_to_candidate
gap_candidate_to_floor
role_theoretical_share
is_prefill_compute_bound_or_memory_bound
```

P0 must answer:

1. What is the practical prefill ceiling at ctx512/1024/2048/4096/8192?
2. Is the existing `pipe_tm2_tn2` candidate near that ceiling or still far away?
3. Which roles mathematically dominate the floor?
4. Is the ceiling set by GEMM, attention, launch/graph, or memory?
5. Is the old aggressive bound stale or still valid?

Verdicts:

```text
PREFILL_P0_PASS_CEILING_PINNED
PREFILL_P0_INCONCLUSIVE_STALE_ARTIFACTS
PREFILL_P0_BLOCKED_MISSING_BLAS_FLOOR
```

Stop condition:

```text
If P0 cannot pin a ceiling, stop. Do not search candidates.
```

## Phase P1: Authority Baseline Refresh

Goal: re-measure the current prefill baseline under one clean authority harness.

Build or reuse:

```text
extra/qk_prefill_emit_search.py
extra/qk_prefill_graph_gemm_route.py
```

If needed, build:

```text
extra/qk_prefill_authority_refresh.py
```

Write:

```text
bench/qk-prefill-authority-refresh/latest.json
bench/qk-prefill-authority-refresh/summary.md
bench/qk-prefill-authority-refresh/current_default_by_ctx.json
bench/qk-prefill-authority-refresh/eightwave_guard.json
bench/qk-prefill-authority-refresh/noise_profile.json
bench/qk-prefill-authority-refresh/route_attribution.json
```

Required contexts:

```text
512
1024
2048
4096
8192
```

Required arms:

```text
current_default
eightwave_off
known_candidate_pipe_tm2_tn2 if safe
```

P1 must verify:

```text
route_bound
output/logit equivalence or accepted prefill equivalence gate
determinism
no hidden Tensile/BLAS fallback unless intentionally selected
chunk schedule
noise / spread
```

Verdicts:

```text
PREFILL_P1_PASS_AUTHORITY_BASELINE_PINNED
PREFILL_P1_BLOCKED_NOISY_OR_STALE
PREFILL_P1_BLOCKED_ROUTE_ATTRIBUTION
PREFILL_P1_BLOCKED_CORRECTNESS
```

## Phase P2: Whole-Prefill Role Attribution

Goal: identify the actual wall buckets in whole prefill, not isolated chunks.

Build:

```text
extra/qk_prefill_whole_role_attribution.py
```

Write:

```text
bench/qk-prefill-whole-role-attribution/latest.json
bench/qk-prefill-whole-role-attribution/summary.md
bench/qk-prefill-whole-role-attribution/per_role_by_ctx.json
bench/qk-prefill-whole-role-attribution/per_chunk_by_ctx.json
bench/qk-prefill-whole-role-attribution/route_coverage.json
bench/qk-prefill-whole-role-attribution/unknown_bucket.json
```

Buckets:

```text
ffn_gate_up
ffn_down
attn_qo
attn_kv
attention_qk
attention_pv
attention_softmax
norm_rope_elementwise
copy_cast_sync
graph_launch_sync
unknown
```

Required:

```text
unknown bucket < 10%
role timing by ctx
role timing by chunk start_pos
route label per role
shape per role
effective TFLOPS per GEMM role
```

P2 must explicitly check prior flagged rows:

```text
ffn_gate_up dominant share
ffn_down deeper-K shape option
kv_proj WG-starved shape-key issue
attention_or_KV growth with ctx
non_gemm integration overhead
```

Verdicts:

```text
PREFILL_P2_PASS_ROLE_ATTRIBUTION_PINNED
PREFILL_P2_INCONCLUSIVE_UNKNOWN_BUCKET
PREFILL_P2_BLOCKED_OOM_OR_PROFILING
PREFILL_P2_BLOCKED_ROUTE_LABELS
```

## Phase P3: Candidate Search Space / Refutation Ledger

Goal: define the candidate space and refuse stale/refuted knobs.

Build:

```text
extra/qk_prefill_candidate_space_audit.py
```

Write:

```text
bench/qk-prefill-candidate-space/latest.json
bench/qk-prefill-candidate-space/summary.md
bench/qk-prefill-candidate-space/candidates.json
bench/qk-prefill-candidate-space/refuted_axes.json
bench/qk-prefill-candidate-space/selected_next.json
```

Candidate families:

```text
eightwave_existing_default
old_plra_secondary
pipe_tm2_tn2
pipe_tm4_tn2
graph_gemm_shape_key_fix
kv_proj_wg_starvation_fix
ffn_down_deeper_k_shape
wmma_mfma_tile_search
lds_double_buffering
kloop_schedule_template
external_blas_or_tensile_bridge
native_isa_generated_gemm
attention_prefill_tile
graph_fusion_or_chunk_schedule
```

Selection rule:

```text
select candidate with:
  role share >= 10% OR whole-prefill gain evidence >= TIER_B
  correctness route is testable
  expected gain clears tiered policy
  not already refuted by long-context/no-regression artifact
```

Special handling:

```text
pipe_tm2_tn2:
  already showed +11-19% in aggressive proof
  must be revalidated under authority and long-context gates
  cannot promote from old artifact alone

eightwave:
  already promoted default
  do not re-search except as baseline/rollback

old_plra:
  secondary; interaction with eightwave regressed
  do not combine with eightwave unless new evidence specifically clears it
```

Verdicts:

```text
PREFILL_P3_PASS_NEXT_CANDIDATE_SELECTED
PREFILL_P3_PASS_NO_LIVE_CANDIDATE
PREFILL_P3_INCONCLUSIVE_NEEDS_ROLE_ATTRIBUTION
```

## Phase P4: Candidate Route Design

Goal: design the selected candidate before implementation.

Build one design tool per selected candidate, for example:

```text
extra/qk_prefill_pipe_tm2_tn2_design.py
extra/qk_prefill_kv_proj_shape_fix_design.py
extra/qk_prefill_wmma_tile_design.py
```

Write:

```text
bench/qk-prefill-route-design/latest.json
bench/qk-prefill-route-design/summary.md
bench/qk-prefill-route-design/current_route.json
bench/qk-prefill-route-design/candidate_route.json
bench/qk-prefill-route-design/risk_register.json
bench/qk-prefill-route-design/implementation_plan.json
```

Design must answer:

```text
which roles are changed
which shapes are changed
which contexts are protected
which old flags are reused
which new flags are needed
what route labels prove firing
what correctness gate applies
what speed tier is expected
what rollback path exists
```

Verdicts:

```text
PREFILL_P4_PASS_DESIGN_READY
PREFILL_P4_BLOCKED_ROUTE_MAPPING
PREFILL_P4_BLOCKED_CORRECTNESS_CONTRACT
PREFILL_P4_BLOCKED_LOW_AMDAHL
```

## Phase P5: Correctness / Equivalence Gate

Goal: prove the candidate is correct before speed.

Prefill correctness should use:

```text
logit equivalence
or hidden-state equivalence
or byte/relative-error gate appropriate for prefill GEMM/attention
```

Build:

```text
extra/qk_prefill_candidate_correctness_gate.py
```

Write:

```text
bench/qk-prefill-candidate-correctness/latest.json
bench/qk-prefill-candidate-correctness/summary.md
bench/qk-prefill-candidate-correctness/route_attribution.json
bench/qk-prefill-candidate-correctness/equivalence.json
```

Required:

```text
route_bound = true
hidden_fallback = false
deterministic = true
equivalence_pass = true
rollback_works = true
default_off_until_speed = true
```

Contexts:

```text
512
1024
2048
4096
8192 if feasible
```

Verdicts:

```text
PREFILL_P5_PASS_CORRECTNESS
PREFILL_P5_BLOCKED_ROUTE_BINDING
PREFILL_P5_BLOCKED_EQUIVALENCE
PREFILL_P5_BLOCKED_FALLBACK
```

## Phase P6: Speed / Long-Context Gate

Goal: measure whole-prefill speed with no-regression across context.

Build:

```text
extra/qk_prefill_candidate_speed_gate.py
```

Write:

```text
bench/qk-prefill-candidate-speed/latest.json
bench/qk-prefill-candidate-speed/summary.md
bench/qk-prefill-candidate-speed/wd_or_tok_table.json
bench/qk-prefill-candidate-speed/role_before_after.json
bench/qk-prefill-candidate-speed/long_context_slope.json
bench/qk-prefill-candidate-speed/amdahl_vs_measured.json
```

Arms:

```text
current_default
candidate
rollback
```

Contexts:

```text
512
1024
2048
4096
8192
```

Tiered promotion:

```text
TIER_A_MAJOR:
  >=5% whole-prefill improvement, no protected ctx regression >2%

TIER_B_RESIDUAL:
  >=2% and <5%, mechanism proof clean, no protected ctx regression >1%

TIER_C_EQUIVALENT_CLEANUP:
  -1% to +2%, only if route is purer/simpler or retires owned/external dependency
```

For `pipe_tm2_tn2`, because old evidence showed +11-19%, require:

```text
TIER_A_MAJOR expected
ctx512/1024/2048/4096 all positive
8192 no regression beyond tier
role counters explain gain
```

Verdicts:

```text
PREFILL_P6_PASS_SPEED
PREFILL_P6_PASS_TIER_B_RESIDUAL
PREFILL_P6_CORRECT_BUT_NOT_FAST
PREFILL_P6_REGRESSION
PREFILL_P6_INCONCLUSIVE_NOISE
```

## Phase P7: BubbleBeam / Search Binding

Goal: make the selected route search-owned, not manual-flag-owned.

Build:

```text
extra/qk_prefill_search_binding_gate.py
```

Write:

```text
bench/qk-prefill-search-binding/latest.json
bench/qk-prefill-search-binding/summary.md
bench/qk-prefill-search-binding/search_space_update.json
bench/qk-prefill-search-binding/ledger.json
bench/qk-prefill-search-binding/rollback.json
```

Required:

```text
candidate in search manifest
shape guards
context guards
target guards
rollback flags
refuted axes ledgered
route attribution labels stable
```

Verdicts:

```text
PREFILL_P7_PASS_SEARCH_BOUND
PREFILL_P7_BLOCKED_MANUAL_FLAG_ONLY
PREFILL_P7_BLOCKED_ROUTE_ATTRIBUTION
```

## Phase P8: Promotion Package

Goal: promote only if correctness, speed, and search binding all pass.

Write:

```text
bench/qk-prefill-promotion/latest.json
bench/qk-prefill-promotion/summary.md
bench/qk-prefill-promotion/default_policy.json
bench/qk-prefill-promotion/regression_guard.json
docs/prefill-promotion-result-YYYYMMDD.md
```

Promotion requirements:

```text
correctness pass
speed tier pass
long-context no-regression
search binding pass
rollback pass
policy consistency pass
decode untouched
```

Verdicts:

```text
PREFILL_P8_PROMOTION_PASS
PREFILL_P8_CORRECT_BUT_NOT_FAST
PREFILL_P8_BLOCKED_REGRESSION
PREFILL_P8_BLOCKED_POLICY
```

## Phase P9: Generalize Beyond Qwen3-8B / gfx1100

Goal: make prefill search quant/shape/target aware.

This phase should use the future-work architecture from:

```text
docs/quant-shape-target-agnostic-search-future-work-20260629.md
```

Generalize across:

```text
model sizes: 8B, 14B, 32B, 70B
quant formats: Q4_K, Q6_K, Q8_0, FP16/BF16, GPTQ/AWQ
targets: AMD gfx1100, NVIDIA, Metal
workloads: decode, prefill, batch prefill
```

Required specs:

```text
QuantSpec
ShapeSpec
TargetSpec
RouteCandidate
CostFloor
PromotionDecision
```

Verdicts:

```text
PREFILL_P9_PASS_GENERIC_SEARCH_SUBSTRATE
PREFILL_P9_BLOCKED_SCHEMA
PREFILL_P9_BLOCKED_BACKEND_TARGET
```

## First Executable Task

Run P0-P2 only. The next prompt should not implement kernels.

Expected first deliverable:

```text
extra/qk_prefill_theoretical_ceiling_audit.py
extra/qk_prefill_authority_refresh.py       # only if existing harness cannot refresh cleanly
extra/qk_prefill_whole_role_attribution.py
```

Minimum first verdict:

```text
PREFILL_P0_PASS_CEILING_PINNED
PREFILL_P1_PASS_AUTHORITY_BASELINE_PINNED
PREFILL_P2_PASS_ROLE_ATTRIBUTION_PINNED
```

Only then select P3 candidate.

## Claude Prompt

```text
You are working in /home/ubuntu/tinygrad-arkey.

Task: execute prefill P0-P2 audit only. Do not implement kernels and do not change defaults.

Read:
- docs/prefill-pure-machine-search-roadmap-20260629.md
- docs/prefill-long-context-no-regression-audit-result-20260623.md
- docs/decode-campaign-final-synthesis-20260623.md

Source artifacts:
- bench/qk-prefill-external-blas/ceiling.json
- bench/qk-prefill-long-context-harness-authority-role-tax/baseline_whole_prefill_by_ctx.json
- bench/qk-prefill-long-context-harness-authority-role-tax/per_role_time_tax_by_ctx.json
- bench/qk-prefill-long-context-harness-authority-role-tax/graphgemm_vs_tensile_integration_by_role.json
- bench/qk-prefill-long-context-harness-authority-role-tax/decision.json
- bench/qk-prefill-aggressive-target-proof-20260624/decision.json
- bench/qk-prefill-aggressive-target-proof-20260624/whole_prefill_baseline.json
- bench/qk-prefill-aggressive-target-proof-20260624/whole_prefill_candidates.json

Build:
- extra/qk_prefill_theoretical_ceiling_audit.py
- extra/qk_prefill_whole_role_attribution.py
- extra/qk_prefill_authority_refresh.py only if needed

Write:
- bench/qk-prefill-theoretical-ceiling/{latest.json,summary.md,role_flops.json,roofline_floor.json,whole_prefill_floor_by_ctx.json,source_artifacts.json}
- bench/qk-prefill-authority-refresh/{latest.json,summary.md,current_default_by_ctx.json,eightwave_guard.json,noise_profile.json,route_attribution.json}
- bench/qk-prefill-whole-role-attribution/{latest.json,summary.md,per_role_by_ctx.json,per_chunk_by_ctx.json,route_coverage.json,unknown_bucket.json}

Required:
- Compute practical prefill ceiling from per-role FLOPs and measured BLAS/Tensile ceilings.
- Refresh or verify current default prefill across ctx512/1024/2048/4096/8192.
- Attribute whole-prefill time by role and chunk, with unknown bucket <10%.
- Compare current default, old aggressive pipe_tm2_tn2 evidence, and theoretical floor.
- Select no implementation candidate yet; P3 selection is a separate phase.

Verdicts:
- PREFILL_P0_PASS_CEILING_PINNED / INCONCLUSIVE / BLOCKED
- PREFILL_P1_PASS_AUTHORITY_BASELINE_PINNED / BLOCKED
- PREFILL_P2_PASS_ROLE_ATTRIBUTION_PINNED / INCONCLUSIVE / BLOCKED

Stop after P2 and report:
- theoretical ceiling by ctx,
- current baseline by ctx,
- role-level wall stack,
- live candidate families for P3,
- whether pipe_tm2_tn2 remains plausible or was refuted.
```

