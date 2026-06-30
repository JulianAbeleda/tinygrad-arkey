# AMD ISA lm_head / Q6_K Route Scope

Date: 2026-06-29

Status: scoped subtrack. This is separate from the general Q6_K direct route because lm_head has a different measured profile.

## Source Of Truth

This scope is based on Q6K-0:

```text
verdict: AMD_ISA_Q6K_RESIDUAL_PASS_DIRECT_ROUTE_JUSTIFIED
commit: 1a4055125
source artifacts:
  bench/amd-isa-backend-q6k-residual-math/latest.json
  bench/amd-isa-backend-q6k-residual-math/reduce_role_split.json
  bench/amd-isa-backend-q6k-residual-math/q6k_route_candidates.json
  bench/amd-isa-backend-system-residual-ceiling/kernel_taxonomy.json
```

Relevant measured rows:

```text
ctx512:
  lm_head bucket                  = 5.7% GPU-time
  lm_head effective bandwidth      = 761.4 GB/s
  firm lm_head/Q6_K reduce kernels = r_32_4_1187 + r_32_4_1187n1
  firm reduce duration             = 131.6us + 129.58us
  firm reduce class                = q6k_lm_head_reduce_FIRM

ctx4096:
  lm_head bucket                  = 5.3% GPU-time
  lm_head effective bandwidth      = 761.7 GB/s
  firm lm_head/Q6_K reduce share   = 3.17% GPU-time
```

The key interpretation:

```text
lm_head GEMV is not obviously slow.
lm_head coop reduce is the firm removable row.
```

So this subtrack must not blindly optimize the lm_head GEMV. The first target is the lm_head route shape: eliminate or reduce the separate coop partials+sum path while preserving Q6_K semantics.

## Why lm_head Needs Its Own Scope

The general Q6_K route has two separable components:

```text
1. q6k_gemv bandwidth gap:
   Q6_K current route ~503 GB/s
   Q4_K G3 route      ~650 GB/s

2. lm_head firm reduce overhead:
   prod == 151936 reductions
   class == q6k_lm_head_reduce_FIRM
```

lm_head is special because:

- its GEMV row already reports high effective bandwidth, around `761 GB/s`,
- its shape includes the large vocab/output dimension,
- its firm removable overhead is the reduce, not necessarily the GEMV,
- a direct/warp route may need a different topology from hidden-layer Q6_K GEMVs.

## Track Goal

Prove and, if justified, implement a route that removes lm_head Q6_K coop reduce overhead without regressing token correctness.

Target:

```text
replace:
  lm_head Q6_K coop partials + r_32_4_1187 / r_32_4_1187n1 reduce

with:
  direct/warp or fewer-pass lm_head route
```

Not target:

```text
lm_head quant demotion
lm_head quality change
generic Q4_K layout work
attention
RMSNorm reductions
ambiguous prod==4096 reductions
```

## Phase LH0: lm_head Route Audit

Goal: isolate lm_head's exact route and prove what can be removed.

Build:

```text
extra/amd_isa_lm_head_q6k_route_audit.py
```

Write:

```text
bench/amd-isa-backend-lm-head-q6k-route/latest.json
bench/amd-isa-backend-lm-head-q6k-route/summary.md
bench/amd-isa-backend-lm-head-q6k-route/current_route.json
bench/amd-isa-backend-lm-head-q6k-route/reduce_rows.json
bench/amd-isa-backend-lm-head-q6k-route/amdahl.json
```

Required inputs:

```text
bench/amd-isa-backend-q6k-residual-math/latest.json
bench/amd-isa-backend-q6k-residual-math/reduce_role_split.json
bench/amd-isa-backend-weight-path-ceiling/route_attribution.json
bench/amd-isa-backend-system-residual-ceiling/kernel_taxonomy.json
```

Required audit rows:

```text
lm_head_gemv:
  kernel name
  quant
  calls per step
  duration
  bytes
  effective bandwidth
  route family

lm_head_reduce:
  r_32_4_1187
  r_32_4_1187n1
  calls per step
  duration
  product 151936
  class q6k_lm_head_reduce_FIRM

other_vocab_shape_rows:
  any row containing 151936 or 1187
  whether it is reduce, GEMV, copy, or other
```

Required math:

```text
p_lm_head_gemv(ctx)
p_lm_head_reduce_firm(ctx)
p_lm_head_total_firm(ctx) = p_lm_head_gemv(ctx) + p_lm_head_reduce_firm(ctx)
p_lm_head_removable(ctx) = p_lm_head_reduce_firm(ctx) unless design proves GEMV is also removable

S = 1 / (1 - p_lm_head_removable * r)
R_new = R0 * S
```

Use:

```text
r in {0.25, 0.50, 1.00}
ctx in {512, 4096}
```

LH0 verdicts:

```text
AMD_ISA_LM_HEAD_Q6K_AUDIT_PASS_REDUCE_TARGET_PINNED
AMD_ISA_LM_HEAD_Q6K_AUDIT_PASS_GEMV_AND_REDUCE_TARGET_PINNED
AMD_ISA_LM_HEAD_Q6K_AUDIT_BLOCKED_ROUTE_NOT_ROLE_RESOLVED
AMD_ISA_LM_HEAD_Q6K_AUDIT_REJECT_LOW_AMDAHL
```

Pass criteria:

```text
exact lm_head GEMV row identified
exact firm reduce rows identified
reduce rows tied to lm_head, not ambiguous prod==4096 rows
standalone lm_head Amdahl upside computed
implementation target selected
```

Stop condition:

```text
If LH0 cannot prove lm_head reduce ownership, do not implement a standalone lm_head route. If standalone upside is below the residual-tier threshold in the promotion policy, fold lm_head back into the broader Q6_K direct route.
```

## Phase LH1: Route Design

Only start if LH0 passes.

Goal: design a lower-pass lm_head route.

Design candidates:

```text
LH1A: direct lm_head warp route
  one warp/lane-map group produces final vocab stripe output without separate r_* reduce

LH1B: two-stage but cheaper reduce
  keep partials but fuse or shrink the r_32_4_1187/r_32_4_1187n1 reduce

LH1C: reuse Q4_K/G3-style lane map for Q6_K lm_head
  preserve Q6_K layout but use single-pass route shape

LH1D: reject standalone lm_head
  if GEMV is already near ceiling and reduce elimination cannot be done without a bigger Q6_K route
```

Required design artifact:

```text
bench/amd-isa-backend-lm-head-q6k-design/latest.json
bench/amd-isa-backend-lm-head-q6k-design/summary.md
bench/amd-isa-backend-lm-head-q6k-design/candidate_routes.json
bench/amd-isa-backend-lm-head-q6k-design/implementation_plan.json
bench/amd-isa-backend-lm-head-q6k-design/risk_register.json
```

Design must answer:

1. Can lm_head output be produced without the firm reduce?
2. If yes, which axis is assigned to warp lanes?
3. Does the route preserve Q6_K unpack/dequant semantics?
4. Does it increase global memory traffic?
5. Does it increase register/LDS pressure enough to erase the reduce win?
6. Is it generated/search-owned or hand-owned?
7. What flags select and roll it back?
8. Does it compose with the broader Q6_K direct route?

LH1 verdicts:

```text
AMD_ISA_LM_HEAD_Q6K_DESIGN_PASS_READY
AMD_ISA_LM_HEAD_Q6K_DESIGN_BLOCKED_NO_SINGLE_PASS_MAPPING
AMD_ISA_LM_HEAD_Q6K_DESIGN_BLOCKED_QUANT_LAYOUT
AMD_ISA_LM_HEAD_Q6K_DESIGN_BLOCKED_RESOURCE_PRESSURE
AMD_ISA_LM_HEAD_Q6K_DESIGN_REJECT_FOLD_INTO_Q6K_GENERAL
```

## Phase LH2: Minimal Correctness Implementation

Only start if LH1 passes.

Goal: route-bind lm_head candidate default-off and prove token correctness.

Rules:

```text
default-off
rollback flag required
no Q6_K demotion
no output dtype change
no hidden fallback
no default replacement before speed gate
```

Suggested flags:

```text
Q6K_LM_HEAD_DIRECT=1
Q6K_LM_HEAD_DIRECT_DISABLE=1
```

Use repo naming conventions if better flags already exist.

Required artifacts:

```text
bench/amd-isa-backend-lm-head-q6k-correctness/latest.json
bench/amd-isa-backend-lm-head-q6k-correctness/summary.md
bench/amd-isa-backend-lm-head-q6k-correctness/route_attribution.json
bench/amd-isa-backend-lm-head-q6k-correctness/token_gate.json
```

Required gates:

```text
token_match true
route_bound true
deterministic true
hidden_fallback false
rollback true
firm reduce row removed or reduced
lm_head GEMV row not regressed badly
```

Contexts:

```text
ctx512
ctx1024
ctx2048
ctx4096
```

LH2 verdicts:

```text
AMD_ISA_LM_HEAD_Q6K_PASS_CORRECTNESS
AMD_ISA_LM_HEAD_Q6K_BLOCKED_TOKEN_MISMATCH
AMD_ISA_LM_HEAD_Q6K_BLOCKED_ROUTE_BINDING
AMD_ISA_LM_HEAD_Q6K_BLOCKED_FALLBACK
AMD_ISA_LM_HEAD_Q6K_BLOCKED_REDUCE_NOT_REMOVED
```

## Phase LH3: Speed Gate

Only start if LH2 passes.

Goal: measure whether standalone lm_head route movement is real.

Build:

```text
extra/amd_isa_lm_head_q6k_speed_gate.py
```

Write:

```text
bench/amd-isa-backend-lm-head-q6k-speed/latest.json
bench/amd-isa-backend-lm-head-q6k-speed/summary.md
bench/amd-isa-backend-lm-head-q6k-speed/wd_table.json
bench/amd-isa-backend-lm-head-q6k-speed/kernel_taxonomy_before_after.json
bench/amd-isa-backend-lm-head-q6k-speed/amdahl_vs_measured.json
```

Compare:

```text
baseline current lm_head/Q6_K route
candidate lm_head direct route
rollback current route
```

Required speed checks:

```text
firm reduce rows:
  r_32_4_1187
  r_32_4_1187n1

must move:
  duration down or count eliminated

must not regress:
  lm_head GEMV duration
  token_match
  total W==D
```

LH3 verdicts:

```text
AMD_ISA_LM_HEAD_Q6K_PASS_SPEED
AMD_ISA_LM_HEAD_Q6K_CORRECT_BUT_NOT_FAST
AMD_ISA_LM_HEAD_Q6K_REGRESSION
AMD_ISA_LM_HEAD_Q6K_INCONCLUSIVE_NOISY_WD
```

Promotion threshold:

```text
standalone lm_head route:
  TIER_A_MAJOR: >=5.0% W==D improvement
  TIER_B_RESIDUAL: >=2.0% and <5.0% W==D improvement with firm reduce removal proven and no protected context regression >1.0%
  TIER_C_EQUIVALENT_CLEANUP: -1.0% to +2.0% W==D only if it materially simplifies/removes firm reduce route for broader Q6_K promotion

no context regression beyond the selected tier
```

Important: because lm_head firm reduce is roughly 2.2-2.4% GPU-time in the standalone LH0 audit, a perfect standalone lm_head-only speedup is a residual-tier win at best. Do not reject it solely for missing the old 5% bar. Promote standalone only if it cleanly removes the firm reduce and clears TIER_B_RESIDUAL; otherwise fold it into the broader Q6_K direct route.

## Phase LH4: Promotion / Integration

Only start if LH3 passes.

Goal: integrate lm_head route into the broader Q6_K search path.

Required:

```text
BubbleBeam/search candidate for lm_head Q6_K
quant guard = Q6_K
shape guard = lm_head/vocab shape
target guard = measured supported GPU
rollback to current route
route attribution labels
ledger entries for refuted lm_head axes
```

Artifacts:

```text
bench/amd-isa-backend-lm-head-q6k-promotion/latest.json
bench/amd-isa-backend-lm-head-q6k-promotion/summary.md
bench/amd-isa-backend-lm-head-q6k-promotion/search_space_update.json
```

LH4 verdicts:

```text
AMD_ISA_LM_HEAD_Q6K_PROMOTION_PASS
AMD_ISA_LM_HEAD_Q6K_PROMOTION_FOLD_INTO_Q6K_DIRECT
AMD_ISA_LM_HEAD_Q6K_PROMOTION_BLOCKED_SEARCH_BINDING
```

## Relationship To General Q6_K Route

lm_head can be handled in three ways:

```text
Option A: standalone lm_head route
  Use if LH3 clears TIER_A_MAJOR or TIER_B_RESIDUAL, or if TIER_C cleanup materially simplifies the broader Q6_K route with no regression.

Option B: fold into general Q6_K direct route
  Use if standalone lm_head is correct but too small by itself.

Option C: do not implement lm_head separately
  Use if LH0/LH1 shows the firm reduce cannot be removed without a larger route rewrite.
```

Preferred order:

```text
1. Run LH0 audit.
2. If standalone upside is enough, run LH1 design.
3. If standalone upside is too small, fold lm_head into Q6K-1 general design instead of building a separate route.
```

## Claude Prompt

```text
You are working in /home/ubuntu/tinygrad-arkey.

Task: execute LH0 only: lm_head/Q6_K route audit. Do not implement kernels.

Read:
- docs/archive/amd-isa-lm-head-q6k-route-scope-20260629.md
- docs/amd-isa-q6k-direct-route-full-scope-20260629.md
- docs/archive/amd-isa-q6k-residual-amdahl-math-20260629.md

Inputs:
- bench/amd-isa-backend-q6k-residual-math/latest.json
- bench/amd-isa-backend-q6k-residual-math/reduce_role_split.json
- bench/amd-isa-backend-q6k-residual-math/q6k_route_candidates.json
- bench/amd-isa-backend-weight-path-ceiling/route_attribution.json
- bench/amd-isa-backend-system-residual-ceiling/kernel_taxonomy.json

Build:
- extra/amd_isa_lm_head_q6k_route_audit.py

Write:
- bench/amd-isa-backend-lm-head-q6k-route/latest.json
- bench/amd-isa-backend-lm-head-q6k-route/summary.md
- bench/amd-isa-backend-lm-head-q6k-route/current_route.json
- bench/amd-isa-backend-lm-head-q6k-route/reduce_rows.json
- bench/amd-isa-backend-lm-head-q6k-route/amdahl.json

Required:
- Identify exact lm_head GEMV row and effective bandwidth.
- Identify exact firm lm_head reduce rows: r_32_4_1187 and r_32_4_1187n1.
- Compute standalone lm_head Amdahl upside.
- Decide whether lm_head should be standalone or folded into the broader Q6_K direct route.
- Do not credit ambiguous prod==4096 reductions.
- Do not optimize lm_head GEMV if the data still shows it is bandwidth-healthy.

Verdicts:
- AMD_ISA_LM_HEAD_Q6K_AUDIT_PASS_REDUCE_TARGET_PINNED
- AMD_ISA_LM_HEAD_Q6K_AUDIT_PASS_GEMV_AND_REDUCE_TARGET_PINNED
- AMD_ISA_LM_HEAD_Q6K_AUDIT_BLOCKED_ROUTE_NOT_ROLE_RESOLVED
- AMD_ISA_LM_HEAD_Q6K_AUDIT_REJECT_LOW_AMDAHL

Stop after LH0 and report whether lm_head gets its own route or folds into Q6K-1.
```
