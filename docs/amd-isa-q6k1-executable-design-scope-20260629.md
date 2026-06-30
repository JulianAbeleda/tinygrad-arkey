# AMD ISA Q6K-1 Executable Design Scope

Date: 2026-06-29

Status: next executable phase. Design-only. No kernel implementation in Q6K-1.

## One-Line Goal

Design the Q6_K direct/warp route that replaces the current coop-partial + separate-reduce path with a lower-pass, search-owned route, while preserving Q6_K quant semantics and carrying the lm_head reduce as a folded-in TIER_B residual.

## What "Q6" Would Do

In this thread, "Q6" means the Q6_K route work, not changing model quality or demoting quantization.

The current Q6_K decode path appears to do:

```text
Q6_K weight GEMV
  -> coop partial outputs
  -> separate r_* reduce / sum
  -> final projection output
```

The proposed Q6_K direct route should do:

```text
Q6_K packed weight load
  -> Q6_K bit unpack + scale/dequant
  -> dot / accumulate
  -> in-warp or lane-map reduction
  -> final output
```

The practical difference:

```text
old route:
  split/coop partials + extra reduce kernels

new route:
  direct or fewer-pass warp/lane-map route with less partial traffic and less separate reduce
```

What it must not do:

```text
do not change Q6_K bits
do not demote Q6_K to Q4_K
do not change dtypes / quality semantics
do not touch Q4_K G3 while parity holds
do not optimize lm_head GEMV blindly; it is bandwidth-healthy
```

## Why Q6_K Is The Next Route

Q6K-0 proved:

```text
commit: 1a4055125
verdict: AMD_ISA_Q6K_RESIDUAL_PASS_DIRECT_ROUTE_JUSTIFIED

ctx512:
  affected q6k_gemv + lm_head share = 19.44% GPU-time
  firm removable share              = 6.53% GPU-time
  conservative W==D gain            = +7.0%

ctx4096:
  affected q6k_gemv + lm_head share = 17.75% GPU-time
  firm removable share              = 5.98% GPU-time
  conservative W==D gain            = +6.4%
```

LH0 then reclassified lm_head:

```text
commit: 6f7aa00a7
lm_head GEMV: bandwidth-healthy, ~761 GB/s
lm_head firm reduce: valid TIER_B residual
standalone lm_head: not preferred
decision: fold lm_head reduce into general Q6_K route
```

So Q6K-1 should design one general route that covers:

```text
1. Q6_K GEMV bandwidth gap:
   current Q6_K route ~503 GB/s
   Q4_K G3 reference  ~650 GB/s

2. Q6_K coop partial / reduce removal:
   firm lm_head reduce included
   ambiguous prod==4096 reductions not credited unless proven
```

## Branch / Merge Guidance

Current repository state at the time of this scope:

```text
branch: master
master is ahead of origin/master by 54 commits
origin/master is old relative to local work
there is a local psp-top-table branch, but it is behind by 1638 commits and is unrelated
worktree has pre-existing dirty bench artifacts and untracked docs
```

Why not merge "the branch" immediately:

1. The current useful work is already on local `master`, not on a clean feature branch.
2. `origin/master` is far behind this local stack, so a normal merge does not solve anything.
3. The visible `psp-top-table` branch is stale/unrelated, so merging it would add risk without helping Q6_K.
4. The worktree is dirty with pre-existing artifacts; merging now would make conflict attribution harder.
5. The Q6_K route is not implemented yet. Merging a design/audit stack upstream before the implementation gate passes would mix speculative docs/tools with production code.

Recommended branch strategy:

```text
1. Keep current local master as the integration scratch stack.
2. Before Q6K-2 implementation, create a dedicated branch from the current local HEAD:

   git switch -c q6k-direct-route

3. Commit Q6K-1 design artifacts there.
4. Implement Q6K-2/Q6K-3 on that branch.
5. Only merge back after correctness + speed gates pass.
6. If sharing externally, push the feature branch instead of force-pushing local master.
```

If the user explicitly wants a merge now, first run:

```text
git status --short --branch
git log --oneline --decorate --graph --max-count=30 --all
git branch --contains 6f7aa00a7
```

Then decide whether the target is:

```text
local master
origin/master
a new feature branch
an upstream PR branch
```

Do not merge stale `psp-top-table` into this stack unless it is explicitly required for Q6_K.

## Q6K-1 Deliverable

Build an audit/design tool:

```text
extra/amd_isa_q6k_direct_route_design.py
```

Write artifacts:

```text
bench/amd-isa-backend-q6k-direct-route-design/latest.json
bench/amd-isa-backend-q6k-direct-route-design/summary.md
bench/amd-isa-backend-q6k-direct-route-design/current_route.json
bench/amd-isa-backend-q6k-direct-route-design/candidate_routes.json
bench/amd-isa-backend-q6k-direct-route-design/implementation_plan.json
bench/amd-isa-backend-q6k-direct-route-design/risk_register.json
bench/amd-isa-backend-q6k-direct-route-design/merge_plan.json
```

Q6K-1 is complete only when it returns one of:

```text
AMD_ISA_Q6K_DIRECT_DESIGN_PASS_READY
AMD_ISA_Q6K_DIRECT_DESIGN_BLOCKED_NO_ROUTE_MAPPING
AMD_ISA_Q6K_DIRECT_DESIGN_BLOCKED_QUANT_LAYOUT_UNCLEAR
AMD_ISA_Q6K_DIRECT_DESIGN_BLOCKED_ROLE_ATTRIBUTION
AMD_ISA_Q6K_DIRECT_DESIGN_BLOCKED_LOW_AMDAHL_AFTER_DESIGN
AMD_ISA_Q6K_DIRECT_DESIGN_BLOCKED_BRANCH_OR_ARTIFACT_STATE
```

## Required Inputs

Use:

```text
bench/amd-isa-backend-q6k-residual-math/latest.json
bench/amd-isa-backend-q6k-residual-math/q6k_route_candidates.json
bench/amd-isa-backend-q6k-residual-math/reduce_role_split.json
bench/amd-isa-backend-lm-head-q6k-route/latest.json
bench/amd-isa-backend-lm-head-q6k-route/amdahl.json
bench/amd-isa-backend-system-residual-ceiling/latest.json
bench/amd-isa-backend-system-residual-ceiling/kernel_taxonomy.json
bench/amd-isa-backend-weight-path-ceiling/route_attribution.json
bench/amd-isa-backend-g3-weight-promotion/latest.json
```

Read docs:

```text
docs/amd-isa-q6k-direct-route-full-scope-20260629.md
docs/amd-isa-lm-head-q6k-route-scope-20260629.md
docs/amd-isa-q6k-next-task-scope-20260629.md
docs/amd-isa-q6k-residual-amdahl-math-20260629.md
```

## Q6K-1 Required Analysis

### 1. Current Route Inventory

Identify exact current Q6_K route rows:

```text
kernel_name
role
quant
shape
calls_per_step
duration_per_step_us
bytes_per_step
effective_bw
route_family
is_gemv
is_reduce
is_lm_head
is_ambiguous
```

Separate rows into:

```text
q6k_gemv_proven
lm_head_gemv
lm_head_firm_reduce
q6k_likely_reduce
ambiguous_reduce_prod4096
other_reduce
not_q6k
```

Rules:

```text
lm_head GEMV is not a target unless new data refutes ~761 GB/s.
lm_head firm reduce is a folded-in target.
prod==4096 reductions are not credited unless role attribution proves they are Q6_K-owned.
```

### 2. Q6_K Quant Semantics

The design must locate and cite the current Q6_K unpack/dequant code path.

It must record:

```text
block size
payload layout
high-bit layout
scale layout
zero/min behavior, if any
accumulator dtype
output dtype
rounding/cast behavior
vectorization assumptions
current kernel path
```

If the current code does not make this clear, Q6K-1 must return:

```text
AMD_ISA_Q6K_DIRECT_DESIGN_BLOCKED_QUANT_LAYOUT_UNCLEAR
```

### 3. Candidate Topologies

Evaluate at least these route candidates:

```text
Candidate A: single_pass_warp_q6k
  one warp/lane group produces final output without separate partial reduce

Candidate B: q6k_lanemap_g3_like
  adapt Q4_K G3 lane-map ideas to Q6_K packing/dequant

Candidate C: two_stage_less_reduce_q6k
  keep partials but reduce fewer times or fuse the firm reduce

Candidate D: lm_head_folded_direct
  do not build standalone lm_head; include firm lm_head reduce in Candidate A/B/C

Candidate E: reject_current
  if no route can preserve Q6_K semantics while clearing TIER_B/TIER_A
```

For each candidate, output:

```text
route_id
roles_covered
kernels_replaced
kernels_remaining
firm_removable_pct
ambiguous_removable_pct
expected_WD_gain
promotion_tier
required_primitives
implementation_complexity
correctness_risk
performance_risk
rollback_plan
```

### 4. Primitive Checklist

For the selected candidate, prove these primitives exist or specify the blocker:

```text
packed Q6_K load
Q6_K low/high-bit extraction
scale/dequant math
accumulation dtype
lane shuffle / lane map
in-register reduction
final output store
shape guard for supported roles
route attribution label
rollback flag
```

If a primitive is missing, return a blocker verdict instead of writing pseudocode that assumes it exists.

### 5. Amdahl Classification

Use the tiered policy:

```text
TIER_A_MAJOR:
  >=5.0% W==D

TIER_B_RESIDUAL:
  >=2.0% and <5.0% W==D
  requires clean mechanism proof and no protected context regression >1.0%

TIER_C_EQUIVALENT_CLEANUP:
  -1.0% to +2.0% W==D
  not a speed win; only useful for purity/simplification
```

The selected Q6_K general route should target:

```text
TIER_A_MAJOR using firm combined Q6K-0 removables
```

But if Q6K-1 selects a smaller first implementation:

```text
TIER_B_RESIDUAL is acceptable only with clean mechanism proof
```

### 6. Implementation Plan For Q6K-2

Write a concrete Q6K-2 plan:

```text
files_to_edit
new_flags
route guards
new labels
correctness gates
rollback path
expected artifacts
first microgate
first full-model token gate
stop conditions
```

Do not implement this plan in Q6K-1.

## Q6K-2 Preview

Q6K-2, if unlocked, would implement the smallest default-off route from Q6K-1.

Likely behavior:

```text
Q6K_DIRECT_ROUTE=1
  route selected Q6_K roles through the new direct route
  emit route labels
  preserve current route as fallback
  token-match against baseline
```

Expected first correctness gate:

```text
single Q6_K role microgate
then ctx512 token gate
then ctx512/1024/2048/4096 route-bound token gate
```

Expected first speed gate:

```text
Q6K-3 compares:
  baseline current Q6_K route
  candidate direct route
  rollback route
```

## Q6K-1 Success Criteria

Pass only if all are true:

```text
exact current route inventoried
Q6_K quant semantics cited
candidate route selected
firm removables mapped to route mechanism
expected gain classified under tiered policy
rollback strategy defined
Q6K-2 implementation plan is concrete
branch/merge plan is explicit
```

## Claude Prompt

```text
You are working in /home/ubuntu/tinygrad-arkey.

Task: execute Q6K-1 only: exhaustive direct/warp Q6_K route design. Do not implement kernels.

Read:
- docs/amd-isa-q6k1-executable-design-scope-20260629.md
- docs/amd-isa-q6k-direct-route-full-scope-20260629.md
- docs/amd-isa-lm-head-q6k-route-scope-20260629.md
- docs/amd-isa-q6k-residual-amdahl-math-20260629.md

Inputs:
- bench/amd-isa-backend-q6k-residual-math/latest.json
- bench/amd-isa-backend-q6k-residual-math/q6k_route_candidates.json
- bench/amd-isa-backend-q6k-residual-math/reduce_role_split.json
- bench/amd-isa-backend-lm-head-q6k-route/latest.json
- bench/amd-isa-backend-lm-head-q6k-route/amdahl.json
- bench/amd-isa-backend-system-residual-ceiling/latest.json
- bench/amd-isa-backend-system-residual-ceiling/kernel_taxonomy.json
- bench/amd-isa-backend-weight-path-ceiling/route_attribution.json
- bench/amd-isa-backend-g3-weight-promotion/latest.json

Build:
- extra/amd_isa_q6k_direct_route_design.py

Write:
- bench/amd-isa-backend-q6k-direct-route-design/latest.json
- bench/amd-isa-backend-q6k-direct-route-design/summary.md
- bench/amd-isa-backend-q6k-direct-route-design/current_route.json
- bench/amd-isa-backend-q6k-direct-route-design/candidate_routes.json
- bench/amd-isa-backend-q6k-direct-route-design/implementation_plan.json
- bench/amd-isa-backend-q6k-direct-route-design/risk_register.json
- bench/amd-isa-backend-q6k-direct-route-design/merge_plan.json

Required:
- Inventory exact current Q6_K kernels and reduce rows.
- Locate/cite Q6_K quant layout and dequant semantics.
- Evaluate single_pass_warp_q6k, q6k_lanemap_g3_like, two_stage_less_reduce_q6k, and folded lm_head direct route.
- Do not credit ambiguous prod==4096 reductions unless proven Q6_K-owned.
- Do not optimize lm_head GEMV unless new evidence refutes its ~761 GB/s health.
- Select one candidate or return a precise blocker.
- Classify expected gain as TIER_A/TIER_B/TIER_C.
- Produce concrete Q6K-2 implementation plan.
- Include branch/merge guidance. Current local master is ahead of origin; do not merge stale psp-top-table.

Verdicts:
- AMD_ISA_Q6K_DIRECT_DESIGN_PASS_READY
- AMD_ISA_Q6K_DIRECT_DESIGN_BLOCKED_NO_ROUTE_MAPPING
- AMD_ISA_Q6K_DIRECT_DESIGN_BLOCKED_QUANT_LAYOUT_UNCLEAR
- AMD_ISA_Q6K_DIRECT_DESIGN_BLOCKED_ROLE_ATTRIBUTION
- AMD_ISA_Q6K_DIRECT_DESIGN_BLOCKED_LOW_AMDAHL_AFTER_DESIGN
- AMD_ISA_Q6K_DIRECT_DESIGN_BLOCKED_BRANCH_OR_ARTIFACT_STATE

Stop after Q6K-1. Do not implement Q6K-2.
```

