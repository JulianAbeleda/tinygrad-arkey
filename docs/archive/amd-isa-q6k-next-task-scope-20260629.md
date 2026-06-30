# AMD ISA Q6_K Next Task Scope

Date: 2026-06-29

Status: high-level scope for the next track. This is intentionally staged: prove the target first, then implement only if the proof gate clears.

## Executive Summary

The Q4_K weight path is no longer the live residual:

- G3 LaneMap is speed-equivalent to owned for the major Q4_K decode roles.
- Q4_K G3 GEMV is already near its practical dequant-GEMV bandwidth ceiling.
- The raw `820 GB/s` memcpy number is not the correct full-decode ceiling.

The next suspect is the Q6_K route, because the system-residual audit shows:

```text
ctx512:
  q6k_gemv       = 13.6% GPU-time
  lm_head        =  5.7% GPU-time
  proven share   = 19.3%

  reduce_partial = 22.4% GPU-time, but mixed / not role-resolved

ctx4096:
  q6k_gemv       = 12.5% GPU-time
  lm_head        =  5.3% GPU-time
  proven share   = 17.8%

  reduce_partial = 20.7% GPU-time, but mixed / not role-resolved
```

The correct next task is therefore:

```text
Prove how much of the remaining decode wall is genuinely Q6_K route overhead, then implement a direct/warp Q6_K route only if the measured Amdahl upside clears the threshold.
```

Do not start by writing the kernel. Start by proving the math.

## Source Artifacts

Use these current artifacts as inputs:

```text
bench/amd-isa-backend-system-residual-ceiling/latest.json
bench/amd-isa-backend-system-residual-ceiling/kernel_taxonomy.json
bench/amd-isa-backend-system-residual-ceiling/probe_matrix.json
bench/amd-isa-backend-g3-weight-promotion/latest.json
bench/amd-isa-backend-weight-path-ceiling/route_attribution.json
bench/amd-isa-backend-weight-path-ceiling/latest.json
docs/archive/amd-isa-q6k-residual-amdahl-math-20260629.md
```

## Phase Q6K-0: Residual Proof Gate

Goal: determine whether Q6_K direct/warp routing is justified before implementing it.

Build:

```text
extra/amd_isa_q6k_residual_math_gate.py
```

Write:

```text
bench/amd-isa-backend-q6k-residual-math/latest.json
bench/amd-isa-backend-q6k-residual-math/summary.md
bench/amd-isa-backend-q6k-residual-math/reduce_role_split.json
bench/amd-isa-backend-q6k-residual-math/amdahl_sensitivity.json
bench/amd-isa-backend-q6k-residual-math/q6k_route_candidates.json
```

Required analysis:

1. Load current system-residual and route-attribution artifacts.
2. Compute conservative Q6_K share:

```text
p_q6k_proven(ctx) = p_q6k_gemv(ctx) + p_lm_head(ctx)
```

3. Role-resolve `reduce_partial` as far as the existing route attribution allows.
4. Define:

```text
a(ctx) = fraction of reduce_partial attributable to Q6_K coop partials+sum
p_q6k_total(ctx) = p_q6k_proven(ctx) + a(ctx) * p_reduce_partial(ctx)
```

5. Compute Amdahl projections:

```text
S(ctx)     = 1 / (1 - p_q6k_total(ctx) * r)
R_new(ctx) = R0(ctx) * S(ctx)
```

for:

```text
a in {0, 0.25, 0.5, 1.0}
r in {0.25, 0.50, 1.00}
ctx in {512, 4096}
```

6. Identify exact candidate routes:

```text
current_q6k_route:
  q6k_gemv kernel names
  partial kernels
  reduce kernels
  call counts
  bytes
  dur_per_step
  effective bandwidth

candidate_direct_route:
  which roles it would replace
  whether it can be single-pass like Q4_K G3
  whether it preserves quality / dtype / quant semantics
  expected removed fraction r
```

Verdicts:

```text
AMD_ISA_Q6K_RESIDUAL_PASS_DIRECT_ROUTE_JUSTIFIED
  Q6_K affected share >= 10% at ctx512 or ctx4096, and a credible direct route can remove >=25% of that affected share.

AMD_ISA_Q6K_RESIDUAL_INCONCLUSIVE_REDUCE_NOT_ROLE_RESOLVED
  q6k_gemv+lm_head is visible, but reduce_partial cannot be assigned and route upside is not bounded tightly enough.

AMD_ISA_Q6K_RESIDUAL_PASS_RECLASSIFY_TARGET
  another bucket has higher proven W==D upside.
```

Stop condition:

```text
If Q6K-0 does not return PASS_DIRECT_ROUTE_JUSTIFIED, stop. Do not implement Q6_K.
```

## Phase Q6K-1: Direct/Warp Route Design

Only start if Q6K-0 passes.

Goal: design a Q6_K route equivalent in spirit to Q4_K G3 LaneMap:

```text
single-pass or fewer-pass route
less coop partial traffic
less separate reduce time
same token correctness
same quant semantics
BubbleBeam/search-owned selection
rollback to current route one flag away
```

Required design artifact:

```text
bench/amd-isa-backend-q6k-direct-route-design/latest.json
bench/amd-isa-backend-q6k-direct-route-design/summary.md
```

The design must answer:

1. Which exact Q6_K roles are targeted?
2. Is `lm_head` actually part of the route, or already near ceiling?
3. Is the target Q6_K GEMV-only, Q6_K+reduce, or Q6_K+lm_head?
4. What current kernels are replaced?
5. What current kernels remain?
6. Is the new route pure/generated/search-owned, or hand-owned?
7. What flags select it?
8. What flags roll it back?
9. What is the predicted Amdahl gain?

Verdicts:

```text
AMD_ISA_Q6K_DIRECT_DESIGN_PASS_READY
AMD_ISA_Q6K_DIRECT_DESIGN_BLOCKED_NO_SINGLE_PASS_MAPPING
AMD_ISA_Q6K_DIRECT_DESIGN_BLOCKED_QUANT_LAYOUT
AMD_ISA_Q6K_DIRECT_DESIGN_BLOCKED_QUALITY_OR_DTYPE
```

## Phase Q6K-2: Minimal Correctness Route

Only start if Q6K-1 passes.

Goal: implement the smallest route that proves correctness for one Q6_K role.

Rules:

- Default-off.
- No owned default replacement.
- No quant demotion.
- No quality change.
- No broad refactor.
- No autogen churn unless truly required by local codegen convention.

Required gates:

```text
route_bound = true
token_match = true
deterministic = true
hidden_fallback = false
rollback_flag_works = true
```

Artifacts:

```text
bench/amd-isa-backend-q6k-direct-correctness/latest.json
bench/amd-isa-backend-q6k-direct-correctness/summary.md
```

Verdicts:

```text
AMD_ISA_Q6K_DIRECT_PASS_CORRECTNESS
AMD_ISA_Q6K_DIRECT_BLOCKED_ROUTE_BINDING
AMD_ISA_Q6K_DIRECT_BLOCKED_TOKEN_MISMATCH
AMD_ISA_Q6K_DIRECT_BLOCKED_FALLBACK
```

Stop condition:

```text
If correctness fails, stop and record the blocker. Do not continue to speed.
```

## Phase Q6K-3: Speed Gate

Only start if Q6K-2 passes.

Goal: measure whether the route actually moves W==D and whether the measured movement matches the Q6K-0 Amdahl prediction.

Required contexts:

```text
ctx512
ctx1024
ctx2048
ctx4096
```

Compare:

```text
baseline current Q6_K route
candidate direct/warp Q6_K route
rollback route
```

Required outputs:

```text
bench/amd-isa-backend-q6k-direct-speed/latest.json
bench/amd-isa-backend-q6k-direct-speed/summary.md
bench/amd-isa-backend-q6k-direct-speed/wd_table.json
bench/amd-isa-backend-q6k-direct-speed/route_counts.json
bench/amd-isa-backend-q6k-direct-speed/amdahl_vs_measured.json
```

Promotion thresholds:

```text
PASS_SPEED:
  token_match true at all contexts
  route_bound true at all contexts
  hidden_fallback false
  TIER_A_MAJOR: >=5.0% median W==D improvement, no context regresses >2.0%
  TIER_B_RESIDUAL: >=2.0% and <5.0% median W==D improvement, mechanism proof clean, no context regresses >1.0%

PASS_SPEED_EQUIVALENT:
  TIER_C_EQUIVALENT_CLEANUP: -1.0% to +2.0% median W==D movement
  useful only if it retires owned/hand code, removes a known residual, or simplifies search

CORRECT_BUT_NOT_FAST:
  correctness passes but does not clear TIER_B_RESIDUAL and has no TIER_C cleanup value

REGRESSION:
  any context loses beyond the selected tier's allowed regression or token mismatch/fallback appears
```

## Phase Q6K-4: Search Binding / Promotion

Only start if Q6K-3 passes speed.

Goal: make BubbleBeam/search select the route without manual forced flags.

Required:

1. Add route candidate to the search manifest.
2. Add ledger entries for refuted Q6_K axes.
3. Keep rollback flags.
4. Update route attribution to distinguish:

```text
q6k_current_coop
q6k_direct_warp
q6k_reduce_partial
lm_head_q6k
fallback
```

Artifacts:

```text
bench/amd-isa-backend-q6k-promotion/latest.json
bench/amd-isa-backend-q6k-promotion/summary.md
bench/amd-isa-backend-q6k-promotion/search_space_update.json
```

Verdicts:

```text
AMD_ISA_Q6K_PROMOTION_PASS
AMD_ISA_Q6K_PROMOTION_CORRECT_BUT_NOT_FAST
AMD_ISA_Q6K_PROMOTION_BLOCKED_SEARCH_BINDING
```

## What Not To Do

Do not:

- Re-open Q4_K layout reshuffle unless G3 parity is refuted.
- Treat `reduce_partial` as Q6_K-owned without role attribution.
- Use Q6_K bit-demotion as the solution; that is a quality change, not route efficiency.
- Claim the full `163 tok/s` memcpy ceiling is reachable by a Q6_K kernel.
- Optimize attention again unless a new whole-step audit says attention is the dominant wall.
- Start implementation before Q6K-0 selects the route.

## Expected Outcomes

Conservative:

```text
Q6_K proven share ~18-19%
remove 25-50% of that share
expected W==D: ~99-115 tok/s depending on ctx
```

If half of `reduce_partial` is Q6_K-owned:

```text
affected share ~28-31%
remove 25-50%
expected W==D: ~102-123 tok/s depending on ctx
```

If all of `reduce_partial` is Q6_K-owned:

```text
affected share ~38-42%
remove 25-50%
expected W==D: ~104-131 tok/s depending on ctx
```

The first implementation target should not claim more than the conservative case until `reduce_partial` is role-resolved.

## Claude Prompt

```text
You are working in /home/ubuntu/tinygrad-arkey.

Task: execute the next Q6_K track in stages. Start with Q6K-0 only. Do not implement kernels unless Q6K-0 returns PASS_DIRECT_ROUTE_JUSTIFIED.

Read first:
- docs/archive/amd-isa-q6k-residual-amdahl-math-20260629.md
- docs/archive/amd-isa-q6k-next-task-scope-20260629.md

Inputs:
- bench/amd-isa-backend-system-residual-ceiling/latest.json
- bench/amd-isa-backend-system-residual-ceiling/kernel_taxonomy.json
- bench/amd-isa-backend-system-residual-ceiling/probe_matrix.json
- bench/amd-isa-backend-g3-weight-promotion/latest.json
- bench/amd-isa-backend-weight-path-ceiling/route_attribution.json
- bench/amd-isa-backend-weight-path-ceiling/latest.json

Phase Q6K-0:
- Build extra/amd_isa_q6k_residual_math_gate.py.
- Write bench/amd-isa-backend-q6k-residual-math/{latest.json,summary.md,reduce_role_split.json,amdahl_sensitivity.json,q6k_route_candidates.json}.
- Compute p_q6k_proven = q6k_gemv + lm_head.
- Role-resolve reduce_partial as far as current attribution allows.
- Compute p_q6k_total(a) = p_q6k_proven + a * reduce_partial for a in {0, 0.25, 0.5, 1.0}.
- Compute Amdahl projections for r in {0.25, 0.50, 1.00} at ctx512 and ctx4096.
- Identify exact current Q6_K route kernels, partial kernels, reduce kernels, call counts, bytes, time, and effective bandwidth.

Verdicts:
- AMD_ISA_Q6K_RESIDUAL_PASS_DIRECT_ROUTE_JUSTIFIED
- AMD_ISA_Q6K_RESIDUAL_INCONCLUSIVE_REDUCE_NOT_ROLE_RESOLVED
- AMD_ISA_Q6K_RESIDUAL_PASS_RECLASSIFY_TARGET

Discipline:
- Audit only in Q6K-0.
- Do not claim reduce_partial belongs to Q6_K unless role attribution proves it.
- Do not treat 820 GB/s memcpy as the full decode target.
- Do not reopen Q4_K layout while G3 parity holds.
- Stop after Q6K-0 and report the verdict, Amdahl table, exact route candidates, and whether implementation is justified.
```
