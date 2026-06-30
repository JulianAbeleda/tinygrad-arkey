# AMD ISA Q6_K Direct/Warp Route Full Scope

Date: 2026-06-29

Status: implementation track scope, unlocked by Q6K-0. This document scopes Q6K-1 through Q6K-4, but the next executable phase is Q6K-1 design only.

## Source Of Truth

This track is justified by Q6K-0:

```text
verdict: AMD_ISA_Q6K_RESIDUAL_PASS_DIRECT_ROUTE_JUSTIFIED
commit: 1a4055125
tool: extra/amd_isa_q6k_residual_math_gate.py
artifacts:
  bench/amd-isa-backend-q6k-residual-math/latest.json
  bench/amd-isa-backend-q6k-residual-math/summary.md
  bench/amd-isa-backend-q6k-residual-math/reduce_role_split.json
  bench/amd-isa-backend-q6k-residual-math/amdahl_sensitivity.json
  bench/amd-isa-backend-q6k-residual-math/q6k_route_candidates.json
```

Measured Q6K-0 facts:

```text
ctx512:
  q6k_gemv + lm_head affected share = 19.44% GPU-time
  firm Q6_K reduce share             = 3.44% GPU-time
  ambiguous reduce share             = 5.70% GPU-time
  firm removable share               = 6.53% GPU-time
  conservative W==D gain             = +7.0%

ctx4096:
  q6k_gemv + lm_head affected share = 17.75% GPU-time
  firm Q6_K reduce share             = 3.17% GPU-time
  ambiguous reduce share             = 5.25% GPU-time
  firm removable share               = 5.98% GPU-time
  conservative W==D gain             = +6.4%
```

The implementation lever is:

```text
direct/warp single-pass Q6_K route eliminating coop partials+sum
```

This is route efficiency. It is not quant demotion, not Q4_K layout reshuffle, and not attention.

## Why This Track Exists

The current system state:

- Q4_K G3 LaneMap is speed-equivalent to owned.
- Q4_K is near its practical dequant-GEMV ceiling.
- Decode attention was measured as low leverage after the native route work.
- The remaining system residual points to Q6_K route inefficiency.
- Q6K-0 proved a conservative `+6-7%` W==D opportunity from firm removables alone.

The Q6_K problem is structurally different from the closed Q4_K track:

```text
Q4_K:
  generated G3 route
  single-pass / speed-equivalent to owned
  no separate live coop reduce residual

Q6_K:
  coop partial route
  separate r_* reduce for at least lm_head
  q6k_gemv effective bw ~503 GB/s vs Q4_K G3 ~650 GB/s
  firm removable subset clears implementation threshold
```

## Hard Non-Goals

Do not:

- reduce Q6_K precision or demote to fewer bits,
- re-open Q4_K layout reshuffle while G3 parity holds,
- optimize attention in this track,
- claim the raw `820 GB/s` memcpy number as the decode target,
- credit ambiguous per-layer `prod==4096` reductions as Q6_K unless new role attribution proves it,
- replace defaults before correctness + speed gates pass,
- delete rollback flags or owned/current fallback routes.

## Track Overview

```text
Q6K-1: Direct/warp route design
Q6K-2: Minimal correctness implementation
Q6K-3: Speed and attribution gate
Q6K-4: BubbleBeam/search binding and promotion package
```

Each phase must stop at the first hard blocker. A later phase cannot pass because an earlier microgate passed.

## Q6K-1: Direct/Warp Route Design

Goal: prove a concrete route design before writing the kernel.

Build:

```text
extra/amd_isa_q6k_direct_route_design.py
```

Write:

```text
bench/amd-isa-backend-q6k-direct-route-design/latest.json
bench/amd-isa-backend-q6k-direct-route-design/summary.md
bench/amd-isa-backend-q6k-direct-route-design/current_route.json
bench/amd-isa-backend-q6k-direct-route-design/candidate_routes.json
bench/amd-isa-backend-q6k-direct-route-design/risk_register.json
bench/amd-isa-backend-q6k-direct-route-design/implementation_plan.json
```

Required inputs:

```text
bench/amd-isa-backend-q6k-residual-math/latest.json
bench/amd-isa-backend-q6k-residual-math/q6k_route_candidates.json
bench/amd-isa-backend-q6k-residual-math/reduce_role_split.json
bench/amd-isa-backend-system-residual-ceiling/latest.json
bench/amd-isa-backend-weight-path-ceiling/route_attribution.json
bench/amd-isa-backend-g3-weight-promotion/latest.json
```

Required design questions:

1. Which roles are targeted first?

```text
candidate roles:
  q6k_gemv
  lm_head q6k route
  lm_head firm coop reduce
  ambiguous per-layer reduce only if role attribution proves ownership
```

2. Which current kernels are replaced?

The design must name exact current kernel names, call counts, duration per step, bytes, and effective bandwidth.

3. Which current kernels remain?

The design must not accidentally claim to remove RMSNorm, flash reduce, or unrelated `r_*` kernels.

4. What is the candidate topology?

Acceptable route-family candidates:

```text
single_pass_warp_q6k
lanemap_q6k
direct_lm_head_q6k
two_stage_less_reduce_q6k
generated_native_isa_q6k
```

5. What primitives are required?

The design must list required primitives:

```text
packed q6k load
q6k bit unpack
scale/dequant
dot accumulation
lane map / shuffle
in-register reduction
output write
optional split handling
```

6. What exact quant semantics must be preserved?

The design must cite:

```text
storage layout
scale layout
high/low bit layout
accumulator dtype
rounding / cast behavior
output dtype
token-match expectation
```

7. What is the expected removable fraction?

Use the Q6K-0 Amdahl model:

```text
firm_removable_pct_gpu:
  ctx512  = 6.53%
  ctx4096 = 5.98%

expected W==D:
  ctx512  ~= +7.0%
  ctx4096 ~= +6.4%
```

8. What is the smallest correctness-first implementation?

The design should choose one of:

```text
lm_head firm-reduce elimination first
q6k_gemv bw route first
combined q6k_gemv + firm-reduce route
```

The preferred choice should maximize firm removable share while minimizing correctness blast radius.

Q6K-1 verdicts:

```text
AMD_ISA_Q6K_DIRECT_DESIGN_PASS_READY
AMD_ISA_Q6K_DIRECT_DESIGN_BLOCKED_NO_ROUTE_MAPPING
AMD_ISA_Q6K_DIRECT_DESIGN_BLOCKED_QUANT_LAYOUT_UNCLEAR
AMD_ISA_Q6K_DIRECT_DESIGN_BLOCKED_ROLE_ATTRIBUTION
AMD_ISA_Q6K_DIRECT_DESIGN_BLOCKED_LOW_AMDAHL_AFTER_DESIGN
```

Q6K-1 pass criteria:

```text
exact current kernels named
exact candidate route named
firm removable rows mapped to implementation mechanism
quant semantics preserved
rollback flags identified
predicted Amdahl gain clears the tiered promotion policy below
minimal Q6K-2 implementation plan written
```

Stop condition:

```text
If Q6K-1 does not return PASS_READY, stop. Do not implement Q6K-2.
```

## Q6K-2: Minimal Correctness Implementation

Only start if Q6K-1 returns `AMD_ISA_Q6K_DIRECT_DESIGN_PASS_READY`.

Goal: implement the smallest default-off Q6_K direct route that proves token correctness and route binding.

Expected files to inspect before editing:

```text
tinygrad/renderer/amd.py
tinygrad/renderer/isa/amd.py
tinygrad/renderer/amd/elf.py
tinygrad/codegen/*
tinygrad/nn/state.py
tinygrad/tensor.py
extra/*q4k*
extra/*q6k*
extra/*g3*
```

Actual files to edit must be determined by Q6K-1. Do not assume these are all required.

Implementation rules:

```text
default_off = true
rollback_flag = required
owned/current fallback retained
no quant demotion
no quality change
no broad refactor
no autogen churn unless local convention requires it
```

Suggested flags:

```text
Q6K_DIRECT_ROUTE=1
Q6K_DIRECT_ROUTE_DISABLE=1
Q6K_DIRECT_ROUTE_FORCE_CURRENT=1
```

Use existing naming conventions if the repo already has better flag names.

Required gates:

```text
route_bound = true
token_match = true
deterministic = true
hidden_fallback = false
rollback_flag_works = true
current_default_unchanged = true
```

Required artifacts:

```text
bench/amd-isa-backend-q6k-direct-correctness/latest.json
bench/amd-isa-backend-q6k-direct-correctness/summary.md
bench/amd-isa-backend-q6k-direct-correctness/route_attribution.json
bench/amd-isa-backend-q6k-direct-correctness/token_gate.json
```

Required correctness matrix:

```text
ctx512
ctx1024
ctx2048
ctx4096
```

Minimum route attribution:

```text
q6k_current_coop_count
q6k_direct_count
q6k_reduce_current_count
q6k_reduce_removed_count
lm_head_current_count
lm_head_direct_count
fallback_count
owned_count
unknown_count
```

Q6K-2 verdicts:

```text
AMD_ISA_Q6K_DIRECT_PASS_CORRECTNESS
AMD_ISA_Q6K_DIRECT_BLOCKED_ROUTE_BINDING
AMD_ISA_Q6K_DIRECT_BLOCKED_TOKEN_MISMATCH
AMD_ISA_Q6K_DIRECT_BLOCKED_NONDETERMINISM
AMD_ISA_Q6K_DIRECT_BLOCKED_HIDDEN_FALLBACK
AMD_ISA_Q6K_DIRECT_BLOCKED_QUANT_SEMANTICS
```

Stop condition:

```text
If correctness fails, stop. Do not run speed promotion.
```

## Promotion Threshold Policy

The old flat `>=5% W==D` threshold was useful while the search was finding large structural misses. At this stage, the route is near the practical model/GPU ceiling and remaining wins will often be smaller. Promotion should now use a tiered rule that combines wall movement with mechanism proof.

Promotion tiers:

```text
TIER_A_MAJOR:
  >=5.0% median W==D improvement
  ordinary PASS_SPEED if correctness/route gates pass

TIER_B_RESIDUAL:
  >=2.0% and <5.0% median W==D improvement
  can promote only if:
    mechanism counter moves in the predicted direction
    targeted kernel/reduce time drops by enough to explain the wall gain
    no protected context regresses >1.0%
    route is simpler, more search-owned, or removes a known residual
    rollback flag works

TIER_C_EQUIVALENT_CLEANUP:
  -1.0% to +2.0% median W==D movement
  can promote only if it retires owned/current special-case code or materially improves search purity
  cannot be sold as a speed win

REJECT_OR_DEFER:
  any protected context regresses >1.0% for residual-tier work
  or mechanism counters do not explain the measured gain
  or wall movement is inside noise with no structural simplification
```

For Q6_K specifically:

```text
Q6K-0 predicted conservative gain:
  ctx512  +7.0%
  ctx4096 +6.4%

So Q6_K direct route should still target TIER_A_MAJOR.
But if implementation captures only a firm subset, TIER_B_RESIDUAL is acceptable if the q6k/reduce mechanism proof is clean.
```

This policy prevents two failure modes:

```text
false negative:
  rejecting a real 2-4% residual win after the large levers are exhausted

false positive:
  promoting noisy wall movement with no route/mechanism proof
```

## Q6K-3: Speed And Attribution Gate

Only start if Q6K-2 returns `AMD_ISA_Q6K_DIRECT_PASS_CORRECTNESS`.

Goal: prove that measured W==D moves in the direction predicted by Q6K-0/Q6K-1.

Build:

```text
extra/amd_isa_q6k_direct_speed_gate.py
```

Write:

```text
bench/amd-isa-backend-q6k-direct-speed/latest.json
bench/amd-isa-backend-q6k-direct-speed/summary.md
bench/amd-isa-backend-q6k-direct-speed/wd_table.json
bench/amd-isa-backend-q6k-direct-speed/route_counts.json
bench/amd-isa-backend-q6k-direct-speed/amdahl_vs_measured.json
bench/amd-isa-backend-q6k-direct-speed/kernel_taxonomy_before_after.json
```

Compare arms:

```text
baseline_current_q6k
candidate_direct_q6k
rollback_current_q6k
```

Contexts:

```text
ctx512
ctx1024
ctx2048
ctx4096
```

Required measurements:

```text
tok_s_median
tok_s_spread
token_match
deterministic
route_bound
hidden_fallback
per_kernel_gpu_time
q6k_effective_bw
reduce_partial_role_split
firm_removable_before_after
```

Noise handling:

- W==D wall spread can be large due to AMD clock behavior.
- Use median and repeated contexts.
- Do not claim speed from one noisy context.
- Use route counts and per-kernel GPU-time to confirm the mechanism.

Speed pass thresholds:

```text
PASS_SPEED:
  token_match true all contexts
  route_bound true all contexts
  hidden_fallback false
  TIER_A_MAJOR or TIER_B_RESIDUAL by the promotion threshold policy
  no protected context regresses beyond the tier's allowed threshold
  q6k/reduce bucket moves in predicted direction

PASS_SPEED_EQUIVALENT:
  correctness passes
  TIER_C_EQUIVALENT_CLEANUP by the promotion threshold policy
  only useful if route is more pure/search-owned or retires owned/current special route

CORRECT_BUT_NOT_FAST:
  correctness passes
  W==D movement does not clear TIER_B_RESIDUAL and no TIER_C cleanup value exists

REGRESSION:
  any protected context regresses beyond the tier's allowed threshold
  or token_match/fallback/determinism fails
```

Q6K-3 verdicts:

```text
AMD_ISA_Q6K_DIRECT_PASS_SPEED
AMD_ISA_Q6K_DIRECT_PASS_SPEED_EQUIVALENT
AMD_ISA_Q6K_DIRECT_CORRECT_BUT_NOT_FAST
AMD_ISA_Q6K_DIRECT_REGRESSION
AMD_ISA_Q6K_DIRECT_INCONCLUSIVE_NOISY_WD
```

Stop condition:

```text
If Q6K-3 is CORRECT_BUT_NOT_FAST or REGRESSION, do not promote. Record why the Amdahl prediction failed.
```

## Q6K-4: BubbleBeam/Search Binding And Promotion

Only start if Q6K-3 returns `AMD_ISA_Q6K_DIRECT_PASS_SPEED`.

Goal: make the route search-owned and promotable without manual forced flags.

Required outputs:

```text
bench/amd-isa-backend-q6k-promotion/latest.json
bench/amd-isa-backend-q6k-promotion/summary.md
bench/amd-isa-backend-q6k-promotion/search_space_update.json
bench/amd-isa-backend-q6k-promotion/rollback.json
```

Required search-space changes:

```text
add q6k_direct_route candidate
add quant guard = Q6_K only
add shape guards for proven roles
add target guard = gfx1100 / supported targets only
add route attribution labels
ledger refuted axes
retain rollback to current q6k route
```

Required route labels:

```text
q6k_current_coop
q6k_direct_warp
q6k_firm_reduce_removed
q6k_ambiguous_reduce
lm_head_q6k
fallback
```

Promotion requirements:

```text
correctness:
  token_match true
  deterministic true
  quant semantics unchanged

route:
  route_bound true
  no hidden fallback
  rollback works

speed:
  clears TIER_A_MAJOR or TIER_B_RESIDUAL
  no protected context regression beyond the selected tier
  measured mechanism matches Q6K-0/Q6K-3

search:
  BubbleBeam selects route without forced diagnostic flag
  manifest updated
  ledger updated
  refuted axes retained
```

Q6K-4 verdicts:

```text
AMD_ISA_Q6K_PROMOTION_PASS
AMD_ISA_Q6K_PROMOTION_CORRECT_BUT_NOT_FAST
AMD_ISA_Q6K_PROMOTION_BLOCKED_SEARCH_BINDING
AMD_ISA_Q6K_PROMOTION_BLOCKED_ROUTE_ATTRIBUTION
AMD_ISA_Q6K_PROMOTION_BLOCKED_ROLLBACK
```

## Regression Ladder

Every implementation phase must preserve:

```text
G3 Q4_K promotion gate
system residual artifact assumptions
Q6K-0 math gate still valid or explicitly refreshed
attention native/default route unchanged
default AMD path unchanged unless explicitly promoted
token generation W==D
```

At minimum, run or refresh:

```text
extra/amd_isa_q6k_residual_math_gate.py
extra/amd_isa_g3_weight_promotion_gate.py
extra/amd_isa_system_residual_ceiling_audit.py
new q6k correctness/speed gates
```

If any source artifact becomes stale after implementation, regenerate it and cite the new commit/artifact.

## Risk Register

### R1: Q6_K layout may not map to single-pass cleanly

The Q6_K bit layout may require more unpack/high-bit handling than Q4_K. Q6K-1 must prove the route is legal before implementation.

### R2: lm_head GEMV is already bandwidth-efficient

Q6K-0 says the firm lm_head removable is the reduce, not the GEMV. Do not optimize lm_head GEMV if its bandwidth remains ~761 GB/s.

### R3: ambiguous per-layer reduces may be RMSNorm

Do not credit `prod==4096` reduces until role attribution proves they are Q6_K-owned.

### R4: W==D noise can mask a 5-7% win

Use per-kernel mechanism counters and multi-context median. If wall is inconclusive but mechanism is strong, verdict should be inconclusive, not pass.

### R5: direct route may reduce GPU-time but not wall

If graph overlap or another bucket dominates, Q6K-3 must record Amdahl mismatch and stop promotion.

### R6: search binding could route too broadly

Quant and shape guards must prevent applying the route to unproven quant formats or shapes.

## Expected Outcome

Conservative expected speed if the firm Q6K-0 removables are captured:

```text
ctx512:  ~103.9 tok/s -> ~111.2 tok/s
ctx4096: ~94.4 tok/s  -> ~100.4 tok/s
```

This corresponds to:

```text
ctx512:  +7.0%
ctx4096: +6.4%
```

If ambiguous reduces later prove Q6_K-owned, upside can be larger, but that must be treated as a second-stage bonus, not the basis for Q6K-2.

## Claude Prompt

```text
You are working in /home/ubuntu/tinygrad-arkey.

Task: execute Q6K-1 only: direct/warp Q6_K route design. Do not implement kernels yet.

Read:
- docs/amd-isa-q6k-direct-route-full-scope-20260629.md
- docs/amd-isa-q6k-next-task-scope-20260629.md
- docs/amd-isa-q6k-residual-amdahl-math-20260629.md

Source artifacts:
- bench/amd-isa-backend-q6k-residual-math/latest.json
- bench/amd-isa-backend-q6k-residual-math/q6k_route_candidates.json
- bench/amd-isa-backend-q6k-residual-math/reduce_role_split.json
- bench/amd-isa-backend-system-residual-ceiling/latest.json
- bench/amd-isa-backend-weight-path-ceiling/route_attribution.json
- bench/amd-isa-backend-g3-weight-promotion/latest.json

Build:
- extra/amd_isa_q6k_direct_route_design.py

Write:
- bench/amd-isa-backend-q6k-direct-route-design/latest.json
- bench/amd-isa-backend-q6k-direct-route-design/summary.md
- bench/amd-isa-backend-q6k-direct-route-design/current_route.json
- bench/amd-isa-backend-q6k-direct-route-design/candidate_routes.json
- bench/amd-isa-backend-q6k-direct-route-design/risk_register.json
- bench/amd-isa-backend-q6k-direct-route-design/implementation_plan.json

Required:
- Name exact current q6k/lm_head/reduce kernels, call counts, durations, bytes, bandwidth.
- Identify firm removable rows from Q6K-0 and map each to a concrete implementation mechanism.
- Do not credit ambiguous prod==4096 reduces unless role attribution proves they are Q6_K-owned.
- Propose the minimal Q6K-2 route: which role first, which kernels replaced, which flags, which rollback.
- Preserve quant semantics; no bit demotion.
- Compute predicted Amdahl gain and classify it under TIER_A_MAJOR / TIER_B_RESIDUAL / TIER_C_EQUIVALENT_CLEANUP.

Verdicts:
- AMD_ISA_Q6K_DIRECT_DESIGN_PASS_READY
- AMD_ISA_Q6K_DIRECT_DESIGN_BLOCKED_NO_ROUTE_MAPPING
- AMD_ISA_Q6K_DIRECT_DESIGN_BLOCKED_QUANT_LAYOUT_UNCLEAR
- AMD_ISA_Q6K_DIRECT_DESIGN_BLOCKED_ROLE_ATTRIBUTION
- AMD_ISA_Q6K_DIRECT_DESIGN_BLOCKED_LOW_AMDAHL_AFTER_DESIGN

Stop after Q6K-1 and report the verdict. Do not implement Q6K-2 in this phase.
```
