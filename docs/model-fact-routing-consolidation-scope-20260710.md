# Model Fact Routing Consolidation Scope

## Goal

Do not create one route script or runtime path per model size. Loading a model should derive facts once, and QK routes
should bind from those facts:

```text
GGUF metadata -> ModelFacts -> ModelRoutePlan -> primitive install -> decode/prefill candidates
```

Routes should depend on quant, role, shape, phase, arch capability, memory, and explicit policy. They should not depend
on labels like 8B, 14B, or 32B.

## Current State

- `model_profiles.py` contains reusable search/gate shape data for known Qwen profiles.
- `model_facts.py` normalizes GGUF metadata into tensor/module facts.
- `model_route_plan.py` maps facts into primitive install entries.
- `Transformer.from_gguf` now builds facts/plan once and passes the plan to Q4/Q6 primitive install.
- Q4/Q6 install no longer calls legacy policies inside the install loop when a plan is supplied.

## Remaining Risks

- The route-plan path needs stronger old-vs-new parity proof.
- Primitive linears still mostly expose only `name`, so decode/prefill role helpers parse names as fallback authority.
- Some live QK scripts still carry model-size labels or hardcoded defaults that should be profile/role parameters.
- `RouteCandidateRegistry` should not grow until route parity and fact ownership prove it is earning its weight.

## This Slice

1. Add route-plan parity tests from `ModelFacts`, not just raw metadata.
2. Carry route role from `ModelRoutePlan` onto installed Q4/Q6 primitive linears.
3. Make decode and direct-packed prefill role helpers prefer carried role before parsing names.
4. Audit live `extra/qk` scripts with 8B/14B/32B references and convert safe cases to `--profile` / `--role`.
5. Keep legacy policy functions as compatibility adapters for now.

## Non-Goals

- Do not promote 14B WMMA; the scheduler-owned tile loop remains blocked.
- Do not delete legacy policy functions in this slice.
- Do not split CLI model aliases; aliases are user-facing convenience, not route authority.
- Do not add new dispatch framework layers.

## Required Verification

- Focused model facts / route-plan / decode / prefill tests.
- Full route/model matrix.
- Full `python3 -m pytest`.
- `git diff --check`.
- doc link check.
- size gate.
