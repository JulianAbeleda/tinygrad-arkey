# Tinygrad Pure Search / Codegen Audit And Resolution Scope

Date: 2026-07-01.

## Verdict

`TINYGRAD_PURE_SEARCH_CODEGEN_AUDIT_FAIL`

The current tinygrad route surface is **partly pure-search/codegen, but not
fully pure by default**.

The important distinction:

- ordinary tinygrad graph lowering is already generated enough;
- Q4_K G3 decode GEMV is the successful machine-search direction;
- several hot default routes still come from handwritten or specialized route
  bodies;
- several route decisions still live as hardcoded policy in `model.py` instead
  of as BoltBeam-generated route policy.

Do **not** make the audit pass by turning off fast routes. The resolution is to
make generated/search-authored replacements pass correctness and W==D gates, then
move handwritten routes to rollback/oracle status.

## Definitions

| class | final default? | meaning |
|---|---:|---|
| `tinygrad_scheduler_generated` | yes | normal tinygrad graph lowering, no custom hot route |
| `machine_authored_generated` | yes | route emitted from profile/quant/target facts, grammar/search candidate, and generated lowering |
| `hand_authored_uop_template` | transitional only | Python UOp `custom_kernel` body written by humans |
| `external_handwritten_kernel` | no | HIP/C++/ASM/precompiled binary or explicit instruction emitter used as a route kernel |
| `rollback_oracle` | yes, behind rollback only | handwritten/specialized route retained for comparison, fallback, or diagnosis |

Human-written code may implement reusable IRs, renderers, schedulers, grammars,
and emitters. Human-written code may not encode the final hot route as a fixed
kernel body and then call it search.

## Evidence Read

| evidence | result |
|---|---|
| `extra/pure_machine_search_default_path_census.py` | ran successfully; reports 4 non-tinygrad-generated default hot routes, only one search/codegen-generated |
| `extra/qk_route_manifest.py` | contains the default route ledger, but needs refresh for the latest 14B/32B route work |
| `tinygrad/llm/model.py:252-299` | Q4_K G3 generated route is default-on and structurally generalized by `DECODE_Q4K_G3_ANYSHAPE=1` |
| `tinygrad/llm/model.py:500-514` | Q6_K coop route is default-on and calls `q6k_coop_partial_kernel` |
| `tinygrad/llm/model.py:1086-1199` | decode attention chooses generated flash routes, default-off G=5 block tile experiments, and owned HIP tile for the validated 8B long-context shape |
| `tinygrad/llm/model.py:1483-1501` | Q4_K/Q6_K primitive policy auto-enables on AMD GGUF paths, not from a BoltBeam policy artifact |
| `extra/q6_k_gemv_primitive.py:170-191` | Q6_K coop implementation is a hand-authored UOp route template |
| `extra/qk_owned_flash_decode_graph_node.py:1-8` | owned attention route injects precompiled HIP/AMDGCN binary and explicitly skips codegen |
| `extra/qk_prefill_graph_gemm_route.py:58-70` and `:117-125` | prefill pipe defaults on and emits an instruction list through `build_gemm_pipe(...)` |

## Audit Findings

### F0: Route Census Is Useful But Not Yet Authoritative

The existing census says:

```text
PMS_R0_PASS_CENSUS_PINNED
4 kernels on the default path are non-tinygrad-generated.
1 is search/codegen-generated; 3 are hand-owned.
```

That is still directionally correct, but the census/manifest is not enough as a
future gate because the live model has moved:

- `DECODE_Q4K_G3_ANYSHAPE=1` is now default-on in `model.py`;
- `DECODE_ROUTE_ATTN_K` is currently read with default `1`;
- `DECODE_Q6K_FFN_DOWN_LONGK=1` is default-on for large Q6_K ffn_down;
- `qk_route_manifest.py` still describes parts of the older 8B-centered state.

Resolution: refresh the route manifest and census schema before relying on them
as a CI gate.

### F1: Q4_K G3 Is The Positive Control

`tinygrad/llm/model.py:252-299` routes eligible Q4_K decode GEMVs through
`extra/qk_gemv_g3_codegen_lowering.py:q4k_g3_lanemap_gemv_kernel`.

This is the model to preserve:

```text
profile/shape facts -> LaneMap/TopologySpec -> generated UOp route -> W==D proof
```

Remaining debt is not the kernel body. The debt is policy coupling:

- structural eligibility lives directly in `model.py`;
- `QK_GENERATED_POLICY` exists but is not the primary default policy authority;
- 14B/32B route policy is still encoded by env guards and model-side branches.

Resolution: keep G3 default, but make BoltBeam-generated policy the authority
for which Q4_K tensors get the G3 route.

### F2: Q6_K Coop Is Correct And Fast Enough To Ship, But Not Final-Pure

`tinygrad/llm/model.py:500-514` routes Q6_K lm_head / ffn_down / long-K ffn_down
through `extra/q6_k_gemv_primitive.py:q6k_coop_partial_kernel`.

The route is not external HIP, but the UOp body is hand-authored. It should be
treated as:

```text
hand_authored_uop_template
```

not:

```text
machine_authored_generated
```

Resolution: build a Q6_K route grammar/spec that can losslessly re-emit the
current coop route from quant/shape/target facts. Only after that proof should
the shipped Q6_K route be reclassified as generated.

### F3: Decode Attention Still Has External Handwritten Default Debt

For the validated Qwen3-8B/gfx1100 long-context shape, `model.py:1156-1180`
calls `amdgcn_flash_decode(...)`, which is backed by
`extra/qk_owned_flash_decode.hip` through
`extra/qk_owned_flash_decode_graph_node.py`.

That graph node explicitly injects a precompiled binary:

```text
Ops.PROGRAM(... SOURCE, BINARY ...)
```

and its own docstring says it skips codegen. This is not pure machine search.

Resolution: replace it with a generated attention route from a GQA/flash Tile IR
and generic UOp/ISA lowering. Existing generated candidates and G=5 experiments
are useful, but the owned HIP route must remain default until a generated
replacement passes W==D.

### F4: Prefill Pipe Is A Specialized Assembly Emitter

`extra/qk_prefill_graph_gemm_route.py:58-70` defaults the role-selective pipe on
and calls `ref.build_gemm_pipe(...)`. The route wraps the returned instruction
list in `Ops.LINEAR` at `:117-125`.

This is a performance win, but it is not yet pure-search/codegen in the strict
sense. The schedule choices are encoded in Python and instruction lists instead
of being emitted from a generated schedule spec.

Resolution: turn the current role-selective pipe into a schedule IR that can
losslessly re-emit the current route, then make candidate search author the
schedule.

### F5: Policy Is Still Too Hardcoded In Tinygrad

The most important non-kernel debt is policy location.

Examples:

- `_q4k_policy(...)` and `_q6k_policy(...)` select tensor coverage in
  `model.py`;
- route-specific env flags decide behavior at call sites;
- `QK_GENERATED_POLICY` is present, but it is an optional override rather than
  the default authority for generated search decisions.

Resolution: move route selection toward:

```text
GGUF/ProfileIR + TargetProfile + BoltBeam candidate ledger
-> generated runtime route policy
-> tinygrad installs/runs the requested route
```

Tinygrad should still own execution, validation, fallback, and measurement. It
should not own the search decision as scattered hardcoded branches.

## Required Tinygrad Work

### TG-P0: Refresh Route Authority

Goal: make tinygrad's route manifest and census match the live code.

Tasks:

1. Update `extra/qk_route_manifest.py` for current default-on routes:
   - `DECODE_Q4K_G3_ANYSHAPE=1`;
   - `DECODE_ROUTE_ATTN_K=1`;
   - `DECODE_Q6K_FFN_DOWN_LONGK=1`;
   - current prefill role-selective pipe;
   - current attention default/rollback.
2. Extend `extra/pure_machine_search_default_path_census.py` with provenance:
   - `machine_authored_generated`;
   - `tinygrad_scheduler_generated`;
   - `hand_authored_uop_template`;
   - `external_handwritten_kernel`;
   - `rollback_oracle`.
3. Make the census multi-profile:
   - 8B;
   - 14B;
   - 32B;
   - prefill authority profile.
4. Fail loudly when manifest and live guards drift.

Verdicts:

- `TG_P0_PASS_ROUTE_AUTHORITY_REFRESHED`
- `TG_P0_BLOCKED_MANIFEST_CODE_DRIFT`
- `TG_P0_BLOCKED_CENSUS_PROFILE_GAP`

### TG-P1: Make BoltBeam Policy The Default Search Authority

Goal: stop encoding search policy primarily in `model.py`.

Tasks:

1. Promote `QK_GENERATED_POLICY` from optional override to the preferred policy
   surface when a BoltBeam policy artifact is available.
2. Define a stable runtime policy schema consumed by tinygrad:

```text
qk_route_policy.v1:
  model_id
  target_id
  tensors:
    tensor_name
    role
    quant
    shape
    route_family
    route_params
    rollback_route
    evidence_refs
```

3. Keep old env flags as rollback/diagnostic, not as the main source of truth.
4. Add strict mode:

```text
QK_GENERATED_POLICY_STRICT=1
```

which errors if a policy-selected tensor silently falls back.

Verdicts:

- `TG_P1_PASS_GENERATED_POLICY_AUTHORITY`
- `TG_P1_BLOCKED_POLICY_SCHEMA_INCOMPLETE`
- `TG_P1_BLOCKED_HIDDEN_FALLBACK`

### TG-P2: Lock Q4_K G3 As The Generated Positive Control

Goal: keep the current success and remove policy hardcoding around it.

Tasks:

1. Route Q4_K G3 from the generated policy artifact for all eligible shapes.
2. Keep `BUBBLEBEAM_FUTURESIGHT=0` and owned Q4_K routes as rollback only.
3. Keep `DECODE_Q4K_G3_ANYSHAPE=1` as compatibility/diagnostic until the policy
   route is proven.
4. Add a route-bound gate proving Q4_K does not fall back to the owned warp on
   generated-policy runs.

Verdicts:

- `TG_P2_PASS_Q4K_G3_POLICY_DRIVEN`
- `TG_P2_BLOCKED_G3_POLICY_DRIFT`

### TG-P3: Generate Q6_K Coop From A Route Spec

Goal: replace the hand-authored Q6_K UOp template with a machine-authored
version while preserving shipped behavior.

Required IR:

```text
Q6KGEMVRouteSpec:
  quant = Q6_K
  rows
  k
  row_tile
  lane_extent
  pos_axis = local
  block_axis = reduce
  reduction = external_sum | in_kernel_wave
  storage = packed_u16
```

Tasks:

1. Build a spec that losslessly re-emits current `q6k_coop_partial_kernel`.
2. Prove the emitted kernel name/key differs from fallback but matches numeric
   output.
3. Route lm_head, 8B ffn_down, and 14B/32B long-K ffn_down through the generated
   spec.
4. Preserve the current hand-authored route as rollback.

Verdicts:

- `TG_P3_PASS_Q6K_GENERATED_COOP`
- `TG_P3_BLOCKED_Q6K_IR_CANNOT_REEMIT`
- `TG_P3_REFUTE_Q6K_GENERATED_REGRESSION`

### TG-P4: Generate Prefill GEMM Schedule

Goal: replace the specialized prefill assembly pipe with a generated schedule.

Required IR:

```text
PrefillGEMMScheduleSpec:
  tile_m
  tile_n
  tile_k
  waves_m
  waves_n
  wm
  wn
  pipeline_depth
  role_policy
  waitcnt_policy
  target_capabilities
```

Tasks:

1. Losslessly represent the current role-selective pipe.
2. Generate the instruction schedule from the spec rather than a fixed
   `build_gemm_pipe` body.
3. Run prefill authority gates at `pp512/1024/2048/4096/8192` where supported.
4. Preserve `PREFILL_PIPE_ROLE_SELECTIVE=0` and `PREFILL_GEMM_PIPELINE=0` as
   rollback chain.

Verdicts:

- `TG_P4_PASS_PREFILL_GENERATED_SCHEDULE`
- `TG_P4_BLOCKED_SCHEDULE_IR_CANNOT_REEMIT`
- `TG_P4_REFUTE_PREFILL_WD_REGRESSION`

### TG-P5: Replace Owned Decode Attention With Generated Route

Goal: move `DECODE_ATTN_AMDGCN_TILE` from default route to rollback/oracle.

The route must follow the generated-primitive boundary:

```text
GQAFlashTileSpec -> generated UOp/ISA lowering -> route-bound W==D evidence
```

It must not be:

```text
new hand-written HIP/ASM/RDNA3 fixed kernel
```

Tasks:

1. Continue the generated G=5/GQA primitive track.
2. Add a generated candidate axis for K-only staging if GP1's diagnosis holds.
3. Prove correctness and resources in microgate.
4. Only bind in-model after the generated primitive is materially faster than
   the current generated G=5 block tile.
5. Promote only if W==D has no protected-context regression.

Verdicts:

- `TG_P5_PASS_ATTENTION_GENERATED_DEFAULT`
- `TG_P5_BLOCKED_RENDERER_OR_IR_CAPABILITY`
- `TG_P5_REFUTE_GENERATED_ATTENTION_SLOWER`

### TG-P6: Add Pure-Search Diagnostic Mode

Goal: make purity debt visible without changing the shipped default.

Add:

```text
PURE_MACHINE_SEARCH_ONLY=1
```

Rules:

- forbids external handwritten kernels as selected defaults;
- forbids hidden fallback to owned routes;
- permits rollback only when explicitly requested;
- prints or exports a route census for the run.

Verdicts:

- `TG_P6_PASS_PURE_SEARCH_DIAGNOSTIC_MODE`
- `TG_P6_BLOCKED_HIDDEN_HANDWRITTEN_ROUTE`
- `TG_P6_BLOCKED_POLICY_MISSING_GENERATED_ROUTE`

### TG-P7: Final Default Flip

Only after TG-P3, TG-P4, and TG-P5 pass:

1. make generated Q6_K default;
2. make generated prefill schedule default;
3. make generated attention default for applicable profiles;
4. move old routes to rollback/oracle;
5. rerun route census and W==D authority gates.

Verdicts:

- `TG_P7_PASS_PURE_SEARCH_CODEGEN_DEFAULT`
- `TG_P7_BLOCKED_PURITY_DEBT_REMAINING`
- `TG_P7_BLOCKED_PROTECTED_CONTEXT_REGRESSION`

## Acceptance Gates

The final pass requires:

- route manifest and census agree with live guards;
- no selected default route has provenance `external_handwritten_kernel`;
- no selected default route has provenance `hand_authored_uop_template` unless
  the phase explicitly marks it transitional;
- generated policy route-bound checks pass;
- token/logit correctness passes for protected models;
- W==D has no protected-context regression;
- handwritten routes remain available as rollback/oracles.

Expected current result:

```text
TINYGRAD_PURE_SEARCH_CODEGEN_AUDIT_FAIL
```

That is correct. The point is to make the failure precise and close it route by
route, not to hide it behind flags.

## Immediate Next Step

Start with **TG-P0 + TG-P1**.

Reason:

- the manifest/census is the tinygrad-side authority, and it currently lags the
  live route surface;
- BoltBeam already has the model/quant/target-agnostic brain;
- tinygrad needs to consume generated policy as the normal route authority before
  Q6_K, prefill, and attention replacements can be promoted cleanly.

