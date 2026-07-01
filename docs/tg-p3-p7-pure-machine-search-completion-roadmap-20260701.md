# TG-P3 to TG-P7: Pure Machine Search Completion Roadmap

Date: 2026-07-01

Status: execution roadmap for Claude/Codex after TG-P2. This scope completes
the migration from "fast routes selected by model-side code and flags" to
"BoltBeam owns route selection, tinygrad executes generated/search-owned routes,
handwritten routes remain rollback/oracles only."

## Starting Point

TG-P2 is the immediate prerequisite. It should make Q4_K G3 policy-driven:

```text
BoltBeam route policy -> tinygrad QK_ROUTE_POLICY -> Q4_K G3 selected tensors
```

After TG-P2 passes, the expected census is still not pure:

```text
TINYGRAD_DEFAULT_PURITY_FAIL

generated defaults:
  - decode_q4k_g3_generated
  - decode_flash_block_tile_g5_konly

remaining purity debt:
  - decode_q6k_coop_shipped             hand_authored_uop_template
  - decode_attention_owned_two_kernel   external_handwritten_kernel
  - prefill_pipe_role_selective_default external_handwritten_kernel
```

This roadmap scopes the remaining debt.

## Source Citations

Claude must read these before implementation:

| claim | citation |
|---|---|
| Current pure-search audit and phase list | `docs/tinygrad-pure-search-codegen-audit-and-resolution-20260701.md` |
| TG-P2 handoff | `docs/tg-p2-q4k-g3-policy-driven-selection-scope-20260701.md` |
| Current census | `bench/pure-machine-search-default-path-census/summary.md`, `extra/pure_machine_search_default_path_census.py` |
| Route manifest/provenance | `extra/qk_route_manifest.py`, `bench/qk-search-spaces/default_route_manifest.json` |
| Q6_K shipped route | `extra/q6_k_gemv_primitive.py`, `tinygrad/llm/model.py` Q6_K route branches |
| Q6_K direct refutation | `bench/amd-isa-backend-q6k-direct-speed/latest.json`, `bench/amd-isa-backend-q6k-direct-speed/summary.md` |
| Prefill pipe route | `extra/qk_prefill_graph_gemm_route.py`, `extra/qk_prefill_whole_synced.py` |
| Prefill role-selective proof | `bench/qk-prefill-pipe-role-selective/latest.json`, `bench/qk-prefill-pipe-role-selective/summary.md` |
| 8B owned attention route | `extra/qk_owned_flash_decode_graph_node.py`, `extra/qk_owned_flash_decode.hip`, `tinygrad/llm/model.py` attention route branches |
| Attention ceiling and closure evidence | `bench/amd-isa-backend-decode-attention-ceiling/latest.json`, `docs/qwen-14b-32b-attention-combine-inkernel-result.md`, `docs/attention-combine-reachability-audit-20260701.md` if present |
| G5 generated attention work | `bench/gp-track/gp4_latest.json`, `bench/gp-track/gp3_microgate.json`, `extra/qk_flash_decode.py` |
| BoltBeam route policy and candidate data | `/home/ubuntu/BoltBeam/boltbeam/policy/emit.py`, `/home/ubuntu/BoltBeam/boltbeam/data/candidates.json`, `/home/ubuntu/BoltBeam/boltbeam/manifest.py` |

## North-Star Final State

The final route census must say:

```text
TINYGRAD_DEFAULT_PURITY_PASS
0 selected defaults with provenance external_handwritten_kernel
0 selected defaults with provenance hand_authored_uop_template
all selected defaults are machine_authored_generated or tinygrad_scheduler_generated
handwritten routes retained only as rollback_oracle
```

The user-facing behavior should become:

```text
select model
  -> BoltBeam/tinygrad derive route policy from model + quant + target + context + VRAM
  -> tinygrad loads and executes selected generated routes
  -> inference
```

not:

```text
select model
  -> remember model-specific env flags
  -> hope the intended route fired
```

## Global Rules

- No new handwritten hot kernels as final defaults.
- No default flip without rollback.
- No hidden fallback under strict policy.
- No speed-only promotion if provenance remains handwritten.
- No provenance-only promotion if W==D regresses protected contexts.
- Existing handwritten routes may stay as rollback/oracle.
- If an exact candidate loses, ledger it as refuted.
- If the needed knob is not expressible, classify as
  `SEARCH_SPACE_INCOMPLETE` or `CODEGEN_CAPABILITY_BLOCKED`, not "refuted."

## Phase TG-P3: Generate Q6_K Coop From A Route Spec

### Purpose

Replace the shipped Q6_K hand-authored UOp template with a machine-authored
generated route that preserves current behavior.

This phase is primarily a **provenance conversion**:

```text
before:
  decode_q6k_coop_shipped = hand_authored_uop_template

after:
  decode_q6k_coop_generated = machine_authored_generated
  decode_q6k_coop_shipped = rollback_oracle
```

Do not chase the refuted Q6_K direct route in this phase.

### Required Route Spec

Create a route/spec representation capable of losslessly describing the current
coop route:

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
  role
  target
```

The spec must be data, not an alternate hardcoded branch.

### Implementation Tasks

Repository split:

- BoltBeam:
  - add candidate/spec data for Q6_K coop generation;
  - emit selected policy rows for Q6_K roles once generated route passes;
  - ledger the old route as rollback/oracle.
- tinygrad:
  - add the route-spec emitter;
  - re-emit the current `q6k_coop_partial_kernel` / `q6k_gemv_partial_kernel`
    behavior from the spec;
  - bind via `QK_ROUTE_POLICY`;
  - preserve old route behind rollback.

### Required Gates

| gate | requirement |
|---|---|
| lossless emit | generated route numerically matches current Q6_K coop route |
| route identity | generated route is distinct from fallback and route-bound |
| policy emit | BoltBeam selects generated Q6_K rows from profile/role/quant facts |
| strict fallback | `QK_ROUTE_POLICY_STRICT=1` fails if selected Q6_K falls back |
| W==D | no protected-context regression vs current default |
| census | `decode_q6k_coop_*` no longer appears as `hand_authored_uop_template` selected default |

### Artifacts

```text
bench/tg-p3-q6k-generated-coop/
  latest.json
  summary.md
  route_policy.json
  microgate.json
  wd.json
```

### Verdicts

```text
TG_P3_PASS_Q6K_GENERATED_COOP
TG_P3_BLOCKED_Q6K_IR_CANNOT_REEMIT
TG_P3_BLOCKED_POLICY_SCHEMA_INCOMPLETE
TG_P3_BLOCKED_HIDDEN_FALLBACK
TG_P3_REFUTE_Q6K_GENERATED_REGRESSION
```

### Stop Rules

Stop if the generated route cannot losslessly re-emit the current coop route.
Do not replace it with the previously refuted half-warp direct route.

## Phase TG-P4: Generate Prefill GEMM Schedule

### Purpose

Replace the specialized prefill pipe schedule with a generated schedule spec.

Current state:

```text
prefill_pipe_role_selective_default = external_handwritten_kernel / specialized schedule debt
```

Target:

```text
prefill_pipe_role_selective_generated = machine_authored_generated
prefill_pipe_role_selective_default old schedule = rollback_oracle
```

This is also mostly provenance/ownership first. The current role-selective pipe
already has the speed proof.

### Required Schedule Spec

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
  protected_roles
```

`role_policy` must express the current fact that pipe helps several roles but
excludes saturated `ffn_gate_up`.

### Implementation Tasks

Repository split:

- BoltBeam:
  - add schedule candidate/spec data;
  - emit role-selective prefill route policy from model/target/memory facts;
  - account for VRAM fit before selecting prefill routes.
- tinygrad:
  - represent the current prefill pipe as a spec;
  - emit the schedule from the spec rather than a fixed `build_gemm_pipe` body;
  - bind via route policy;
  - preserve `PREFILL_PIPE_ROLE_SELECTIVE=0` and `PREFILL_GEMM_PIPELINE=0` as
    rollback chain.

### Required Gates

| gate | requirement |
|---|---|
| lossless schedule | generated schedule reproduces current role-selective route |
| prefill authority | `extra/qk_prefill_whole_synced.py` passes at supported contexts |
| correctness | logits/token equivalence vs current route |
| memory fit | route not selected if fp16 realization would exceed VRAM policy |
| rollback | old prefill route remains one flag away |
| census | prefill selected default no longer classified as `external_handwritten_kernel` |

### Contexts

Run where supported:

```text
pp512
pp1024
pp2048
pp4096
pp8192
```

### Artifacts

```text
bench/tg-p4-prefill-generated-schedule/
  latest.json
  summary.md
  route_policy.json
  schedule_spec.json
  wd_by_ctx.json
  memory_fit.json
```

### Verdicts

```text
TG_P4_PASS_PREFILL_GENERATED_SCHEDULE
TG_P4_BLOCKED_SCHEDULE_IR_CANNOT_REEMIT
TG_P4_BLOCKED_MEMORY_FIT_POLICY
TG_P4_BLOCKED_HIDDEN_FALLBACK
TG_P4_REFUTE_PREFILL_WD_REGRESSION
```

## Phase TG-P5: Replace 8B Owned Decode Attention With Generated Route

### Purpose

Remove the remaining external handwritten default route:

```text
decode_attention_owned_two_kernel = external_handwritten_kernel
```

The replacement must be generated:

```text
GQAFlashTileSpec
  -> generated UOp/ISA lowering
  -> route-bound W==D evidence
```

It must not be a new fixed HIP/ASM/RDNA3 kernel.

### Important Context

Attention has repeatedly been lower leverage than weight/prefill, and several
combine/fusion paths are already refuted. Do not reopen attention broadly. This
phase is specifically about final-default purity for the 8B owned attention
route.

Existing generated G5 work proves some generated attention routes can ship for
specific shapes. The 8B shape still has owned HIP default debt.

### Candidate Families

Allowed:

- generated GQA flash tile spec;
- generated K-only/LDS staging variants if expressible as data/spec;
- generated route using existing tinygrad/UOp/ISA lowering;
- policy-selected route scoped by model shape/head geometry.

Forbidden:

- new handwritten HIP/ASM fixed kernel;
- hardcoded 8B-only branch marked as "search";
- hidden fallback to owned HIP under strict mode.

### Required Gates

| gate | requirement |
|---|---|
| microgate | generated attention numerically matches reference |
| route-bound | owned HIP does not fire under selected generated route |
| resource gate | VGPR/LDS/scratch sane; no accidental spill |
| W==D | no protected-context regression vs owned default |
| rollback | `DECODE_ATTN_AMDGCN_TILE=1` or equivalent remains rollback/oracle |
| census | selected attention default no longer `external_handwritten_kernel` |

### Artifacts

```text
bench/tg-p5-attention-generated-default/
  latest.json
  summary.md
  route_policy.json
  microgate.json
  resources.json
  wd_by_ctx.json
```

### Verdicts

```text
TG_P5_PASS_ATTENTION_GENERATED_DEFAULT
TG_P5_BLOCKED_RENDERER_OR_IR_CAPABILITY
TG_P5_BLOCKED_POLICY_SCHEMA_INCOMPLETE
TG_P5_BLOCKED_HIDDEN_OWNED_FALLBACK
TG_P5_REFUTE_GENERATED_ATTENTION_SLOWER
```

### Stop Rules

If a generated attention route is correct but slower, keep owned HIP as default
and report `TG_P5_REFUTE_GENERATED_ATTENTION_SLOWER`. Do not force purity by
making the model slower.

If the winning structure requires an unimplemented primitive, report
`TG_P5_BLOCKED_RENDERER_OR_IR_CAPABILITY` and name the primitive.

## Phase TG-P6: Pure-Search Diagnostic Mode

### Purpose

Add a runtime/audit mode that proves whether a run is pure-search selected.

Add:

```text
PURE_MACHINE_SEARCH_ONLY=1
```

### Rules

When enabled:

- selected defaults with provenance `external_handwritten_kernel` are forbidden;
- selected defaults with provenance `hand_authored_uop_template` are forbidden;
- hidden fallback to owned routes is forbidden;
- rollback/oracle routes are allowed only when explicitly requested;
- route census is exported or printed for the run;
- failure messages must name the route and replacement scope.

### Implementation Tasks

- tinygrad:
  - load route manifest/census at startup or route-selection time;
  - enforce provenance for selected hot routes;
  - expose diagnostics in stderr or `/runtime/status` if runtime server is active.
- BoltBeam:
  - optionally emit `purity_required: true` in route policy;
  - ensure policy rows include provenance and rollback.

### Required Gates

| gate | requirement |
|---|---|
| fail-current | mode fails on current default if TG-P3/P4/P5 not done |
| pass-after | mode passes after all debts are converted |
| explicit rollback | rollback routes require explicit env/policy reason |
| route report | output names selected route/provenance/reason |

### Artifacts

```text
bench/tg-p6-pure-search-diagnostic/
  latest.json
  summary.md
  fail_current.json
  pass_candidate.json
```

### Verdicts

```text
TG_P6_PASS_PURE_SEARCH_DIAGNOSTIC_MODE
TG_P6_BLOCKED_MANIFEST_RUNTIME_BINDING
TG_P6_BLOCKED_HIDDEN_HANDWRITTEN_ROUTE
TG_P6_BLOCKED_POLICY_MISSING_GENERATED_ROUTE
```

## Phase TG-P7: Final Default Flip

### Purpose

Make the generated/search-owned routes the normal default path and move all old
handwritten/specialized routes to rollback/oracle status.

### Prerequisites

All must pass:

```text
TG_P2_PASS_Q4K_G3_POLICY_DRIVEN
TG_P3_PASS_Q6K_GENERATED_COOP
TG_P4_PASS_PREFILL_GENERATED_SCHEDULE
TG_P5_PASS_ATTENTION_GENERATED_DEFAULT
TG_P6_PASS_PURE_SEARCH_DIAGNOSTIC_MODE
```

### Tasks

1. Update `extra/qk_route_manifest.py`:
   - generated Q6_K route = `promoted_default`;
   - generated prefill route = `promoted_default`;
   - generated attention route = `promoted_default`;
   - old Q6_K/prefill/owned attention = `rollback_reference` or
     `superseded_rollback`.
2. Update `extra/pure_machine_search_default_path_census.py` and artifacts.
3. Make BoltBeam route policy the normal selector authority.
4. Keep diagnostic env flags only as rollback/override.
5. Run protected W==D gates.
6. Run `PURE_MACHINE_SEARCH_ONLY=1`.

### Final Gates

```bash
cd /home/ubuntu/tinygrad-arkey
PYTHONPATH=. python3 extra/pure_machine_search_default_path_census.py --check --strict-final-default
PYTHONPATH=. python3 -m pytest -q test/unit/test_qk_route_purity.py
```

Plus the route-specific W==D authority gates from TG-P3/TG-P4/TG-P5.

### Artifacts

```text
bench/tg-p7-pure-search-default/
  latest.json
  summary.md
  final_census.json
  wd_protected_contexts.json
  rollback_matrix.json
```

### Verdicts

```text
TG_P7_PASS_PURE_SEARCH_CODEGEN_DEFAULT
TG_P7_BLOCKED_PURITY_DEBT_REMAINING
TG_P7_BLOCKED_PROTECTED_CONTEXT_REGRESSION
TG_P7_BLOCKED_ROLLBACK_MISSING
```

## Final Acceptance

The end state is accepted only when:

```text
TINYGRAD_DEFAULT_PURITY_PASS
```

and:

- no selected default route has provenance `external_handwritten_kernel`;
- no selected default route has provenance `hand_authored_uop_template`;
- generated policy route-bound checks pass;
- token/logit correctness passes for protected models;
- W==D has no protected-context regression;
- old handwritten/specialized routes remain available as rollback/oracles;
- BoltBeam can explain every selected route with evidence refs.

## Recommended Execution Order

Do not parallelize GPU-heavy W==D gates. The safe order is:

1. TG-P3 Q6_K generated coop.
2. TG-P4 prefill generated schedule.
3. TG-P5 generated 8B attention replacement.
4. TG-P6 pure-search diagnostic mode.
5. TG-P7 final default flip.

Rationale:

- Q6_K and prefill are mostly lossless re-emission/provenance conversion.
- Attention is harder and historically lower leverage.
- Diagnostic mode should be added before the final flip, but after enough routes
  exist for it to pass.
- Final flip should be a packaging/gating phase, not a place to invent kernels.
