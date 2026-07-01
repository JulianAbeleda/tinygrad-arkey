# Pure Machine Search Default Migration Analysis

Date: 2026-07-01.

## Verdict

`PURE_SEARCH_DEFAULT_MIGRATION_SCOPED`

The repo now has enough accounting to say exactly what remains. The path to a
pure-machine-search default is **not** "turn off every handwritten route." That
would make the default purer and slower. The correct path is:

```text
profile facts -> BoltBeam candidate/policy -> generated tinygrad route
-> route-bound/token/W==D gate -> default promotion
```

Current strict gate:

```text
PYTHONPATH=. python3 extra/pure_machine_search_default_path_census.py --strict-final-default
=> TINYGRAD_DEFAULT_PURITY_FAIL
```

Default-purity debt is now explicit:

| route | current class | move needed |
|---|---|---|
| `decode_q6k_coop_shipped` | `hand_authored_uop_template` | generate the same Q6_K coop route from a route spec |
| `decode_attention_owned_two_kernel` | `external_handwritten_kernel` | replace with a generated attention route where W==D passes |
| `prefill_pipe_role_selective_default` | `external_handwritten_kernel` | generate the prefill pipe schedule from a schedule spec |

`decode_q4k_g3_generated` is the positive control: it is already
`machine_authored_generated` and allowed as a final default.

## Fresh Evidence Since The Audit

The GP track produced a new generated attention candidate:

| evidence | result |
|---|---|
| tinygrad commit `3058b03c9` | `DECODE_FLASH_BLOCK_TILE_G5_KONLY=0`: K-only LDS staging for G=5 flash |
| `docs/gp5-final-report.md` | `GP4_PASS_TIER_A`; 14B ctx512 `49.9 -> 53.8` tok/s, ctx2048 `46.9 -> 53.8` tok/s |
| BoltBeam commit `783ed13` | candidate `decode_flash_block_tile_g5_konly` marked promoted/TIER_A |
| `docs/g5-generated-isa-primitive-route-scope-20260701.md` | purity boundary: generated UOps are allowed; handwritten G=5 ISA/HIP is forbidden |

This changes the attention plan for 14B:

- there is now a generated default-off route with measured Tier-A movement;
- it is not yet a default-purity fix because tinygrad still does not select it
  from policy by default;
- it has not yet proven 32B transfer or protected-context safety.

## Direct Answer: Switch Something On Or Add Primitives?

### Q4_K Decode GEMV

**Switch needed:** no.

Q4_K G3 is already default-on for eligible shapes through
`DECODE_Q4K_G3_ANYSHAPE=1`. The remaining work is policy cleanup, not
performance: BoltBeam should become the default route-policy authority instead
of leaving the structural decision primarily in `model.py`.

### 14B G=5 Attention

**Switch candidate exists, but do not global-flip yet.**

Candidate flags:

```text
DECODE_FLASH_BLOCK_TILE_G5=1
DECODE_FLASH_BLOCK_TILE_G5_KONLY=1
```

This is generated-UOp code, not a handwritten kernel. It is the closest current
attention path to a pure-search promotion. But it needs a promotion track:

1. rerun route-bound/token/W==D on 14B at ctx128/512/2048/4096;
2. test transfer on 32B (`Hq=64,Hkv=8,G=8`);
3. verify 8B is not accidentally routed into the G=5 path;
4. emit a BoltBeam policy row selecting it only for the proven profile/shape;
5. update `extra/qk_route_manifest.py` provenance once it is selected by policy.

If these pass, this can replace the handwritten attention default **for the
large-model GQA shape only**. It does not automatically solve 8B owned attention.

### 8B Owned Attention

**Primitive/search work still needed.**

The current default `decode_attention_owned_two_kernel` is external HIP/AMDGCN.
Existing generated 8B/native attention routes were correct but not fast enough
(`decode_attention_native_correct_not_fast`). Do not disable the owned route just
to pass purity. The pure path needs a generated attention route that matches the
owned primitive boundary:

- split occupancy;
- cache identity;
- online softmax;
- PV;
- combine/lifecycle economics;
- no protected-context regression.

The G5 K-only result may teach a reusable `stage_k=true, stage_v=false` attention
tile parameter, but 8B still needs its own W==D proof.

### Q6_K Decode GEMV

**No obvious new hardware primitive required.**

The route is already tinygrad UOps and performant enough to ship. The problem is
provenance: `extra/q6_k_gemv_primitive.py` is a hand-authored route template. The
right move is a lossless route-spec generator:

```text
Q6KRouteSpec(quant facts, role, shape, target) -> generated UOp route
```

Acceptance is not a speed win. Acceptance is:

- generated route re-emits the current coop route exactly or numerically
  equivalently;
- route-bound in-model;
- no W==D regression;
- old hand-authored Q6_K route demoted to rollback/oracle.

### Prefill Pipe

**Schedule primitive/spec work required.**

The role-selective prefill pipe is a real performance win but is still a
specialized assembly/instruction-list emitter. The replacement is not a new
kernel handwritten by a human. It is a schedule IR:

```text
PrefillGemmScheduleSpec(tile_m, tile_n, pipe_depth, role_filter, target)
-> generated schedule/instruction lowering
```

First gate should be lossless re-emit of the current role-selective pipe. Only
after that should search vary schedule knobs.

## Migration Scope

### PM0: State Pin

Run and archive:

```text
PYTHONPATH=. python3 extra/qk_route_manifest.py
PYTHONPATH=. python3 extra/pure_machine_search_default_path_census.py --check
PYTHONPATH=. python3 extra/pure_machine_search_default_path_census.py --strict-final-default
```

Expected today:

- manifest/check pass;
- strict-final-default fails with the three debt routes above.

Verdicts:

- `PM0_PASS_STATE_PINNED`
- `PM0_BLOCKED_MANIFEST_DRIFT`

### PM1: Promote Generated G5 K-Only By Policy, Not By Global Flag

Goal: convert the GP4 result into a policy-selected generated route for the
profiles where it is proven.

Tasks:

1. Re-run 14B authority W==D at ctx128/512/2048/4096 with:
   `DECODE_FLASH_BLOCK_TILE_G5=1 DECODE_FLASH_BLOCK_TILE_G5_KONLY=1`.
2. Run the same route on 32B to test the `G=8` transfer claim.
3. Prove route-bound evidence: no owned attention tile, no hidden fallback.
4. Add/update BoltBeam policy output for only the passing model/shape.
5. Add a tinygrad strict policy mode check so policy-selected G5 cannot silently
   fall back.

Verdicts:

- `PM1_PASS_G5_KONLY_POLICY_PROMOTABLE`
- `PM1_REFUTE_32B_TRANSFER`
- `PM1_BLOCKED_ROUTE_ATTRIBUTION`
- `PM1_BLOCKED_PROTECTED_CONTEXT_REGRESSION`

### PM2: Make BoltBeam Policy The Default Authority

Goal: remove hardcoded route selection as the primary source of truth.

Tasks:

1. Define/load `qk_route_policy.v1` from BoltBeam:
   model id, target id, tensor/role/quant/shape, candidate id, route params,
   rollback, evidence refs.
2. Tinygrad consumes this policy before environment heuristic branches.
3. Keep env flags as rollback/diagnostic only.
4. Add:

```text
QK_GENERATED_POLICY_STRICT=1
```

which errors if a selected route does not fire.

Verdicts:

- `PM2_PASS_POLICY_AUTHORITY`
- `PM2_BLOCKED_POLICY_SCHEMA_INCOMPLETE`
- `PM2_BLOCKED_HIDDEN_FALLBACK`

### PM3: Generate Q6_K Coop From A Route Spec

Goal: remove `decode_q6k_coop_shipped` from final-default purity debt without
regressing speed.

Tasks:

1. Create `Q6KRouteSpec` from quant facts and target capabilities.
2. Losslessly re-emit the current coop/partial route.
3. Gate lm_head, ffn_down, long-K ffn_down, and attn_v.
4. Demote the hand-authored route to rollback/oracle.

Verdicts:

- `PM3_PASS_Q6K_SPEC_REEMITS_SHIPPED_ROUTE`
- `PM3_BLOCKED_Q6K_SPEC_GAP`
- `PM3_REFUTE_WD_REGRESSION`

### PM4: Generate Prefill Pipe Schedule From A Schedule Spec

Goal: replace the specialized prefill assembly emitter as the final default.

Tasks:

1. Define `PrefillGemmScheduleSpec`.
2. Re-emit the role-selective pipe from the spec.
3. Preserve the current role exclusion: gate/up stays on the faster non-pipe
   path unless search proves otherwise.
4. Run prefill authority at ctx512/1024/2048/4096/8192.

Verdicts:

- `PM4_PASS_PREFILL_SCHEDULE_REEMIT`
- `PM4_PASS_PREFILL_GENERATED_DEFAULT`
- `PM4_BLOCKED_SCHEDULE_IR_GAP`
- `PM4_REFUTE_WD_REGRESSION`

### PM5: Replace 8B Owned Attention Or Keep It As Explicit Debt

Goal: eliminate `decode_attention_owned_two_kernel` as a final default.

Tasks:

1. Reuse the generated attention Tile IR from PM1 if it transfers to G=4.
2. If it does not transfer, record the missing primitive precisely:
   occupancy, split-combine coordination, v_dot2, cache identity, or scheduler.
3. Do not promote a slower generated route only to satisfy purity.

Verdicts:

- `PM5_PASS_8B_GENERATED_ATTENTION_DEFAULT`
- `PM5_CORRECT_BUT_NOT_FAST_KEEP_OWNED_ORACLE`
- `PM5_BLOCKED_PRIMITIVE_GAP`

### PM6: Enforce Final Gate

Only after PM1-PM5 resolve the relevant rows:

```text
PYTHONPATH=. python3 extra/pure_machine_search_default_path_census.py --strict-final-default
```

must pass.

Verdicts:

- `PM6_PASS_PURE_SEARCH_DEFAULT`
- `PM6_FAIL_REMAINING_DEFAULT_DEBT`

## Priority Order

1. **PM1 G5 K-only policy promotion**: already has Tier-A evidence; cheapest
   route to move a real default debt row for 14B/32B.
2. **PM2 policy authority**: prevents future route drift and hidden fallback.
3. **PM3 Q6_K spec re-emit**: likely provenance work, not a performance fight.
4. **PM4 prefill schedule spec**: high value, but more compiler/scheduler work.
5. **PM5 8B owned attention replacement**: hardest; do not sacrifice W==D for
   purity.

## Non-Goals

- Do not write a new HIP/ASM/ISA kernel and call it pure search.
- Do not disable fast shipped routes just to make the strict gate green.
- Do not promote a generated route without route-bound evidence and rollback.
- Do not claim G5 K-only solves all attention shapes until 32B and 8B are
  measured separately.

## Clean End State

The target statement is:

```text
All default hot routes are either ordinary tinygrad-generated or
machine-authored/generated from BoltBeam policy; handwritten kernels remain only
as rollback oracles.
```

The repo now has the gate to enforce that. The remaining work is making the
three debt routes satisfy it without giving back the performance wins.
