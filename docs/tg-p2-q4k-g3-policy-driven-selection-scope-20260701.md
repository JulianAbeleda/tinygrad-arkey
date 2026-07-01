# TG-P2: Q4_K G3 Policy-Driven Selection Scope

Date: 2026-07-01

Status: execution scope for Claude/Codex. This is the next step toward making
tinygrad's default path pure machine search by default. Do not write a new
kernel in this phase. The Q4_K G3 kernel already won; this phase moves its
selection authority out of model-side hardcoding and into BoltBeam route policy.

## Goal

Make Q4_K G3 tensor coverage policy-driven:

```text
GGUF profile + target profile + BoltBeam candidate ledger
  -> boltbeam.route_policy.v1 selected Q4_K G3 rows
  -> tinygrad consumes those rows
  -> Q4_K G3 fires for selected tensors
  -> owned warp remains rollback/oracle
```

This phase should end with:

```text
TG_P2_PASS_Q4K_G3_POLICY_DRIVEN
```

or a precise blocker:

```text
TG_P2_BLOCKED_G3_POLICY_DRIFT
TG_P2_BLOCKED_POLICY_SCHEMA_INCOMPLETE
TG_P2_BLOCKED_HIDDEN_FALLBACK
TG_P2_REFUTE_POLICY_SELECTED_REGRESSION
```

## Current Actuals

The current route census says:

```text
PMS_R0_PASS_CENSUS_PINNED
5 default hot routes
2 generated defaults:
  - decode_q4k_g3_generated
  - decode_flash_block_tile_g5_konly
3 final-default purity debts:
  - decode_q6k_coop_shipped
  - decode_attention_owned_two_kernel
  - prefill_pipe_role_selective_default
```

Q4_K G3 is already the positive control:

- generated UOp route;
- speed-equivalent to owned for tracked Q4_K decode roles;
- owned warp retained as rollback/oracle;
- current debt is selector policy, not kernel quality.

G5 K-only already proves the policy bridge shape:

```text
BoltBeam candidate manifest
  -> selected_route=decode_flash_block_tile_g5_konly
  -> QK_ROUTE_POLICY
  -> tinygrad route selection
```

TG-P2 applies that same pattern to Q4_K G3.

## Source Citations

Claude must read these before editing:

| claim | citation |
|---|---|
| Current pure-search audit and remaining phases | `docs/tinygrad-pure-search-codegen-audit-and-resolution-20260701.md` |
| Current default-route census | `bench/pure-machine-search-default-path-census/summary.md`, `extra/pure_machine_search_default_path_census.py` |
| Current route manifest and provenance rules | `extra/qk_route_manifest.py`, `bench/qk-search-spaces/default_route_manifest.json` |
| Q4_K G3 default route and rollback | `tinygrad/llm/model.py`, `Q4KLinear.__call__` G3 branch and owned-warp fallback branches |
| Generated G3 emitter | `extra/qk_gemv_g3_codegen_lowering.py` |
| G2/G3 lane-map provenance | `extra/qk_gemv_g2_lanemap.py`, `extra/qk_lanemap_template.py` |
| G3 speed-equivalence authority | `bench/amd-isa-backend-g3-weight-promotion/latest.json`, `bench/amd-isa-backend-g3-weight-promotion/summary.md` |
| Current route-policy consumer pattern | `tinygrad/llm/model.py`, `_load_qk_route_policy`, `_qk_route_policy_selected`, G5 branch |
| BoltBeam selected-route emitter | `/home/ubuntu/BoltBeam/boltbeam/policy/emit.py` |
| BoltBeam candidate data | `/home/ubuntu/BoltBeam/boltbeam/data/candidates.json` |
| BoltBeam candidate loader | `/home/ubuntu/BoltBeam/boltbeam/manifest.py` |
| BoltBeam GGUF profile metadata | `/home/ubuntu/BoltBeam/boltbeam/profile/gguf.py` |
| Existing policy tests | `/home/ubuntu/BoltBeam/tests/test_policy_guard.py`, `test/unit/test_qk_route_purity.py` |

## Non-Goals

- Do not change the Q4_K G3 kernel body.
- Do not hand-write a new Q4_K kernel.
- Do not remove owned warp rollback.
- Do not promote or modify Q6_K, prefill, or 8B owned attention in this phase.
- Do not claim final pure-search default; Q6_K, prefill, and 8B attention still
  remain purity debt after TG-P2.
- Do not make policy selection depend on model names like `Qwen3-8B`.

## Design Requirements

### 1. Extend BoltBeam Policy Emission

BoltBeam should emit selected policy rows for the promoted Q4_K G3 route when
the profile roles match validated G3 eligibility.

The selected row shape should be data-driven from role/tensor facts:

```json
{
  "selected_route": "decode_q4k_g3_generated",
  "status": "promoted",
  "provenance": "machine_authored_generated",
  "role": "ffn_gate_up",
  "quant": "Q4_K",
  "shape": {"rows": 12288, "cols": 4096},
  "route_family": "q4k_g3_lanemap",
  "route_params": {
    "BUBBLEBEAM_FUTURESIGHT": "1"
  },
  "rollback": {
    "BUBBLEBEAM_FUTURESIGHT": "0"
  },
  "evidence_refs": [
    "bench/amd-isa-backend-g3-weight-promotion/latest.json"
  ]
}
```

Do not hardcode the Qwen model name. Eligibility should come from:

- workload = `decode`;
- architecture compatible with dense decoder;
- quant = `Q4_K`;
- role in the promoted Q4_K role set;
- shape matches validated G3 shape rules or the existing structural anyshape
  rule:

```text
tracked 8B shapes:
  ffn_gate_up: K=4096, N=12288
  ffn_down:    K=12288, N=4096
  attn_qo:     K=4096, N=4096

structural anyshape:
  (K // 256) % 4 == 0
  N % 32 == 0
```

If BoltBeam cannot express that eligibility from current profile/search data,
return:

```text
TG_P2_BLOCKED_POLICY_SCHEMA_INCOMPLETE
```

and add the missing field explicitly.

### 2. Extend tinygrad Route Policy Consumer

tinygrad currently supports only:

```text
decode_flash_block_tile_g5_konly
```

in `QK_ROUTE_POLICY`. Add:

```text
decode_q4k_g3_generated
```

Consumer rules:

- If `QK_ROUTE_POLICY` is present, policy-selected Q4_K tensors must use G3.
- If `QK_ROUTE_POLICY` is absent, current default behavior remains unchanged.
- If `QK_ROUTE_POLICY_STRICT=1` and a selected Q4_K tensor cannot bind to G3,
  fail loudly before/at route time.
- Existing flags remain rollback/diagnostic:

```text
BUBBLEBEAM_FUTURESIGHT=0
Q4K_GEMV_WARP=1
Q4K_GEMV_WARP_PROJ=1
```

Do not silently fall back to owned warp for a selected tensor under strict mode.

### 3. Preserve Proven Behavior

This phase is a selector-authority migration. It should preserve output and
speed.

Required gates:

| gate | requirement |
|---|---|
| BoltBeam policy tests | policy emits Q4_K G3 selected rows for eligible synthetic GGUF profile |
| Negative policy test | ineligible shape or quant does not select G3 |
| tinygrad unit test | `_load_qk_route_policy` accepts `decode_q4k_g3_generated` rows and shape/role scopes correctly |
| route-bound smoke | selected policy row causes G3 route to fire, not owned warp/fallback |
| rollback smoke | `BUBBLEBEAM_FUTURESIGHT=0` still forces owned warp/reference path |
| census | generated default still counted as `machine_authored_generated`; no new purity debt |
| W==D | no protected-context regression vs current default; speed-equivalent is sufficient |

If GPU time is limited, at minimum run the unit/census gates and use the
existing G3 speed authority as cited evidence. Do not claim fresh W==D movement
without running the authority harness.

## Recommended Phase Plan

### TG-P2A: BoltBeam Candidate/Policy Rows

Repository: `/home/ubuntu/BoltBeam`

Tasks:

1. Add route-policy data for `decode_q4k_g3_generated` to
   `boltbeam/data/candidates.json` if missing:
   - `status: promoted`
   - `default_on: true`
   - `provenance: machine_authored_generated`
   - `route_family: q4k_g3_lanemap`
   - evidence refs to G3 promotion artifacts
   - rollback to `BUBBLEBEAM_FUTURESIGHT=0`
2. Extend `boltbeam/policy/emit.py` so selected default routes can be emitted
   per tensor/role, not only once per attention shape.
3. Add tests in `tests/test_policy_guard.py` or a new route-policy test:
   - eligible Q4_K role emits selected G3;
   - Q6_K role does not emit selected G3;
   - unsupported target/profile fails closed.

Verification:

```bash
cd /home/ubuntu/BoltBeam
python3 -m pytest -q
```

Commit and push BoltBeam.

### TG-P2B: tinygrad Policy Consumer

Repository: `/home/ubuntu/tinygrad-arkey`

Tasks:

1. Extend `_SUPPORTED_QK_ROUTE_IDS` with `decode_q4k_g3_generated`.
2. Extend `_load_qk_route_policy` validation:
   - allowed params for Q4_K G3;
   - reject unsupported route params;
   - reject malformed shape/role rows.
3. Thread selected policy rows into Q4_K primitive installation / routing.
   Prefer the least invasive path:
   - map selected policy rows into the existing `QK_GENERATED_POLICY`-style
     lookup, or
   - add a parallel route-policy lookup that `Q4KLinear.__call__` checks before
     model-side `_q4k_policy`/shape defaults.
4. In strict mode, selected tensors must not silently fall back.
5. Keep no-policy behavior byte-identical to the current default.

Verification:

```bash
cd /home/ubuntu/tinygrad-arkey
PYTHONPATH=. python3 -m pytest -q test/unit/test_qk_route_purity.py
PYTHONPATH=. python3 extra/pure_machine_search_default_path_census.py --check
python3 -m py_compile tinygrad/llm/model.py extra/qk_route_manifest.py extra/pure_machine_search_default_path_census.py
```

If feasible, run one route-bound model gate on a small/fast profile with
`QK_ROUTE_POLICY=<artifact>` and confirm G3 fires.

Commit and push tinygrad.

### TG-P2C: Authority Artifact

Produce an artifact directory:

```text
bench/tg-p2-q4k-g3-policy-driven/
  latest.json
  summary.md
  route_policy.json
  route_bound.json
```

`latest.json` must record:

```json
{
  "verdict": "TG_P2_PASS_Q4K_G3_POLICY_DRIVEN",
  "policy_schema": "boltbeam.route_policy.v1",
  "selected_route": "decode_q4k_g3_generated",
  "token_match": true,
  "route_bound": true,
  "hidden_fallback": false,
  "rollback_available": true,
  "wd_basis": "existing_g3_authority_or_fresh_measurement",
  "remaining_purity_debt": [
    "decode_q6k_coop_shipped",
    "decode_attention_owned_two_kernel",
    "prefill_pipe_role_selective_default"
  ]
}
```

If any requirement fails, use the blocker/refute verdict instead and do not
paper over it.

## Acceptance Criteria

TG-P2 passes only if:

1. BoltBeam can emit selected Q4_K G3 route-policy rows from profile/role/shape
   facts.
2. tinygrad can consume those rows.
3. Policy-selected Q4_K tensors route to G3.
4. Hidden fallback is detected and fails in strict mode.
5. No-policy default behavior remains unchanged.
6. Owned warp remains one rollback away.
7. Route census still passes and does not introduce new purity debt.

## Stop Rules

Stop and report `TG_P2_BLOCKED_POLICY_SCHEMA_INCOMPLETE` if BoltBeam profile data
cannot identify Q4_K eligible tensors without model-name hardcoding.

Stop and report `TG_P2_BLOCKED_HIDDEN_FALLBACK` if tinygrad cannot make a
policy-selected Q4_K tensor fail-loud on fallback.

Stop and report `TG_P2_REFUTE_POLICY_SELECTED_REGRESSION` if policy-selected G3
changes tokens/logits or regresses protected contexts beyond the tiered policy.

Do not proceed to Q6_K or prefill generation inside this phase.

## What This Changes If It Passes

TG-P2 does not increase tok/s by itself. It changes ownership:

```text
before:
  generated Q4_K G3 wins, but model.py still owns most selection logic

after:
  BoltBeam owns Q4_K G3 selection, tinygrad executes it, owned warp is rollback
```

That is the required bridge before the harder phases:

- TG-P3: Q6_K coop route-spec generation;
- TG-P4: prefill schedule-spec generation;
- TG-P5: generated replacement for 8B owned attention.

The final pure-search target remains:

```text
TINYGRAD_DEFAULT_PURITY_PASS
0 selected defaults with provenance external_handwritten_kernel
0 selected defaults with provenance hand_authored_uop_template
handwritten routes retained only as rollback_oracle
```
