# TG-P9 Scope: Pure Generated Attention Primitive Route

Date: 2026-07-01.

Goal: unblock full `TINYGRAD_DEFAULT_PURITY_PASS` by adding the missing **generic generated-code primitives** needed for 8B decode attention to match the owned HIP route, without hand-writing an attention kernel.

TG-P8 closed the tuning path. The remaining problem is not route policy, not flags, not `L`, and not generic "try harder." The generated 8B attention route is correct and route-bound but slower because it lacks two owned-route capabilities:

1. **Live-context split geometry**: fixed split count for occupancy, but runtime per-split length scales with live `Tc`, not `MAXC`.
2. **Split-preserving combine lifecycle**: combine/log-sum-exp must avoid the current generated 3-kernel tax without collapsing the parallelism that makes flash decode fast.

TG-P9 is the primitive route: teach tinygrad/BoltBeam enough reusable IR/codegen/search vocabulary to express those capabilities, then let the generated route compete. If the primitives cannot be expressed, produce a precise `EMITTER_BLOCKED` / `PRIMITIVE_MISSING` ledger entry.

## Current Evidence

Primary artifacts:

- `bench/tg-p5-attention-generated-default/latest.json`
  - `TG_P5_REFUTE_GENERATED_ATTENTION_SLOWER`
  - generated G4 attention is token-identical and route-bound, but 87.6% of owned at ctx512 and 95.6% at ctx4096.
- `bench/tg-p8-generated-8b-attention-parity/`
  - TG-P8.0/P8.1/P8.2 evidence and geometry refutation.
  - generated tile is flat across context because it launches against `MAXC`.
  - `L` search refuted; `L=128` is already optimal.
- `bench/tg-p7-pure-search-default/summary.md`
  - strict default purity fails only on `decode_attention_owned_two_kernel`.
- `bench/pure-machine-search-default-path-census/summary.md`
  - 4 of 5 hot routes are generated/policy-owned; only 8B attention remains external-handwritten.
- `/home/ubuntu/BoltBeam/boltbeam/data/candidates.json`
  - `decode_attention_g5_8b_refuted` records the reopen condition:
    - symbolic per-split length generated tile;
    - new non-collapse combine primitive.

Important refutations:

- Do not re-run `L` geometry tuning. TG-P8.2 refuted it.
- Do not promote current generated G4/G5-8B attention. TG-P5 refuted it on speed.
- Do not re-chase Hq-only fused combine. It removes combine but loses occupancy.
- Do not re-chase merged combine that collapses `Hq*Hd` combine parallelism. It regresses.
- Do not add HIP/ASM/inline ISA kernels. The purpose is pure generated codegen/search.

## Definition of "Primitive Route"

Allowed:

- generic tinygrad UOp / scheduler / lowering improvements;
- parameterized route specs;
- generated UOp kernels emitted from specs;
- BoltBeam candidate families and policy rows;
- microgates and oracle comparisons against owned HIP.

Not allowed:

- new handwritten HIP;
- new handwritten AMDGCN/ISA kernel as the route implementation;
- copying `extra/qk_owned_flash_decode.hip` into another external kernel;
- making the slower generated route default just to pass purity;
- special-casing only the exact Qwen3-8B tensor names without a generic shape/geometry reason.

The route can be implemented by humans at the **compiler primitive / emitter / spec** layer. The generated attention kernel itself must be produced from reusable primitives or route specs.

## Target Geometry

Protected shape:

```text
model: Qwen3-8B-Q4_K_M
target: amd_gfx1100 / RX 7900 XTX
attention: B=1, Hq=32, Hkv=8, G=4, Hd=128
contexts: 512 and 4096 protected; 128/1024/2048 optional sanity
```

Owned default route:

```text
decode_attention_owned_two_kernel
owned_flash_tile_gqa_whole + owned_flash_combine
external handwritten HIP/AMDGCN
```

Current generated refuted route:

```text
flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128
flash_state_gmax_32_128
flash_state_combine_32_128
generated UOp, correct, route-bound, slower
```

## Phase TG-P9.0: Primitive Gap Confirmation

Purpose: load TG-P8 evidence into a machine-readable primitive backlog before changing code.

Create:

- `bench/tg-p9-pure-attention-primitive-route/primitive_gap.json`
- `bench/tg-p9-pure-attention-primitive-route/summary.md`

Required rows:

| primitive | status | evidence |
|---|---|---|
| `live_tc_split_geometry` | expected `EMITTER_BLOCKED` | generated route launches `ceildiv(MAXC,L)` splits; owned scales per-split length to live `Tc` |
| `split_preserving_attention_lse_combine` | expected `EMITTER_BLOCKED` or `PRIMITIVE_MISSING` | current generated 3-kernel lifecycle loses 556us/fwd; collapse attempts refuted |
| `owned_external_attention_route` | `FORBIDDEN_FINAL_DEFAULT` | external HIP; allowed only as rollback/oracle |

Also update BoltBeam only if needed with a structured backlog candidate such as:

```text
decode_attention_live_split_generated
decode_attention_split_preserving_lse_combine
```

Verdicts:

- `TG_P9_0_PASS_PRIMITIVE_BACKLOG_PINNED`
- `TG_P9_0_BLOCKED_EVIDENCE_MISSING`
- `TG_P9_0_BLOCKED_LEDGER_DRIFT`

Stop if the evidence cannot reproduce TG-P8's two blockers.

## Phase TG-P9.1: Live-Context Split Geometry IR

Purpose: represent owned-like split geometry in generated code.

Required capability:

```text
S = fixed or policy-selected split count for occupancy
per = ceildiv(Tc, S) or equivalent runtime live-context split length
split_start = split_id * per
split_end = min(Tc, split_start + per)
tile loop only covers live tokens in this split
```

Key constraint: preserve enough split parallelism at ctx512 while eliminating the `MAXC` over-launch tax.

Deliverable:

- a tiny route/spec object or UOp pattern that can express:
  - live `Tc`;
  - fixed `S`;
  - dynamic per-split bounds;
  - masked tail;
  - no model-name hardcode.

Acceptance microgates:

1. For a synthetic attention split, generated split ranges cover `[0, Tc)` exactly once.
2. No split reads beyond live `Tc`.
3. Workgroup count is policy-selected `S`, not `ceildiv(MAXC,L)`, when this primitive is selected.
4. Existing generated G5/G4 route still works unchanged when the primitive is disabled.

Verdicts:

- `TG_P9_1_PASS_LIVE_TC_SPLIT_IR`
- `TG_P9_1_BLOCKED_UOP_RANGE_MODEL`
- `TG_P9_1_BLOCKED_SYMBOLIC_BOUNDS`

Do not proceed to full W==D until the split-range microgate passes.

## Phase TG-P9.2: Generated Tile Using Live Split Geometry

Purpose: build the first generated attention tile candidate using TG-P9.1.

Candidate rules:

- generated UOp only;
- default-off;
- route-bound;
- owned HIP disabled during candidate measurement;
- compare against owned only as oracle/reference.

Suggested flag:

```text
DECODE_ATTN_LIVE_SPLIT_GENERATED=0
```

Suggested candidate id:

```text
decode_attention_live_split_generated
```

Measurements:

- ctx512 and ctx4096;
- token/logit equivalence;
- route attribution;
- tile wall time;
- gmax/combine wall time;
- total W==D.

Expected result:

- ctx512 should improve if live split geometry is the real short-context blocker.
- ctx4096 may remain blocked by combine lifecycle.

Verdicts:

- `TG_P9_2_PASS_LIVE_SPLIT_TILE`
- `TG_P9_2_REFUTE_LIVE_SPLIT_NO_MOVEMENT`
- `TG_P9_2_BLOCKED_CORRECTNESS`
- `TG_P9_2_BLOCKED_ROUTE_ATTRIBUTION`

If ctx512 does not move, reclassify TG-P8's split-geometry conclusion and stop.

## Phase TG-P9.3: Split-Preserving LSE Combine Primitive Design

Purpose: design a generated combine primitive that reduces lifecycle cost without collapsing parallelism.

The primitive must preserve at least one of the parallelism levels that previous attempts lost:

- partial stage roughly `Hq*S`;
- combine/output stage enough `Hq*Hd` or equivalent d-sharded parallelism;
- no Hq-only collapse unless a measured occupancy proof says Hq is enough.

Required analysis:

| option | allowed? | reason |
|---|---|---|
| Hq-only fused in-workgroup combine | no | refuted: -88% |
| merged gmax/den/combine to one per-head kernel | no as-is | refuted: collapses `Hq*Hd`, -16% |
| split-preserving two-stage generated combine | yes | target |
| global atomic/grid-sync LSE | only if backend primitive exists | likely `PRIMITIVE_MISSING` on AMD |
| d-sharded generated combine with fewer launches but preserved `Hq*Hd` work | yes | target |

Deliverable:

- `docs/tg-p9-split-preserving-lse-combine-design.md` or equivalent artifact;
- BoltBeam classification: `REACHABLE_NOW`, `EMITTER_BLOCKED`, or `PRIMITIVE_MISSING`.

Verdicts:

- `TG_P9_3_PASS_COMBINE_PRIMITIVE_DESIGN`
- `TG_P9_3_BLOCKED_PRIMITIVE_MISSING`
- `TG_P9_3_REFUTE_NO_PARALLELISM_PRESERVING_DESIGN`

Do not build if the design collapses to a previously refuted shape.

## Phase TG-P9.4: Combine Primitive Microgate

Run only if TG-P9.3 returns `REACHABLE_NOW` or a bounded `EMITTER_BLOCKED` fix was implemented.

Microgate requirements:

- input: synthetic per-split `(m, l, pv)` or equivalent online-softmax partials;
- output: exact LSE-merged attention output within tolerance;
- compare against numpy/Python reference;
- test at several `S`, `Hd`, and `Hq` values including 8B shape;
- report workgroup geometry.

Acceptance:

- correct within tolerance;
- generated UOp only;
- route has no hidden external HIP/ASM;
- no collapse to Hq-only shape.

Verdicts:

- `TG_P9_4_PASS_COMBINE_MICROGATE`
- `TG_P9_4_BLOCKED_NUMERIC`
- `TG_P9_4_REFUTE_PARALLELISM_COLLAPSE`

## Phase TG-P9.5: Full Generated Attention Candidate

Run only after TG-P9.2 and TG-P9.4 pass.

Candidate:

```text
live-context split tile + split-preserving generated LSE combine
```

Protected gate:

| context | required |
|---:|---|
| 512 | >=98% of owned |
| 4096 | >=98% of owned |

Also require:

- token/logit equivalence;
- route-bound generated attention;
- no owned HIP fallback;
- no hidden fallback to current slower TG-P5 route unless that is the selected tile;
- rollback to owned one flag away;
- no regression in default fast mode until promotion.

Verdicts:

- `TG_P9_5_PASS_GENERATED_ATTENTION_PARITY`
- `TG_P9_5_REFUTE_STILL_SLOW`
- `TG_P9_5_BLOCKED_POLICY_OR_ROUTE`
- `TG_P9_5_BLOCKED_CORRECTNESS`

## Phase TG-P9.6: Promotion and Final Purity

Run only if TG-P9.5 passes.

Actions:

1. Add/update BoltBeam candidate as promoted.
2. Emit route policy selecting the generated attention route for Qwen3-8B/gfx1100 shape.
3. Update tinygrad route-policy consumer if a new route id is needed.
4. Keep owned HIP route as rollback/oracle.
5. Run:

```bash
PYTHONPATH=. python3 extra/pure_machine_search_default_path_census.py --check
PYTHONPATH=. python3 extra/pure_machine_search_default_path_census.py --strict-final-default
PYTHONPATH=. python3 -m pytest -q test/unit/test_qk_route_purity.py
```

6. Run BoltBeam tests:

```bash
cd /home/ubuntu/BoltBeam && pytest -q
```

Verdicts:

- `TG_P9_6_PASS_TINYGRAD_DEFAULT_PURITY_PASS`
- `TG_P9_6_BLOCKED_STRICT_CENSUS`
- `TG_P9_6_BLOCKED_BOLTBEAM_POLICY`

## Phase TG-P9.7: Honest Blocker Ledger

Run if any prior phase blocks or refutes.

Record in tinygrad and BoltBeam:

- exact primitive missing;
- whether it is `EMITTER_BLOCKED`, `PRIMITIVE_MISSING`, `REFUTED_BY_LEDGER`, or `LOW_AMDAHL`;
- do-not-retry axis;
- reopen condition;
- owned route remains default.

The acceptable terminal blocked states are:

| verdict | meaning |
|---|---|
| `TG_P9_BLOCKED_LIVE_SPLIT_GEOMETRY` | cannot express owned-like dynamic per-split bounds |
| `TG_P9_BLOCKED_SPLIT_PRESERVING_COMBINE` | cannot express non-collapse LSE combine |
| `TG_P9_REFUTED_GENERATED_ATTENTION_STILL_SLOW` | primitives built, but still below 98% owned |
| `TG_P9_PASS_FULL_PURITY` | generated attention promoted and strict default purity passes |

## Claude Handoff Prompt

Use this exact prompt for a fresh Claude context:

> Continue tinygrad pure-machine-search from TG-P9. Goal: remove the last external-handwritten default route, `decode_attention_owned_two_kernel`, by adding generic generated-code primitives, not by hand-writing a kernel. TG-P8 proved current generated 8B attention is correct and route-bound but slower: 87.6% of owned at ctx512, 95.6% at ctx4096. TG-P8 identified two blockers: (1) live-context split geometry: generated route launches `ceildiv(MAXC,L)` splits and over-launches at ctx512; owned uses fixed split count with runtime per-split length scaled to live `Tc`; (2) split-preserving combine lifecycle: generated 3-kernel gmax+combine costs too much at ctx4096, but previous combine-collapse attempts are refuted because they lose `Hq*S` or `Hq*Hd` parallelism. Start with TG-P9.0 primitive backlog and TG-P9.1 live-context split IR. Do not implement a full candidate until microgates prove the primitive. No HIP/ASM/handwritten attention kernel. Promotion requires generated route >=98% of owned at ctx512 and ctx4096, token/logit equivalence, route-bound, rollback to owned, BoltBeam candidate/policy update, and strict default purity pass. If blocked, ledger the exact primitive/emitter gap and leave owned default.

## Expected End Result

If successful, tinygrad reaches:

```text
TINYGRAD_DEFAULT_PURITY_PASS
```

with:

- Q4_K decode GEMV generated;
- Q6_K decode generated;
- prefill generated schedule;
- G5/G4 attention generated;
- owned/handwritten kernels retained only as rollback/oracle.

If unsuccessful, the project still improves because the final impurity is reduced to a precise compiler primitive gap rather than a vague performance problem.

