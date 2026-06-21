# Decode-Attention Route A / Route B Full Execution Scope

Date: 2026-06-21

Status: **`DECODE_ATTENTION_ROUTE_AB_FULL_SCOPE_READY`**

Companion to `docs/decode-attention-primitive-spec-and-route-scope-20260621.md`. That document defines the
`decode_attention_llama_flash_tile` primitive and recommends **Route B escape hatch first**. This document fully scopes
**both** implementation routes so either can be executed deliberately:

- **Route A:** native tinygrad codegen/renderer capability.
- **Route B:** AMDGCN/HSACO escape hatch wrapped by the evaluator/lifecycle system.

Boundary: scope only. No kernel built, no `tinygrad/` change, no model/default route, no benchmark rerun.

## Shared Target

Both routes target the same primitive:

```text
decode_attention_llama_flash_tile
T=1 GQA decode attention
KV-split workgroups
GQA query-head column packing
LDS-staged K/V
v_dot2_f32_f16 vector dot body
register online softmax + PV in one tile
efficient split combine/fixup
env-gated default-off in-model route with fallback to gqa_coop_vec
```

Shared comparator: `gqa_coop_vec`, the current tinygrad decode-attention winner.

Shared first performance gate:

- correctness: `rel_rmse <= 1e-3` vs numpy/reference or greedy byte-identical/dNLL `<= 0.01` in model;
- local A/B: at least `1.5x` vs `gqa_coop_vec` @ctx1024;
- W==D: at least `+5%` @ctx1024 and `+7%` @ctx4096 whole-decode, no ctx512 regression;
- artifact: harness-contract stamped, machine-readable, linked to ledger/refutation;
- policy: default-off until owner promotion; unsupported shapes fall back.

## Why Both Routes Exist

| evidence | implication |
|---|---|
| llama oracle: `flash_attn_tile` is `5-6x` faster standalone | the target is real and worth pursuing |
| fused-flash concrete gate: tinygrad matmul path loses `0.965x` and emits register-tiled global-load code, not LDS/v_dot2 tile | bounded tinygrad graph tricks are exhausted |
| low-level attribution: tinygrad `flash_partial` has scalar fp16 loads, `0 v_dot2`, `0 LDS`; llama tile has LDS + `v_dot2` | gap is codegen/control-surface quality |
| lifecycle/evaluator/search now exist | any route must plug into `decode_eval`, lifecycle-search, and refutation memory |

Route B is the shortest way to prove runtime/lifecycle value. Route A is the proper long-term tinygrad-native ownership
path, but only justified after the target and W==D ceiling are proven.

## Route B — AMDGCN/HSACO Escape Hatch

Decision role: **recommended first**.

Route B owns the primitive below tinygrad's current renderer. It is DeepSeek-style: drop to the lowest responsible
layer only because the normal stack cannot express the required schedule.

### Route B Deliverables

| phase | deliverable | files likely touched | promotion eligible? |
|---|---|---|---|
| B0 | binding/candidate update for Route-B variants | `bench/qk-decode-eval/candidates.json`, `binding_templates.json`, lifecycle templates | no |
| B1 | vendored llama `.co` reference runner through HCQ bridge | `extra/qk_llama_flash_attn_tile_hcq_ab.py`, `bench/qk-*` artifacts | no, reference-only |
| B2 | in-model default-off de-risk route using vendored `.co` | minimal model-route flag or route adapter, evaluator W==D runner | no, vendored |
| B3 | hand-authored AMDGCN/HSACO tile matching the primitive | `extra/qk_decode_attention_amd_gcn_*`, `.s`/`.co` artifacts | yes, if owned |
| B4 | in-model default-off owned route + fallback | route flag, shape guard, W==D runner | yes, owner decision only |
| B5 | search-template expansion over HSACO parameters | lifecycle templates, candidate generation, evaluator bindings | candidate-dependent |

### B0 — Binding And Candidate Setup

Goal: make Route-B candidates explicit and non-ambiguous.

Add candidate families:

- `reference_oracle_hcq_llama_tile`: vendored llama `.co`, non-promotable;
- `owned_amdgcn_flash_tile`: hand-authored owned kernel, promotable only after W==D.

Required metadata:

- source status: `vendored_reference` or `owned_escape_hatch`;
- kernel provenance: source path, compile command, ISA hash, `.co` hash;
- shape support: Hd/Hq/Hkv/G, dtype, ctx buckets;
- default eligibility: false for vendored, gated for owned.

Gate:

- lifecycle-search distinguishes vendored/non-promotable from owned/promotable;
- policy guard rejects vendored default promotion.

Stop:

- if candidate metadata cannot express vendored vs owned, fix policy/schema before building kernels.

### B1 — Vendored Llama `.co` HCQ Local A/B

Goal: prove the actual llama-class kernel can be launched from the tinygrad process and still wins locally.

Work:

- extract or build the exact non-WMMA decode `flash_attn_tile<128,128,*,4,false>` and combine kernel for gfx1100;
- avoid broad llama runtime dependencies: hardcode Qwen3-8B decode shape first;
- launch via proven HCQ/Buffer bridge;
- compare against `gqa_coop_vec` in the same process where possible;
- produce a stamped local A/B artifact.

Gate:

- correctness `rel_rmse <= 1e-3`;
- local A/B `>=1.5x` @ctx1024 and no ctx4096 regression;
- artifact carries kernel hash, ISA summary, workgroups/splits/LDS/VGPR, timing authority, comparator reason.

Stop:

- if extraction becomes a broad llama runtime port, stop and classify `NEEDS_DEEPER_PORT`;
- if local A/B misses `1.5x`, stop: standalone oracle did not transfer into the HCQ bridge.

### B2 — Vendored Llama `.co` W==D De-Risk

Goal: answer the decisive lifecycle question: does a llama-class tile improve whole-decode under tinygrad's runtime?

Work:

- add a default-off route flag for the vendored tile;
- shape-guard strictly to the known Qwen3-8B/GFX1100 case;
- fallback to `gqa_coop_vec` on unsupported shapes or failure;
- run `decode_eval` W==D as a non-promotable `reference_oracle_route`.

Gate:

- W==D `>= +5%` @ctx1024 and `>= +7%` @ctx4096;
- no ctx512 regression;
- greedy output byte-identical or dNLL `<=0.01`;
- route fallback tested.

Stop:

- if local wins but W==D misses, do not author the owned kernel yet; classify the gap as integration/lifecycle overhead
  and audit route overhead first;
- if W==D passes, proceed to B3 because the value is proven.

### B3 — Owned Hand-AMDGCN/HSACO Tile

Goal: replace the vendored reference with an owned escape-hatch primitive.

Work:

- author the kernel in AMDGCN/HSACO or a minimal HIP/asm source controlled by this repo;
- implement the same primitive boundary: KV splits, GQA packing, LDS K/V, `v_dot2`, online softmax+PV, combine;
- keep the launcher and artifact format identical to B1/B2;
- compare ISA/resource profile against the vendored/reference tile.

Gate:

- correctness `rel_rmse <= 1e-3`;
- local A/B at least `1.5x` vs `gqa_coop_vec`;
- owned tile is at least `80%` of vendored llama oracle local speed at ctx1024, or has a clear resource reason;
- no spills, LDS/VGPR occupancy within the planned budget.

Stop:

- if owned tile is much slower than vendored and ISA shows missing `v_dot2`/LDS structure, fix kernel structure once;
- after one structural fix, if still below gate, bank refutation and reconsider Route A vs REST.

### B4 — Owned In-Model Route

Goal: make the owned escape hatch a legitimate default-off tinygrad primitive.

Work:

- add a single env-gated route, e.g. `DECODE_ATTN_AMDGCN_TILE=1`;
- route only supported shape/device/dtype;
- fallback automatically to `gqa_coop_vec`;
- register in `decode_eval` with local + W==D rungs.

Gate:

- W==D `>= +5%` @ctx1024 and `>= +7%` @ctx4096;
- no ctx512 regression;
- correctness/quality passes;
- policy guard confirms no default flip;
- contract artifact conforms or carries explicit exception metadata.

Stop:

- if local passes but W==D fails, keep it as research-only and do not promote;
- if route is fragile across contexts, restrict it or refute it.

### B5 — Route-B Machine Search

Goal: turn the hand point into a searched primitive.

Knobs:

- `kv_split_count`;
- `K_tile_size`;
- GQA pack width;
- LDS layout/padding;
- vector form;
- combine strategy;
- workgroup size;
- VGPR/LDS occupancy cap;
- ctx threshold.

Gate:

- generated candidates run through lifecycle-search and `decode_eval`;
- search finds a candidate at least as good as the hand point or records why the hand point is the optimum;
- no search result bypasses W==D/policy.

Stop:

- if compile/run time explodes, reduce to a small manually-ranked grid;
- if no generated candidate beats the hand point, keep the hand point and bank the search refutation.

## Route A — Native Tinygrad Codegen/Renderer Capability

Decision role: **north-star follow-on**, not recommended before Route-B de-risk unless the owner wants pure tinygrad
ownership at higher risk.

Route A changes tinygrad so the native compiler can emit the llama-style primitive directly.

### Route A Deliverables

| phase | deliverable | files likely touched | promotion eligible? |
|---|---|---|---|
| A0 | exact codegen gap spec and golden UOp/ISA fixtures | docs + tests + tiny fixtures | no |
| A1 | minimal renderer intrinsic path for packed fp16 dot / LDS tile idiom | renderer/codegen tests | no |
| A2 | fused reduction/tile scheduling capability for decode flash | `tinygrad/codegen/*`, `tinygrad/renderer/*` | no |
| A3 | standalone native candidate vs `gqa_coop_vec` | evaluator child harness | no until W==D |
| A4 | default-off in-model native route | `tinygrad/llm/model.py`/attention route | owner-gated |
| A5 | template/search knobs over native codegen | lifecycle templates | candidate-dependent |

### A0 — Native Capability Spec

Goal: freeze what native codegen must emit before editing core compiler code.

Required fixtures:

- minimal UOp graph or custom-kernel expression for T=1 decode flash;
- expected structural ISA properties:
  - `ds_read`/LDS use for K/V;
  - `v_dot2_f32_f16` body;
  - no scalar per-fp16 V load loop;
  - no score/prob materialization to HBM;
  - one tile kernel plus bounded combine;
- correctness fixture vs numpy.

Gate:

- a failing test demonstrates current renderer does not emit the structural ISA target;
- the test is narrow and skip-safe outside AMD/gfx1100.

Stop:

- if the target cannot be expressed as a stable UOp/custom-kernel fixture, do not edit the compiler yet; use Route B
  as the oracle fixture first.

### A1 — Intrinsic/Renderer Building Blocks

Goal: make the renderer capable of the instruction/locality pieces without solving full flash at once.

Work:

- add or expose a safe way to request packed fp16 dot form where legal;
- add test coverage that validates emitted ISA contains `v_dot2_f32_f16`;
- add a minimal LDS tile load/store/reuse fixture with `ds_read`/`ds_write` evidence.

Gate:

- unit/golden tests pass;
- no behavior change for ordinary tensor programs;
- AMD-only path guarded.

Stop:

- if renderer changes affect broad codegen or non-AMD backends, stop and reduce to custom-kernel escape path.

### A2 — Fused Flash Scheduling Capability

Goal: unify the two currently separated capabilities:

```text
tiled-GEMM LDS/vectorization
AND
.set/.after register-state online-softmax fusion
```

Work:

- schedule a hand-reduction over K tiles with LDS-staged K/V;
- keep `(m,l,acc[D])` in registers;
- avoid the two-granularity store wall where possible;
- if needed, introduce a narrow AMD-only custom-kernel lowering form rather than general compiler surgery.

Gate:

- generated single tile kernel has the structural ISA target;
- standalone correctness passes;
- local A/B `>=1.5x` @ctx1024.

Stop:

- if the only solution is broad UOp reduce semantics or general linearizer redesign, pause for an explicit compiler
  design scope; do not hide it inside the decode project.

### A3 — Native Standalone Candidate

Goal: compare the native tinygrad-generated tile against the current winner and the Route-B/reference target.

Work:

- add an `ab_script` candidate for the native tile;
- stamp artifact with harness contract;
- include ISA/resource metadata.

Gate:

- local A/B `>=1.5x` vs `gqa_coop_vec`;
- at least `80%` of Route-B/reference tile local speed, or an explicit resource gap is documented.

Stop:

- if local A/B misses, bank refutation and do not add a model route.

### A4 — Native In-Model Route

Goal: make the native codegen tile a default-off route.

Work:

- env-gated route, shape guard, fallback;
- `decode_eval` W==D runner;
- quality gate.

Gate:

- W==D `>= +5%` @ctx1024 and `>= +7%` @ctx4096;
- no ctx512 regression;
- dNLL/greedy passes;
- policy guard passes.

Stop:

- if W==D fails, keep it as a codegen research artifact only.

### A5 — Native Search

Goal: expose codegen knobs to lifecycle search.

Knobs:

- split count;
- tile sizes;
- LDS layout;
- vector form;
- combine;
- occupancy constraints;
- ctx thresholds.

Gate:

- generated native candidates are legal lifecycle rows;
- search improves or confirms the best hand/native point.

Stop:

- if template expansion becomes compiler-specific churn without W==D value, keep Route-B or REST.

## Route Comparison

| axis | Route A native tinygrad | Route B AMDGCN/HSACO |
|---|---|---|
| first useful answer | slow | fast |
| core risk | high | low initially |
| ownership purity | highest | medium/high if hand-authored; low if vendored |
| proof of W==D value | after major work | early via vendored de-risk |
| reusability beyond decode | high | lower |
| chance of llama-class first candidate | uncertain | higher, because oracle exists |
| recommended order | second | first |

## Interaction Between Routes

Route B is not a substitute for Route A forever. It is the de-risk path:

1. Route B vendored proves whether tinygrad runtime can host a W==D win.
2. Route B owned proves whether the project can own the primitive below the renderer.
3. Route A uses the owned/reference kernel as a golden target for native codegen.

If Route B fails W==D, Route A should not start unless the failure is clearly in vendored integration rather than in
the primitive's Amdahl value.

If Route B succeeds, Route A becomes much better specified: match this kernel, this ISA, this resource table, and this
W==D ceiling.

## Required Guardrails

- The vendored llama kernel is never promotable.
- No default route changes before W==D and owner approval.
- No closed-lane reopen: WMMA decode, MMVQ, FLASH_L promotion, fused tail, matmul-PV, warp tile remain closed.
- No benchmark headline from local A/B.
- Any new harness must follow `bench/qk-decode-eval/HARNESS_GUIDE.md`.
- Any failed candidate must write a refutation row.

## Acceptance For This Scope

| gate | status |
|---|---|
| Route A has phases, files, gates, stops, and fallback | PASS |
| Route B has phases, files, gates, stops, and fallback | PASS |
| vendored vs owned promotability is explicit | PASS |
| both routes share the same primitive boundary and evaluator ladder | PASS |
| machine-search deepening is scoped for both routes | PASS |
| no `tinygrad/` changes | PASS |

## Recommended Next Executable Task

Run **Route B B0-B1 only**:

```text
register reference_oracle_hcq_llama_tile
extract/build llama tile .co for gfx1100
launch via HCQ bridge
local A/B vs gqa_coop_vec
stamp artifact
stop before W==D unless local >=1.5x
```

This is the cheapest decisive gate. It either proves the escape hatch is worth integrating, or it prevents both routes
from consuming a month on a target that does not transfer into this runtime.
