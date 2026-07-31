# FP16 prefill routing: one decision, one authority

Date: 2026-07-31

Status: scoped, not implemented. Branch boundary: all work begins on `nvidia-bringup-20260731` per the NVIDIA
campaign rule (branch -> selective merge to `exp` -> promote to `dev` only after NVIDIA is proven end-to-end).
This scope does not authorize promotion to `dev`/`master` by itself, and it adds no new selection-policy table,
registry, or control plane.

## 1. End goal

**One-sentence reduction:** *fp16 residency ("overlay or not") becomes one decision computed once from
facts-as-data, with capability and promotion as separate TG3 questions, so NV, AMD, and Metal share one routing
path with no duplicated coverage lists, no dead parameters, and no three-hop replanning.*

The fp16 routing today is written as an accident of history: the "should this load use the fp16 overlay" answer
is smeared across `model.py`, `admission.py`, and `prefill_candidate_runtime.py`, derived twice, threaded through
a parameter nobody reads, and gated by a promotion artifact that is fused with capability so that a new target
(NV) is structurally invisible rather than loudly un-promoted. The end state:

```text
which tensors are fp16-covered   -> one role list + one byte estimate (data, not suffix matching)
can this target EXPRESS it?      -> renderer/device capability facts
is this candidate PROMOTED here? -> BoltBeam registry (unchanged authority)
what fits, and what wins?        -> one pure planner call, no replanning
```

This is a unification, not a new subsystem. It reuses the inventory the runtime already derives, the capability
facts the device scan already publishes, and the registry admission that already exists.

## 2. Pinned evidence

All numbers measured 2026-07-31 on RTX 5090 (GB202, sm_120), 32 GB, `nvidia-bringup-20260731`.

| result | value | where |
| --- | ---: | --- |
| 8B Q4_K_M decode floor (generic dequant fallback) | 4.49 tok/s, 21.5 GB/s = 1.3% of BW | `bench/models/qwen/data/nv-rtx5090/qwen3-8b.json` |
| fused Q4_K path once `wave_size=32` is declared (probe, monkeypatched facts) | 253/253 primitives installed+admitted; first token identical `50994`; 7.8s -> 4.7s | `/tmp/nv_q4k_lower_probe2.py`, re-run 2026-07-31 |
| fp16 2048x2048 GEMM through real `CUDARenderer` | emits `HMMA.16816.F32`, numerically correct to fp16 rounding | Phase-2 compile-only probe |
| 8B fp16 overlay size | 16.4 GB -> fits 32 GB -> overlay-shaped | `docs/bringing-up-a-new-target-20260731.md` section 9 |
| 14B fp16 overlay size | 29.5 GB -> does not fit -> Metal-shaped | same |

`CUDARenderer` now declares `wave_size=32` (landed with this scope's prerequisite commit), so the fused-path
row above is expected to reproduce without monkeypatching; the e2e re-measurement is this scope's baseline
gate. The fp16 routing refactor itself is target-neutral: it fixes the same three-hop structure for AMD, Metal,
and NV ("a fix made for one target should fix all three", section 9 of the bring-up doc).

## 3. The current architecture: every fp16 selection point

### 3.1 Coverage definition (what counts as fp16)

| file:line | what it does | problem |
| --- | --- | --- |
| `tinygrad/llm/model.py:884` | `_PREFILL_V2_LINEARS` tuple of linear names | hardcoded role list (the one honest authority, but private to the model class) |
| `tinygrad/llm/model.py:1078` | `_cov = tuple(f"{n}.weight" for n in Transformer._PREFILL_V2_LINEARS)` | converts roles to name suffixes for metadata matching |
| `tinygrad/llm/model.py:896` | `_prefill_v2_covered()` walk of the built model | the realization authority for warmstart/realize; admission cannot use it (pre-construction) |

**Two authorities for "which tensors are fp16-covered":** the name-suffix list at admission time and the model
walk at realization time. They can drift (a renamed tensor, a new role, an arch that omits a linear). The
comment at `model.py:900` claims a single source, but the admission estimate does not use it.

### 3.2 The byte estimate (computed twice)

| file:line | path | expression |
| --- | --- | --- |
| `tinygrad/llm/model.py:1105` | GGUF metadata | `sum(prod(dims) * 2 for name, dims, _, _ in _admit_meta["tensor_infos"] if any(name.endswith(s) for s in _cov))` |
| `tinygrad/llm/model.py:1200` | state dict | `sum(t.numel() * 2 for k, t in state_dict.items() if any(k.endswith(s) for s in _cov))` |

Identical semantics, two data shapes, two call sites, one suffix-matching mechanism. Meanwhile the selected-GGUF
inventory (`derive_selected_gguf_prefill_inventory`, `model.py:151`) already derives per-tensor **roles and
shapes** from the same metadata; the coverage bytes are recomputable from the inventory in one pass, with no
suffix matching.

### 3.3 The admission switch (a parameter nobody reads)

| file:line | what it does |
| --- | --- |
| `tinygrad/llm/model.py:1108`, `:1203` | `resident_fp16_admit=False` hardcoded at both call sites |
| `tinygrad/llm/admission.py:215-232` | field carried through `AdmissionInputs` |
| `tinygrad/llm/admission.py:362` | `replace(inp, ..., resident_fp16_admit=explicit_overlay)` overwrites it |
| `tinygrad/llm/admission.py:260` | `_plan_context_admission` never reads it |

`resident_fp16_admit` is dead weight: written `False`, overwritten by `replace`, never read. It manufactures the
impression that fp16 admission is a caller decision when the real switch is `overlay_requested` (3.4).

### 3.4 The planner and its tri-state override

| file:line | what it does |
| --- | --- |
| `tinygrad/llm/admission.py:360` | `explicit_overlay = overlay_requested is True` |
| `tinygrad/llm/admission.py:240-241` | `ContextMemoryTerms.from_inputs`: weights = q4 + est_fp16 iff `resident_fp16` |
| `tinygrad/llm/admission.py:374-377` | `full-resident-overlay` candidate, `supported=inp.v2_on` |
| `tinygrad/llm/admission.py:378-379` | `direct-packed-baseline` candidate |
| `tinygrad/llm/admission.py:381` | `override = FULL_RESIDENT_OVERLAY if explicit_overlay else (DIRECT_PACKED_FALLBACK if overlay_requested is False else None)` |

The tri-state `overlay_requested` (None/True/False) encodes "decide for me / force overlay / force baseline";
three imperative modes instead of one policy input. The planner itself is otherwise a clean pure function.

### 3.5 The orchestration (three-hop sequencing, up to three planner calls)

`model.py:1093-1130` (GGUF path):

1. `select_memory_adaptive_runtime_policy(...)` always returns `DIRECT_PACKED_FALLBACK` for normal loads
   (`"no production policy collector available"`, `model.py:220`).
2. If baseline and `measured is False`: `_automatic_overlay_policy = automatic_promoted_prefill_graph_policy(...)`
   does a registry lookup on `(backend, arch, wave_size)`; the pinned artifact is AMD gfx1100 only
   (`prefill_candidate_runtime.py:179`, `:269`).
3. `_overlay_request = prefill_policy_uses_overlay(_runtime_policy)` (False).
4. `v2_on = _automatic_overlay_policy is not None or _overlay_request is not False`; capability is derived from
   promotion (backwards).
5. First planner call with `overlay_requested=None`; if `FULL_RESIDENT_OVERLAY in feasible_strategies`, replan
   with `overlay_requested=True`; else replan with `overlay_requested=False` (`model.py:1116-1130`).

So the "should we use fp16" answer is the residue of a registry miss, not a decision. On NV the registry returns
`None`, `v2_on=False`, the overlay candidate is `supported=False`, and the whole route is silently unreachable
regardless of VRAM.

### 3.6 The KV fp16 decision (separate planner)

| file:line | what it does |
| --- | --- |
| `tinygrad/llm/admission.py:270-325` | `_resolve_max_context_admission`: `kv-fp16-exact` vs `kv-q8-exact` vs `kv-fp16-ring` |

The KV representation choice is capacity math over the same scanned budget, in a second candidate language
(`_ContextCandidate`) inside the same module. The composition with the weights overlay is correct (`terms.weights`
already includes `est_fp16` when the overlay is explicit, and KV capacity is computed against that), but the
composition is implicit and untested as a single "fp16 spend" question.

### 3.7 The runtime `v2_on` double meaning

| file:line | what it does |
| --- | --- |
| `tinygrad/llm/model.py:1146`, `:1207` | `_v2_on = prefill_policy_strategy(_runtime_policy) in ("FULL_RESIDENT_OVERLAY", "BOUNDED_PACKED_TILES", "DIRECT_PACKED_FALLBACK")` |
| `tinygrad/llm/model.py:1215` | `prefill_concrete_kv_auto_decision(_workload_reuse, _v2_on)` |
| `tinygrad/llm/model.py:1258` | `prefill_v2=_v2_on` |

The admission-time `v2_on` (capability-ish) and the runtime `_v2_on` (always True for every executed strategy,
since REFUSE raises) are the same name for different meanings. The runtime flag is effectively constant after
admission and only feeds the concrete-KV default and config; a naming lie that costs a reader a trace.

## 4. Named failure modes (mapped to the repo's principles)

| # | failure | principle it violates |
| --- | --- | --- |
| F1 | two fp16-coverage authorities (suffix list vs model walk) can drift | tell-it-or-show-it: a fact written twice is not a fact |
| F2 | `resident_fp16_admit` threaded but never read | narrow-optimization-collapse section 5: an unbuilt/vestigial mitigation manufactures confidence |
| F3 | decision is a three-hop side-effecting sequence with replanning, not a function | codebase-organization: deep modules; the planner should be pure and called once |
| F4 | capability derived from promotion; registry miss is a silent `None` | honest-ratcheting: a gate that cures into an invisible fallback is a lie; missing promotion must be loud and labeled |
| F5 | fp16 spend split across two candidate languages with implicit composition | consistency: one budget question, one ledger |

## 5. Target design

### 5.1 Coverage as data (kills F1)

- Keep ONE role list, but move it out of the model class into the facts layer (e.g. `tinygrad/llm/model_facts.py`)
  as `PREFILL_V2_ROLES` (the same names as today's `_PREFILL_V2_LINEARS`).
- GGUF path: `derive_selected_gguf_prefill_inventory` already produces role+shape rows. Add
  `covered_fp16_bytes` to the inventory dict, computed once at derivation:
  `sum(row["shape"]["n"] * row["shape"]["k"] * 2 for row in rows if row["role"] in PREFILL_V2_ROLES)`, with
  `lm_head` included only under an explicit `lm_head_route == "resident_fp16"` workload policy (matching the
  existing `_prefill_v2_covered` boundary at `model.py:917-920`).
- State-dict path: one shared helper `estimate_prefill_fp16_bytes(names_and_numels)` that resolves roles the
  same way the inventory does; both call sites disappear.
- Delete `_cov` (`model.py:1078`) and both `_est_fp16` expressions (`:1105`, `:1200`).
- Ratchet test: for a fixture arch, inventory-derived coverage bytes equal the `_prefill_v2_covered()` walk bytes
  on the built model.

### 5.2 One policy input, one planner call (kills F3, F2)

- Replace `overlay_requested: bool|None` with `policy: Mapping|None` on `plan_selected_model_memory`:
  - `policy is None` -> planner decides from feasibility (today's `None`).
  - `policy` with strategy `DIRECT_PACKED_FALLBACK` -> forced baseline (today's `False`).
  - `policy` with strategy `FULL_RESIDENT_OVERLAY` -> forced overlay (today's `True`).
  - The tri-state survives as data, not as three call shapes.
- `explicit_overlay = policy is not None and prefill_policy_strategy(policy) == "FULL_RESIDENT_OVERLAY"`;
  `ContextMemoryTerms.from_inputs` keeps `resident_fp16=explicit_overlay` internally; delete `resident_fp16_admit`
  from `AdmissionInputs`, `from_model_metadata`, both call sites, and the `replace` at `admission.py:362`.
- `model.py` calls the planner exactly once per path. The auto-promotion lookup stays a caller concern (it answers
  the promotion question), but its result is passed in; no `None -> True -> False` replan loop.

### 5.3 Capability from facts, promotion from registry (kills F4)

- Capability: prefill-v2 dense fp16 needs tensor cores and fp16 dtype. Both are already published facts:
  `DeviceCapabilities.supports_tensor_cores` (`device_facts.py:228`) and `renderer.supported_dtypes()`.
  Compute `v2_on` (admission capability) from those, never from the promotion artifact.
- Promotion: unchanged authority, `automatic_promoted_prefill_graph_policy` and the BoltBeam registry.
- Registry miss becomes **loud**: when capability says expressible but promotion says no artifact, the load
  continues on `DIRECT_PACKED_FALLBACK` with a labeled census entry ("prefill-v2 expressible; no promoted
  candidate for (backend, arch, wave_size)"), exactly the honest-ratchet shape the TG3 package already uses
  for Q4_K (`QKPrimitiveRouteAdmission.admitted = capability.satisfied and target_promoted`).
- On NV today this yields `DIRECT_PACKED_FALLBACK` plus a loud "no promoted candidate": same behavior as now,
  but observable, and it stops being a structural impossibility the moment a promoted artifact exists.

### 5.4 One fp16 spend ledger (kills F5)

- Keep the existing composition (it is correct) but make it explicit and tested: add one derived field to the
  admission report, `fp16_spend_gb` = overlay bytes (if elected) + KV bytes (per the elected KV representation),
  and a unit test asserting the overlay + KV fp16 composition against a single scanned budget. No new planner,
  no new candidate language.
- `_resolve_max_context_admission` stays the KV-representation owner (deep module: it already answers its one
  question).

### 5.5 Truthful runtime naming (kills the v2_on lie)

- Rename the runtime flag at `model.py:1146/1207` to `_prefill_v2_active` (it is True for every executed strategy)
  or fold it; the admission-time capability keeps the `v2_on` name. No behavior change, one renamed variable and
  its consumers (`prefill_concrete_kv_auto_decision`, `prefill_v2=` config).

## 6. Authority table (one owner per concern)

| concern | owner |
| --- | --- |
| fp16-covered roles | `PREFILL_V2_ROLES` data (model_facts) |
| fp16-covered bytes | inventory derivation (GGUF) / one shared helper (state dict) |
| can this target express prefill-v2 fp16 | renderer/device facts (`tensor_cores`, `supported_dtypes`, `wave_size`) |
| is this candidate promoted here | BoltBeam registry via `automatic_promoted_prefill_graph_policy` |
| memory plan (strategy, feasibility, context) | `plan_selected_model_memory` (pure, called once) |
| KV representation | `_resolve_max_context_admission` (unchanged) |
| runtime execution hooks | `model.py` (unchanged) |

No new capability registry, no new selection table, no new control plane (target-capability scope section 3.1).

## 7. Slices with verification

Each slice keeps the NV e2e bench green (honest ratchet: always one runnable artifact with a real number) and
adds unit tests at module boundaries.

| slice | change | verification |
| --- | --- | --- |
| S0 (landed) | `CUDARenderer.wave_size = 32` | device facts report 32; fused Q4_K installs; 8B e2e re-measure (baseline gate for this scope) |
| S1 | coverage as data (5.1) | inventory bytes == model-walk bytes test; NV e2e unchanged strategy/digits |
| S2 | delete `resident_fp16_admit` (5.2) | `test_prefill_memory_plan_integration.py` constructors updated; full llm test batch |
| S3 | policy input + single planner call (5.2) | planner pure-function tests for policy=None/baseline/overlay; model.py call-count review |
| S4 | capability from facts + loud registry-miss census (5.3) | test: NV facts + no artifact -> baseline + census; AMD facts + artifact -> overlay when feasible; census visible in admission report |
| S5 | fp16 spend ledger (5.4) | composition test vs one budget; report field present |
| S6 | `_prefill_v2_active` rename (5.5) | no behavior change; config/report consumers updated |

Post-S4, the NV overlay route is no longer structurally impossible: the missing piece becomes the promoted
candidate artifact (BoltBeam export + policy collector), which is the next campaign step and is out of scope here.

## 8. Out of scope / parked gates

- `BOUNDED_PACKED_TILES`, the measurement authority (`_MEMORY_ADAPTIVE_MEASUREMENT_AUTHORITY`), and
  `select_memory_adaptive_runtime_policy` itself: unchanged.
- The NV promoted prefill candidate artifact and its memory_adaptive policy run: a separate step after S4
  (this scope only makes the miss loud).
- 14B Metal-shaped fused path and the layer-resident-overlay lifecycle
  (`layer-resident-overlay-lifecycle-scope-20260731.md`): separate scopes; this refactor is their prerequisite
  routing cleanup, not their implementation.
- KV ring / streaming semantics: unchanged.
- No env-switch, no new flag, no promotion to `dev`/`master` by this scope.

## 9. Current vs proposed: exhaustive comparison

One row per fp16-routing concern. "Current" is the code as it exists on this branch (pinned above in section 3);
"proposed" is section 5 expressed side by side; the principle column names the knowledge-base principle the
proposal serves (full mapping in section 10).

| concern | current | proposed | principle |
| --- | --- | --- | --- |
| fp16-covered roles | `_PREFILL_V2_LINEARS` hardcoded on `Transformer` (`model.py:884`); `_cov` name-suffix list (`:1078`); separate model-walk authority `_prefill_v2_covered()` (`:896`) | one `PREFILL_V2_ROLES` data list in `model_facts`; coverage derived from inventory role+shape rows; model walk kept but tested equal to the inventory bytes | tell-it-or-show-it (a fact written once), minimization |
| fp16 byte estimate | two expressions with suffix matching: GGUF metadata (`:1105`) and state dict (`:1200`) | one computation at inventory derivation (GGUF) + one shared helper (state dict); both `_est_fp16` expressions deleted | locality, minimization |
| fp16 admission switch | `resident_fp16_admit` field, hardcoded `False` at both call sites, overwritten by `replace`, never read | parameter deleted; the policy input is the decision | narrow-optimization-collapse (a vestigial gate manufactures confidence), minimization |
| planner override | tri-state `overlay_requested` None/True/False = decide/force-overlay/force-baseline (`admission.py:360-381`) | `policy: Mapping|None`; the tri-state survives as data (the policy's strategy field) | deep modules, consistency |
| orchestration | three-hop sequence, up to three planner calls with replanning (`model.py:1093-1130`) | one planner call; promotion lookup is a caller concern whose result is passed in | deep modules, locality |
| capability vs promotion | fused: `v2_on` derived from promotion-artifact presence; registry miss is a silent `None`; NV structurally invisible | split: `v2_on` from renderer/device facts (`supports_tensor_cores` + fp16 dtype); promotion stays in the BoltBeam registry; miss becomes a loud labeled census entry | TG3 (the repo's own Q4_K pattern), honest-ratcheting |
| KV representation | separate candidate language in `_resolve_max_context_admission` (`admission.py:270-325`); composition with overlay implicit | owner unchanged (deep module); composition made explicit and tested via one `fp16_spend_gb` ledger field | deep modules, constrained-memory |
| runtime `v2_on` | same name for admission capability and a runtime flag that is always True post-admission (`model.py:1146/1207`) | runtime flag renamed `_prefill_v2_active`; no behavior change | comprehensibility, consistency |
| LM-head fp16 | explicit workload policy `lm_head_route == "resident_fp16"` (`model.py:917-920`) | unchanged; the coverage computation merely respects it | restraint (the honest part stays) |
| promotion authority | `automatic_promoted_prefill_graph_policy` + pinned AMD gfx1100 artifact | unchanged authority; capability decoupled so a target with a future artifact becomes reachable | one authority per concern |
| planner purity | side-effecting sequence mixing policy collector, registry, and planner | `plan_selected_model_memory` is a pure function of (inputs, facts, policy) | deep modules, honesty |
| parked subsystems | `BOUNDED_PACKED_TILES`, measurement authority, KV ring, policy collector | unchanged, listed as parked gates | honest-ratcheting (parked pieces stay loud) |

## 10. Principles alignment

The design honors the durable knowledge-base principles (`/home/ubuntu/knowledge_base/principles`) and the repo's
established TG pattern. Every claim cites where in this scope it lands.

| principle | how the design honors it |
| --- | --- |
| minimization: smallest honest surface, delete what you don't run | deletes `_cov`, both `_est_fp16` expressions, `resident_fp16_admit`, and the replan loop; adds no new table, registry, or control plane (section 8) |
| codebase organization: deep modules, locality, restraint | `plan_selected_model_memory` becomes one pure function called once (5.2); KV keeps its one question (5.4); no premature abstraction — the tri-state becomes data, not a new type system (5.2) |
| consistency beats cleverness | policy-as-data reuses the existing `immutable_prefill_policy` shape; no second candidate language; the same coverage list feeds both GGUF and state-dict paths (5.1) |
| honest ratcheting: runnable artifact green at every checkpoint; parked pieces are loud labeled gates | S0 landed and measured (decode 34.8x, correctness-qualified); each slice keeps the NV e2e green (section 7); the prefill delta is parked and labeled, not explained (section 2); a registry miss becomes a census entry, never a silent `None` (5.3) |
| narrow optimization collapse: locality is an assumption, probe off-target, gates during not after | each slice's verification runs the whole-model NV bench plus unit tests, so a routing change cannot hide behind a green planner; the decode win is measured alongside the prefill delta, not cherry-picked (section 2) |
| tell-it-or-show-it: facts are written down, not duplicated | coverage roles and bytes are data computed once (5.1); capability is read from existing facts (`supports_tensor_cores`, `supported_dtypes`), never re-derived (5.3) |
| constrained memory: know exact cost, spend the scarce tier only where it pays | one `fp16_spend_gb` ledger makes the overlay + KV composition an explicit budget question (5.4); overlay feasibility stays in the planner against the scanned budget |
| TG3 / one authority per concern | capability (renderer/device facts) and promotion (BoltBeam registry) are separate questions, mirroring `QKPrimitiveRouteAdmission.admitted` for Q4_K (5.3); authority table in section 6 |
| authored-machine primitives: per-chip idioms as facts, not flat translation | NV is admitted the same way AMD and Metal are — facts decide expressibility, artifacts decide promotion; no NV-specific branch anywhere in the design |
| repo rules: branch -> exp -> dev only after e2e proof | all work lands on `nvidia-bringup-20260731`; selective merge to `exp`; `dev` only after NVIDIA is proven end-to-end (header) |

The single claim the design rests on is testable: after S1, inventory-derived `covered_fp16_bytes` must equal the
`_prefill_v2_covered()` model walk on a fixture arch, so "the two authorities can drift" stops being possible by
construction, not by convention.
