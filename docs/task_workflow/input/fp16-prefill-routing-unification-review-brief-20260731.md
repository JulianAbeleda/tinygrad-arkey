# FP16 prefill routing unification: review brief

Date: 2026-07-31

Companion to `docs/task_workflow/input/fp16-prefill-routing-unification-scope-20260731.md` (the exhaustive
current-vs-proposed design, committed `c5b8365ea`). This brief is the implementation request for review: the
process, the per-slice design, the exact code anchors, and the questions a reviewer should answer before code
is written.

## 1. Context

Repo: `tinygrad-arkey`, branch `nvidia-bringup-20260731`. The NVIDIA campaign is mid-flight: decode is proven
(4.49 -> 156.2 tok/s via the one-line `CUDARenderer.wave_size = 32` declaration, commit `044c9be17`), the 8B
prefill delta (87.4 -> 66.3 tok/s) is parked as an unisolated gate, and the fp16 overlay route for 8B is the
next prize. This refactor is the routing cleanup that makes the overlay reachable on NV; it is target-neutral
and fixes the same structure for AMD and Metal.

Branch rules: no commits to `master`; all work on `nvidia-bringup-20260731`; selective merge to `exp`; promote
to `dev` only after NVIDIA is proven end-to-end. Commit prefixes in use: `[nv]`, `[docs]`, `[bench]`, `[test]`.
This refactor proposes `[refactor]` for its slices (a new prefix; confirm or map to `[nv]`).

## 2. Process

1. Six slices (S1-S6, section 3), each landed as its own commit in dependency order S1 -> S6.
2. Per slice: unit tests for the behavior change land in the same commit; the llm unit batch must pass; slices
   touching admission/model load (S1-S4) also run the NV 8B e2e after the change (honest ratchet: decode must
   stay ~156 tok/s, correctness-qualified, prefill delta unchanged in direction).
3. The registry API is deliberately untouched: `automatic_promoted_prefill_graph_policy` keeps returning `None`
   on a miss so the existing registry unit tests stay green; loudness is added at the `model.py` call site.
4. No new subsystems, no new flags, no env switches, no changes to `dev`/`master`.

Verification matrix:

| slice | unit coverage | e2e |
| --- | --- | --- |
| S1 | fixture-arch coverage-bytes equality test; inventory schema test | NV 8B e2e (strategy/digits unchanged) |
| S2 | `test_prefill_memory_plan_integration.py` constructor updates; grep for field readers | none required (pure deletion) |
| S3 | planner pure-function tests for policy=None/baseline/overlay; model.py call-count review | NV 8B e2e |
| S4 | NV-facts+no-artifact -> baseline+census; AMD-facts+artifact -> overlay when feasible | NV 8B e2e (census visible in report) |
| S5 | composition test vs one budget; report field present | none required |
| S6 | config/report consumers updated; behavior unchanged | none required |

## 3. Design per slice

### S1 - coverage as data

- Move `_PREFILL_V2_LINEARS` (`tinygrad/llm/model.py:884`) to `PREFILL_V2_ROLES` in `tinygrad/llm/model_facts.py`.
- `derive_selected_gguf_prefill_inventory` (`model.py:151`) already produces role+shape rows; add
  `covered_fp16_bytes` to the inventory dict, computed once at derivation: sum of `row["shape"]["n"] *
  row["shape"]["k"] * 2` for rows whose role is in `PREFILL_V2_ROLES`, with `lm_head` included only under an
  explicit `lm_head_route == "resident_fp16"` policy (same boundary as `_prefill_v2_covered`, `model.py:917-920`).
- State-dict path: one shared helper `estimate_prefill_fp16_bytes(names_and_numels)` resolving roles like the
  inventory; both `_est_fp16` expressions (`model.py:1105`, `:1200`) and `_cov` (`:1078`) are deleted.
- Ratchet test: inventory-derived `covered_fp16_bytes` equals `_prefill_v2_covered()` walk bytes on the real
  Qwen3 8B GGUF fixture (not a toy arch).

Reviewer questions: does the inventory role vocabulary (`normalize_route_role`, `QK_ROUTE_ROLES`,
`model_facts.py:49-68`) cover every `_PREFILL_V2_LINEARS` name including the `_shexp` variants? Is the
state-dict helper's role resolution provably equivalent to the GGUF-path resolution?

### S2 - delete `resident_fp16_admit`

- Remove the field from `AdmissionInputs` (`admission.py:215-232`), `from_model_metadata`, both call sites
  (`model.py:1108`, `:1203`), and the `replace` at `admission.py:362`. It is written `False`, overwritten, and
  never read; the policy input (S3) is the decision.

Reviewer question: grep says no other reader exists; confirm before the deletion is accepted.

### S3 - one policy input, one planner call

- `plan_selected_model_memory(inp, facts, *, direct_packed_supported, policy: Mapping|None)`:
  - `policy is None` -> planner decides from feasibility (today's `overlay_requested=None`).
  - `policy` with strategy `DIRECT_PACKED_FALLBACK` -> forced baseline (today's `False`).
  - `policy` with strategy `FULL_RESIDENT_OVERLAY` -> forced overlay (today's `True`).
  - `explicit_overlay = policy is not None and prefill_policy_strategy(policy) == "FULL_RESIDENT_OVERLAY"`.
- `model.py` collapses the replan loop (`:1093-1130`, up to three calls) to one call; the promotion lookup
  stays a caller concern whose result is passed in.

Reviewer questions: `test_explicit_safe_overlay_is_selected_and_serialized` semantics must survive
(`plan.decision is FULL_RESIDENT_OVERLAY`, serialized payload). Does the tri-state-as-data shape preserve
`candidate_controlled` route coverage and the `plan.decision is None` deferred-selection case?

### S4 - capability from facts, loud census

- `v2_on` becomes a capability derived from existing device facts: `supports_tensor_cores` (`device_facts.py:228`)
  and fp16 in `renderer.supported_dtypes()`; never from the promotion artifact.
- On a registry miss with expressible capability, `model.py` records a labeled census entry in the admission
  report (`prefill_v2_promotion: "no-promoted-candidate"`) and continues `DIRECT_PACKED_FALLBACK`.
- Promotion authority unchanged (`prefill_candidate_runtime.py:248-269`, pinned AMD artifact at `:179`).

Reviewer questions: this is the only user-visible behavior change; confirm it cannot flip any existing AMD/Metal
path. Should the census also surface as a log line, or is the report field sufficient?

### S5 - one fp16 spend ledger

- Add `fp16_spend_gb` to the admission report: overlay bytes (if elected) + KV bytes (per the elected KV
  representation), with a composition test against one scanned budget. `_resolve_max_context_admission`
  (`admission.py:270-325`) stays the KV owner; no planner changes.

### S6 - truthful runtime naming

- Rename the runtime flag `_v2_on` (`model.py:1146`, `:1207`) to `_prefill_v2_active` (it is True for every
  executed strategy) and update consumers (`prefill_concrete_kv_auto_decision`, `prefill_v2=` config). No
  behavior change.

## 4. Non-goals (reviewer should hold us to this)

- `BOUNDED_PACKED_TILES`, the measurement authority (`_MEMORY_ADAPTIVE_MEASUREMENT_AUTHORITY`),
  `select_memory_adaptive_runtime_policy`, KV ring/streaming: untouched.
- The NV promoted prefill candidate artifact (BoltBeam export + policy collector): out of scope; S4 only makes
  the miss loud so the route becomes reachable once an artifact exists.
- 14B Metal-shaped path and layer-resident-overlay lifecycle: separate scopes.
- No env-switch, no new flag, no promotion to `dev`/`master`.

## 5. Open questions with defaults

| question | default |
| --- | --- |
| `policy: Mapping|None` vs a typed wrapper | plain Mapping (repo convention is `immutable_prefill_policy` dicts; a typed wrapper is premature) |
| census as report field only vs log line too | report field only (audit surface is the report; logs are noisy) |
| coverage-equality test on toy arch vs real GGUF | real Qwen3 8B GGUF fixture (drift is exactly what we are eliminating; a toy arch cannot prove it) |
| commit prefix for refactor slices | `[refactor]` unless reviewer prefers mapping to `[nv]` |

## 6. Files a reviewer should open

- `tinygrad/llm/model.py` (884, 896, 917-920, 1078, 1093-1130, 1105/1108, 1146/1207, 1200/1203)
- `tinygrad/llm/admission.py` (215-232, 240-241, 260, 270-325, 352-385)
- `tinygrad/llm/prefill_candidate_runtime.py` (179, 248-269)
- `tinygrad/llm/model_facts.py` (49-68, 234)
- `tinygrad/llm/device_facts.py` (225-228)
- `test/unit/test_prefill_memory_plan_integration.py`
- `test/unit/test_promoted_prefill_candidate_runtime.py`
- `test/unit/test_memory_adaptive_route_manifest.py`
- `test/unit/test_llm_route_module_contract.py`
- `test/unit/test_qk_capability_policy_gate.py`
