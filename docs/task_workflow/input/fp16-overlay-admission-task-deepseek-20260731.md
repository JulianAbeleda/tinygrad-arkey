# TASK (deepseek): fp16 overlay admission unification — S1–S6

Read this ENTIRE file before touching anything. This is **admission/planner refactor work**, not codegen and not
kernels. You will work in three Pieces, each ending at a **HARD STOP** for review. Do not run ahead.

Repo: `/home/ubuntu/tinygrad-arkey` · Python: `/home/ubuntu/tinygrad-arkey/.venv/bin/python` ·
Branch: `nvidia-bringup-20260731`

## §0 Authority order — read this first

Three documents govern this task. When they disagree, **later in this list wins**:

1. `docs/task_workflow/input/fp16-prefill-routing-unification-scope-20260731.md` — the design (sections 5, 6)
2. `docs/task_workflow/input/fp16-prefill-routing-unification-review-brief-20260731.md` — the slices S1–S6
3. `docs/task_workflow/input/fp16-prefill-routing-unification-review-notes-20260731.md` — the review, R1–R6

**The scope's section 5.2 and the brief's S3 are SUPERSEDED by R6 in the review notes.** The corrected S3 spec is
§4 of this file. Do not implement 5.2/S3 as written in the scope — it contains a known defect that the review
caught after the scope was updated.

## §1 HARD BANS — violating any is task failure

1. ❌ **No commits to `master`, `dev`, or `exp`.** All work on `nvidia-bringup-20260731`. Commit prefix
   `[refactor]`.
2. ❌ **No new subsystem.** No new registry, no new selection table, no new control plane, no new env var, no new
   flag. This task is a net deletion. If your diff adds a module, you have misread it.
3. ❌ **Do not touch `tinygrad/llm/prefill_routes.py`** or anything on the per-call dispatch path. That is an
   explicitly parked scope (scope section 8). This task governs how the decision is *made*, never how it is
   *dispatched*.
4. ❌ **Do not touch** `BOUNDED_PACKED_TILES`, `_MEMORY_ADAPTIVE_MEASUREMENT_AUTHORITY`,
   `select_memory_adaptive_runtime_policy`, KV ring/streaming, or `automatic_promoted_prefill_graph_policy`'s
   *authority* (you may change who calls it and what is done with its result; you may not change what it returns).
5. ❌ **Do not "simplify" the two-evaluation structure in §4 back into one call.** That is the exact defect R6
   documents. If you find yourself thinking "this could be one call", re-read §4 and R6.
6. ❌ **No conclusions without artifacts.** Every claim is a pytest result, an exact exception
   `type + file:line + traceback`, or a measured number. Never write "this should be fine" — run it.
7. ❌ **Do not delete a test to make a change pass.** `test_explicit_overlay_cannot_bypass_shared_byte_budget` is
   *replaced* by a specified test in the same commit (§4), not removed.

## §2 What is already established — do NOT re-derive

Verified by review, in code, on this branch:

- `automatic_promoted_prefill_graph_policy` (`prefill_candidate_runtime.py:248`) keys on
  `(backend, arch, wave_size)` + exact inventory shapes and does **no VRAM check**. Its docstring says the memory
  planner must still admit the overlay.
- `plan_prefill_memory`'s `override` **restricts** the candidate set (`prefill_memory_plan.py:138` docstring,
  `:153`). An infeasible forced overlay therefore yields `REFUSE` → `RuntimeError` (`admission.py:383-384`).
- `ContextMemoryTerms.from_inputs` sets `weights = q4_bytes + (est_fp16 if resident_fp16 else 0)`
  (`admission.py:240-241`), and `max_context` is resolved from `budget - weights`
  (`admission.py:260-266`). **`max_context` depends on the residency decision.**
- Feasibility depends on `max_context` (kv_cache and prefill_activations are `per_tok * max_context`,
  `admission.py:364-370`). Residency ↔ context is a **fixpoint**; today's "three hops" evaluate it in two steps.
- `resident_fp16_admit` has exactly four sites (`admission.py:215-232`, `:362`, `model.py:1108`, `:1203`) and
  **no reader**. Deletion is safe.
- All ten `_PREFILL_V2_LINEARS` names (`model.py:884`) resolve via the exact-leaf alias table
  (`roles.py:8-11`). `normalize_program_role` (`roles.py:20-24`) is exact-leaf only — it does **not** substring
  match, so MoE expert tensors (`ffn_gate_exps`) are correctly excluded. S1's prerequisite is met.
- Capability is fp16 dtype only (R2a resolution). Tensor cores are a **promotion/perf** question, not a
  capability question — the overlay bottoms out in an ordinary fp16 `.linear()` (`prefill_routes.py:222`).
- **Correction (review, before execution): the fp16 capability needs a facts owner.** `v2_on` must read a
  published device-fact, not call a renderer method at admission time. Add `supports_fp16: bool | None = None`
  to `DeviceCapabilities` (`device_facts.py:40`) and publish it in the capabilities dict of the device-facts
  scan as `"supports_fp16": None if renderer is None else (dtypes.half in renderer.supported_dtypes())`,
  following the exact `supports_tensor_cores` pattern (`device_facts.py:225-230`). S4 then computes
  `v2_on = capabilities.supports_fp16 is True`.

## §3 PIECE 1 — S1 + S2 (safe, independent). HARD STOP after.

### S1 — coverage as data

Per scope 5.1, with the R4 names:

- Move `_PREFILL_V2_LINEARS` (`model.py:884`) to `PREFILL_OVERLAY_ROLES` in `tinygrad/llm/model_facts.py`.
- Add `overlay_bytes` to `derive_selected_gguf_prefill_inventory` (`model.py:151`), computed once at derivation:
  sum of `row["shape"]["n"] * row["shape"]["k"] * 2` for rows whose role is in `PREFILL_OVERLAY_ROLES`, with
  `lm_head` included only under `lm_head_route == "resident_fp16"` (same boundary as `_prefill_v2_covered`,
  `model.py:917-920`).
- One shared helper `estimate_prefill_overlay_bytes(names_and_numels)` for the state-dict path.
- **Delete** `_cov` (`model.py:1078`) and both `_est_fp16` expressions (`:1105`, `:1200`).

**Required test (the ratchet):** inventory-derived `overlay_bytes` **equals** the `_prefill_v2_covered()` walk
bytes on the **real Qwen3 8B GGUF fixture**. Not a toy arch — drift is exactly what this slice eliminates, and a
toy arch cannot prove it. If the fixture is unavailable in your environment, **STOP and report**; do not
substitute a toy arch.

### S2 — delete `resident_fp16_admit`

Remove the field and all four sites listed in §2. Update `test/unit/test_prefill_memory_plan_integration.py`
constructors (`_inputs()` at `:20-23`, and `from_model_metadata` at `:31`).

### Piece 1 verification

```
.venv/bin/python -m pytest test/unit/test_prefill_memory_plan_integration.py test/unit/test_prefill_memory_plan.py \
  test/unit/test_llm_context_admission.py test/unit/test_llm_route_module_contract.py -q
```

Plus the NV 8B e2e using **the same invocation used for `044c9be17`** (check your campaign history; if you do not
have it, STOP and ask — do not invent a bench command). Strategy and first-token digits must be unchanged.

**Correction (addition): record a §9 line in `docs/bringing-up-a-new-target-20260731.md` at each of the Piece 1
and Piece 2 HARD STOPs** (raw bench JSON is gitignored; the durable record is §9). The line states strategy,
decode tok/s, and first-token digits, and marks the refactor as non-moving. Same for Piece 3's census entry as
it appears in the bench row.

**HARD STOP.** Report: the two commits, the pytest output, the e2e strategy + digits, and the `overlay_bytes`
equality number from the ratchet test.

## §4 PIECE 2 — S3, corrected per R6. HARD STOP after.

**This supersedes scope 5.2 and brief S3. Read R6 in the review notes before writing code.**

The failure being avoided has two halves:

- **R1:** a preferred-but-infeasible overlay must **degrade to packed**, never `REFUSE`.
- **R6:** the degraded packed load must get the **packed-sized `max_context`**, not a context sized against an
  overlay that is never allocated.

### Required shape

```
admit(inp, facts, *, direct_packed_supported, resident_fp16: bool) -> (AdmissionPlan, PrefillMemoryPlan, Strategy)
#   pure. no registry, no policy, no side effects. keeps the existing 3-tuple return shape
#   (Correction: the required tests assert `effective is Strategy...`, so the Strategy must travel with the plan)
overlay, packed = admit(..., resident_fp16=True), admit(..., resident_fp16=False)
choose(overlay if overlay_preferred and overlay_is_feasible else packed)
```

- `overlay_preferred = policy is not None and prefill_policy_strategy(policy) == "FULL_RESIDENT_OVERLAY"`.
- **Never pass a single-strategy `override` into `plan_prefill_memory`.** Its restrict semantics stay exactly as
  they are; the preference is applied by choosing among results, not by narrowing `allowed`.
- `REFUSE` survives only for its honest meaning: nothing fits.
- **Correction: the public entry keeps the `plan_selected_model_memory` name** with signature
  `(inp, facts, *, direct_packed_supported, policy: Mapping|None)`; it internally runs the two `admit`
  evaluations and `choose`. `admit(resident_fp16=True)` whose residency cannot fit returns a REFUSE-carrying
  plan with labeled reasons **instead of raising** (today `_resolve_max_context_admission` raises at
  `admission.py:335`/`:348` when `q4_bytes + est_fp16` exceeds budget — that would reintroduce R1's dead load
  one line lower). The user-facing raise survives only when **both** residencies refuse (nothing fits), keeping
  today's message for the packed-only case so non-overlay loads (policy None, NV today) are unchanged.
- The promotion lookup stays a caller concern in `model.py`; its result is passed in as `policy`.
- Delete the `None → True → False` imperative loop at `model.py:1115-1130`.

Two calls to a pure function is **not** replanning — it is evaluating a function over a two-element domain. F3 is
still killed: no side effects, no registry mid-sequence, no imperative re-entry.

### Required tests, in this commit (not a follow-up)

`test_explicit_overlay_cannot_bypass_shared_byte_budget`
(`test/unit/test_prefill_memory_plan_integration.py:47`) is the **only** thing currently asserting the
infeasible-overlay safety property. It is replaced, in the same commit, by:

```
test_infeasible_overlay_degrades_instead_of_refusing:
  overlay-preferred policy + budget too small for the overlay
    -> effective is Strategy.DIRECT_PACKED_FALLBACK           (R1: no REFUSE)
    -> a labeled reason naming the byte shortfall is present  (loudness)
    -> max_context == admit(resident_fp16=False).max_context  (R6: context not amputated)
```

The third assertion is the one that catches R6. A version of this test asserting only the strategy **passes while
the bug is live** — do not ship that version.

`test_explicit_safe_overlay_is_selected_and_serialized` (`:54`) semantics must survive: a feasible preferred
overlay still selects `FULL_RESIDENT_OVERLAY` and serializes it.

### Piece 2 verification

Same pytest batch as Piece 1, plus the NV 8B e2e. Decode must stay ~156 tok/s, correctness-qualified; the parked
prefill delta must be unchanged in direction.

**HARD STOP.** Report: the diff of `admission.py` + `model.py`, both test bodies, pytest output, e2e numbers.

## §5 PIECE 3 — S4 + S5 + S6. HARD STOP after.

- **S4** — capability = **fp16 dtype only** (`renderer.supported_dtypes()`), never the promotion artifact and
  **never `supports_tensor_cores`** (R2a). Registry miss with expressible capability →
  `DIRECT_PACKED_FALLBACK` plus a labeled census entry `prefill_overlay_promotion: "no-promoted-candidate"`,
  carried in **both** the admission report **and** the e2e bench row (R3). Promotion authority unchanged.
  (Correction: read the new `supports_fp16` device-fact per §2; the census field name
  `prefill_overlay_promotion` is intentional — it is R4's domain naming, not a typo of the scope's older
  `prefill_v2_promotion`. Do not "fix" it back.)
- **S5** — add `fp16_spend_gb` (overlay bytes if elected + KV bytes per the elected KV representation) to the
  admission report, with a composition test against one scanned budget. No planner changes;
  `_resolve_max_context_admission` stays the KV owner.
- **S6** — the runtime `_v2_on` (`model.py:1146`, `:1207`) is **folded to `True`**, not renamed: it is True for
  every executed strategy. Update `prefill_concrete_kv_auto_decision` and the `prefill_v2=` config consumer. The
  admission-time capability keeps the `v2_on` name.

**Required test for S4:** NV-shaped facts + no artifact → baseline + census visible in the report **and** the
bench row. AMD-shaped facts + artifact → overlay when feasible.

**HARD STOP.** Report all three diffs, pytest output, and the census entry as it appears in the bench row.

## §6 Report format

For each Piece, exactly this:

```
PIECE n
commits:     <sha> <subject>   (one line each)
pytest:      <command> -> N passed, M failed   (paste failures verbatim)
e2e:         strategy=<...> decode=<...> tok/s first_token=<digits>
artifacts:   <the specific numbers this Piece was supposed to produce>
deviations:  <anything you did differently from this file, and why>
blocked on:  <or "nothing">
```

Report numbers. Do not conclude. If you believe a slice needs a design change, **STOP and say so with the exact
failing line** — do not redesign in place. The last three reviews of this design each found a defect that looked
like an implementation detail and was not.
