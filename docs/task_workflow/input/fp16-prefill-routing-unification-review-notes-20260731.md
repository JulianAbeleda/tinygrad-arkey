# FP16 prefill routing unification: review notes

Date: 2026-07-31
Reviewing: `c5b8365ea` (scope) + `faad43264` (review brief)
Branch: `nvidia-bringup-20260731`

Verdict: **diagnosis accepted, land it, with S3 changed.** F1-F5 are real and correctly named, the authority
table (scope section 6) is the right artifact, and the restraint in section 8 is the best judgement in the
document. Notes below are ordered by what should change before code is written.

---

## R1 (blocking) - S3 converts a graceful fallback into a load failure

**Claim, verified by reading, not inferred.**

`automatic_promoted_prefill_graph_policy` (`prefill_candidate_runtime.py:248`) keys on
`(backend, arch, wave_size)` plus exact inventory shapes. It performs **no VRAM check**. Its own docstring
says so:

> "This is structural admission only. The model's existing memory planner must still admit the complete fp16
> overlay before installing this policy."

S3 passes that policy into a single planner call, so:

1. `explicit_overlay = True` (`admission.py:360`, S3 form)
2. `override = FULL_RESIDENT_OVERLAY` (`admission.py:381`)
3. `plan_prefill_memory` treats override as **restrict, not prefer** (`prefill_memory_plan.py:138` docstring;
   `:153` emits `"excluded by explicit strategy override"`), so `direct-packed-baseline` leaves the candidate set
4. If the overlay does not fit the scanned budget, `feasible` is empty -> `decision = REFUSE`
   (`prefill_memory_plan.py:174`) -> `RuntimeError(f"{inp.model_label}: memory plan refused load: ...")`
   (`admission.py:383-384`)

Today's replan loop is precisely what prevents this: `overlay_requested=None` probes feasibility first, and
overlay is forced **only if** `FULL_RESIDENT_OVERLAY in feasible_strategies` (`model.py:1115-1118`). The
"three-hop sequencing" that scope section 3.5 identifies as a smell is load-bearing.

**Reachable case.** 8B on gfx1100 (artifact shapes match, so the registry returns a policy) at a context large
enough that `packed_weights + dense_fp16_overlay + kv_cache + prefill_activations + flash_scratch` exceeds the
24 GB scanned budget. Today: packed, loads, slower. After S3: refuses to load. The regression is not a slowdown,
it is a dead load.

**Fix.** Do not collapse the hops by forcing. Change the contract so the caller never has to force:

```
plan_prefill_memory(...) -> tuple[CandidateDecision, ...]   # pure: what fits. no decision, no override
choose(feasible, policy, capability) -> Strategy            # ranking: what wins
```

The promotion policy becomes a **preference over the feasible set**, not a restriction on the candidate set.
This is strictly simpler than the proposed S3, not more complex: `decision`, `override`, the tri-state, the
`allowed` filter, and the REFUSE-via-override path all delete, and F3 ("the decision should be a function") is
actually killed rather than relocated. `REFUSE` survives only for its honest meaning: nothing fits.

If you prefer to keep `policy: Mapping|None` as the interface shape, the minimum acceptable version is an
explicit, tested rule: *an infeasible policy degrades to the best feasible candidate and records a census entry.*
Today's three hops give you that behavior for free; one call must not lose it silently.

**Test to add either way:** overlay-strategy policy + a budget too small for the overlay -> plan returns
`DIRECT_PACKED_FALLBACK` with a labeled reason, and the load succeeds. This test fails on S3 as currently
written, which is the point.

---

## R2 - `supports_tensor_cores` is a performance test wearing a capability costume

S4 defines capability as `supports_tensor_cores` (`device_facts.py:230`) + fp16 in `supported_dtypes()`.
But `FULL_RESIDENT_OVERLAY` does not need tensor cores to be **expressible**: the overlay path bottoms out in an
ordinary fp16 `.linear()` (`prefill_routes.py:222`), which runs anywhere fp16 runs. Tensor cores decide whether
it is **fast**.

This re-fuses two questions one slice after F4 separated them, and it has a concrete victim: per the comment at
`device_facts.py:227`, `MetalRenderer` conditions `tensor_cores` on Apple7+. Pre-Apple7 Metal then silently
loses the overlay - the exact silent-invisibility failure F4 names, relocated rather than removed.

Pick one and say which:

- **(a)** capability = fp16 dtype only; the tensor-core question moves to promotion, where perf questions belong.
- **(b)** keep the tensor-core requirement, but the census must state *why*: "expressible; not promoted without
  tensor cores." A labeled decision, not an unreported one.

(a) is more principle-consistent. (b) is defensible if there is a measurement saying the overlay loses to packed
without tensor cores - if that measurement exists, cite it in the scope; if it does not, this is a guess and (a)
is correct by default.

---

## R3 - a report field is not loudness

Brief section 5 defaults the census to "report field only (audit surface is the report; logs are noisy)". Agreed
that logs are the wrong answer, but a field in a JSON blob nobody diffs is the silent `None` with extra steps.
It is read at the post-mortem, and narrow-optimization-collapse #7 is exactly about that ordering.

**Suggested default instead:** carry it in the bench row you already read every campaign step
(`bench/models/qwen/data/...`). Then "NV silently dropped to packed" is visible in the number you already look
at, and a future AMD regression - artifact stops matching after an inventory change - surfaces the same way,
without anyone having to remember to check.

Keep the report field too; it is the audit surface. The bench row is the gate.

---

## R4 - S6 renames the flag but keeps the naming lie

`_v2_on -> _prefill_v2_active` fixes the double meaning. Agreed on the substance. But "v2" names *chronology*,
not domain: it tells a reader when the code was written, not what it decides.

Same issue one level up: the scope is titled "fp16 routing" when all three prefill paths compute in fp16
(`prefill_routes.py:197`, `:222`). The actual axis is **weight residency** - a pre-realized fp16 copy
(`overlay`) versus Q4_K/Q6_K consumed in place (`packed`).

Since S1 and S6 are already renames, make them truthful ones:

| proposed | suggested |
| --- | --- |
| `PREFILL_V2_ROLES` | `PREFILL_OVERLAY_ROLES` |
| `_prefill_v2_active` | `_prefill_overlay_active` |
| `covered_fp16_bytes` | `overlay_bytes` (or keep, if "fp16" reads as the storage format here) |

Codebase-organization principle V.1: the module graph should describe the domain, not the implementation
technique or its version history. Worth noting that the smearing this scope exists to fix plausibly happened
*because* there was no name for the thing being decided.

---

## R5 - the scope is admission-side only; label the gap

Neither document opens `tinygrad/llm/prefill_routes.py`, and the authority table (section 6) has no row for
"who dispatches the call." The dispatch side still re-derives per linear, per layer, per forward: a contextvar
lookup, an env parse (`prefill_route_mode`), and a full structural re-validation of facts that were immutable at
load (`_attached_production_route`, `prefill_routes.py:49`).

This is **correctly out of scope** - it is not needed to make NV reachable, and it touches the proven AMD path.
But it should be a labeled park in section 8, not an unmentioned hole, so nobody later reads the authority table
as complete. Suggested wording:

> Prefill **dispatch** (`prefill_routes.py`): the per-call route branch, env override, and structural
> re-validation are unchanged by this scope, which governs only how the decision is *made*. Binding the route
> as load-time data rather than a per-call branch is a separate scope.

Note for whoever picks that up later: it is a comprehensibility and authored-line argument, **not** a perf one.
Prefill is one forward pass; ~250 Python dispatches is not measurable. Do not let it into a scope on perf
grounds without a measurement.

---

## Accepted as written

- **S1** - coverage as data. The ratchet test (inventory-derived bytes == `_prefill_v2_covered()` walk on a real
  Qwen3 8B GGUF, not a toy arch) is the right probe and can genuinely fail. The choice of a real fixture over a
  toy arch is correct and worth keeping when it gets inconvenient.
- **S2** - delete `resident_fp16_admit`. Grep confirms `admission.py:215-232`, `:362`, `model.py:1108`, `:1203`
  are the only sites and none read it. Safe deletion; the brief's reviewer question is answered yes.
- **S5** - one fp16 spend ledger. No new candidate language, no new planner; `_resolve_max_context_admission`
  keeps its one question. Correct restraint.
- **Section 8 non-goals.** Carving a routing cleanup out of a live NVIDIA campaign and refusing to bundle bounded
  tiles, the NV artifact, and the 14B Metal path is the best judgement in the document.

## Brief reviewer questions still open

- **S1:** whether the inventory role vocabulary (`normalize_route_role`, `QK_ROUTE_ROLES`,
  `model_facts.py:49-68`) covers every `_PREFILL_V2_LINEARS` name including the `_shexp` variants - not checked
  in this review. This is a prerequisite for S1, not a consequence of it: if a role does not resolve, the
  equality test fails and the cause will look like a byte-math bug.
- **Commit prefix:** `[refactor]` is fine.

## Suggested revision order

1. Rewrite S3 per R1 (contract change: restrict -> prefer). This is the only blocking item.
2. Settle R2 (a) or (b) and state the choice in scope section 5.3.
3. Fold R3 into the S4 verification row and R4 into S1/S6.
4. Add the R5 park to section 8.

---

## Resolution (maintainer, 2026-07-31)

All four revision steps accepted and folded into `fp16-prefill-routing-unification-scope-20260731.md` and the
review brief; committed with these notes.

- **R1 (blocking), accepted and verified.** The chain was re-read in code: `plan_prefill_memory` docstring
  "overrides restrict but never bypass checks", `allowed` filter excludes the baseline, `REFUSE if not feasible`.
  `test_explicit_overlay_cannot_bypass_shared_byte_budget` (8/8 passing) proves forced overlay + undersized
  budget raises today, i.e. the replan loop is load-bearing. S3 is rewritten as a **preference over the feasible
  set**: infeasible overlay degrades to the best feasible candidate with a labeled reason, never REFUSE;
  `plan_prefill_memory` semantics untouched; the old budget-bypass test is replaced by the degradation test, and
  the census assertion lands in the same commit as the S3 change.
- **R2 (a), accepted.** Capability = fp16 dtype only; tensor cores moved to promotion/perf. Verified the overlay
  bottoms out in an ordinary fp16 `.linear()` (`prefill_routes.py:222`) and the pre-Apple7 Metal victim is real
  (`device_facts.py:227`). No measurement exists for the no-TC overlay-vs-packed question, so (a) by default.
- **R3, accepted.** Census carried in the admission report (audit) AND the e2e bench row (the gate read each
  campaign step). No log line.
- **R4, accepted with one correction.** `PREFILL_OVERLAY_ROLES` and `overlay_bytes` adopted. The runtime flag is
  **folded to `True`**, not renamed: it is True for every executed strategy including `DIRECT_PACKED_FALLBACK`,
  so `_prefill_overlay_active` would be a new lie. The admission-time capability keeps the `v2_on` name.
- **R5, accepted.** Dispatch park added to scope section 8 and brief non-goals with the suggested wording,
  including the "comprehensibility, not perf" warning verbatim.
- **`_shexp` question, answered.** Alias table at `tinygrad/llm/roles.py:9-10` maps all three `_shexp` names to
  canonical roles; `normalize_route_role` covers every `_PREFILL_V2_LINEARS` name. S1 prerequisite met.

One maintainer caveat added to R1: the existing test is the only thing asserting the "cannot force an infeasible
overlay" safety property today, so the degradation path needs its census assertion in the S3 commit itself.

---

## R6 (blocking, second round) - the one-call form sizes max_context against an overlay that is never allocated

Raised after re-reading `4359d7b55` against the code. **This defect is downstream of R1's wording**: R1 said "one
planner call", the brief implemented one call faithfully, and one call is the part that is wrong. R1's contract
change (restrict -> prefer) is correct and stands; the call-count reduction does not.

**The coupling that was missed.**

- `ContextMemoryTerms.from_inputs(inp, resident_fp16=...)` sets
  `weights = q4_bytes + (est_fp16 if resident_fp16 else 0)` (`admission.py:240-241`)
- `_plan_context_admission` passes `terms.weights` into `_resolve_max_context_admission`
  (`admission.py:260-266`), which resolves `max_context` from `budget - weights`

So `max_context` is a function of the residency decision, and the residency decision's feasibility is a function
of `max_context` (kv_cache and prefill_activations are `per_tok * max_context`, `admission.py:364-370`). This is a
**fixpoint**, and the "three hops" are the two-step evaluation of it.

**The failure under revised 5.2.** With an overlay-strategy policy preferred:

1. `overlay_preferred = True` -> context admitted as if `est_fp16` were resident
2. plan evaluates; overlay is infeasible; caller degrades to `DIRECT_PACKED_FALLBACK` (R1 fix works)
3. the returned `AdmissionPlan.max_context` is still the overlay-sized one - **the degraded packed load runs with
   a context sized against 16.4 GB that was never allocated**

Today this cannot happen: the probe call passes `overlay_requested=None`, and
`explicit_overlay = overlay_requested is True` (`admission.py:360`) is therefore **False**, so the probe and the
`False` replan both size context without the overlay. The degraded path gets the correct, larger context.

**Magnitude, 8B.** `kv_per_tok = 2 * 8 * 128 * 2 * 36` ~= 144 KB/token, so 16.4 GB of phantom overlay is ~116k
tokens of context. And when `q4_bytes + est_fp16` exceeds the budget outright, `_resolve_max_context_admission`
does not shrink - it raises (`admission.py:350`). Note the existing test already matches both refusal strings
(`"memory plan refused load|requested --max_context"`,
`test_prefill_memory_plan_integration.py:48`), which is direct evidence that the context path refuses too. So R6
is either a silent context amputation or **R1's dead load reintroduced at a different line**.

**Root cause worth recording.** Scope section 3.5 names the defect as *call count* ("up to three planner calls").
It is not. The defect is that the sequence is **side-effecting, with a registry lookup in the middle**. Call count
is a symptom of a real circularity that must be evaluated, not removed.

**Fix - keep two evaluations, make them pure:**

```
admit(inp, facts, resident_fp16: bool) -> AdmissionPlan    # pure; no registry, no policy, no side effects
overlay, packed = admit(..., True), admit(..., False)
choose(overlay if overlay_preferred and overlay_feasible else packed)
```

Two calls to a pure function is not "replanning" - it is evaluating a function over a two-element domain. F3 is
still killed: no side effects, no registry mid-sequence, no `None -> True -> False` imperative loop, and the
promotion lookup stays a caller concern. R6 cannot exist in this shape because each candidate carries its own
context admission.

**Test the current degradation test does not catch.** As specced, the S3 degradation test asserts only the
selected strategy, so it passes while R6 is live. It must also assert the context:

```
overlay-preferred policy + budget too small for the overlay
  -> effective is DIRECT_PACKED_FALLBACK          (R1: no REFUSE)
  -> max_context == admit(resident_fp16=False).max_context   (R6: context not amputated)
```

**Revision order, second round:**

1. Re-spec S3 per R6 (two pure evaluations, not one call). Scope 5.2 and brief S3.
2. Add the `max_context` assertion to the S3 degradation test.
3. Correct scope section 3.5 to name the defect as side-effecting sequencing rather than call count, so a future
   reader does not re-derive "fewer calls is better" and land here again.

S1, S2, S4, S5, S6 are unaffected by R6.
