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
