# FEEDBACK → deepseek: realign the composite reduce with the repo (machine-native, not hardcoded)

**Context:** your composite-reduce work is a good *base* — the primitive lowers, M4/M5 pass, and your blocker analysis is honest. Keep it. But it **diverged from its own Phase 0 design and from how the rest of tinygrad works**: the combine is a hardcoded special case, and composite reduces are carved *out* of the optimizer. This scope realigns it. The hardcoded version stays as the reference to diff against; the realigned version is what ships.

## Why this matters — measured against the repo's principles
(From `knowledge_base/principles/codebase-organization-principles.md`.)

1. **Consistency — "looks like it was written by one person with one set of conventions."** tinygrad's convention: *everything is a UOp flowing through PatternMatcher rewrites and one generic optimizer.* A REDUCE carries a generic op (ADD/MAX/MUL). Your changes break this two ways:
   - `reduce_to_acc` has `if composite.combine_fn == "online_softmax":` followed by the hand-coded formula (LOG2E/EXP2/MAX/correction). The combine is **code baked into the lowering, keyed on a string** — nothing else in tinygrad is shaped like that.
   - `bab4f31aa` ("skip all opts for composite kernels") and `433da4696` ("skip UNROLL collapse for composite REDUCE") special-case composite reduces **out of** the generic optimizer/expander. Every other op flows *through* it.
2. **Clear mental models — "signatures tell you about the domain, not the implementation strategy."** The domain object is *"a reduce carrying a combine."* A string tag + a baked formula is implementation-technique-shaped. The combine should be **domain data** (a UOp sub-graph), so a reader sees "a reduce + its combine," not "a special case for one string."
3. **Information hiding — deep modules.** The combine belongs behind a clean interface: `(state, element) -> new_state` as a UOp sub-graph. Online-softmax is then just *one instance* passed in, not a branch in the lowering. (This is exactly your Phase 0 design, §3: *"the combine is a UOp sub-graph, so it participates in the rewrites."*)
4. **Restraint** — generalizing here is NOT premature abstraction: it *removes* complexity (deletes the string branch + the two skip carve-outs) AND is required for the goal. So it passes the restraint test (it reduces what the next person must hold).

**The self-defeating part (the real reason to fix, not just style):** WMMA *is* an optimizer pass. `skip all opts for composite kernels` therefore **guarantees the composite reduce can never be WMMA'd** — which is why Phase 2 (composite, no WMMA) and Phase 3 (WMMA, no composite) can't merge. The carve-out isn't an unblock; it forecloses the win. Aligning with the repo (composite flows through the optimizer) is the *same thing as* unblocking the goal.

## The realignment work
Keep normal (single-op) reduces byte-identical throughout. Each step commits an artifact Claude re-runs.

- **R1 — Combine as a general UOp sub-graph (per your Phase 0 design).** Replace `combine_fn == "online_softmax"` + the hand-coded formula with a `combine: UOp` field carrying the `(state_slots, element) -> new_state_slots` graph. `reduce_to_acc` **evaluates the generic combine** against each slot's DEFINE_ACC — it must contain NO combine-specific math. Online-softmax becomes a graph constructed at the call site (a helper is fine), passed in as data. Artifact: the same M4/M5 tests pass with the online-softmax combine now supplied as a UOp graph, and a *second* trivial coupled combine (e.g. `(m, l)` only) works through the identical code path with zero new branches in `reduce_to_acc`.
- **R2 — Delete the opt carve-outs.** Remove `bab4f31aa` (skip all opts for composite) and `433da4696` (skip UNROLL). The composite reduce must flow through the same expander/optimizer as everything else.
- **R3 — Make the composite SURVIVE the optimizer generically (the real blocker, done right).** The expander's `fix_reduce_unroll` collapses the RANGE loop into a horizontal vector reduction the composite can't handle. Solve this *generically* — either the composite lowering handles the vectorized/UPCAST'd inputs correctly, or the reduce carries what the expander needs to keep the sequential-state semantics — NOT by exempting composite reduces. This is the hard part and the whole point: a composite reduce that is a first-class citizen of the optimizer, like any REDUCE.
- **R4 — Consistency + comments.** The composite REDUCE should read like the rest of tinygrad: generic op through rewrites, no string dispatch, strategic comments explaining *why* (the online-softmax sequential-state invariant, what the combine contract is) — not narrating the code.

## The realignment GATE (the single config Phase 2/3 couldn't reach)
Prove attention runs in **ONE config** that is simultaneously: (a) using the composite reduce (score-resident, no `T×KV` buffer), (b) **through the full optimizer (`TC_OPT=2`, no `NOOPT`)**, (c) correct (`max_rel_err ≤ 1e-2` vs fp32), and (d) per-kernel WMMA dump showing the QKᵀ and PV contractions on tensor cores **inside the composite-reduce kernel**. That is the realignment succeeding: the composite reduce survived the machine and got optimized like everything else. If it can't reach that config, report the exact generic blocker (file:line) — do not re-add a skip to force it.

## Rules (unchanged)
- No hand kernels (`__builtin_amdgcn`/barriers/LDS/custom_kernel/flash_kernels imports). Enum + lowering + dispatch + UOp-graph construction only.
- Prove per-artifact (per-kernel `__WMMA` dumps, numerics vs fp32/numpy — never vs another composite reduce). Report what does NOT work.
- Run the full suite each commit; keep it green + normal reduces byte-identical. Commit on master, trailer, push. Honest fallback: stop at a real generic wall and report file:line — never re-exempt composite reduces from the machine to fake progress.

## One line
**Turn the hardcoded composite reduce into a first-class citizen of the machine: combine = a general UOp sub-graph (your Phase 0 design), delete the "skip opts for composite" carve-outs, make the composite SURVIVE the expander/optimizer generically — so attention runs score-resident AND WMMA'd in ONE `TC_OPT=2` config. Consistency with the repo IS the unblock.**
