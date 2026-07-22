# FEEDBACK → deepseek: realign the composite reduce with the repo (machine-native, not hardcoded)

## ⚠️ THIS IS A DESIGN REFACTOR — not new work, not a rewrite. Read this framing first.

You are **restructuring existing, working code to change its shape, while its behavior stays identical.** Treat it with refactor discipline:
- **Behavior-preserving.** The composite reduce must compute the *same numeric result* before and after. Correctness is the fixed invariant; only the *structure* changes (hardcoded → orthogonal/abstract).
- **The existing tests + the hardcoded version are your ORACLE.** M4/M5 and the composite tests must stay green at every step; the current hardcoded output is what you diff the refactored output against. Do NOT delete the hardcoded version until the refactored one reproduces it exactly — keep it side-by-side as the reference.
- **Incremental, green between each step.** Small commits, full suite green each time. Never a big-bang rewrite.
- **Exactly ONE intended behavior change, and it's a *removal of a restriction*, not a logic change:** the composite reduce stops being exempted from the optimizer (so it can be WMMA'd). Its numeric output is unchanged; it simply now flows through the machine like every other REDUCE.
- **You are not starting over.** The base is good (§ below). This refactor pays down the design debt in it — it does not discard it.

If at any point you find yourself rewriting the combine math or re-deriving online-softmax, stop — that's not a refactor, that's a rewrite, and the numeric oracle already exists in your own passing tests.

---

**Context:** your composite-reduce work is a good *base* — the primitive lowers, M4/M5 pass, and your blocker analysis is honest. Keep it. But it **diverged from its own Phase 0 design and from how the rest of tinygrad works**: the combine is a hardcoded special case, and composite reduces are carved *out* of the optimizer. This refactor realigns it. The hardcoded version stays as the reference oracle to diff against; the realigned version is what ships.

## Why this matters — orthogonality, centralization, modularization, abstraction

The realignment is really one idea in four lenses. Deepseek's build has exactly two violations — a **hardcoded combine** and a **"skip-opts" carve-out** — and each fails all four:

**1. Orthogonality.** Three things must vary independently: *what* is accumulated (the combine — online-softmax), *how* a reduce accumulates (the lowering), and *whether/how it's optimized* (WMMA). Deepseek coupled all three:
   - `if combine_fn == "online_softmax"` bakes the *what* into the *how* → changing the combine means editing `reduce_to_acc`.
   - `skip all opts for composite kernels` couples *being composite* to *not being optimized* → you can't be composite AND WMMA'd.
   - **Orthogonal target:** the combine is data passed in; `reduce_to_acc` is combine-agnostic; the optimizer treats a composite REDUCE like any REDUCE. Each axis moves without touching the others.

**2. Centralization.** There is supposed to be **one** WMMA path (the TC opt on a REDUCE), **one** reduce lowering, **one** optimizer. The carve-out creates a **second regime** — composite reduces that bypass the central optimizer — so now there are two ways a reduce gets (or doesn't get) scheduled. And the online-softmax math lives in a bespoke spot instead of one composable place. **Target:** composite reduces go through the *same* central TC-opt/optimizer as everything; nothing forks the path.

**3. Modularization.** A combine should be a self-contained module you swap in. A string branch + inline formula is the opposite: adding a second combine means surgery inside the lowering. **Target:** a new combine is new *data* (a UOp sub-graph), with **zero** edits to `reduce_to_acc`. The test of success: a second, unrelated combine works through the identical code path with no new branch.

**4. Abstraction.** The combine should be an **interface** — `(state_slots, element) -> new_state_slots` — with online-softmax as one *instance* behind it. Deepseek has no abstraction: the online-softmax formula *is* the code. **Target:** the reduce machinery depends only on the abstract combine contract, never on the concrete online-softmax. (This is exactly your Phase 0 design §3: *"the combine is a UOp sub-graph, so it participates in the rewrites."* The implementation regressed from the design into a concrete special case.)

**These are the same principles the repo's own governing doc encodes** (`minimization-principles.md`: §III.9 "generate, never hand-write"; §III.10 "rules-as-data, not hand-coded passes"; §III.12 "don't fight the compiler / never rebuild it to escape it"; §II.4 "never add an op you can compose"). Orthogonality/centralization/modularization/abstraction are *why* those rules exist.

**And it's self-defeating, not just impure:** WMMA *is* an optimizer pass, so `skip all opts for composite` **guarantees the composite reduce can never be WMMA'd** — which is exactly why Phase 2 (composite, no WMMA) and Phase 3 (WMMA, no composite) can never merge. The orthogonality violation *is* the blocker. Restoring orthogonality (composite flows through the machine) *is* the unblock. Aligning with your principles and reaching the flash win are the same act.

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
