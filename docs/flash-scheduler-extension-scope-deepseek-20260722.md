# PROJECT SCOPE (deepseek): composite-accumulator REDUCE — the flash scheduler extension

**This supersedes** `flash-prefill-piece2-rest-task-deepseek-20260722.md` (that task completed: incremental fusion is walled; verified in `flash-prefill-finish-report-20260722.md`). This is the real remaining project — the one that reaches the measured 2.45× flash win.

**⚠️ Honest risk statement (read first):** this is a research-grade, multi-week compiler project with real chance of a fundamental blocker. It is NOT a knob tweak. Success is not guaranteed. The point of the phasing below is to surface a blocker EARLY (Phase 0 design + Phase 1 toy) before weeks are spent — and to STOP honestly at a wall rather than force a fake (§Fallback). A precisely-reported blocker at Phase 1 is a good outcome.

Repo: `/home/ubuntu/tinygrad-arkey` · Python: `.venv/bin/python` · Env: `DEV=AMD`

## §0 The problem, exactly (verified this session)
- WMMA attaches ONLY to a REDUCE op (`postrange.py:_apply_tc_opt` requires `self.reduceops`).
- Keeping the `T×KV` score resident (not spilled, the whole flash win) requires online-softmax: a KV-block pass carrying running `(m=max, l=sum, acc=Σp·v)`.
- Today that carry can only be expressed as a LOOP (PCONTIG converts REDUCE→LOOP → 0 WMMA) or a hand kernel (banned). **WMMA-needs-REDUCE vs residency-needs-carried-state are mutually exclusive — because REDUCE only supports a SINGLE scalar accumulator today.**
- **The fix (this project): make REDUCE support a COMPOSITE accumulator** `(m, l, acc)` with the online-softmax associative combine. Then the KV axis stays a REDUCE (WMMA attaches to the QKᵀ and PV contractions inside it), the score never materializes, and it's one kernel.

## §1 Why the composite-REDUCE route (a), not WMMA-in-loop (b)
- **(a) composite-accumulator REDUCE (CHOSEN):** extends the EXISTING REDUCE primitive. WMMA keeps attaching to a REDUCE (one WMMA path, centralized). Online-softmax's combine IS associative (a monoid), so it is legitimately a REDUCE, not a loop. Scheduler-native, reuses the whole TC-opt + geometry pipeline.
- **(b) WMMA-on-loop-accumulation (REJECTED):** a second, parallel WMMA mechanism (emit WMMA inside a hand-shaped loop). Decentralized, against the architecture. Do not build this.

## §2 The real code sites (verified — build here, don't guess)
- `tinygrad/codegen/late/devectorizer.py:369 reduce_to_acc` — lowers REDUCE → `DEFINE_ACC` (single reg), init to `identity_element(red.arg[0], dtype)`, update, `end`. **This is where a composite accumulator must be supported** (multiple DEFINE_ACCs / a struct acc, a custom combine, a custom identity).
- `tinygrad/uop/ops.py` — `identity_element`, `RegisterResidentAccumulator`, the REDUCE UOp and its `arg=(op, axes)` (op is currently a single ADD/MAX/MUL). The composite combine needs a representation here.
- `tinygrad/codegen/opt/postrange.py:305-501` — the TC opt, incl. the **already-present "epilogue reduction around the dot-product" hook** (305-307): it can WMMA an inner dot inside an outer reduce. This is the scaffolding that must WMMA the QKᵀ/PV contractions inside the composite reduce. `get_single_element(... tag=="TC")` at 391 assumes ONE TC reduce — will need to handle QKᵀ and PV.
- Online-softmax combine (the monoid) for reference (math only, NOT the kernel): `extra/qk/flash_kernels.py` running m/l/acc + correction. Reuse the MATH, never the kernel body.

## §3 Phases (each commits artifacts; §Fallback if blocked; see §Process for review cadence)

**Phase 0 — DESIGN (deliverable = a committed design doc, NO code).** Specify: (a) how a composite accumulator `(m,l,acc)` is represented on the REDUCE UOp (arg encoding the combine + identity); (b) how `reduce_to_acc` lowers it (multiple DEFINE_ACCs, the combine, the identity elements: m→−∞, l→0, acc→0); (c) how the online-softmax combine maps onto it; (d) how the TC opt attaches WMMA to the QKᵀ and PV contractions inside it; (e) how attention (`softmax(qkᵀ)@v`) is expressed to produce this REDUCE. Write to `docs/flash-composite-reduce-design-<date>.md`. **This is the highest-leverage gate — a wrong primitive design wastes the whole project.**

**Phase 1 — Toy composite reduce, NO WMMA.** Implement the minimal 2-accumulator reduce (e.g. carry `(sum, max)` or `(m, l)`) and prove it lowers + runs correct on a toy (`x` → one reduce producing both). Artifact: correctness vs numpy, the lowered kernel showing 2 accumulators, no separate buffer. This proves the primitive is expressible BEFORE the full online-softmax. If it can't be done, STOP — report the exact blocker in `reduce_to_acc`/REDUCE semantics.

**Phase 2 — Online-softmax as a composite reduce, NO WMMA yet.** Express `softmax(a@b)@c` via the composite `(m,l,acc)` reduce over KV. Prove: correct vs fp32 reference (`max_rel_err ≤ 1e-2`), and the `T×KV` score buffer is GONE (per-kernel buffer list). WMMA not required yet — this proves residency.

**Phase 3 — WMMA on both contractions inside the composite reduce.** Get the TC opt to WMMA the QKᵀ (over Hd) and PV (over KV) contractions inside the composite reduce. Artifact: per-kernel WMMA dump showing TWO `__WMMA` call sites, score buffer still gone, correctness held.

**Phase 4 — Gate + wire.** Measure the composite-reduce attention vs materialized SDPA at `T=KV=2048`: two-ceiling table, absolute `tm` (DEBUG=2, warm ≥200), per-kernel WMMA, `max_rel_err`. If it beats SDPA (target ~2.45×) with correctness → wire into `model.py:583-598` and integration-test 14B prefill. If not → report honest numbers, do not wire in.

## §Fallback (use it — stopping honestly is success)
STOP and write the review package if: the composite accumulator can't be represented/lowered without a hand kernel; WMMA can't attach inside the composite reduce without REDUCE→LOOP; or Phase 4 isn't faster than SDPA. Report the PRECISE code-level blocker (file:line, what the primitive can't express). Never fake, never force, never hand-kernel.

## §Process + review
- Commit each phase's artifacts. Write a running `docs/flash-composite-reduce-report-<date>.md` (per phase: commit, artifacts, correctness, per-kernel WMMA, what-doesn't-work). Claude reviews from committed artifacts.
- **STRONGLY RECOMMENDED: pause for Claude review after Phase 0 (design) and Phase 1 (toy)** before the multi-week Phase 2-3 build — a wrong primitive design or an unbuildable toy is the cheapest possible place to catch a dead end. (If instructed to run straight through, still commit Phase 0/1 artifacts so the end review can catch it.)

## §Rules (unchanged, non-negotiable)
- ❌ No hand kernels (no `custom_kernel`, UOp kernel bodies, `__builtin_amdgcn`/barriers/LDS by hand, `flash_kernels` imports). This is scheduler/codegen work.
- ⛔ Prove WMMA PER-KERNEL (dump `__WMMA` call sites + which kernel) — never aggregate grep counts. Report what does NOT work.
- ❌ Run the relevant test suite before each commit; keep it unregressed (baselines in the finish report). Single GPU lane; `tm` not wall-clock, warm ≥200. `.venv` python. Commit on master, no branches, Co-Authored-By trailer. No BEAM.

## §One-line
**Extend REDUCE to carry a composite `(m,l,acc)` accumulator (online-softmax monoid) so attention is one WMMA kernel with the score never materialized — the real 2.45× flash win. Phase 0 design first (highest-leverage gate), toy in Phase 1, residency in Phase 2, WMMA in Phase 3, gate+wire in Phase 4. Research-grade and multi-week: stop honestly at a wall, prove every WMMA claim per-kernel, no hand kernels.**
