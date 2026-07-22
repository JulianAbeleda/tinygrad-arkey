# HANDOFF → deepseek: switch to the multi-output primitive route; you own all of it

**Decision (2026-07-22):** we commit to the **multi-output primitive route** (`Ops.REDUCE_SLOT` — one composite REDUCE yields `(m,l,acc)`). The **two-reduce workaround is ABANDONED** (its uncommitted work is stashed as `stash@{0}` "ABANDONED two-reduce track" — leave it; do not restore). **You (deepseek) do all the work from here.** Claude reviews at the gates and re-runs your tests.

## Where we are — verified state (do NOT re-derive)
- **master HEAD = `a5a24a0d8`** — your M1-M3: `Ops.REDUCE_SLOT` added, spec + dispatch touches (ops.py, spec.py, cstyle.py, postrange.py, simplify.py, uop/__init__.py). `hasattr(Ops,'REDUCE_SLOT')` is True. Working tree CLEAN, `test/unit/test_composite_reduce.py` = **3 passed** (green baseline).
- **BUT M1-M3 is UNPROVEN for the actual goal.** The 3 green tests are Phase A's *single/independent-slot* tests. Nothing yet proves REDUCE_SLOT lets you read TWO slots from ONE reduce, or that attention works. **M4 and M5 (the real gates) are NOT done.**
- **Fully verified so far:** Phase A (composite reduce, independent slots) works; the two-reduce online-softmax math is exact in numpy (formulation sound — but that track is abandoned in favor of this cleaner one).
- **The multi-output value-model wall is real and is exactly why REDUCE_SLOT exists:** vec-dtype packing dies at `rangeify.py:423` (`size=prod(shape)//dtype.count`→0); `Ops.GEP` dies at `symbolic.py:212` (GEP-in-order folds to identity). A dedicated op is the only clean fix — do NOT retry GEP or vec-dtype.

## Your job — finish the multi-output route, end to end
Full scope: **`docs/flash-multi-output-reduce-scope-deepseek-20260722.md`** (read it). You've done M1-M3 (infra). Remaining, in order, each with a committed artifact Claude re-runs:

1. **VERIFY M1-M3 actually works** — before building on it. Prove a `REDUCE_SLOT(composite_reduce, i)` lowers and reads the right accumulator. If your M1-M3 infra can't yet do that, fix it now.
2. **M4 (the gate A.5 failed):** ONE `(ADD, MAX)` composite reduce over `arange(1..16)`, read BOTH slots via two `REDUCE_SLOT`s, assert **sum=136 AND max=16**, and assert exactly ONE `Ops.REDUCE` feeds both reads. Commit in `test_composite_reduce.py`.
3. **M5 (clean attention):** `softmax(q@kᵀ·scale)@v` as ONE 3-slot `(m,l,acc)` online-softmax composite reduce (existing `combine_fn=="online_softmax"`), read `l` and `acc` via REDUCE_SLOT, `out=acc/l`, assert `max_rel_err ≤ 1e-2` vs fp32. No QKᵀ recompute (that's the win over two-reduce), no WMMA yet.
4. **Phase B:** rangeify graph-rewrite emits this one-reduce structure from the `softmax(q@kᵀ·scale+mask)@v` subgraph; score-resident (no `T×KV` buffer); correct.
5. **Phase C:** WMMA on the QKᵀ (over Hd) and PV (over KV) contractions inside the composite reduce (resolve `postrange.py:391` single-tag).
6. **Phase D:** gate vs SDPA at `T=KV=2048` (per-kernel WMMA dump, `tm`, rel-err); if faster + correct, wire into `model.py:583-598` + 14B integration test.

## Rules (hard — these are why you succeeded when honest and failed when not)
- **No hand kernels** (`__builtin_amdgcn`/barriers/LDS/custom_kernel/flash_kernels imports). Enum + lowering + dispatch + UOp-graph construction only.
- **Prove every claim per-artifact**, never prose or aggregate grep counts. WMMA claims = actual `__WMMA` call-site dumps per kernel. Numerics vs numpy/fp32, never vs another composite reduce.
- **Run the full suite before each commit**; paste counts; keep normal reduces byte-identical (baselines: composite `3 passed`; WMMA/packed `54p/10s/5x`; `test_tiny`+sched `38/1`; ignore the 3 pre-existing `test_wmma_emitted_code_fixtures` subfails).
- **Report what does NOT work.** A partial result reported as complete is a failure (it happened before).
- **Honest fallback:** if a step is genuinely impossible without a hand kernel or breaking normal reduces, STOP and report the exact file:line + what it can't express. Do not fake, do not force.
- Commit on master, no branches, `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`, push. Single GPU — no backgrounded benches. Don't touch the existing user stashes.

## One line
**You own the multi-output route end to end. Baseline: master a5a24a0d8, REDUCE_SLOT added but UNPROVEN. Prove M1-M3, then M4 (two slots from one reduce), M5 (attention acc/l ≤1e-2), then Phase B rangeify emission, C WMMA, D gate+wire. Per-artifact proof, no hand kernels, stop honestly at a real wall. Claude re-runs your gate tests.**

---

## UPDATE (2026-07-22) — intel from the abandoned two-reduce agent that helps YOU

The two-reduce track was abandoned but its agent left two findings on master (`90a12cc20`, `829e7e9be`, both green) that apply to the multi-output route:

1. **A general composite-machinery bug is now FIXED (keep it) — you need this.** `reduce_to_acc` used to `return results[-1]`, which let DCE delete the stores of every non-surfaced slot (e.g. `m` froze at its identity forever — verified in generated ISA). Fixed by anchoring the return on `.after(*ends)` (ALL slots' ends) so `merge_reduce_ends` keeps them. **This directly enables your multi-output goal:** REDUCE_SLOT reads multiple slots, so the non-"last" slots' stores MUST survive DCE. Confirm your REDUCE_SLOT lowering benefits from / is consistent with this fix.

2. **A real wall you WILL hit at M5 — the vec-input wall.** The `online_softmax` combine takes a `vec2(score, v)` input. Building that vec input *externally at the Tensor/UOp level* is rejected at `tinygrad/uop/symbolic.py:205` (`gep_pushing` GEP-of-GEP fold): vec dtypes are only legitimate when introduced by the optimizer's own UPCAST/CONTRACT/WMMA expander, NOT as a literal externally-constructed input dtype to `Ops.REDUCE`. Documented + machine-checked in `test_online_softmax_acc_vec_input_is_walled` (3 reproductions).
   - **Implication:** a MANUAL M5 test that hand-builds the reduce with a literal vec2(score,v) input will hit this wall.
   - **But likely NOT a wall for Phase B (scheduler-emitted):** when rangeify emits the reduce, `score` (from the QKᵀ matmul) and `v` enter via RANGE-indexed loads inside the kernel, not a literal external vec input — so the scheduler path may sidestep symbolic.py:205 entirely. **Test M5 accordingly:** if the manual vec2 construction is walled, feed score and v as separate scalar/RANGE-indexed inputs to the combine rather than a pre-packed vec, OR validate the combine at the Phase-B (scheduler-emitted) level. Report precisely which path works.
