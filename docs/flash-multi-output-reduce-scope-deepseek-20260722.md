# TASK (deepseek): multi-output REDUCE via `Ops.REDUCE_SLOT` — the clean primitive

**Parallel track.** Claude is proving a two-reduce *workaround* (recomputes QKᵀ) separately. THIS task builds the **clean** primitive: a single composite REDUCE that yields `(m, l, acc)` and lets consumers read each slot — no recompute. If this lands, attention is ONE reduce, not two. Mindset: this is doable — it's adding an op and teaching the dispatch tables about it, not inventing new math.

Repo: /home/ubuntu/tinygrad-arkey · Python: .venv/bin/python · Env DEV=AMD · commit on master, trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## §0 Why a new op, not GEP or vec-dtype (verified — do NOT retry these)
A prior attempt proved both natural encodings collide with pervasive generic assumptions:
- **vec-dtype packing** (`red.dtype = slot.vec(n)`): breaks `tinygrad/schedule/rangeify.py:423` `size = prod(x.shape)//x.dtype.count` → 0-sized buffer.
- **`Ops.GEP(red, i)`**: breaks `tinygrad/uop/symbolic.py:212` "GEP in order is removed" — `GEP(x,(0,))` folds to identity for scalar dtype, destroying the slot index.
The fix is a **dedicated `Ops.REDUCE_SLOT`** so NO existing generic pattern matches it. That is the whole point — it's addressable, survives rewrites, and is handled only where we explicitly teach it.

## §1 Design (M0 — deliverable: `docs/flash-multi-output-design-<date>.md`, no code yet)
Specify precisely:
- **Representation:** the composite REDUCE stays `Ops.REDUCE` with `arg=(CompositeReduce(slots), axes)`. Consumers read slot `i` via a new `REDUCE_SLOT(reduce, arg=i)` whose dtype = `slots[i].dtype`. The REDUCE node itself carries a composite/placeholder dtype (decide: `dtypes.void`, or the last slot's — pick what avoids the size formula cleanly).
- **Lowering:** `reduce_to_acc` already builds one `DEFINE_ACC` per slot. Add a rewrite so `REDUCE_SLOT(reduce, i)` resolves to the final value of `acc[i]` (`acc.after(end_i).index(0)`). The REDUCE itself no longer needs to `return accs[-1]` — each slot is reached via its REDUCE_SLOT.
- **The audit list:** enumerate every exhaustive `Ops`-dispatch site that must learn REDUCE_SLOT or safely pass it through (start from: `tinygrad/uop/ops.py` Ops enum + GroupOp buckets + `identity_element`; `tinygrad/uop/spec.py:190`; `tinygrad/codegen/late/devectorizer.py`; `tinygrad/codegen/late/expander.py`; `tinygrad/uop/symbolic.py`; `tinygrad/schedule/rangeify.py:423`; the renderers `tinygrad/renderer/cstyle.py` + `tinygrad/renderer/isa/`). For each: what it does with REDUCE today and what REDUCE_SLOT needs there.

## §2 Build (each an artifact; keep normal reduces byte-identical)
- **M1 — add the op.** Add `Ops.REDUCE_SLOT` to the Ops enum (near `Ops.REDUCE`/`Ops.GEP`), register in the right GroupOp buckets, add its spec rule (`spec.py`). Make a hand-built `REDUCE_SLOT(composite_reduce, i)` pass `type_verify`. Artifact: a spec/verify unit test.
- **M2 — lowering.** In `reduce_to_acc`, resolve `REDUCE_SLOT(reduce, i)` → `acc[i]` final value. Fix `rangeify.py:423` size handling for a composite REDUCE (its logical size is per-slot, not `//dtype.count`). Artifact: a composite REDUCE lowers without the 0-size assert.
- **M3 — dispatch audit.** Walk the §1 audit list; teach or safe-pass REDUCE_SLOT everywhere. Artifact: the full test suite stays green (baselines: WMMA/packed `54p/10s/5x`; `test_tiny`+scheduler `38/1`; composite `3 passed`; ignore the 3 pre-existing `test_wmma_emitted_code_fixtures` subfails).
- **M4 — MULTI-OUTPUT TEST (the gate A.5 couldn't pass).** One `(ADD, MAX)` composite REDUCE over `arange(1..16)`, read BOTH slots via two `REDUCE_SLOT`s, assert sum=136 AND max=16 — and assert there is exactly ONE `Ops.REDUCE` feeding both reads. Commit in `test/unit/test_composite_reduce.py`.
- **M5 — clean attention.** Express `softmax(q@kᵀ·scale)@v` as ONE 3-slot `(m,l,acc)` online-softmax composite reduce (the existing `combine_fn=="online_softmax"` combine), read `l` and `acc` via REDUCE_SLOT, `out = acc/l`. Assert `max_rel_err ≤ 1e-2` vs fp32 reference. No QKᵀ recompute, no WMMA yet. Artifact: the passing test + the numeric error.

## §3 Rules (hard)
- **No hand kernels** (`__builtin_amdgcn`/barriers/LDS/custom_kernel/flash_kernels imports). This is enum + lowering + dispatch work. Composite-REDUCE/REDUCE_SLOT UOp construction is required and fine.
- **Prove per-artifact**, never prose: the M4 two-slot-from-one-reduce test and the M5 attention error are the gates. Prove numerics against numpy/fp32, never against another composite reduce (no circular proof).
- **Normal (single-op) reduces must stay byte-identical.** If a change perturbs them, revert and rescope.
- **Run the full suite before each commit**; paste counts.
- **Honest fallback:** if REDUCE_SLOT genuinely cannot be made to survive some dispatch pass without a hand kernel or breaking normal reduces, STOP and report the exact file:line + what it can't express. Do not fake, do not fall back to accs[-1].
- Claude reviews from committed artifacts and re-runs the M4/M5 tests.

## §4 One-line
**Add `Ops.REDUCE_SLOT` so a single composite REDUCE exposes all its slots (dodging the GEP-identity and vec-dtype-size assumptions that blocked packing), lower it, audit the dispatch tables, then prove: two slots read from ONE reduce (M4), and clean one-reduce attention `acc/l ≤ 1e-2` vs fp32 (M5). No hand kernels, per-artifact proof, stop honestly at a real wall.**
