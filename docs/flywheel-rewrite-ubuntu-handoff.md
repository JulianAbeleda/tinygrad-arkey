# Flywheel Rewrite — Ubuntu Execution Handoff

Audience: the dev agent on the Ubuntu AMD box. This is the execution handoff for
the parts of the flywheel rewrite that **could not be verified on the Mac**
because they touch the generation path (need the gguf model + AMD device). The
authoritative plan is `docs/flywheel-judging-rewrite-scope.md` — read it first;
this doc only covers what's left and why it's yours.

## Current state (what already landed, Mac-side)

Commits `652f0c086..05d35bea1` on `origin/master`, all green, zero regressions:
- **Scorer locked** with golden tests (`test/external/test_flywheel_scorer_golden.py`) — byte-identical scoring is now pinned. Treat these as untouchable invariants.
- **Cost-model merged** to one module (962→544 LOC), both backends preserved.
- **CLI unified** (`extra/qk_flywheel_cli.py`).
- **Row-builders partially consolidated** (shared `assemble_row()`), golden-locked — but NOT collapsed into a full factory (see "Row-builder note" below).
- **Steps 4–5 of the scope doc deferred to you.**

## Why these are yours and not the Mac's

The Mac has **no AMD device and no gguf model**, so ~6 tests fail there
pre-existing (missing `Qwen3-8B-Q4_K_M.gguf`, no `/dev/kfd`). Anything that
touches `model.generate` / the rollout harness / the shadow staging pipeline
**cannot be proven behavior-preserving on the Mac** — but it *can* on your box,
where those tests run for real. That's the whole reason this handoff exists.

## Step 0 — Sync and establish the REAL baseline

```
cd /home/ubuntu/tinygrad-arkey && git fetch -q origin && git checkout master && git merge --ff-only origin/master
.venv/bin/python -m pytest test/external/ -q 2>&1 | tail -8
```
Record the baseline. **On your box the ~6 Mac-failures should PASS** (you have
the model + AMD). That fuller green set is your real verification floor — no
test that passes here may regress. (`test_qk_loop_live.py` needs `xgboost`;
ensure it's installed via `uv sync --extra costmodel` if you want it in scope.)

## The tasks (in order, each its own commit, never commit red)

### Task A (scope step 4) — Unify generation into one module
Extract the shared env-setup + generation loop from `extra/llm_rollout.py` and
the harness child in `extra/llm_eval_harness.py` into one `generate.py`
(per the scope doc). **Preserve BOTH entry points** — the in-process loop AND
the subprocess-isolated child (that isolation is irreducible: it gives clean
per-run AMD/JIT device state and a JSON summary over stdout — do NOT remove it).
The env-ordering invariant is sacred: `DEV`/`JIT`/`QK_PRIMITIVE_STORAGE` must be
set **before** `from tinygrad import ...`. Verify: run the rollout + eval tests
that exercise real generation (they only pass on your box) and confirm byte-
identical generated tokens vs a pre-change run on a fixed seed. Commit:
`[test] NFC - unify flywheel generation into generate.py`.

### Task B (scope step 5) — Fold + split shadow
`extra/qk_flywheel_shadow.py` (~1,005 LOC) triplicates Phase 4.1/4.2/4.3 staged
eval (~80% copy-paste). Two things, in order:
1. **De-triplicate** the staging into one generic staged-eval driven by a stage
   list (the `freeze_*`/`run_*`/`build_*_outcomes`/`score_*` repetition).
2. **Then split** the file into dataset/score/train/report siblings per the
   scope's monolith plan (this was the S4 split deferred earlier for the same
   model/AMD reason).
Its test `test/external/test_qk_flywheel_phase4.py` has tests that only pass with
the model — those are your byte-proof that the fold/split is NFC. If any can't be
made green, DEFER that sub-step and report. Commit each: `[test] NFC - ...`.

### Task C — Re-attempt the row-builder collapse (now byte-provable here)
On the Mac, the `kernel-triage-v0` builder couldn't be byte-proven because the
committed artifacts embed **this box's** absolute paths in row IDs
(`home-ubuntu-...`), which don't match a Mac checkout. **On your box they match**,
so you can finally byte-prove a `build_row(source_spec)` factory.
- BUT heed the Mac finding: the ~26 builders are **NOT clones** — each reads a
  different artifact schema/gain-convention/stage. Only collapse builders whose
  extraction is genuinely identical; for divergent ones, keep the shared
  `assemble_row()` and leave them. Conservative > complete.
- The golden test `test_flywheel_dataset_golden.py` is your gate — stay byte-
  identical. Commit: `[test] NFC - collapse identical row-builders into factory`.

### Task D — Fix the portability bug (do this regardless)
`kernel-triage-v0` embeds an **absolute machine path in `accepted_runtime` row
IDs** — the same absolute-paths-in-artifacts defect the overrides forbid. Make
the row IDs repo-relative / machine-independent (derive from a stable key, not
the checkout path), regenerate the affected artifacts, and update the golden
anchor. Commit: `[test] make kernel-triage row IDs machine-independent`.

## Verification gate (every commit)
- The Mac golden tests (`test_flywheel_scorer_golden`, `test_flywheel_dataset_golden`)
  stay **byte-identical** — they are the cross-machine invariant.
- Full `test/external/` suite green at your real (model+AMD) baseline — no
  previously-passing test regresses.
- Fixed-seed generation produces identical tokens before/after Tasks A/B.
- `py_compile` clean; `git diff --check` clean. **Pull-rebase before push.**

## Discipline
- **`tinygrad/` core, renderer, codegen, uop = `[codegen]`/`[runtime]`, never
  `[test]`** (a prefix gap we already flagged — don't repeat it).
- One owning prefix per commit; `NFC` tag for behavior-free moves; never mix NFC
  with a functional change.
- **Anti-re-sprawl rule** (record in `tinygrad-coding-overrides.md`): a new
  experiment adds a *row to the source table*, not a new file or `build_*`
  function. New scoring axes extend the scorer; new backends extend the cost
  model. Copy-pasting a `main()` or row-builder is the re-sprawl — stop.

## Done definition
- Generation unified (Task A), shadow de-triplicated + split (Task B), identical
  row-builders collapsed (Task C), the path-in-row-ID bug fixed (Task D).
- All golden + reproduce-from-artifact tests byte-identical; full model+AMD suite
  green; no regressions.
- Report back: per task what landed vs deferred, LOC before/after, and the
  fuller green count so the Mac side can confirm the cross-machine invariants
  held.
