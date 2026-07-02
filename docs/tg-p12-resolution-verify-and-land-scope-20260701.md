# TG-P12 Resolution Scope: Verify And Land Manual-Accumulator Widening

Date: 2026-07-01.

Goal: resolve Claude's interrupted TG-P12 attempt safely. The working tree contains a default-off compiler fix for manual `END/AFTER` scalar-REG accumulators. Codex must treat it as **untrusted code** until it passes the microgate, BoltBeam classifier, and default-off route regression gates.

This is a verify-and-land scope, not a new design scope.

## Current Situation

Claude implemented a gated fix but could not run the Python verification ladder because its command safety classifier was unavailable.

Intentional working-tree changes reported by Claude:

- `tinygrad/codegen/late/devectorizer.py`
  - adds `reduce_acc_upcast_fix` and `pm_reduce_acc_upcast_fix`;
  - imports `AxisType`;
  - rewrites manual `END/AFTER` scalar-REG accumulator stores into widened accumulator storage plus horizontal reduce;
  - supports `ADD`, `MAX`, `MUL`;
  - is intended to fail closed.
- `tinygrad/codegen/__init__.py`
  - wires the pass under `REDUCE_ACC_UPCAST_FIX=1`;
  - adds the flag to `to_program_cache`.
- `extra/qk_tg_p11_reduce_upcast_microgate.py`
  - fixes the invalid-store regex so `make_floatN(...) = ...` is actually detected.
- `bench/tg-p12-manual-end-accumulator-fold-guard/HANDOFF.md`
  - Claude's local handoff.

Important: these changes are **not validated** until Codex reruns the gates.

## Updated Technical Finding

The preferred strategy in `tg-p12-manual-end-accumulator-fold-guard-scope-20260701.md` was "prevent the bad load fold first." Claude's run found that this is insufficient:

```text
The bad vector accumulator is created by expander/upcast of the reduce axis before load folding.
Even trivial sum_s x[h,s] fails at baseline.
Blocking a later load fold cannot fix the root failure.
```

So the resolution path is now:

```text
manual END/AFTER accumulator
-> widen accumulator to the real vector width
-> horizontally reduce reduce-axis lanes
-> preserve invariant output lanes
-> verify through microgate and route ladder
```

The old scalar-fold-guard document remains useful provenance, but this resolution scope supersedes its preferred strategy.

## Principles

- **Do not trust unverified code.** Re-run every gate locally before committing.
- **Tiny and gated.** `REDUCE_ACC_UPCAST_FIX=1` remains default-off.
- **No hand kernels.** No HIP/ASM/ISA kernel may be added.
- **No model-specific fix.** No Qwen, attention, `Hq`, `Hd`, `S`, or kernel-name branches.
- **Fail closed.** If accumulator layout cannot be proven exactly, do not rewrite.
- **No promotion in this scope.** Owned HIP attention remains default. Promotion is TG-P12.7/.8 only after W==D parity.
- **Protect the dirty tree.** Do not commit unrelated pre-existing dirty artifacts.

## Pre-Flight

Confirm Python works in Codex:

```bash
python3 -c "print('python-ok')"
```

Inspect the diff:

```bash
cd /home/ubuntu/tinygrad-arkey
git diff -- tinygrad/codegen/late/devectorizer.py tinygrad/codegen/__init__.py extra/qk_tg_p11_reduce_upcast_microgate.py
```

Expected uncommitted intentional files:

- `tinygrad/codegen/late/devectorizer.py`
- `tinygrad/codegen/__init__.py`
- `extra/qk_tg_p11_reduce_upcast_microgate.py`
- `bench/tg-p12-manual-end-accumulator-fold-guard/HANDOFF.md`
- `bench/tg-p12-manual-end-accumulator-fold-guard/baseline.json`
- `bench/tg-p12-manual-end-accumulator-fold-guard/summary.md`

Do not include these unrelated files in a TG-P12 commit unless they are intentionally refreshed by the verification:

- `bench/qk-decode-runtime-overhead/result.json`
- `bench/qk-search-spaces/default_route_manifest.json`
- `bench/qk-search-spaces/refuted_axes.json`
- `bench/tg-p10-reg-scalar-combine-lowering/reg_scalar_lowering.json`
- `bench/tg-p8-generated-8b-attention-parity/baseline.json`
- `bench/tg-p8-generated-8b-attention-parity/summary.md`

## Phase R0: Baseline Reproduction

Run with the fix off:

```bash
cd /home/ubuntu/tinygrad-arkey
DEV=AMD PYTHONPATH=. python3 extra/qk_tg_p11_reduce_upcast_microgate.py
```

Expected based on Claude's updated finding:

- verdict starts with `TG_P11_1_PASS`;
- all four cases fail to compile at baseline (`cfail`);
- the regex detects the invalid `make_floatN(...) = ...` source when present.

If only the older two-case failure appears, record that as drift but do not treat it as fatal if the fix-on gate still passes.

Verdicts:

- `TG_P12R_R0_PASS_BASELINE_REPRODUCED`
- `TG_P12R_R0_BLOCKED_BASELINE_DRIFT`

## Phase R1: Fix-On Microgate

Run:

```bash
DEV=AMD REDUCE_ACC_UPCAST_FIX=1 PYTHONPATH=. python3 extra/qk_tg_p11_reduce_upcast_microgate.py
```

Acceptance:

- all four cases compile;
- all four cases are numeric-correct;
- no invalid `make_floatN(...) = ...` accumulator store appears;
- no NaN;
- default-off baseline remains unchanged.

Verdicts:

- `TG_P12R_R1_PASS_MICROGATE`
- `TG_P12R_R1_BLOCKED_COMPILE`
- `TG_P12R_R1_BLOCKED_NUMERIC`

If this fails, stop. Do not run route gates.

## Phase R2: TG-P10 Repro And BoltBeam Classifier

Run:

```bash
DEV=AMD REDUCE_ACC_UPCAST_FIX=1 PYTHONPATH=. python3 extra/qk_tg_p10_reg_scalar_repro.py
cd /home/ubuntu/BoltBeam
PYTHONPATH=. python3 -m pytest -q tests/test_reg_lowering.py
```

Acceptance:

- fixed TG-P10 artifact becomes reachable or at least no longer reports the invalid accumulator-store failure;
- old artifact remains classifiable as `EMITTER_BLOCKED`;
- BoltBeam tests pass;
- no schema/verdict literal drift.

Verdicts:

- `TG_P12R_R2_PASS_CLASSIFIER`
- `TG_P12R_R2_BLOCKED_TG_P10_REPRO`
- `TG_P12R_R2_BLOCKED_BOLTBEAM`

## Phase R3: Default-Off Census

Run:

```bash
cd /home/ubuntu/tinygrad-arkey
PYTHONPATH=. python3 extra/pure_machine_search_default_path_census.py --check
```

Acceptance:

- census passes;
- owned HIP attention remains default;
- `REDUCE_ACC_UPCAST_FIX` off causes no route behavior change.

Verdicts:

- `TG_P12R_R3_PASS_DEFAULT_OFF_CENSUS`
- `TG_P12R_R3_BLOCKED_DEFAULT_ROUTE_CHANGED`

## Phase R4: Protected Route Smoke Gates

Run only lightweight protected gates in this scope. Full W==D promotion remains out of scope.

Protect at least:

- Q4_K G3 decode GEMV;
- Q6_K generated coop decode;
- generated prefill schedule smoke;
- generated G5 K-only attention smoke if available;
- owned attention default smoke.

Acceptance:

- token/logit equality where the existing gate supports it;
- no hidden fallback;
- no NaN;
- no route ownership change with the fix off.

Verdicts:

- `TG_P12R_R4_PASS_ROUTE_SMOKE`
- `TG_P12R_R4_BLOCKED_ROUTE_REGRESSION`

## Phase R5: Commit Or Terminal

If R0-R4 pass, commit the code and only the relevant new/updated TG-P12 artifacts.

Suggested commit:

```text
[codegen] TG-P12 widen manual END accumulators for upcast reductions
```

Include:

```text
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

Do not commit unrelated stale benchmark/search artifacts.

If any gate fails, do **not** commit the codegen fix as-is. Instead write a terminal artifact:

- `bench/tg-p12-manual-end-accumulator-fold-guard/latest.json`
- `bench/tg-p12-manual-end-accumulator-fold-guard/summary.md`

Terminal verdicts:

- `TG_P12R_BLOCKED_MICROGATE`
- `TG_P12R_BLOCKED_CLASSIFIER`
- `TG_P12R_BLOCKED_DEFAULT_ROUTE_REGRESSION`
- `TG_P12R_BLOCKED_ROUTE_SMOKE`

## What This Does Not Do

This resolution does not promote generated 8B attention.

Promotion requires a later TG-P12.7/.8 phase:

- split-preserving generated combine;
- token/logit equivalence;
- route-bound;
- ctx512 and ctx4096 generated attention >= 98% of owned;
- rollback oracle retained;
- BoltBeam ledger updated;
- strict final purity pass.

## Codex Handoff Prompt

```text
Execute docs/tg-p12-resolution-verify-and-land-scope-20260701.md.

Claude left a default-off codegen fix in the working tree but could not run Python verification. Treat the code as untrusted. First run the P11 microgate with REDUCE_ACC_UPCAST_FIX off and on. If fix-on is not 4/4 compile+numeric, stop and write a terminal artifact. If it passes, run the TG-P10 repro, BoltBeam reg-lowering tests, default-off purity census, and protected route smoke gates. Commit only the codegen fix, microgate regex fix, and TG-P12 artifacts if all gates pass. Do not promote generated attention and do not commit unrelated dirty benchmark/search artifacts.
```
