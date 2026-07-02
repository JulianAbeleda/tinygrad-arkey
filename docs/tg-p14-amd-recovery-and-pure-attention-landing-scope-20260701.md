# TG-P14 Scope: AMD Recovery, Manual Accumulator Landing, And Pure Attention Finish

Date: 2026-07-01.

Goal: resume the pure-machine-search blocker from the last valid terminal state without over-trusting dirty scratch code. TG-P14 starts with AMD runtime recovery, then either lands the generic manual `END/AFTER` accumulator lowering fix or records the next exact blocker. Only after the compiler fix is proven may it reopen the split-preserving generated attention combine and attempt full default purity.

This is a **landing scope**, not a new route-search scope. The work must preserve the core rule:

```text
No handwritten attention kernel. No model-name branch. No hidden BoltBeam runtime dependency.
```

Tinygrad should be able to run after generation without BoltBeam installed. BoltBeam is the audit brain and ledger, not a runtime dependency.

## Current Starting Point

Committed references:

- `docs/tg-p13-manual-end-accumulator-inplace-rewrite-scope-20260701.md`
- `bench/tg-p13-manual-end-accumulator-inplace-rewrite/summary.md`
- `bench/tg-p13-manual-end-accumulator-inplace-rewrite/latest.json`
- BoltBeam classifier paths:
  - `/home/ubuntu/BoltBeam/boltbeam/diagnostics/reg_lowering.py`
  - `/home/ubuntu/BoltBeam/tests/test_reg_lowering.py`
  - `/home/ubuntu/BoltBeam/boltbeam/data/candidates.json`

Latest committed verdict:

```text
TG_P13_BLOCKED_AMD_DEVICE_INIT
```

Meaning:

- dirty compiler code exists, but is not trusted;
- before the final dirty patch, the P11 microgate passed with `REDUCE_ACC_UPCAST_FIX=1`;
- before the final dirty patch, the P10 shared-weight combine compiled but was numerically wrong (`rel ~= 1.82`);
- before the final dirty patch, the P10 inline-gmax combine compiled but returned NaN;
- after the final dirty patch, AMD device initialization hung before the ladder could run;
- owned HIP attention remains default.

Known dirty files that must be treated as scratch until proven:

- `tinygrad/codegen/late/devectorizer.py`
- `tinygrad/codegen/__init__.py`
- `extra/qk_tg_p11_reduce_upcast_microgate.py`
- `bench/tg-p10-reg-scalar-combine-lowering/reg_scalar_lowering.json`
- `bench/tg-p11-reduce-upcast-accumulator/invariant_microgate.json`

Known unrelated dirty artifacts that must not be bundled:

- `bench/qk-decode-runtime-overhead/result.json`
- `bench/qk-search-spaces/default_route_manifest.json`
- `bench/qk-search-spaces/refuted_axes.json`
- `bench/tg-p8-generated-8b-attention-parity/baseline.json`
- `bench/tg-p8-generated-8b-attention-parity/summary.md`

## Why TG-P14 Exists

TG-P13 was not a compiler verdict. It was an infrastructure terminal:

```text
AMD_DEVICE_INIT_HANG
```

Therefore the next run must not jump directly to commit or promotion. It must first prove:

1. the AMD device is usable again;
2. the dirty compiler patch still reproduces the expected fix-off / fix-on behavior;
3. the P10 combine-shaped repro is actually fixed, not merely compiling;
4. protected default routes do not regress;
5. BoltBeam flips the blocker from `EMITTER_BLOCKED` to `REACHABLE` only when the evidence supports it;
6. generated 8B attention reaches the speed bar before default promotion.

## Phase TG-P14.0: AMD Recovery Gate

Purpose: prove that the environment can run AMD work before testing compiler behavior.

Do not run model benchmarks or compiler ladders until this gate passes.

Check for stuck Python/AMD workers:

```bash
cd /home/ubuntu/tinygrad-arkey
ps -eo pid,ppid,stat,etime,cmd | rg "python3 -|qk_tg|AMDKFD|tinygrad" || true
```

If any process is in `D` uninterruptible sleep on the AMD path, stop and write:

```text
TG_P14_0_BLOCKED_AMD_PROCESS_STILL_STUCK
```

If no stuck process is present, run only a timeout-guarded smoke:

```bash
timeout 45s bash -lc 'DEV=AMD PYTHONPATH=. python3 - <<'"'"'PY'"'"'
from tinygrad import Tensor
print((Tensor([1.0], device="AMD") + 1).realize().numpy())
PY'
```

Acceptance:

- exits within 45 seconds;
- prints `[2.]` or equivalent;
- leaves no stuck Python process.

Artifacts:

- `bench/tg-p14-amd-recovery-and-pure-attention-landing/amd_recovery.json`
- `bench/tg-p14-amd-recovery-and-pure-attention-landing/summary.md`

Verdicts:

- `TG_P14_0_PASS_AMD_RECOVERED`
- `TG_P14_0_BLOCKED_AMD_DEVICE_INIT`
- `TG_P14_0_BLOCKED_AMD_PROCESS_STILL_STUCK`

## Phase TG-P14.1: Scratch-State Census

Purpose: make the dirty code state explicit before trusting or editing it.

Run:

```bash
git status --short
git diff -- tinygrad/codegen/late/devectorizer.py tinygrad/codegen/__init__.py extra/qk_tg_p11_reduce_upcast_microgate.py > bench/tg-p14-amd-recovery-and-pure-attention-landing/scratch_codegen.diff
```

Classify the scratch:

| class | meaning | action |
|---|---|---|
| `EMPTY` | no compiler scratch is present | rebuild from TG-P13 scope |
| `TG_P13_SCRATCH_PRESENT` | dirty implementation exists | verify before editing further |
| `DRIFTED_UNRELATED` | scratch touches unrelated codegen surfaces | stop and scope narrower |

Acceptance:

- only codegen/test files related to TG-P13 are considered for the landing;
- unrelated bench artifacts are explicitly excluded from commits;
- no destructive cleanup is performed without explicit user approval.

Verdicts:

- `TG_P14_1_PASS_SCRATCH_CENSUS`
- `TG_P14_1_BLOCKED_UNRELATED_DIRTY_SURFACE`

## Phase TG-P14.2: Reproduce The Baseline Contract

Purpose: prove that the test harness still has the same semantics.

Run with the fix off:

```bash
DEV=AMD PYTHONPATH=. python3 extra/qk_tg_p11_reduce_upcast_microgate.py
```

Expected:

- P11 baseline still fails in the known way;
- all four cases are `cfail`;
- failure is an invalid vectorized scalar-REG accumulator store, not a different compile/runtime failure.

Run with the fix on:

```bash
DEV=AMD REDUCE_ACC_UPCAST_FIX=1 PYTHONPATH=. python3 extra/qk_tg_p11_reduce_upcast_microgate.py
```

Expected:

- all four P11 microgate cases pass;
- no invalid `make_floatN(...) = ...`;
- no NaN.

Write:

- `bench/tg-p14-amd-recovery-and-pure-attention-landing/p11_microgate.json`

Verdicts:

- `TG_P14_2_PASS_P11_CONTRACT`
- `TG_P14_2_BLOCKED_FIX_OFF_DRIFT`
- `TG_P14_2_BLOCKED_FIX_ON_P11`
- `TG_P14_2_BLOCKED_AMD_RUNTIME`

## Phase TG-P14.3: P10 Fixed-Mode Repro

Purpose: prove that the real combine-shaped blocker is fixed, not just the minimal P11 reproducer.

Run:

```bash
DEV=AMD REDUCE_ACC_UPCAST_FIX=1 PYTHONPATH=. python3 extra/qk_tg_p10_reg_scalar_repro.py
```

Required case-level results:

| case | required status |
|---|---|
| shipped per-d combine control | compile ok + numeric ok |
| shared-weight combine | compile ok + numeric ok |
| inline/fused-gmax combine | compile ok + numeric ok, or a new non-accumulator blocker recorded |
| old `REG_STORE_DEVEC` NaN path | not used as a success path |

If a case compiles but returns wrong output, record the first exact mismatch:

- case id;
- max absolute error;
- relative error;
- NaN/Inf count;
- whether numerator, denominator, or max path is wrong;
- whether the generated source contains invalid or suspicious REG stores.

Artifacts:

- `bench/tg-p14-amd-recovery-and-pure-attention-landing/p10_fixed_mode.json`
- update or supersede `bench/tg-p10-reg-scalar-combine-lowering/reg_scalar_lowering.json` only if the artifact schema remains valid.

Verdicts:

- `TG_P14_3_PASS_P10_FIXED_MODE`
- `TG_P14_3_BLOCKED_SHARED_WEIGHT_NUMERIC`
- `TG_P14_3_BLOCKED_INLINE_GMAX_NUMERIC`
- `TG_P14_3_BLOCKED_COMPILE`
- `TG_P14_3_BLOCKED_NEW_NON_ACCUMULATOR_LOWERING`

Stop here if P10 does not pass. Do not proceed to attention route tests with a numerically wrong combine repro.

## Phase TG-P14.4: Fix Strategy If P10 Still Fails

Purpose: constrain the debugging path so it stays generic and small.

Allowed fixes:

1. Correct in-place replacement of the matched accumulator `STORE` while preserving the original `END` boundary.
2. Correct grouping of multiple accumulators that share the same reduce range.
3. Correct widening of manual scalar-REG accumulator storage into distinct scalar slots.
4. Correct horizontal reduction of vectorized reduce-axis lanes before storing to scalar accumulator slots.
5. A narrow devectorizer for **distinct-slot REG stores only**, not broad `REG_STORE_DEVEC`.

Forbidden fixes:

- no attention-kernel name branch;
- no Qwen/model-name branch;
- no handwritten HIP/ASM/ISA;
- no disabling broad optimizer passes globally;
- no default-on codegen change before the regression ladder passes;
- no "compile ok" success if numeric correctness fails.

If P10 numeric failure persists, write a terminal result with the exact failing lane/grouping assumption. Likely classes:

- `TG_P14_4_BLOCKED_REDUCE_LANE_GROUPING`
- `TG_P14_4_BLOCKED_MULTI_ACCUMULATOR_ALIAS`
- `TG_P14_4_BLOCKED_GMAX_ACCUMULATOR`
- `TG_P14_4_BLOCKED_UNSAFE_PATTERN_MATCH`

## Phase TG-P14.5: BoltBeam Reachability Flip

Purpose: keep the audit brain honest.

Run in BoltBeam:

```bash
cd /home/ubuntu/BoltBeam
PYTHONPATH=. python3 -m pytest -q tests/test_reg_lowering.py
```

Then normalize/classify the fixed tinygrad artifact through the existing adapter/classifier path:

- artifact kind: `tinygrad.reg_scalar_lowering.v1`;
- adapter: `boltbeam/artifacts/tinygrad.py`;
- classifier: `boltbeam/diagnostics/reg_lowering.py`;
- candidate: `decode_attention_g5_8b_refuted` or the current canonical generated-attention candidate id in `boltbeam/data/candidates.json`.

Acceptance:

- old blocked artifact still classifies as `EMITTER_BLOCKED`;
- fixed artifact classifies as `REACHABLE`;
- classifier does not infer success from missing rows;
- no schema-id or verdict string escapes `boltbeam/vocab.py`;
- candidate ledger reopen condition points to TG-P14 evidence.

Verdicts:

- `TG_P14_5_PASS_BOLTBEAM_REACHABLE`
- `TG_P14_5_BLOCKED_CLASSIFIER_FALSE_POSITIVE`
- `TG_P14_5_BLOCKED_SCHEMA_DRIFT`

## Phase TG-P14.6: Default-Off Route Regression Ladder

Purpose: prove the compiler fix does not silently corrupt other generated routes.

Run default-off census first:

```bash
cd /home/ubuntu/tinygrad-arkey
PYTHONPATH=. python3 extra/pure_machine_search_default_path_census.py --check
```

Then run the protected route gates with `REDUCE_ACC_UPCAST_FIX=1`:

| route | required check |
|---|---|
| Q4_K G3 decode GEMV | token/logit or role microgate, route-bound |
| Q6_K generated coop decode | token/logit or route microgate |
| generated prefill schedule | byte/logit equivalence on synced harness |
| generated G5 K-only attention | token/logit equivalence |
| owned HIP attention default | unchanged with fix off |
| P11/P13/P10 compiler repros | still pass |

Acceptance:

- fix-off default route behavior unchanged;
- fix-on protected routes do not regress;
- no hidden fallback;
- no NaN;
- no token mismatch;
- no route manifest drift unless explicitly justified.

Verdicts:

- `TG_P14_6_PASS_DEFAULT_OFF_REGRESSION`
- `TG_P14_6_BLOCKED_PROTECTED_ROUTE_REGRESSION`
- `TG_P14_6_BLOCKED_HIDDEN_FALLBACK`

## Phase TG-P14.7: Land The Compiler Fix

Only run if TG-P14.2, TG-P14.3, TG-P14.5, and TG-P14.6 pass.

Commit structure:

1. `[codegen]` tinygrad compiler fix:
   - `tinygrad/codegen/late/devectorizer.py`
   - `tinygrad/codegen/__init__.py`
2. `[test]` tinygrad repro/gate artifacts:
   - `extra/qk_tg_p11_reduce_upcast_microgate.py`
   - any new TG-P14 microgate tools
   - forced bench artifacts for TG-P14 only
3. BoltBeam commit only if classifier/ledger/candidate files changed.

Do not commit unrelated dirty bench outputs.

Verdicts:

- `TG_P14_7_PASS_COMPILER_FIX_LANDED`
- `TG_P14_7_BLOCKED_COMMIT_HYGIENE`

## Phase TG-P14.8: Reopen Split-Preserving Combine

Only after the compiler fix is committed.

Re-run generated-UOp combine shapes that were previously blocked:

- shared softmax-weight combine;
- inline/fused gmax combine;
- fexp-free weighted-sum combine;
- split-preserving shape that keeps both `Hq*S` and `Hq*Hd` occupancy.

Acceptance:

- generated UOp only;
- no `REG_STORE_DEVEC` dependency;
- numeric correctness vs shipped combine/reference;
- route-bound;
- combine timing improves enough to justify full W==D.

Artifacts:

- `bench/tg-p14-amd-recovery-and-pure-attention-landing/combine_reopen.json`

Verdicts:

- `TG_P14_8_PASS_COMBINE_REOPENED`
- `TG_P14_8_REFUTE_COMBINE_SPEED`
- `TG_P14_8_BLOCKED_COMBINE_CORRECTNESS`

If combine is correct but not faster, ledger it as refuted and stop. Do not promote.

## Phase TG-P14.9: Generated 8B Attention W==D

Only if TG-P14.8 passes.

Candidate:

```text
live-split generated tile
+ split-preserving generated combine
+ REDUCE_ACC_UPCAST_FIX
```

Protected contexts:

- ctx512;
- ctx4096.

Acceptance:

- token/logit equivalent;
- route-bound;
- no hidden fallback;
- generated attention is at least 98% of owned at both contexts;
- no protected-context regression over 1%;
- benchmark noise/spread is reported and not used to hide a regression.

Verdicts:

- `TG_P14_9_PASS_GENERATED_ATTENTION_PARITY`
- `TG_P14_9_REFUTE_GENERATED_ATTENTION_SPEED`
- `TG_P14_9_BLOCKED_ROUTE_CORRECTNESS`

## Phase TG-P14.10: Final Default Purity

Only if TG-P14.9 passes.

Promotion:

- generated 8B attention becomes default;
- owned HIP attention remains rollback/oracle;
- rollback flag documented;
- route manifest updated;
- BoltBeam ledger updated;
- strict final purity passes.

Run:

```bash
PYTHONPATH=. python3 extra/pure_machine_search_default_path_census.py --check
PYTHONPATH=. python3 extra/pure_machine_search_default_path_census.py --strict-final-default
```

Verdicts:

- `TG_P14_10_PASS_TINYGRAD_DEFAULT_PURITY`
- `TG_P14_10_BLOCKED_PROMOTION_POLICY`

## Final Success Definition

The full scope succeeds only if:

```text
TG_P14_7_PASS_COMPILER_FIX_LANDED
TG_P14_8_PASS_COMBINE_REOPENED
TG_P14_9_PASS_GENERATED_ATTENTION_PARITY
TG_P14_10_PASS_TINYGRAD_DEFAULT_PURITY
```

Partial success is still useful:

- `TG_P14_7_PASS_COMPILER_FIX_LANDED` means the compiler invariant is solved, even if generated attention speed later refutes.
- `TG_P14_8_REFUTE_COMBINE_SPEED` means purity remains blocked by route speed, not compiler correctness.
- `TG_P14_9_REFUTE_GENERATED_ATTENTION_SPEED` means generated attention is correct but still cannot replace owned under the 98% bar.

## Stop Conditions

Stop and write a terminal artifact if:

- AMD init hangs again;
- P11 fix-off no longer reproduces the expected failure;
- P11 fix-on fails;
- P10 fixed mode compiles but is numerically wrong;
- BoltBeam classifies missing or bad evidence as reachable;
- any protected route regresses;
- generated combine is correct but slower;
- generated attention remains under 98% of owned.

Do not weaken the final purity bar. Do not ship a slower generated attention default for aesthetic purity.

## Expected Next Result

Most likely near-term result:

```text
TG_P14_7_PASS_COMPILER_FIX_LANDED
```

That would be a real compiler win and should be committed even if final attention purity still fails later.

Most likely full-purity blocker if the compiler fix lands:

```text
TG_P14_9_REFUTE_GENERATED_ATTENTION_SPEED
```

because the codegen fix is necessary for split-preserving combine correctness, but speed parity still has to be measured.

## Claude Handoff

```text
Execute docs/tg-p14-amd-recovery-and-pure-attention-landing-scope-20260701.md.

Start with TG-P14.0. Do not run compiler/model ladders while AMD init is hanging. Treat the current dirty compiler files as unverified scratch, not a feature. Verify P11 fix-off/fix-on, then prove the P10 combine-shaped repro is numerically fixed with REDUCE_ACC_UPCAST_FIX=1. If P10 is still numerically wrong, stop and write the exact blocker. If P10 passes, run BoltBeam reg-lowering classification, default-off route regression, then commit the compiler fix. Only after the compiler fix lands may you reopen the generated split-preserving combine and attempt generated 8B attention parity. No handwritten kernels, no model-name branches, no runtime dependency on BoltBeam, and no default promotion below 98% of owned at ctx512 and ctx4096.
```
