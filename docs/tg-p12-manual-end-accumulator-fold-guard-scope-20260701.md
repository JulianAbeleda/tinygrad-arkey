# TG-P12 Scope: Manual END-Accumulator Fold Guard

Date: 2026-07-01.

Goal: unblock the final pure-machine-search route by fixing the compiler invariant that TG-P11 reduced to a minimal repro: a manual `END/AFTER` scalar-REG accumulator must not receive a vectorized reduce-body value unless that value is horizontally folded or the load fold is suppressed.

This is a **codegen correctness scope**, not an attention-kernel scope. Do not write a handwritten HIP, ASM, ISA, or model-specific attention kernel. The solution must be generic, gated, reversible, and proven first on a tiny attention-free UOp repro.

## Why This Is Worth Doing

Pure-machine-search status:

- Four of five hot default routes are generated/search-owned.
- The remaining default handwritten route is 8B decode attention.
- TG-P9 solved the generated live-split tile half: ctx512 generated attention moved from 87.7% to 96.7% of owned.
- The remaining combine half is blocked by one compiler invariant, not by attention math.

If this invariant is fixed, the generated split-preserving combine becomes reachable again. If that combine then clears the W==D gate, `decode_attention_owned_two_kernel` can move from default to rollback oracle and strict final purity can pass.

## Principles For This Scope

- **Tiny first.** Start with the smallest safe fix: prevent the bad fold in scalar manual accumulators. Only attempt algebraic vector horizontal-reduction if the scalar-preserving path is too slow or cannot be made precise.
- **No hand kernels.** The output must still be generated UOp lowered through tinygrad.
- **No model constants.** No branching on Qwen, `Hq`, `Hd`, context length, or a kernel name.
- **Default-off until proven.** Use `REDUCE_ACC_UPCAST_FIX=1` while developing. Add it to the program cache key.
- **Correctness before speed.** The minimal microgate must pass before any attention route is touched.
- **Fail closed.** If pattern detection is not exact, emit a blocked verdict instead of a heuristic rewrite.
- **Rollback kept.** Owned HIP attention remains default until the full W==D and strict-purity gates pass.

## Evidence To Read First

Current terminal artifacts:

- `bench/tg-p11-reduce-upcast-accumulator/summary.md`
- `bench/tg-p11-reduce-upcast-accumulator/latest.json`
- `extra/qk_tg_p11_reduce_upcast_microgate.py`
- `bench/tg-p10-reg-scalar-combine-lowering/summary.md`
- `extra/qk_tg_p10_reg_scalar_repro.py`

Code locations:

- `tinygrad/codegen/late/devectorizer.py`
  - `load_store_folding`: line 136 area, where expanded/stacked loads become vectorized load forms.
  - `horizontal_reduce`: line 303 area, the first-class `Ops.REDUCE` path that already handles vector lanes.
  - `reduce_to_acc`: line 311 area, the path manual accumulators bypass.
- `tinygrad/codegen/late/expander.py`
  - `fix_reduce_unroll`: line 116 area, useful comparison for first-class `Ops.REDUCE`.
- `tinygrad/codegen/__init__.py`
  - devectorize pass order: line 128 area.
  - `to_program_cache` key: line 276 area.

Current refined root cause from TG-P11:

```text
Manual flash/GEMV reductions use acc.set(acc.after(s)+v, end=s), not Ops.REDUCE.
That graph has END/AFTER/STORE but no Ops.REDUCE, so reduce_to_acc/horizontal_reduce never runs.
When a contiguous reduce-body load folds to float4, scalar acc + float4 broadcasts acc.
The renderer emits invalid make_float4(acc,acc,acc,acc) = <4 partials>.
REG_STORE_DEVEC scalarizes the store but aliases all lanes into slot 0, producing NaN.
```

## Preferred Strategy

Do **not** start with the clever algebraic fix.

Start with a scalar-preserving fold guard:

```text
If a foldable contiguous load contributes to a manual scalar REG accumulator update,
do not fold that load into float2/float4 under REDUCE_ACC_UPCAST_FIX=1.
```

Reasoning:

- It is correctness-simple: it keeps the reduce body scalar, matching the already-working gmax case.
- It avoids op-specific algebra for `add`, `max`, `mul`, etc.
- It is smaller and easier to audit than widening accumulator storage for every manual idiom.
- It is default-off, so there is no default-route blast radius.
- The fexp-free combine can still win because the large saving is eliminating repeated `exp`, not necessarily vectorizing the reduce-body load.

Only if the scalar-preserving path compiles but misses speed should Claude attempt the harder op-specific horizontal-reduce rewrite.

## Phase TG-P12.0: Pin Baseline

Run before editing:

```bash
cd /home/ubuntu/tinygrad-arkey
DEV=AMD PYTHONPATH=. python3 extra/qk_tg_p11_reduce_upcast_microgate.py
DEV=AMD PYTHONPATH=. python3 extra/qk_tg_p10_reg_scalar_repro.py
PYTHONPATH=. python3 extra/pure_machine_search_default_path_census.py --check
```

Expected:

- P11 microgate reproduces the failure in `varies_upcast` / `mixed_var_inv`.
- TG-P10 repro remains classified as blocked.
- default route census still passes with owned attention default.

Write:

- `bench/tg-p12-manual-end-accumulator-fold-guard/baseline.json`
- `bench/tg-p12-manual-end-accumulator-fold-guard/summary.md`

Verdicts:

- `TG_P12_0_PASS_BASELINE_PINNED`
- `TG_P12_0_BLOCKED_REPRO_DRIFT`

## Phase TG-P12.1: Locate The Exact Fold Site

Before implementing, identify which lowering rule creates the vector value in the P11 microgate.

Candidate sites:

- `fold_expanded_index`
- `GEP after LOAD`
- `PTRCAT after LOAD`
- `split_load_store` / `correct_load_store`

Acceptance:

- Record the responsible rule in `latest.json`.
- Record the exact failing generated-source fingerprint.
- Do not change behavior yet.

Verdicts:

- `TG_P12_1_PASS_FOLD_SITE_PINNED`
- `TG_P12_1_BLOCKED_FOLD_SITE_UNKNOWN`

## Phase TG-P12.2: Implement The Scalar-Preserving Guard

Add a gated fix:

```text
REDUCE_ACC_UPCAST_FIX=1
```

Requirements:

1. Add the env var to `to_program_cache` key in `tinygrad/codegen/__init__.py`.
2. Keep the flag default-off.
3. The fix must be generic:
   - no kernel-name checks;
   - no Qwen checks;
   - no `Hq/Hd/S` constants;
   - no attention-only branch.
4. Scope the guard to the precise hazard:
   - target is a scalar `AddrSpace.REG` accumulator;
   - update is a manual loop-carried `END/AFTER` accumulator store;
   - the folded value would widen the scalar store data;
   - the scalar path is known to compile.
5. If the graph cannot prove those properties, do not rewrite. Return a blocked verdict.

Preferred implementation shape:

- Add small helper predicates in `tinygrad/codegen/late/devectorizer.py`.
- Keep helpers local and named after the invariant, not after attention.
- Avoid broad disabling of all `load_store_folding`; block only the fold that feeds the scalar manual accumulator hazard.

Do **not** use `REG_STORE_DEVEC` as the fix. It already compiles by aliasing lanes and is numerically wrong for this case.

Run:

```bash
DEV=AMD REDUCE_ACC_UPCAST_FIX=1 PYTHONPATH=. python3 extra/qk_tg_p11_reduce_upcast_microgate.py
```

Acceptance:

- all P11 microgate cases compile;
- all P11 microgate cases are numeric-correct;
- generated source has no invalid `make_float4(...) = ...` accumulator store;
- no slot-0 lane aliasing;
- default-off run remains unchanged.

Verdicts:

- `TG_P12_2_PASS_SCALAR_FOLD_GUARD`
- `TG_P12_2_BLOCKED_DETECTION_NOT_EXACT`
- `TG_P12_2_BLOCKED_MICROGATE_STILL_FAILS`

## Phase TG-P12.3: Re-Run Existing Classifiers

Run:

```bash
DEV=AMD REDUCE_ACC_UPCAST_FIX=1 PYTHONPATH=. python3 extra/qk_tg_p10_reg_scalar_repro.py
cd /home/ubuntu/BoltBeam && PYTHONPATH=. python3 -m pytest -q tests/test_reg_lowering.py
```

Acceptance:

- TG-P10 repro flips from emitter-blocked to reachable under the fixed artifact.
- BoltBeam classifier behavior remains stable:
  - old artifact: `EMITTER_BLOCKED`;
  - fixed artifact: `REACHABLE`;
  - no schema drift.

Verdicts:

- `TG_P12_3_PASS_CLASSIFIER_REACHABLE`
- `TG_P12_3_BLOCKED_BOLTBEAM_DRIFT`

## Phase TG-P12.4: Split-Preserving Combine Microgate

Only run after TG-P12.2 and TG-P12.3 pass.

Goal: prove the fexp-free / weight-sharing split-preserving combine compiles and is numerically correct with `REDUCE_ACC_UPCAST_FIX=1`.

Acceptance:

- no handwritten combine;
- route-bound generated UOp only;
- numeric close to the shipped combine reference;
- no collapse of `Hq*S` or `Hq*Hd` parallelism;
- no `REG_STORE_DEVEC` dependency.

Also measure local combine timing. This is not a promotion gate, but it decides whether the scalar-preserving guard is enough.

Verdicts:

- `TG_P12_4_PASS_COMBINE_MICROGATE`
- `TG_P12_4_REFUTE_SCALAR_GUARD_TOO_SLOW`
- `TG_P12_4_BLOCKED_COMBINE_STILL_MISLOWERS`

## Phase TG-P12.5: Secondary Fix Only If Needed

Run this phase only if TG-P12.4 says the scalar-preserving guard is too slow or cannot safely classify the fold.

Secondary approach: exact op-specific horizontal reduction for manual scalar REG accumulators.

Allowed ops for the first pass:

- `ADD`
- `MAX`

Rules:

- Match only exact scalar-accumulator broadcast forms.
- For `MAX`, `hmax(max(acc, v0..vn))` is valid.
- For `ADD`, do **not** use a heuristic if acc/contrib separation is ambiguous. Exact form only:

  ```text
  acc + horizontal_sum(contrib_vec)
  ```

- If the graph reassociates the add tree so exact separation is impossible, stop with `BLOCKED_PATTERN_NOT_EXACT`.
- Do not support `MUL` or other ops in this phase.

Verdicts:

- `TG_P12_5_PASS_EXACT_HREDUCE_REWRITE`
- `TG_P12_5_BLOCKED_PATTERN_NOT_EXACT`
- `TG_P12_5_BLOCKED_OP_UNSUPPORTED`

## Phase TG-P12.6: Route Regression Ladder

Only run after the compiler microgates pass.

Run with the fix on and off where applicable:

```bash
PYTHONPATH=. python3 extra/pure_machine_search_default_path_census.py --check
DEV=AMD REDUCE_ACC_UPCAST_FIX=1 PYTHONPATH=. python3 extra/qk_tg_p11_reduce_upcast_microgate.py
DEV=AMD REDUCE_ACC_UPCAST_FIX=1 PYTHONPATH=. python3 extra/qk_tg_p10_reg_scalar_repro.py
DEV=AMD JIT=1 REG_STORE_DEVEC=1 PYTHONPATH=. python3 extra/qk_decode_cache_identity_index_gate.py
```

Then run route-specific token / logit gates for:

- Q4_K G3 decode GEMV;
- Q6_K generated coop decode;
- generated prefill schedule;
- generated G5 K-only attention route;
- owned default attention route.

Acceptance:

- no default-off behavior change;
- no token/logit regression in protected routes;
- no NaNs;
- no new hidden fallback.

Verdicts:

- `TG_P12_6_PASS_ROUTE_REGRESSION_LADDER`
- `TG_P12_6_BLOCKED_ROUTE_REGRESSION`

## Phase TG-P12.7: Full Generated 8B Attention W==D

Only run after the regression ladder passes.

Candidate route:

```text
live-split generated tile
+ split-preserving generated combine
+ REDUCE_ACC_UPCAST_FIX=1
```

Gate:

- token/logit equivalence;
- route-bound;
- no hidden fallback;
- ctx512 and ctx4096;
- generated attention >= 98% of owned attention at both protected contexts;
- full W==D no protected-context regression.

Verdicts:

- `TG_P12_7_PASS_GENERATED_ATTENTION_PARITY`
- `TG_P12_7_REFUTE_GENERATED_ATTENTION_SPEED`
- `TG_P12_7_BLOCKED_ROUTE_CORRECTNESS`

## Phase TG-P12.8: Promotion And Strict Purity

Run only if TG-P12.7 passes.

Promotion:

- generated attention becomes default;
- owned HIP remains rollback/oracle;
- rollback flag documented;
- route manifest updated;
- BoltBeam candidate/ledger updated;
- strict census passes.

Run:

```bash
PYTHONPATH=. python3 extra/pure_machine_search_default_path_census.py --check
PYTHONPATH=. python3 extra/pure_machine_search_default_path_census.py --strict-final-default
```

Verdicts:

- `TG_P12_8_PASS_TINYGRAD_DEFAULT_PURITY`
- `TG_P12_8_BLOCKED_PROMOTION_POLICY`

## Honest Terminal Conditions

Stop and record a terminal artifact if any of these happen:

- fold site cannot be identified;
- scalar guard requires a broad global load-fold disable;
- exact manual-accumulator detection cannot be proven;
- microgate passes but route regression fails;
- combine compiles but is too slow to make generated attention competitive;
- generated attention still misses 98% after the compiler fix.

Do not weaken the 98% bar for the final purity route. Do not promote correctness-only attention.

## Expected Outcome

Most likely useful outcome:

```text
TG_P12_2_PASS_SCALAR_FOLD_GUARD
-> TG_P12_4_PASS_COMBINE_MICROGATE
-> TG_P12_7 either PASS parity or REFUTE speed honestly
```

This is solvable as a bounded codegen task. It is difficult because it touches shared lowering, but the risk is controlled by:

- default-off gate;
- attention-free microgate;
- exact pattern detection;
- route regression ladder;
- owned default retained until the final W==D pass.

## Claude Handoff Prompt

Use this exact framing:

```text
Goal: execute TG-P12 from docs/tg-p12-manual-end-accumulator-fold-guard-scope-20260701.md.

Start from TG-P11's terminal finding: manual END/AFTER scalar-REG accumulators bypass Ops.REDUCE horizontal_reduce. A foldable contiguous reduce-body load becomes float4, scalar acc + float4 broadcasts acc, and the renderer emits invalid make_float4(acc,acc,acc,acc)=...; REG_STORE_DEVEC aliases lanes and returns NaN.

Implement the preferred solution first: REDUCE_ACC_UPCAST_FIX=1 scalar-preserving fold guard for the exact manual scalar-REG accumulator hazard. Do not write a handwritten kernel. Do not special-case Qwen, attention, Hq/Hd/S, or a kernel name. Add the env var to the program cache key. Prove the fix on extra/qk_tg_p11_reduce_upcast_microgate.py before touching attention. Only attempt the op-specific horizontal-reduce rewrite if the scalar guard compiles but is too slow or cannot be made exact.

Stop at the first failed gate and write the artifact. Owned HIP attention remains default unless generated attention passes token/logit equivalence, route-bound, >=98% of owned at ctx512 and ctx4096, rollback, BoltBeam ledger, and strict final purity.
```
