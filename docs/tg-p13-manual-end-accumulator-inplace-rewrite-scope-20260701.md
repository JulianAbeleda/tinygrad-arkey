# TG-P13 Scope: In-Place Manual END-Accumulator Rewrite

Date: 2026-07-01.

Goal: finish the compiler-lowering fix that TG-P12 partially proved, without repeating TG-P12's failure. The fix must handle manual `END/AFTER` scalar-REG accumulators whose update store is already wrapped by an existing `END`, and it must do so by rewriting the accumulator update in-place rather than creating a second `END` over the same reduce range.

This scope is the next step toward full pure-machine-search. It is still a **codegen correctness** track first. Generated 8B attention promotion is allowed only after the compiler fix, combine microgate, W==D gate, BoltBeam ledger, and strict-purity census all pass.

## Current Terminal State

TG-P12 stopped at:

```text
TG_P12R_BLOCKED_TG_P10_REPRO
```

The evidence is committed in:

- `bench/tg-p12-manual-end-accumulator-fold-guard/summary.md`
- `bench/tg-p12-manual-end-accumulator-fold-guard/latest.json`
- `docs/tg-p12-resolution-verify-and-land-scope-20260701.md`

What TG-P12 proved:

- Baseline P11 microgate reproduces the manual accumulator failure: all four cases compile-fail with the fix off.
- Claude's `REDUCE_ACC_UPCAST_FIX=1` prototype makes the minimal P11 microgate pass 4/4.
- The same prototype fails the real combine-shaped P10 repro.

Precise TG-P12 failure:

```text
tinygrad/codegen/late/linearizer.py:162
assert y.src[1] not in x.backward_slice_with_self
```

Codex diagnostic:

```text
The matched accumulator is denominator REG 243 in flash_fused_gmax_combine_kernel.
The original accumulator update is already under END(reduce range (4, AxisType.REDUCE)).
The prototype creates a second END over the same reduce range.
CFGContext sees nested same-range END and asserts.
```

Therefore:

```text
Do not create a fresh END for an already-ended manual accumulator update.
Rewrite the matched store in-place and let the existing END remain the control-flow boundary.
```

## Dirty-Tree Warning

The working tree may still contain Claude's failed prototype:

- `tinygrad/codegen/late/devectorizer.py`
- `tinygrad/codegen/__init__.py`
- `extra/qk_tg_p11_reduce_upcast_microgate.py`

Treat those changes as a failed attempt, not as committed truth.

The next agent may repair them in place, but must not commit them unless this scope's gates pass. If starting from a clean branch is preferred, first save the diff as a scratch artifact and get explicit user approval before discarding it.

Do not commit unrelated dirty artifacts, especially:

- `bench/qk-decode-runtime-overhead/result.json`
- `bench/qk-search-spaces/default_route_manifest.json`
- `bench/qk-search-spaces/refuted_axes.json`
- `bench/tg-p10-reg-scalar-combine-lowering/reg_scalar_lowering.json`
- `bench/tg-p11-reduce-upcast-accumulator/invariant_microgate.json`
- `bench/tg-p8-generated-8b-attention-parity/baseline.json`
- `bench/tg-p8-generated-8b-attention-parity/summary.md`

## Principles

- **Generic lowering, not attention code.** No handwritten HIP/ASM/ISA and no model/kernel-name branch.
- **In-place first.** The fix must reuse existing `END` nodes when they already exist.
- **Exact detection.** If the pass cannot prove the accumulator update shape, it must fail closed.
- **Default-off first.** Keep `REDUCE_ACC_UPCAST_FIX=1` while proving correctness.
- **Minimal reproducer first.** Add a microgate for the nested-END failure before changing behavior.
- **BoltBeam decides reachability.** Tinygrad fixes/runs; BoltBeam classifier and ledger verify the compiler blocker moved from `EMITTER_BLOCKED` to `REACHABLE`.
- **No promotion until W==D.** Owned HIP attention remains default until generated attention clears the full parity gate.

## Target Invariant

Manual reductions often look like:

```python
acc = acc.after(ctx)[0].set(identity)
upd = acc[0].store(op(acc.after(r)[0], contrib)).end(r)
out = acc.after(upd)[0]
```

When the reduce body is vectorized, the store may become:

```text
STORE(STACK(reg[0], reg[0], reg[0], reg[0]), op(STACK(acc, acc, acc, acc), contrib_vec4))
```

Correct lowering must:

1. identify the manual accumulator update;
2. widen accumulator storage to the true output-lane width;
3. horizontally reduce reduce-axis lanes into each output lane;
4. preserve invariant accumulators as scalar where appropriate;
5. reuse the existing `END` if the original store is already ended;
6. never create nested `END`s over the same reduce range.

## Phase TG-P13.0: Pin Baseline And Failed Prototype

Run:

```bash
cd /home/ubuntu/tinygrad-arkey
DEV=AMD PYTHONPATH=. python3 extra/qk_tg_p11_reduce_upcast_microgate.py
DEV=AMD REDUCE_ACC_UPCAST_FIX=1 PYTHONPATH=. python3 extra/qk_tg_p11_reduce_upcast_microgate.py
DEV=AMD REDUCE_ACC_UPCAST_FIX=1 PYTHONPATH=. python3 extra/qk_tg_p10_reg_scalar_repro.py
```

Expected if the failed prototype is still present:

- P11 baseline: all four `cfail`;
- P11 fix-on: all four `ok`;
- P10 fix-on: `TG_P10_1_BLOCKED_REPRO_NOT_MINIMAL`;
- stack points to `linearizer.py:162`.

Write:

- `bench/tg-p13-manual-end-accumulator-inplace-rewrite/baseline.json`
- `bench/tg-p13-manual-end-accumulator-inplace-rewrite/summary.md`

Verdicts:

- `TG_P13_0_PASS_BASELINE_PINNED`
- `TG_P13_0_BLOCKED_REPRO_DRIFT`

## Phase TG-P13.1: Add The Already-Ended Accumulator Microgate

Add a new minimal microgate before modifying the compiler fix further.

It must reproduce the TG-P12 failure without importing attention:

```text
one accumulator update already wrapped by END(reduce_range)
same reduce range appears in the matched manual accumulator store
fix prototype creates nested same-range END
CFGContext assertion or explicit nested-END detector fires
```

Required cases:

| case | purpose |
|---|---|
| `ended_scalar_acc_add` | scalar accumulator, reduce range already ended |
| `ended_vec_acc_add` | vector body with output-lane width > 1 |
| `ended_two_acc_same_reduce` | numerator + denominator pattern, same reduce range |
| `ended_max_then_add` | gmax-like max accumulator followed by add accumulator |

The gate must check both:

- compile/numeric correctness;
- graph/control-flow safety: no nested same-range `END` created by the pass.

Suggested tool:

```text
extra/qk_tg_p13_manual_end_inplace_microgate.py
```

Suggested artifact:

```text
bench/tg-p13-manual-end-accumulator-inplace-rewrite/inplace_microgate.json
```

Verdicts:

- `TG_P13_1_PASS_NESTED_END_REPRO_READY`
- `TG_P13_1_BLOCKED_MICROGATE_NOT_MINIMAL`

## Phase TG-P13.2: Audit The Existing Prototype

Before editing, inspect the current prototype's matched accumulator rows.

For every match, record:

- REG id;
- op (`ADD`, `MAX`, `MUL`);
- target width `N`;
- accumulator init width `W`;
- reduce-lane grouping `R = N / W`;
- reduce range ids;
- whether the original store is already enclosed by `END`;
- whether a new `END` would duplicate an existing reduce range;
- whether output reads are after the original `END`, the new `END`, or both.

Write:

```text
bench/tg-p13-manual-end-accumulator-inplace-rewrite/match_audit.json
```

Verdicts:

- `TG_P13_2_PASS_MATCH_AUDIT_PINNED`
- `TG_P13_2_BLOCKED_MATCH_SHAPE_UNKNOWN`

## Phase TG-P13.3: Implement In-Place Rewrite

Fix strategy:

```text
Replace the matched STORE node itself.
Do not create a fresh END if the original STORE is already wrapped by END(reduce_range).
Let the existing END remain the control-flow boundary.
Redirect reads after that existing END to the widened accumulator read.
```

Implementation rules:

1. Keep `REDUCE_ACC_UPCAST_FIX=1` default-off.
2. Keep the env var in `to_program_cache`.
3. Keep or add local helper predicates in `tinygrad/codegen/late/devectorizer.py`.
4. The pass must distinguish:
   - raw manual accumulator stores not yet ended;
   - manual accumulator stores whose consumer is an existing `END`;
   - output reads after the existing `END`;
   - in-loop reads under the reduce range.
5. For already-ended stores, return a substitution for the `STORE` and related reads, not a new `END`.
6. For non-ended stores, creating an `END` is allowed only if the original graph had no suitable existing `END`.
7. If multiple accumulators share the same reduce range, do not create one independent same-range `END` per accumulator if that produces invalid control-flow. Either reuse/merge the existing structure or fail closed.
8. If exact output-read mapping cannot be proven, leave the graph unchanged and report a blocked verdict.

Anti-pattern from TG-P12:

```python
end = reg_wide.index(0).store(...).end(*reduce_range)
out_read = reg_wide.after(end).index(0)
```

This is unsafe when the original store is already wrapped by `END(*reduce_range)`.

Preferred shape for already-ended updates:

```text
old: END(STORE(old_scalar_reg_slot, vectorized_update), reduce_range)
new: END(STORE(wide_reg_slot, horizontally_reduced_update), reduce_range)
```

That is: replace the inner store, not the outer control-flow boundary.

Verdicts:

- `TG_P13_3_PASS_INPLACE_REWRITE`
- `TG_P13_3_BLOCKED_EXISTING_END_MAPPING`
- `TG_P13_3_BLOCKED_MULTI_ACCUMULATOR_MERGE`
- `TG_P13_3_BLOCKED_NOT_GENERIC`

## Phase TG-P13.4: Compiler Microgates

Run:

```bash
DEV=AMD PYTHONPATH=. python3 extra/qk_tg_p11_reduce_upcast_microgate.py
DEV=AMD REDUCE_ACC_UPCAST_FIX=1 PYTHONPATH=. python3 extra/qk_tg_p11_reduce_upcast_microgate.py
DEV=AMD REDUCE_ACC_UPCAST_FIX=1 PYTHONPATH=. python3 extra/qk_tg_p13_manual_end_inplace_microgate.py
```

Acceptance:

- P11 fix-off still reproduces;
- P11 fix-on passes all cases;
- P13 already-ended microgate passes all cases;
- no invalid `make_floatN(...) = ...`;
- no nested same-range `END`;
- no NaN.

Verdicts:

- `TG_P13_4_PASS_COMPILER_MICROGATES`
- `TG_P13_4_BLOCKED_P11_REGRESSION`
- `TG_P13_4_BLOCKED_NESTED_END`
- `TG_P13_4_BLOCKED_NUMERIC`

## Phase TG-P13.5: TG-P10 Repro Fixed Mode

Update `extra/qk_tg_p10_reg_scalar_repro.py` or add a companion fixed-mode gate so the same combine-shaped repro can pass when `REDUCE_ACC_UPCAST_FIX=1`.

Acceptance with fix off:

- old repro still classifies as blocked/emitter-blocked.

Acceptance with fix on:

- `shipped_per_d_combine_compiles`: compile + numeric ok;
- `shared_weight_combine_compile_fails`: now compile + numeric ok;
- `fused_gmax_combine_compile_fails`: now compile + numeric ok, or a precise non-accumulator blocker is recorded;
- `reg_store_devec_compiles_nan`: no longer needed for the fixed route and should not be used as the success path.

Suggested fixed verdict:

```text
TG_P13_5_PASS_TG_P10_REPRO_FIXED
```

Failure verdicts:

- `TG_P13_5_BLOCKED_SHARED_WEIGHT`
- `TG_P13_5_BLOCKED_FUSED_GMAX`
- `TG_P13_5_BLOCKED_NUMERIC`

## Phase TG-P13.6: BoltBeam Classifier And Ledger

Run in BoltBeam:

```bash
cd /home/ubuntu/BoltBeam
PYTHONPATH=. python3 -m pytest -q tests/test_reg_lowering.py
```

Add/update BoltBeam evidence if needed:

- old artifact stays `EMITTER_BLOCKED`;
- fixed artifact flips to `REACHABLE`;
- no schema drift;
- no duplicate verdict strings outside the central vocabulary;
- candidate reopen condition is updated from "manual accumulator lowering blocked" to either:
  - `REACHABLE_UNDER_REDUCE_ACC_UPCAST_FIX`, or
  - a new precise blocker from TG-P13.5.

Verdicts:

- `TG_P13_6_PASS_BOLTBEAM_REACHABLE`
- `TG_P13_6_BLOCKED_CLASSIFIER_DRIFT`

## Phase TG-P13.7: Default-Off Regression Ladder

Run with default flags off first:

```bash
cd /home/ubuntu/tinygrad-arkey
PYTHONPATH=. python3 extra/pure_machine_search_default_path_census.py --check
```

Then run protected smoke/token gates with `REDUCE_ACC_UPCAST_FIX=1` where applicable:

- Q4_K G3 decode GEMV;
- Q6_K generated coop decode;
- generated prefill schedule;
- generated G5 K-only attention;
- owned HIP attention default path;
- P11/P13 compiler microgates.

Acceptance:

- no protected token/logit mismatch;
- no hidden fallback;
- no NaN;
- default route behavior unchanged when the flag is off;
- owned HIP attention remains default.

Verdicts:

- `TG_P13_7_PASS_DEFAULT_OFF_REGRESSION`
- `TG_P13_7_BLOCKED_ROUTE_REGRESSION`

## Phase TG-P13.8: Commit The Compiler Fix

Only if TG-P13.4, TG-P13.5, TG-P13.6, and TG-P13.7 pass.

Commit tinygrad source + test artifacts separately by subsystem:

1. `[codegen]` commit for:
   - `tinygrad/codegen/late/devectorizer.py`
   - `tinygrad/codegen/__init__.py`
   - any core codegen tests.
2. `[test]` commit for:
   - `extra/qk_tg_p11_reduce_upcast_microgate.py`
   - `extra/qk_tg_p13_manual_end_inplace_microgate.py`
   - forced TG-P13 bench artifacts.
3. BoltBeam commit if classifier/ledger changes are needed.

Use `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` only for the repaired codegen commit if the final code materially derives from Claude's prototype.

Do not bundle unrelated dirty bench artifacts.

Verdicts:

- `TG_P13_8_PASS_COMPILER_FIX_LANDED`
- `TG_P13_8_BLOCKED_COMMIT_HYGIENE`

## Phase TG-P13.9: Split-Preserving Combine Reopen

Only after the compiler fix is committed.

Re-run the split-preserving combine candidates that TG-P9/TG-P10 blocked:

- shared weight combine;
- fused gmax combine;
- fexp-free weighted sum;
- any existing no-collapse `Hq*S` and `Hq*Hd` preserving shape.

Acceptance:

- generated UOp only;
- numeric correctness vs shipped combine/reference;
- route-bound;
- no `REG_STORE_DEVEC` dependency;
- local combine timing improves enough to justify W==D.

Verdicts:

- `TG_P13_9_PASS_COMBINE_REOPENED`
- `TG_P13_9_REFUTE_COMBINE_SPEED`
- `TG_P13_9_BLOCKED_COMBINE_CORRECTNESS`

## Phase TG-P13.10: Generated 8B Attention W==D

Only if TG-P13.9 passes.

Candidate:

```text
live-split generated tile
+ split-preserving generated combine
+ REDUCE_ACC_UPCAST_FIX
```

Gate:

- ctx512 and ctx4096;
- token/logit equivalence;
- route-bound;
- no hidden fallback;
- generated attention >= 98% of owned at both protected contexts;
- full W==D no protected regression.

Verdicts:

- `TG_P13_10_PASS_GENERATED_ATTENTION_PARITY`
- `TG_P13_10_REFUTE_GENERATED_ATTENTION_SPEED`
- `TG_P13_10_BLOCKED_ROUTE_CORRECTNESS`

## Phase TG-P13.11: Promotion And Strict Purity

Only if TG-P13.10 passes.

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

- `TG_P13_11_PASS_TINYGRAD_DEFAULT_PURITY`
- `TG_P13_11_BLOCKED_PROMOTION_POLICY`

## Honest Terminal Conditions

Stop and write a terminal artifact if:

- the already-ended microgate cannot be made minimal;
- the in-place rewrite cannot distinguish existing `END` boundaries exactly;
- multi-accumulator same-reduce merging remains ambiguous;
- P10 fixed mode still fails;
- BoltBeam classifier cannot distinguish old blocked vs fixed reachable;
- any protected route regresses;
- generated attention is correct but remains under the 98% parity bar.

Do not weaken the final purity bar. Correctness-only generated attention is not a default.

## Expected Near-Term Outcome

The most likely near-term success is:

```text
TG_P13_8_PASS_COMPILER_FIX_LANDED
```

That would mean the compiler invariant is solved and committed, but generated 8B attention may still need TG-P13.9/TG-P13.10 to prove speed.

Full purity is possible only if:

```text
TG_P13_9_PASS_COMBINE_REOPENED
TG_P13_10_PASS_GENERATED_ATTENTION_PARITY
TG_P13_11_PASS_TINYGRAD_DEFAULT_PURITY
```

## Claude Handoff Prompt

```text
Execute docs/tg-p13-manual-end-accumulator-inplace-rewrite-scope-20260701.md.

TG-P12 verified that Claude's REDUCE_ACC_UPCAST_FIX prototype fixes the minimal P11 microgate but fails the real TG-P10 combine-shaped repro by creating nested same-range END nodes. Do not repeat that shape. First add an already-ended manual accumulator microgate that reproduces the CFGContext failure without attention. Then implement an in-place rewrite: replace the matched STORE and reuse the existing END instead of creating a fresh END. Keep REDUCE_ACC_UPCAST_FIX default-off. No handwritten kernels, no model constants, no attention-specific branch. Stop at the first failed gate and write an artifact. Do not promote generated attention until W==D >=98% at ctx512 and ctx4096 and strict final purity pass.
```
