# TG-P11 Scope: Reduce/Upcast Accumulator Widening

Date: 2026-07-01.

Goal: fix the generic tinygrad codegen invariant that blocks the final pure-machine-search route: generated 8B decode attention with a split-preserving LSE combine.

This is a **core-codegen** scope, not an attention-kernel scope. The intended route is:

```text
generic reduce/upcast accumulator lowering fix
-> TG-P10 REG repro flips reachable
-> split-preserving generated combine microgate
-> generated 8B attention W==D gate
-> strict default purity
```

Do not write a handwritten HIP/ASM/ISA attention kernel. Do not special-case Qwen3-8B, `Hq=32`, `Hd=128`, or a specific kernel name.

## Current Evidence

Primary tinygrad artifacts:

- `extra/qk_tg_p10_reg_scalar_repro.py`
- `bench/tg-p10-reg-scalar-combine-lowering/reg_scalar_lowering.json`
- `bench/tg-p10-reg-scalar-combine-lowering/reg_lowering_diagnosis.json`
- `bench/tg-p10-reg-scalar-combine-lowering/latest.json`
- `bench/tg-p10-reg-scalar-combine-lowering/summary.md`

Primary BoltBeam artifacts:

- `/home/ubuntu/BoltBeam/boltbeam/diagnostics/reg_lowering.py`
- `/home/ubuntu/BoltBeam/tests/test_reg_lowering.py`
- `/home/ubuntu/BoltBeam/boltbeam/data/candidates.json`
- `/home/ubuntu/BoltBeam/docs/tg-p10-reg-scalar-combine-lowering-scope-20260701.md`

Current terminal verdict:

```text
TG_P10_BLOCKED_REG_SCALAR_LOWERING_DIAGNOSED
```

Current generated attention state:

| piece | status |
|---|---|
| live-context split tile | solved; ctx512 87.7% -> 96.7% of owned |
| split-preserving combine | blocked by generic reduce/upcast REG lowering |
| full generated attention | 96.7% / 95.3% of owned at ctx512/4096; below 98% bar |
| owned HIP attention | remains default and rollback/oracle |

## Root Cause To Fix

From TG-P10.3:

```text
When the optimizer UPCASTs an output axis (`d`, by 4), `reduce_to_acc` still creates a size-1 scalar REG accumulator.
But `num = sum_s w * pv[d]` varies along the upcast `d`, so it needs one accumulator slot per upcast lane.
The devectorizer instead emits invalid C: make_float4(acc,acc,acc,acc) = <4 distinct partials>.
REG_STORE_DEVEC=1 scalarizes stores but aliases all lanes into slot 0, causing NaN.
```

Invariant:

```text
A reduce accumulator must be widened over every upcast/unroll output lane that the reduce result varies along.
It must remain scalar over axes the reduce result is invariant along.
```

Likely files:

- `tinygrad/codegen/late/expander.py`
  - `fix_reduce_unroll`
- `tinygrad/codegen/late/devectorizer.py`
  - `reduce_to_acc`
  - reduce END merging / accumulator placeholder sizing
- `tinygrad/codegen/__init__.py`
  - cache-key handling if a temporary flag is used

## Required Safety Shape

This change touches core reduce lowering. It can silently corrupt many kernels. Therefore:

1. Implement behind a temporary default-off gate first, for example:

   ```text
   REDUCE_ACC_UPCAST_FIX=1
   ```

2. Prove correctness under the gate.
3. Only then decide whether to default it on or make it unconditional.
4. Do not route generated attention by default until the full W==D gate passes.

## Phase TG-P11.0: Reproduce And Pin Baseline

Purpose: confirm the starting point before editing core codegen.

Run:

```bash
DEV=AMD PYTHONPATH=. python3 extra/qk_tg_p10_reg_scalar_repro.py
```

Expected current artifact:

```text
TG_P10_1_PASS_REG_REPRO_PINNED
control passes
shared_weight_combine_compile_fails -> invalid_reg_vector_store
fused_gmax_combine_compile_fails -> invalid_reg_vector_store
reg_store_devec_compiles_nan -> nan_output
```

Also run BoltBeam classifier:

```bash
cd /home/ubuntu/BoltBeam
PYTHONPATH=. python3 -m pytest -q tests/test_reg_lowering.py
```

Acceptance:

- current failure reproduces;
- BoltBeam classifies it as `EMITTER_BLOCKED`;
- owned default remains unchanged.

Verdicts:

- `TG_P11_0_PASS_BASELINE_PINNED`
- `TG_P11_0_BLOCKED_REPRO_DRIFT`

## Phase TG-P11.1: Minimal Compiler Invariant Test

Purpose: isolate the codegen invariant away from attention.

Add or extend a tiny generated-UOp microgate with at least these cases:

| case | expected with fix |
|---|---|
| scalar reduce, no upcast | unchanged correct |
| reduce result invariant along upcast axis | accumulator remains scalar |
| reduce result varies along upcast axis | accumulator has one slot per lane |
| mixed case: numerator varies, denominator invariant | numerator widened, denominator scalar |

The test must inspect both:

- numeric output;
- generated code / lowered artifact enough to prove no invalid `make_float4(...) = ...` store and no slot-0 lane aliasing.

Suggested output:

```text
bench/tg-p11-reduce-upcast-accumulator/latest.json
```

Verdicts:

- `TG_P11_1_PASS_INVARIANT_TEST_READY`
- `TG_P11_1_BLOCKED_TEST_NOT_MINIMAL`

## Phase TG-P11.2: Generic Lowering Fix

Purpose: fix the invariant in core codegen.

Implementation rules:

- no model-name branching;
- no attention-kernel branching;
- no shape constants from Qwen3-8B;
- no handwritten external kernel;
- no backend-specific behavior unless the bug is demonstrably AMD renderer/devectorizer specific.

Expected implementation direction:

1. Track which non-reduce upcast/unroll axes survive into the reduce output.
2. Size the REG accumulator placeholder to `prod(varying_upcast_lanes)`.
3. Index accumulator slot by the surviving upcast lane id.
4. Keep invariant accumulators scalar.
5. Ensure END/AFTER merging preserves the widened accumulator identity.

The result should make `REG_STORE_DEVEC=1` unnecessary for this path, or make equivalent devectorization correct without lane aliasing.

Initial gate:

```bash
DEV=AMD REDUCE_ACC_UPCAST_FIX=1 PYTHONPATH=. python3 extra/qk_tg_p10_reg_scalar_repro.py
```

Expected:

```text
all TG-P10 cases compile and numeric_ok=true
```

Verdicts:

- `TG_P11_2_PASS_GENERIC_ACCUM_WIDENING`
- `TG_P11_2_BLOCKED_LOWERING_STILL_WRONG`
- `TG_P11_2_BLOCKED_FIX_NOT_GENERIC`

## Phase TG-P11.3: Focused Regression Gates

Purpose: catch known nearby regressions before full-model tests.

Run at minimum:

```bash
DEV=AMD PYTHONPATH=. python3 extra/qk_tg_p10_reg_scalar_repro.py
DEV=AMD REDUCE_ACC_UPCAST_FIX=1 PYTHONPATH=. python3 extra/qk_tg_p10_reg_scalar_repro.py
DEV=AMD JIT=1 REG_STORE_DEVEC=1 PYTHONPATH=. python3 extra/qk_decode_cache_identity_index_gate.py
```

If the new fix replaces the old `REG_STORE_DEVEC` need, add a no-`REG_STORE_DEVEC` equivalent row instead of weakening the test.

Required:

- TG-P10 repro flips from blocked to reachable under the new fix.
- Previously passing cache/upcast rows stay passing.
- No new NaNs.

Verdicts:

- `TG_P11_3_PASS_FOCUSED_REGRESSIONS`
- `TG_P11_3_BLOCKED_NEARBY_REGRESSION`

## Phase TG-P11.4: Shipped Route Regression Ladder

Purpose: validate core-codegen safety across the hot generated defaults before touching attention promotion.

Run the existing lightweight route gates for:

| route | required proof |
|---|---|
| Q4_K G3 decode GEMV | token/logit gate unchanged |
| Q6_K generated decode route | token/logit gate unchanged |
| G=5 K-only generated attention | token/logit gate unchanged for 14B if feasible |
| prefill generated schedule | correctness/logit gate unchanged |
| owned 8B attention default | default path unchanged with fix disabled |

Do not hand-wave this phase. If an existing gate is missing, add the smallest gate rather than skipping the route.

Suggested commands should be discovered from existing `extra/qk_tg_p*_*.py` gates and recorded in the artifact.

Required artifact:

```text
bench/tg-p11-reduce-upcast-accumulator/regression_ladder.json
```

Verdicts:

- `TG_P11_4_PASS_ROUTE_REGRESSION_LADDER`
- `TG_P11_4_BLOCKED_ROUTE_REGRESSION`

## Phase TG-P11.5: Split-Preserving Combine Microgate

Purpose: resume TG-P10.4 now that the compiler invariant is fixed.

Requirements:

- generated UOp only;
- no HIP/ASM/inline ISA route;
- no Hq-only collapse;
- no Hq*Hd collapse previously refuted;
- compare against Python/numpy LSE reference;
- include Qwen3-8B geometry:

```text
B=1, Hq=32, Hkv=8, G=4, Hd=128, S=36
```

Acceptance:

- all combine variants needed by the final route compile;
- numeric output within tolerance;
- launch geometry preserves split and d-axis parallelism.

Verdicts:

- `TG_P11_5_PASS_SPLIT_PRESERVING_COMBINE`
- `TG_P11_5_REFUTE_COMBINE_NUMERIC`
- `TG_P11_5_BLOCKED_PARALLELISM_COLLAPSE`

## Phase TG-P11.6: Full Generated 8B Attention W==D

Purpose: measure the actual final generated attention candidate.

Candidate should combine:

- TG-P9 live split tile;
- TG-P11 fixed split-preserving combine.

Suggested flag:

```text
DECODE_ATTN_SPLIT_PRESERVING_COMBINE_GENERATED=1
```

Protected contexts:

```text
ctx512
ctx4096
```

Optional sanity:

```text
ctx128, ctx1024, ctx2048
```

Gates:

- token/logit equivalent to owned;
- route-bound generated attention, no hidden fallback;
- generated UOp only;
- W==D vs owned;
- no protected-context regression.

Promotion bar:

```text
>=98% of owned at ctx512
>=98% of owned at ctx4096
```

Verdicts:

- `TG_P11_6_PASS_GENERATED_ATTENTION_PARITY`
- `TG_P11_6_REFUTE_GENERATED_ATTENTION_SPEED`
- `TG_P11_6_BLOCKED_CORRECTNESS`
- `TG_P11_6_BLOCKED_ROUTE_ATTRIBUTION`

## Phase TG-P11.7: Default Promotion And Strict Purity

Run only if TG-P11.6 passes.

Actions:

1. Make the generated attention route the default for the validated 8B geometry.
2. Keep owned HIP as rollback/oracle.
3. Update:

   - `extra/qk_route_manifest.py`
   - `bench/qk-search-spaces/default_route_manifest.json`
   - `bench/qk-search-spaces/refuted_axes.json`
   - BoltBeam `candidates.json` / ledger if needed.

4. Run:

```bash
PYTHONPATH=. python3 extra/pure_machine_search_default_path_census.py --check
PYTHONPATH=. python3 extra/pure_machine_search_default_path_census.py --strict-final-default
```

Acceptance:

```text
TINYGRAD_DEFAULT_PURITY_PASS
```

Verdicts:

- `TG_P11_7_PASS_FINAL_DEFAULT_PURITY`
- `TG_P11_7_BLOCKED_STRICT_CENSUS`

## Phase TG-P11.8: Honest Terminal If It Fails

If the fix cannot safely land or the generated attention still misses the bar, record the terminal precisely.

Required:

- do not promote a slower generated route;
- keep owned default;
- classify whether the remaining blocker is:

```text
LOWERING_STILL_WRONG
COMBINE_NUMERIC
PARALLELISM_COLLAPSE
SPEED_REFUTED
REGRESSION_SURFACE_TOO_BROAD
```

Update BoltBeam so the failed path is not re-tried without a new reopen condition.

## Claude Handoff Prompt

Use this for a fresh context:

```text
You are working in /home/ubuntu/tinygrad-arkey and /home/ubuntu/BoltBeam.

Goal: TG-P11. Fix the generic reduce/upcast accumulator lowering invariant that TG-P10 diagnosed, then resume the generated 8B attention combine path. Do not write a handwritten HIP/ASM/ISA attention kernel. Do not special-case Qwen3-8B or a kernel name.

Starting facts:
- TG-P10 terminal is in tinygrad bench/tg-p10-reg-scalar-combine-lowering/.
- Repro script: extra/qk_tg_p10_reg_scalar_repro.py.
- BoltBeam classifier exists: /home/ubuntu/BoltBeam/boltbeam/diagnostics/reg_lowering.py.
- Root cause: output-axis UPCAST by 4 + reduce_to_acc size-1 scalar REG accumulator. num=sum_s w*pv[d] varies along d and needs one accumulator slot per upcast lane. den=sum_s w*l is invariant and should stay scalar. Current lowering emits invalid make_float4(acc,acc,acc,acc)=<4 partials>; REG_STORE_DEVEC=1 aliases lanes into slot 0 and NaNs.
- Fix location: tinygrad/codegen/late/expander.py fix_reduce_unroll and tinygrad/codegen/late/devectorizer.py reduce_to_acc.

Required sequence:
1. Reproduce TG-P10 baseline.
2. Add a minimal invariant microgate for reduce output varying vs invariant along upcast axes.
3. Implement generic accumulator widening behind a temporary default-off gate such as REDUCE_ACC_UPCAST_FIX=1.
4. Re-run TG-P10 repro under the gate; it must flip all cases to numeric_ok.
5. Run focused regression gates including qk_decode_cache_identity_index_gate.py.
6. Run shipped route regression ladder for Q4_K G3, Q6_K generated, generated prefill schedule, G=5 K-only attention if feasible, and default owned 8B attention.
7. Build/run split-preserving combine microgate.
8. Only if the microgate passes, run full generated 8B attention W==D vs owned at ctx512 and ctx4096.
9. Promote only if generated attention reaches >=98% of owned at both protected contexts and strict final purity passes.

If anything fails, stop and ledger the exact blocker. Do not force purity by making the model slower.
```

## Expected End State

Best case:

```text
TG_P11_7_PASS_FINAL_DEFAULT_PURITY
TINYGRAD_DEFAULT_PURITY_PASS
```

Acceptable terminal:

```text
owned HIP remains default
the remaining blocker is classified to one exact compiler invariant or measured speed refutation
```

