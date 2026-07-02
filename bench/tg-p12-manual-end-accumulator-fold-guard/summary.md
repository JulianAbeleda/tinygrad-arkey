# TG-P12 Resolution: Manual END-Accumulator Widening

Verdict: **TG_P12R_BLOCKED_TG_P10_REPRO**

Codex could run the Python ladder that Claude could not. The result is precise: Claude's default-off `REDUCE_ACC_UPCAST_FIX=1`
implementation fixes the minimal P11 invariant microgate, but it does **not** fix the older combine-shaped TG-P10 repro.
Therefore the codegen fix is not safe to commit as-is.

## Gates Run

### R0 baseline microgate: PASS

Command:

```bash
DEV=AMD PYTHONPATH=. python3 extra/qk_tg_p11_reduce_upcast_microgate.py
```

Result:

```text
TG_P11_1_PASS_INVARIANT_TEST_READY fix= False |
scalar_no_upcast:cfail invariant_upcast:cfail varies_upcast:cfail mixed_var_inv:cfail
```

This confirms the strengthened baseline: all four manual accumulator cases fail without the fix.

### R1 fix-on microgate: PASS

Command:

```bash
DEV=AMD REDUCE_ACC_UPCAST_FIX=1 PYTHONPATH=. python3 extra/qk_tg_p11_reduce_upcast_microgate.py
```

Result:

```text
TG_P11_1_PASS_INVARIANT_TEST_READY fix= True |
scalar_no_upcast:ok invariant_upcast:ok varies_upcast:ok mixed_var_inv:ok
```

So the minimal attention-free invariant is solved.

### R2 TG-P10 repro: FAIL

Command:

```bash
DEV=AMD REDUCE_ACC_UPCAST_FIX=1 PYTHONPATH=. python3 extra/qk_tg_p10_reg_scalar_repro.py
```

Result:

```text
TG_P10_1_BLOCKED_REPRO_NOT_MINIMAL
control_ok=True
fails=['shared_weight_combine_compile_fails', 'reg_store_devec_compiles_nan']
devec_nan=False
```

Full stack on `shared_weight_combine_compile_fails` shows the new rewrite creates an invalid control-flow relationship:

```text
tinygrad/codegen/late/linearizer.py:162
assert y.src[1] not in x.backward_slice_with_self
AssertionError
```

An in-process CFG diagnostic pins it to nested `END` nodes over the same reduce range `(4, AxisType.REDUCE)`.

## Diagnosis

The current fix handles the standalone shape, but the combine-shaped graph has a dependency pattern the pass does not handle:

- the matched accumulator is the denominator `REG 243`;
- it is a scalar manual accumulator broadcast to four lanes;
- the accumulator update already sits under an existing `END` for reduce range `(4, AxisType.REDUCE)`;
- the new pass creates another `END` over the same range instead of reusing/replacing the existing one;
- `CFGContext` then sees a nested same-range `END` and asserts.

This means the implementation is not yet a valid generic lowering. It is a partial prototype, not a commit-ready fix.

## Decision

Per `docs/tg-p12-resolution-verify-and-land-scope-20260701.md`, stop at R2:

- do **not** run route gates;
- do **not** commit the codegen fix;
- do **not** promote generated attention;
- keep owned HIP attention as default.

The next implementation attempt should either:

1. rewrite the matched manual accumulator store in-place and reuse the existing reduce `END`, or
2. make the pass explicitly detect and reject already-ended accumulator stores until an exact in-place rewrite is implemented.

The current uncommitted code should be treated as a failed attempt, not as a landed feature.
