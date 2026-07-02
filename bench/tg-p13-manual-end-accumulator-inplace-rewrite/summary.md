# TG-P13 Terminal: In-Place Manual END-Accumulator Rewrite

Verdict: **TG_P13_BLOCKED_AMD_DEVICE_INIT**

TG-P13 made real codegen progress, but the verification ladder could not be completed because the AMD device began hanging
during initialization. The latest dirty code must be treated as **unverified**.

## What Was Implemented In The Dirty Tree

The failed TG-P12 prototype was repaired in several directions:

- the rewrite now substitutes the matched accumulator `STORE` in place instead of always creating a fresh `END`;
- direct `END`s are rewritten in place;
- same-reduce final `END`s group all enclosed matched stores, so multi-accumulator patterns can keep numerator and denominator
  stores alive;
- widened accumulator storage was changed from vector REG element storage to scalar REG slots;
- a distinct-slot REG store devectorizer was added under `REDUCE_ACC_UPCAST_FIX=1` to scalarize safe vector REG stores without
  the aliasing bug from broad `REG_STORE_DEVEC`.

## Last Verified State Before AMD Init Failed

Before the final direct non-reduce `END` patch, the gate state was:

- P11 fix-on microgate: PASS, all four cases ok.
- Shared-weight P10 combine: compiled, but numeric wrong (`rel ~= 1.82`).
- Inline-gmax P10 combine: compiled but returned NaN.

After the final direct non-reduce `END` patch, the next P11/P10 run hung before producing a result. The traceback from interrupt
showed the process was stuck opening/acquiring the AMD device, before the compiler gate ran.

## Infrastructure Failure

A separate minimal AMD smoke also hung:

```bash
DEV=AMD PYTHONPATH=. python3 - <<'PY'
from tinygrad import Tensor
print((Tensor([1.0], device='AMD') + 1).realize().numpy())
PY
```

`ps` showed the Python process in uninterruptible sleep:

```text
1840274       1 D    python3 -
```

That means the blocker is currently the AMD runtime/driver path, not a completed codegen verdict.

## Decision

Do not commit the dirty codegen changes as a feature.

The current dirty code can be used as a scratch starting point after the AMD device recovers, but it must be revalidated from
TG-P13.4:

```bash
DEV=AMD PYTHONPATH=. python3 extra/qk_tg_p11_reduce_upcast_microgate.py
DEV=AMD REDUCE_ACC_UPCAST_FIX=1 PYTHONPATH=. python3 extra/qk_tg_p11_reduce_upcast_microgate.py
DEV=AMD REDUCE_ACC_UPCAST_FIX=1 PYTHONPATH=. python3 extra/qk_tg_p10_reg_scalar_repro.py
```

If those pass, continue to BoltBeam classifier, default-off census, and protected route smoke gates. If they fail, write the
new exact terminal verdict.

Owned HIP attention remains default.
