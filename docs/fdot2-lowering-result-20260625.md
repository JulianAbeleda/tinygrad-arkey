# P0.1 fdot2 lowering result (2026-06-25)

## Verdict

`P0_1_FDOT2_LOWERING_PASS`

The opt-in AMD `fdot2` primitive exposure is implemented behind `V_DOT2_LOWERING=1`, default-off.

## What changed

- Added `extra/qk_fdot2_lowering.py`.
- Added a conservative matcher for the exact post-devectorize fp16 dot2 idiom:

```text
(float)(a.x*b.x) + (float)(a.y*b.y)
```

- Lowers the idiom to:

```c
__builtin_amdgcn_fdot2(a, b, acc, false)
```

- Integrated the lowering in `tinygrad/codegen/__init__.py` at the linearized UOp stage, where the actual AMD C-style render idiom is present.
- Added `V_DOT2_LOWERING` to the `to_program` cache key so flag-on and flag-off compiles cannot alias.
- Added `test/external/test_fdot2_lowering.py`.

## Scope

This is an instruction-exposure task only.

It does not claim a decode or prefill speedup. It exposes the `v_dot2` primitive so later attention/codegen work can test whether this instruction gap is closed in tinygrad-native generated kernels.

## Gate results

Command:

```bash
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python -m pytest test/external/test_fdot2_lowering.py -q
```

Result:

```text
5 passed in 0.66s
```

Coverage:

- structural exact-pair lowering emits one `fdot2` `CUSTOMI`
- negative structural cases decline
- AMD fp16 dot correctness passes with `rel_rmse <= 1e-2`
- rendered source contains `__builtin_amdgcn_fdot2`
- `extra/qk_amdgpu_isa_primitive_audit.py` reports `has_v_dot2=true`

Default-off sanity command:

```bash
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python - <<'PY'
import os, numpy as np
os.environ.pop('V_DOT2_LOWERING', None)
from tinygrad.helpers import getenv
getenv.cache_clear()
from tinygrad import Tensor
from tinygrad.engine.realize import compile_linear
from tinygrad.uop.ops import Ops
out = (Tensor(np.ones((1,2), dtype=np.float16))*Tensor(np.ones((1,2), dtype=np.float16))).sum(axis=1)
src = '\n'.join(next((u.arg for u in c.src[0].toposort() if u.op is Ops.SOURCE), '') for c in compile_linear(out.schedule_linear()).src if c.src[0].op is Ops.PROGRAM)
assert '__builtin_amdgcn_fdot2' not in src
print('DEFAULT_OFF_NO_FDOT2_PASS')
PY
```

Result:

```text
DEFAULT_OFF_NO_FDOT2_PASS
```

## Kill criteria

Kill did not trigger. The canonical fp16 dot2 form was pinned and the rule fires on the real AMD codegen path.

## Notes

The production hook uses a linearized-UOp replacement pass rather than a graph-level pass. The graph-level matcher remains useful for structural tests, but the production idiom is most reliably present immediately before render.
