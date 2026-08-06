"""Custom-kernel shaped-PARAM fold regression lock (M4 rangeify substrate S3 finding).

`UOp.custom_kernel` builds kernel-body placeholders via `UOp.placeholder_like`, which
reshapes when `len(shape) > 1`: a non-flat arg (e.g. the `(1,1,4096)` opaque block-output
view of the M4 residual fold) surfaces in the body as `RESHAPE(PARAM, shape-STACK)`. No
codegen pass consumes the shape STACK and no renderer has a RESHAPE rule, so the shaped arg
crashed type_verify/rendering before this fix. `pm_index_is_shrink` now folds shaped
GLOBAL-ptr PARAM views to the flat PARAM and loads scalar pointer-typed INDEX value reads in
the explicit `.cast()` spelling; image buffers and multi-index (3+ src) INDEX bases are
untouched.

Kernel bodies must read shaped args as explicit value reads (`buf[i].cast(dtype)` or
`.load()`), the production `epi_resadd` spelling; a bare `buf[i]` used in ALU is a pointer
and was never renderable on CPU.

Scope: `docs/task_workflow/input/m4-resadd-rangeify-substrate-scope-20260806.md` S3. CPU only.
"""
import numpy as np

from tinygrad import Tensor, dtypes, UOp
from tinygrad.helpers import DEV, Target
from tinygrad.uop.ops import Ops, KernelInfo
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.codegen import to_program

DEV.value = "CPU"

N = 256


def _read_kernel():
  def kernel(out: UOp, x: UOp) -> UOp:
    row = UOp.range(N, 0)
    return out[row].store(x[row].cast(dtypes.float32) * 2.0).end(row).sink(
      arg=KernelInfo(name="k_shaped_read", opts_to_apply=()))
  return kernel


def _build(x_tensor: Tensor) -> Tensor:
  out = Tensor.zeros(N, dtype=dtypes.float32)
  ret = UOp.custom_kernel(out.uop, x_tensor.uop, fxn=_read_kernel())
  return Tensor(ret[0])


def test_shaped_custom_arg_is_read_flat_and_executes():
  x = Tensor(np.arange(N, dtype=np.float32).reshape(1, 1, N)).contiguous()
  out = _build(x).realize()
  assert out.numpy()[:3].tolist() == [0.0, 2.0, 4.0]


def test_shaped_arg_matches_flat_arg_byte_for_byte():
  shaped = Tensor(np.arange(N, dtype=np.float32).reshape(1, 1, N)).contiguous()
  flat = Tensor(np.arange(N, dtype=np.float32)).contiguous()
  a, b = _build(shaped).realize().numpy(), _build(flat).realize().numpy()
  assert a.tobytes() == b.tobytes()


def test_fold_drops_reshape_and_expand_from_the_compiled_program():
  x = Tensor(np.arange(N, dtype=np.float32).reshape(1, 1, N)).contiguous()
  t = _build(x)
  calls = t.schedule_linear().src
  ast = next(call.src[0] for call in calls
             if any(u.op is Ops.SINK and isinstance(u.arg, KernelInfo) and u.arg.name == "k_shaped_read"
                    for u in call.src[0].toposort()))
  # The shaped arg genuinely reaches codegen as RESHAPE(PARAM, shape-STACK): the fold is
  # load-bearing, so a revert shows up as RESHAPE in the compiled program below.
  assert Ops.RESHAPE in {u.op for u in ast.toposort()}
  prog = to_program(ast, ClangRenderer(Target.parse("CPU:CLANG:x86_64,znver2")))
  prog_ops = {u.op for u in prog.src[0].toposort()}
  assert Ops.RESHAPE not in prog_ops and Ops.EXPAND not in prog_ops and Ops.STACK not in prog_ops
  assert any(u.op is Ops.SOURCE for u in prog.src)
