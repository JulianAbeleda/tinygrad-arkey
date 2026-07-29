import ast
from pathlib import Path

import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import KernelInfo, Ops, UOp


def _repo_root() -> Path:
  for candidate in Path(__file__).resolve().parents:
    if (candidate / "pyproject.toml").is_file() and (candidate / "tinygrad").is_dir(): return candidate
  raise RuntimeError("could not locate repository root")


def _increment(out:UOp, src:UOp) -> UOp:
  idx = UOp.range(out.numel(), 0)
  return out.flatten()[idx].store(src.flatten()[idx] + 1).end(idx).sink(arg=KernelInfo(name="uop_program_increment"))


def _call(method:str, *, grad_fxn=None) -> list[Tensor]:
  output = Tensor.empty(4, dtype=dtypes.float, device="PYTHON")
  source = Tensor([1, 2, 3, 4], dtype=dtypes.float, device="PYTHON")
  return getattr(output, method)(source, fxn=_increment, grad_fxn=grad_fxn)


def test_legacy_tensor_custom_kernel_api_is_removed():
  assert not hasattr(Tensor, "custom_kernel")


def test_uop_program_is_lazy_multi_output_and_forwards_gradient_callback():
  def gradient(*args): return args
  outputs = _call("uop_program", grad_fxn=gradient)
  assert isinstance(outputs, list) and len(outputs) == 2
  assert [x.shape for x in outputs] == [(4,), (4,)]
  assert all(x.uop.op is Ops.AFTER for x in outputs)
  call = outputs[0].uop.src[1]
  assert call.op is Ops.CALL and outputs[1].uop.src[1] is call
  assert call.arg.grad_fxn is gradient
  np.testing.assert_array_equal(outputs[0].numpy(), np.array([2, 3, 4, 5], dtype=np.float32))


def test_active_source_has_no_legacy_tensor_callers():
  root, callers = _repo_root(), []
  for source_root in ("tinygrad", "test", "extra"):
    for path in (root / source_root).rglob("*.py"):
      if "__pycache__" in path.parts: continue
      tree = ast.parse(path.read_text(), filename=str(path))
      if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
             node.func.attr == "custom_kernel" for node in ast.walk(tree)):
        callers.append(str(path.relative_to(root)))
  assert sorted(callers) == ["tinygrad/nn/__init__.py", "tinygrad/tensor.py"], (
    "custom_kernel call syntax is reserved for direct internal UOp substrate callers")
