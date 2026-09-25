"""Nemotron-H `_Linear`: float32 matvec for decode; a hi/lo bf16 split on the bf16 tensor core for many rows."""
import unittest

import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.codegen.opt import OptOps
from tinygrad.codegen.opt.heuristic import hand_coded_optimizations
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.helpers import Target
from tinygrad.llm.nemotron_h import _Linear
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops


def _bf16_round(values: np.ndarray) -> np.ndarray:
  bits = values.astype(np.float32).view(np.uint32)
  bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000  # round to nearest even
  return bits.view(np.float32)


def _linear(k: int, n: int) -> _Linear:
  layer = _Linear(k, n, bias=False)
  layer.weight = Tensor(_bf16_round(np.random.default_rng(0).standard_normal((n, k), dtype=np.float32)),
                        device="CPU").cast(dtypes.bfloat16).realize()
  return layer


def _nv_opts(layer: _Linear, rows: int) -> list:
  x = Tensor.empty(rows, 1, layer.weight.shape[1], dtype=dtypes.float32, device="CPU")
  (ast,) = [c.src[0] for c in layer(x).schedule_linear().src if any(u.op is Ops.REDUCE for u in c.src[0].toposort())]
  scheduler = Scheduler(ast, CUDARenderer(Target.parse("NV:CUDA:sm_120")))
  scheduler.convert_loop_to_global()
  return hand_coded_optimizations(scheduler).applied_opts


class TestNemotronHLinear(unittest.TestCase):
  def test_float32_output_close_to_float32_activations(self):
    k, n = 64, 32
    layer = _linear(k, n)
    weight = layer.weight.float().numpy().astype(np.float64)
    for rows in (1, _Linear.MATVEC_MAX_ROWS, _Linear.MATVEC_MAX_ROWS + 1, 32, 37):
      x = np.random.default_rng(rows).standard_normal((rows, 1, k), dtype=np.float32)
      out = layer(Tensor(x, device="CPU"))
      self.assertEqual(out.dtype, dtypes.float32)
      want = x.astype(np.float64) @ weight.T
      # matvec: float32 activations. Split path: hi + lo carries the activation to 2**-16; the products are exact in
      # float32 on a tensor core, while CPU (no tensor core) rounds each bf16 product before the float32 sum.
      scale = np.abs(x).sum() / rows * np.abs(weight).max()
      tol = 1e-5 if rows <= _Linear.MATVEC_MAX_ROWS else 2.0 ** -8 * scale
      np.testing.assert_allclose(out.numpy(), want, rtol=1e-5, atol=tol)

  def test_split_carries_the_activation_past_bf16(self):
    # identity weight: every product is exact even without a tensor core, so the output is hi + lo
    layer = _Linear(32, 32, bias=False)
    layer.weight = Tensor(np.eye(32, dtype=np.float32), device="CPU").cast(dtypes.bfloat16).realize()
    x = np.random.default_rng(1).standard_normal((_Linear.MATVEC_MAX_ROWS + 1, 32), dtype=np.float32)
    out = layer(Tensor(x, device="CPU")).numpy()
    self.assertLessEqual(np.abs(out - x).max(), 2.0 ** -16 * np.abs(x).max())
    self.assertGreater(np.abs(_bf16_round(x) - x).max(), 2.0 ** -12 * np.abs(x).max())

  def test_mul_dtype_per_path(self):
    # the matvec multiplies in float32; only the split path feeds MUL(bf16, bf16) to a float32 sum
    layer = _linear(64, 32)
    for rows, operand in ((_Linear.MATVEC_MAX_ROWS, dtypes.float32), (_Linear.MATVEC_MAX_ROWS + 1, dtypes.bfloat16)):
      out = layer(Tensor.empty(rows, 1, 64, dtype=dtypes.float32, device="CPU"))
      (reduce,) = [u for u in out.uop.toposort() if u.op is Ops.REDUCE]
      self.assertEqual(reduce.dtype, dtypes.float32)
      mul = reduce.src[0].src[0] if reduce.src[0].op is Ops.CAST else reduce.src[0]
      self.assertIs(mul.op, Ops.MUL)
      self.assertEqual(mul.dtype, operand)

  def test_tensor_core_selected_for_large_rows(self):
    layer = _Linear(3136, 12544, bias=False)
    layer.weight = Tensor.empty(12544, 3136, dtype=dtypes.bfloat16, device="CPU")
    # 263 rows (a prompt-length recompute) pad to a tensor-core multiple; unpadded, the heuristic falls back to
    # scalar bf16-rounded products
    for rows in (_Linear.MATVEC_MAX_ROWS + 1, 32, 256, 263):
      self.assertIn(OptOps.TC, [o.op for o in _nv_opts(layer, rows)])

  def test_small_rows_keep_the_matvec_schedule(self):
    layer = _Linear(3136, 12544, bias=False)
    layer.weight = Tensor.empty(12544, 3136, dtype=dtypes.bfloat16, device="CPU")
    for rows in (1, 8, _Linear.MATVEC_MAX_ROWS):
      ops = [o.op for o in _nv_opts(layer, rows)]
      self.assertIn(OptOps.GROUP, ops)
      self.assertNotIn(OptOps.TC, ops)


if __name__ == "__main__":
  unittest.main()
