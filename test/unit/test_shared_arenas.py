"""memory planner: JITs captured under `shared_arenas` plan their intermediates into one pool of arenas."""
import unittest

import numpy as np

from tinygrad import Tensor, TinyJit
from tinygrad.dtype import dtypes
from tinygrad.schedule.memory import shared_arenas
from tinygrad.uop.ops import Ops


def _arenas(jit: TinyJit) -> set[int]:
  return {id(u) for u in jit.captured.linear.toposort() if u.op is Ops.BUFFER and u.dtype == dtypes.int8}


class TestSharedArenas(unittest.TestCase):
  def _graphs(self, pool: dict | None):
    x, out_a, out_b = Tensor.rand(64, 64).realize(), Tensor.empty(64, 64).realize(), Tensor.empty(32, 64).realize()
    def f(): out_a.assign(((x @ x).contiguous().relu() @ x).contiguous() + 1).realize()
    def g(): out_b.assign(((x[:32] @ x).contiguous().exp() @ x).contiguous() * 2).realize()
    jits = [TinyJit(f), TinyJit(g)]
    for jit in jits:
      for _ in range(3):
        if pool is None: jit()
        else:
          with shared_arenas(pool): jit()
    return x, out_a, out_b, jits

  def test_two_graphs_share_the_pool_and_stay_correct(self):
    pool: dict = {}
    x, out_a, out_b, (a, b) = self._graphs(pool)
    self.assertTrue(pool)
    self.assertTrue(_arenas(a) and _arenas(b))
    self.assertLessEqual(_arenas(a) | _arenas(b), {id(v) for v in pool.values()})  # no graph has its own arena
    a(); b()
    xs = x.numpy()
    np.testing.assert_allclose(out_a.numpy(), np.maximum(xs @ xs, 0) @ xs + 1, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(out_b.numpy(), np.exp(xs[:32] @ xs) @ xs * 2, rtol=1e-3, atol=1e-2)

  def test_without_a_pool_each_graph_has_its_own(self):
    _, _, _, (a, b) = self._graphs(None)
    self.assertFalse(_arenas(a) & _arenas(b))


if __name__ == "__main__":
  unittest.main()
