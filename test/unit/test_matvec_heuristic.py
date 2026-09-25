"""The hand-coded matvec schedule: GROUP the reduction so a weight row is read coalesced."""
import unittest

from tinygrad import Tensor, dtypes
from tinygrad.codegen.opt import OptOps
from tinygrad.codegen.opt.heuristic import hand_coded_optimizations
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer


def matvec_opts(rows: int, n: int, k: int, weight_dtype=dtypes.bfloat16) -> list:
  weight = Tensor.empty(n, k, dtype=weight_dtype, device="CPU")
  x = Tensor.empty(rows, 1, k, dtype=dtypes.float32, device="CPU")
  (call,) = x.linear(weight.T).schedule_linear().src
  scheduler = Scheduler(call.src[0], CUDARenderer(Target.parse("NV:CUDA:sm_120")))
  scheduler.convert_loop_to_global()
  return hand_coded_optimizations(scheduler).applied_opts


class TestMatvecHeuristic(unittest.TestCase):
  def test_fp32_matvec_groups(self):
    self.assertIn(OptOps.GROUP, [o.op for o in matvec_opts(1, 3072, 3136, dtypes.float32)])

  def test_bf16_weight_cast_matvec_groups(self):
    # the weight operand is CAST(load bf16); it is still a matvec operand
    self.assertIn(OptOps.GROUP, [o.op for o in matvec_opts(1, 3072, 3136)])


if __name__ == "__main__":
  unittest.main()
