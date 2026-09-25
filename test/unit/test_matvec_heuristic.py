"""The hand-coded matvec schedule: GROUP the reduction so a weight row is read coalesced."""
import unittest

from tinygrad import Tensor, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.heuristic import hand_coded_optimizations
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer


def matvec_opts(rows: int, n: int, k: int, weight_dtype=dtypes.bfloat16, bf16_activation=False) -> list:
  weight = Tensor.empty(n, k, dtype=weight_dtype, device="CPU")
  x = Tensor.empty(rows, 1, k, dtype=dtypes.float32, device="CPU")
  out = x.cast(dtypes.bfloat16).dot(weight.T, dtype=dtypes.float) if bf16_activation else x.linear(weight.T)
  (call,) = out.schedule_linear().src
  scheduler = Scheduler(call.src[0], CUDARenderer(Target.parse("NV:CUDA:sm_120")))
  scheduler.convert_loop_to_global()
  return hand_coded_optimizations(scheduler).applied_opts


class TestMatvecHeuristic(unittest.TestCase):
  def test_fp32_matvec_groups(self):
    self.assertIn(OptOps.GROUP, [o.op for o in matvec_opts(1, 3072, 3136, dtypes.float32)])

  def test_bf16_weight_cast_matvec_groups(self):
    # the weight operand is CAST(load bf16); it is still a matvec operand
    self.assertIn(OptOps.GROUP, [o.op for o in matvec_opts(1, 3072, 3136)])

  def test_small_batch_matvec_groups_and_upcasts_the_batch(self):
    # M activation rows share one weight: the weight index lacks the batch range; upcast it so each weight load feeds all M
    for rows in (2, 8, 16):
      opts = matvec_opts(rows, 3136, 12544)
      self.assertIn(OptOps.GROUP, [o.op for o in opts])
      self.assertIn(Opt(OptOps.UPCAST, 0, 0 if rows <= 8 else 8), opts)

  def test_small_batch_bf16_matvec_prefers_matvec_over_a_16_row_tensor_core_tile(self):
    opts = matvec_opts(16, 3136, 12544, bf16_activation=True)
    self.assertNotIn(OptOps.TC, [o.op for o in opts])
    self.assertIn(OptOps.GROUP, [o.op for o in opts])

  def test_bf16_matvec_on_a_bf16_folding_target_splits_rows_across_a_warp(self):
    opts = matvec_opts(1, 3136, 12544)
    self.assertIn(Opt(OptOps.GROUP, 0, 32), opts)
    self.assertIn(Opt(OptOps.UNROLL, 0, 8), opts)
    # K=3136 is not a multiple of 32*8: fall back to 16 lanes x 4
    self.assertIn(Opt(OptOps.GROUP, 0, 16), matvec_opts(1, 12544, 3136))

  def test_bf16_weight_loads_fold_to_16_byte_carriers(self):
    from tinygrad.codegen import to_program
    from tinygrad.uop.ops import Ops
    weight = Tensor.empty(3136, 12544, dtype=dtypes.bfloat16, device="CPU")
    x = Tensor.empty(1, 1, 12544, dtype=dtypes.float32, device="CPU")
    (call,) = x.linear(weight.T).schedule_linear().src
    program = to_program(call.src[0], CUDARenderer(Target.parse("NV:CUDA:sm_120")))
    source = next(u.arg for u in program.src if u.op is Ops.SOURCE)
    self.assertIn("(nv_bfloat168*)", source)
    try: from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
    except Exception: self.skipTest("NVRTC unavailable")
    self.assertTrue(NVRTCCompiler("sm_120", ptx=False, cache_key="bf16_matvec_fold").compile(source))

  def test_large_batch_is_not_a_matvec(self):
    self.assertNotIn(OptOps.GROUP, [o.op for o in matvec_opts(64, 3136, 12544)])


if __name__ == "__main__":
  unittest.main()
