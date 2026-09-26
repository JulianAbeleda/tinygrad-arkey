"""Default opts put consecutive thread lanes on the stride-1 axis: elementwise kernels and row reduces (sm_120)."""
import unittest

from tinygrad import Tensor, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.heuristic import hand_coded_optimizations
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer


def opts(out: Tensor) -> list:
  (call,) = out.schedule_linear().src
  scheduler = Scheduler(call.src[0], CUDARenderer(Target.parse("NV:CUDA:sm_120")))
  scheduler.convert_loop_to_global()
  return hand_coded_optimizations(scheduler).applied_opts


def locals_of(applied: list) -> list:
  return [o for o in applied if o.op is OptOps.LOCAL]


def gated_norm_apply(rows: int) -> Tensor:
  y, scale, weight = Tensor.empty(rows, 7680), Tensor.empty(rows, 8, 1), Tensor.empty(7680)
  return (y.reshape(rows, 8, 960) * scale * weight.reshape(8, 960)).reshape(rows, 7680).cast(dtypes.bfloat16).contiguous()


class TestCoalescingHeuristic(unittest.TestCase):
  def test_broadcast_row_axis_does_not_fill_the_warp(self):
    # Mamba gated-norm apply: [rows, 7680] times a per-row group scale and a per-column weight, to bf16. The weight is
    # broadcast over rows, which ranks the row axis first for locals; at 128 rows it took all 32 lanes of a warp
    # (every lane another row), so the stride-1 column axis becomes the only local
    self.assertEqual(locals_of(opts(gated_norm_apply(128))), [Opt(OptOps.LOCAL, 1, 256)])

  def test_a_warp_with_lanes_on_the_stride_one_axis_keeps_its_schedule(self):
    # at 32 rows the row local is 8 wide: 4 lanes of each warp step along the columns, and the ranked locals stay
    self.assertEqual(locals_of(opts(gated_norm_apply(32))), [Opt(OptOps.LOCAL, 0, 8), Opt(OptOps.LOCAL, 0, 16)])

  def test_hi_lo_operand_rows_stay_coalesced(self):
    y, scale, weight = Tensor.empty(128, 7680), Tensor.empty(128, 8, 1), Tensor.empty(7680)
    n = (y.reshape(128, 8, 960) * scale * weight.reshape(8, 960)).reshape(128, 7680)
    hi = n.cast(dtypes.bfloat16)
    out = hi.cat((n - hi.float()).cast(dtypes.bfloat16))
    self.assertEqual(locals_of(opts(out.contiguous())), [Opt(OptOps.LOCAL, 1, 256)])

  def test_plain_elementwise_keeps_its_schedule(self):
    # no broadcast: the column axis ranks first and shares every warp with an 8-row local, as before
    out = (Tensor.empty(128, 7680) + Tensor.empty(128, 7680)).contiguous()
    self.assertIn(1, [o.axis for o in locals_of(opts(out))])

  def test_one_axis_elementwise_keeps_its_schedule(self):
    # a single axis is already the stride-1 one
    self.assertEqual(locals_of(opts((Tensor.empty(7680) * 2).contiguous())), [Opt(OptOps.LOCAL, 0, 32)])

  def test_row_sum_of_squares_is_a_row_reduce_not_a_matvec(self):
    # x * x shares no operand across rows: interleaved lanes over the row, 16+ elements each (3136 = 64 * 49)
    for rows in (1, 32, 128):
      applied = opts(Tensor.empty(rows, 3136).square().mean(-1, keepdim=True).contiguous())
      self.assertEqual(applied, [Opt(OptOps.GROUP, 0, 64)], f"{rows=}")

  def test_group_sum_of_squares(self):
    applied = opts(Tensor.empty(128, 8, 960).square().mean(-1, keepdim=True).contiguous())
    self.assertEqual(applied, [Opt(OptOps.GROUP, 0, 32)])  # 960 = 32 * 30; 64 would leave 15 per lane

  def test_row_max_reduce(self):
    applied = opts(Tensor.empty(128, 8192).max(-1).contiguous())
    self.assertEqual(applied, [Opt(OptOps.GROUP, 0, 256)])

  def test_reduce_over_a_strided_axis_is_not_a_row_reduce(self):
    # summing down columns: the reduce axis is not stride-1, so lanes cannot interleave over it
    applied = opts(Tensor.empty(3136, 128).sum(0).contiguous())
    self.assertNotIn(OptOps.GROUP, [o.op for o in applied])

  def test_an_operand_shared_through_a_division_is_still_a_matvec(self):
    # the Mamba state readout: state rows [B*heads*head_dim, N] times C, indexed by row // head_dim, so each C row is
    # shared by head_dim state rows; it keeps the matvec schedule (GROUP, rows per block, rows per thread)
    from tinygrad.llm.nemotron_h_decode import _readout
    readout = _readout(Tensor.empty(128, 96, 80, 128), Tensor.empty(128, 96, 128))
    self.assertEqual(opts(readout), [Opt(OptOps.GROUP, 0, 8), Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.UPCAST, 0, 4)])

  def test_a_matvec_keeps_the_matvec_schedule(self):
    weight, x = Tensor.empty(3072, 3136), Tensor.empty(1, 1, 3136)
    applied = opts(x.linear(weight.T).contiguous())
    self.assertIn(OptOps.GROUP, [o.op for o in applied])
    self.assertIn(OptOps.LOCAL, [o.op for o in applied])  # the matvec's rows per block, which a row reduce does not use


if __name__ == "__main__":
  unittest.main()
