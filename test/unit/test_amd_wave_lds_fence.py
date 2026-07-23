import itertools
import pytest

from tinygrad import dtypes
from tinygrad.codegen.opt.compiler_policies import WaveLDSFence
from tinygrad.renderer.cstyle import _render_hip_wait
from tinygrad.renderer.isa.amd import expand_native_row_softmax_repack
from tinygrad.uop.ops import AMDAttentionGridSpec, AMDRowSoftmaxRepackSpec, Ops, UOp


def _expanded(local_size: int) -> UOp:
  grid = AMDAttentionGridSpec(local_size=local_size).validate()
  score = UOp(Ops.WMMA, dtypes.float.vec(8), (
    UOp.const(dtypes.half.vec(16), 0), UOp.const(dtypes.half.vec(16), 0), UOp.const(dtypes.float.vec(8), 0)),
    arg=("__WMMA_16_16_16_half_float", (16, 16, 16), dtypes.float, (32,)))
  state = UOp.const(dtypes.float.vec(8), 0)
  rng = UOp.range(4, 0, 4)
  spec = AMDRowSoftmaxRepackSpec(mode="loop_state_v1", validity_mode="tail_v1", kv_start=-1, valid_kv=64,
                                 dynamic_kv_v1=True, grid=grid)
  return expand_native_row_softmax_repack(itertools.count(),
    UOp(Ops.AMD_ROW_SOFTMAX_REPACK, dtypes.half.vec(16), (score, state, state, rng, UOp.const(dtypes.int, 0)), arg=spec))


def test_wave_lds_fence_is_exact_lgkm_wait():
  fence = WaveLDSFence()
  assert (fence.vmcnt, fence.lgkmcnt, fence.expcnt) == (63, 0, 7)
  assert _render_hip_wait(UOp(Ops.WAIT, dtypes.void, arg=fence)) == f"__builtin_amdgcn_s_waitcnt({fence.simm16});"
  with pytest.raises(ValueError, match="one gfx1100 wave32 workgroup"):
    WaveLDSFence(workgroup_size=64)


def test_native_repack_uses_wave_wait_only_for_one_wave_workgroup():
  admitted = _expanded(32).toposort()
  barriers = [x for x in admitted if x.op is Ops.BARRIER]
  assert len(barriers) == 1 and isinstance(barriers[0].arg, WaveLDSFence)
  assert barriers[0].src[0].op is Ops.GROUP

  fallback = _expanded(64).toposort()
  barriers = [x for x in fallback if x.op is Ops.BARRIER]
  assert len(barriers) == 1 and barriers[0].arg is None
