import itertools
import pytest

from tinygrad import dtypes
from tinygrad.codegen.opt.compiler_policies import WaveLDSFence
from tinygrad.renderer.cstyle import _render_hip_wait
from tinygrad.renderer.isa.amd import expand_native_row_softmax_repack, _validate_fragment_lane_provenance
from tinygrad.uop.ops import AMDAttentionGridSpec, AMDMultiWaveAttentionGridSpec, AMDRowSoftmaxRepackSpec, Ops, UOp


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
  with pytest.raises(ValueError, match="one LDS slice per wave"):
    WaveLDSFence(workgroup_size=64)
  admitted = WaveLDSFence(workgroup_size=64, wave_slices=((0, 512), (512, 512)))
  assert admitted.wave_slices == ((0, 512), (512, 512))
  with pytest.raises(ValueError, match="disjoint"):
    WaveLDSFence(workgroup_size=64, wave_slices=((0, 640), (512, 512)))
  with pytest.raises(ValueError, match="wave scope"):
    WaveLDSFence(scope="workgroup")


def test_native_repack_uses_wave_wait_only_for_one_wave_workgroup():
  admitted = _expanded(32).toposort()
  barriers = [x for x in admitted if x.op is Ops.BARRIER]
  assert len(barriers) == 1 and isinstance(barriers[0].arg, WaveLDSFence)
  assert barriers[0].src[0].op is Ops.GROUP

  fallback = _expanded(64).toposort()
  barriers = [x for x in fallback if x.op is Ops.BARRIER]
  assert len(barriers) == 1 and barriers[0].arg is None


def test_multiwave_repack_proves_disjoint_probability_slices():
  grid = AMDMultiWaveAttentionGridSpec(q_tokens=32, q_heads=4, kv_heads=2, kv_tokens=64).validate()
  score = UOp(Ops.WMMA, dtypes.float.vec(8), (
    UOp.const(dtypes.half.vec(16), 0), UOp.const(dtypes.half.vec(16), 0), UOp.const(dtypes.float.vec(8), 0)),
    arg=("__WMMA_16_16_16_half_float", (16, 16, 16), dtypes.float, (32,)))
  state, rng, group = UOp.const(dtypes.float.vec(8), 0), UOp.range(4, 0, 4), UOp.special(grid.grid_size, "gidx0")
  spec = AMDRowSoftmaxRepackSpec(mode="loop_state_v1", validity_mode="tail_v1", kv_start=-1, valid_kv=64,
                                 dynamic_kv_v1=True, grid=grid)
  topo = expand_native_row_softmax_repack(itertools.count(), UOp(Ops.AMD_ROW_SOFTMAX_REPACK, dtypes.half.vec(16),
    (score, state, state, rng, group), arg=spec)).toposort()
  barriers = [x for x in topo if x.op is Ops.BARRIER]
  assert len(barriers) == 1 and barriers[0].arg == WaveLDSFence(workgroup_size=64, wave_slices=((0, 512), (512, 512)))
  assert any(x.op is Ops.SHR and x.src[1].op is Ops.CONST and x.src[1].arg == 5 for x in topo)
  assert any(x.op is Ops.AND and x.src[1].op is Ops.CONST and x.src[1].arg == 31 for x in topo)


def test_multiwave_fragment_lane_provenance_fails_closed():
  tid = UOp.special(64, "lidx0")
  lane = tid & 31
  wave_id = tid >> 5
  col = lane & 15
  assert _validate_fragment_lane_provenance(lane, wave_id, col, True) is tid
  with pytest.raises(ValueError, match="exact lane"):
    _validate_fragment_lane_provenance(tid & 15, wave_id, col, True)
  with pytest.raises(ValueError, match="exact lane"):
    _validate_fragment_lane_provenance(lane, tid >> 4, col, True)
  with pytest.raises(ValueError, match="exact lane"):
    _validate_fragment_lane_provenance(lane, wave_id, tid & 15, True)
