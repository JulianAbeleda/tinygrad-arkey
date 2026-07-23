from tinygrad import dtypes
from tinygrad.codegen.opt.compiler_policies import WaitCount
from tinygrad.dtype import AddrSpace
from tinygrad.renderer.isa.amd import native_repack_matcher
from tinygrad.uop.ops import AMDAttentionGridSpec, AxisType, Ops, ParamArg, RotatingPVSequentialDrainSpec, RotatingPVStateSpec, UOp, graph_rewrite
from tinygrad.uop.spec import spec_full, type_verify


def test_rotating_pv_state_uops_are_typed_and_compile_only():
  storage = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(2048, AddrSpace.LOCAL), arg=2201)
  state = RotatingPVStateSpec(storage, UOp.special(32, "lidx0"), block=3, generation=1)
  written = state.write(UOp.const(dtypes.float.vec(8), 1.0))
  published = UOp(Ops.WAIT, dtypes.void, (written,), WaitCount(lgkmcnt=0))
  reloaded = state.read(published)
  assert state.block_offset(6).op is Ops.ADD
  assert written.arg == ("rotating_pv_state_write_v1", state)
  assert reloaded.arg == ("rotating_pv_state_read_v1", state)
  type_verify(UOp.sink(reloaded), spec_full)
  lowered = graph_rewrite(UOp.sink(reloaded), native_repack_matcher, bottom_up=True)
  assert not any(u.op is Ops.CUSTOMI and isinstance(u.arg, tuple) and u.arg[:1] in
                 {("rotating_pv_state_write_v1",), ("rotating_pv_state_read_v1",)} for u in lowered.toposort())
  assert sum(u.op is Ops.STORE for u in lowered.toposort()) == 8
  assert sum(u.op is Ops.LOAD for u in lowered.toposort()) == 8


def test_rotating_pv_drain_reloads_blocks_sequentially_compile_only():
  storage = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(2048, AddrSpace.LOCAL), arg=2202)
  lane = UOp.special(32, "lidx0")
  rng = UOp.range(1, 2203, AxisType.REDUCE)
  writes = tuple(RotatingPVStateSpec(storage, lane, block, generation=2).write(UOp.const(dtypes.float.vec(8), 0.0), after=rng) for block in range(8))
  initial_token = token = UOp.group(*writes).end(rng)
  grid = AMDAttentionGridSpec(q_tokens=512, q_heads=32, kv_heads=8, group_ratio=4, kv_tokens=512)
  out = UOp(Ops.PARAM, dtypes.half.ptr(512*32*128), arg=ParamArg(0))
  group = UOp.special(grid.q_heads*grid.q_tiles, "gidx0")
  final_l = UOp.const(dtypes.float.vec(8), 1.0)
  drains = []
  for block in range(8):
    state = RotatingPVStateSpec(storage, lane, block, generation=2)
    drains.append(RotatingPVSequentialDrainSpec(state, out, group, grid, final_l, block).reload(token))
    token = drains[-1]
  drains = tuple(drains)
  assert all(drain.dtype == dtypes.float.vec(8) for drain in drains)
  assert all(drain.src[3:5] == (storage, lane) for drain in drains)
  assert drains[0].src[5] is initial_token and all(drains[block].src[5] is drains[block-1] for block in range(1, 8))
  type_verify(UOp.sink(drains[-1]), spec_full)
  lowered = graph_rewrite(UOp.sink(drains[-1]), native_repack_matcher, bottom_up=True)
  assert not any(u.op is Ops.CUSTOMI and isinstance(u.arg, tuple) and u.arg[:1] == ("rotating_pv_sequential_drain_v1",)
                 for u in lowered.toposort())
  assert sum(u.op is Ops.LOAD for u in lowered.toposort()) == 64
