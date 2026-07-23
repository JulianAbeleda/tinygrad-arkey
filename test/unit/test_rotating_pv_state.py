from tinygrad import dtypes
from tinygrad.codegen.opt.compiler_policies import WaitCount
from tinygrad.dtype import AddrSpace
from tinygrad.renderer.isa.amd import native_repack_matcher
from tinygrad.uop.ops import Ops, RotatingPVStateSpec, UOp, graph_rewrite
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
