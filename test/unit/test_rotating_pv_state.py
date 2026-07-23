from tinygrad import dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import Ops, RotatingPVStateSpec, UOp
from tinygrad.uop.spec import spec_full, type_verify


def test_rotating_pv_state_uops_are_typed_and_compile_only():
  storage = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(2048, AddrSpace.LOCAL), arg=2201)
  state = RotatingPVStateSpec(storage, UOp.special(32, "lidx0"), block=3, generation=1)
  written = state.write(UOp.const(dtypes.float.vec(8), 1.0))
  reloaded = state.read(written)
  assert state.block_offset(6).op is Ops.ADD
  assert written.arg == ("rotating_pv_state_write_v1", state)
  assert reloaded.arg == ("rotating_pv_state_read_v1", state)
  type_verify(UOp.sink(reloaded), spec_full)
