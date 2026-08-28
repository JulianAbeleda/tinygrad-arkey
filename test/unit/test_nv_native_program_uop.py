import pytest
from tinygrad import dtypes
from tinygrad.uop.ops import CallInfo, Ops, UOp
from extra.llm_research.prefill.nv_native_program_uop import native_nv_program
from tinygrad.callify import _native_input_output_routes

ELF=b"\x7fELF"+bytes(32)

def test_native_program_carries_fixed_launch_abi_and_shared_mem():
  p=native_nv_program("native",ELF,global_size=(170,1,1),local_size=(256,1,1),globals=tuple(range(7)),
                      outs=(0,1,2),ins=(3,4,5,6),vals=(512,12288,4096),shared_mem=58880)
  assert p.op is Ops.PROGRAM and p.src[4].op is Ops.BINARY
  assert p.arg.global_size == (170,1,1) and p.arg.local_size == (256,1,1)
  assert p.arg.globals == tuple(range(7)) and p.arg.outs == (0,1,2) and p.arg.ins == (3,4,5,6)
  assert p.arg.aux == (58880,) and tuple(v.arg[1] for v in p.arg.vars) == (512,12288,4096)

def test_native_program_fails_closed_on_non_cubin_and_negative_shared():
  with pytest.raises(ValueError): native_nv_program("bad",b"PTX",global_size=(1,1,1),local_size=(1,1,1),globals=())
  with pytest.raises(ValueError): native_nv_program("bad",ELF,global_size=(1,1,1),local_size=(1,1,1),globals=(),shared_mem=-1)

def test_fixed_native_scalar_vals_need_no_graph_binding():
  p=native_nv_program("fixed",ELF,global_size=(1,1,1),local_size=(1,1,1),globals=(),vals=(512,12288,4096))
  assert p.arg.vals({}) == (512,12288,4096)

def test_native_input_route_exact_slot_and_negatives():
  val=UOp.param(0,dtypes.half,(32,),"NV")
  fn=UOp(Ops.FUNCTION,dtypes.void,(UOp.maketuple(val),val),CallInfo(precompile=True))
  got=fn.gettuple(0).reshape((32,))
  p=native_nv_program("native",ELF,global_size=(1,1,1),local_size=(32,1,1),globals=(0,1),outs=(0,),ins=(1,))
  out=UOp.param(1,dtypes.int8,(32,),"NV")
  assert _native_input_output_routes(UOp.sink(p.call(out,got))) == frozenset({(id(fn),0)})
  assert not _native_input_output_routes(UOp.sink(p.call(out,got.cast(dtypes.float)))) # dtype-changing carrier
  p2=native_nv_program("native2",ELF,global_size=(1,1,1),local_size=(32,1,1),globals=(0,1),outs=(0,),ins=(1,))
  assert not _native_input_output_routes(UOp.sink(p.call(out,got),p2.call(out,got))) # multiple consumers
