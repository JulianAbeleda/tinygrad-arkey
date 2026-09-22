from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import KernelInfo,Ops,UOp
from tinygrad.codegen.late.native_fragment import PackedFragmentSpec, native_q4_a_fragment, NATIVE_FRAGMENT_X4
from tinygrad.codegen.late.expander import expand_native_q4_a_fragment
from tinygrad.dtype import AddrSpace
from extra.llm_research.prefill.nv_native_fragment_gate import emit_native_fragment_imma,emit_native_fragment_readback,emit_q6k_k64_fragment_readback

def test_q6k_packed_fragment_spec_rejects_non_single_phase():
  assert PackedFragmentSpec.q6k_k64().fragment_shape == (8,8)
  try: PackedFragmentSpec("Q6_K", 2, (8,8), "mma_b", phases=4).validate()
  except ValueError: pass
  else: raise AssertionError("multi-phase Q6 fragment contract must fail closed")

def test_q4_a_marker_scalarizes_pointer_before_native_x4():
  lds=UOp.placeholder((20480,),dtypes.char,99,addrspace=AddrSpace.LOCAL)
  expanded=expand_native_q4_a_fragment(native_q4_a_fragment(lds,UOp.const(dtypes.int,80)))
  carrier=expanded.src[0]
  assert carrier.arg==(NATIVE_FRAGMENT_X4,) and carrier.src[0].op is Ops.INDEX
  assert carrier.src[1].op is Ops.CONST and carrier.src[1].arg==0

def test_q4_a_large_local_marker_does_not_emit_allocation_width_vector_types(monkeypatch):
  def kernel(out:UOp) -> UOp:
    lds=UOp.placeholder((20480,),dtypes.char,99,addrspace=AddrSpace.LOCAL)
    ready=UOp.barrier(lds.index(UOp.const(dtypes.int,0)).store(UOp.const(dtypes.char,0)).end())
    fragment=native_q4_a_fragment(lds.after(ready),UOp.const(dtypes.int,0))
    return UOp.sink(out.index(UOp.const(dtypes.weakint,0),dtype=dtypes.char.vec(16)).store(fragment),arg=KernelInfo(name="q4_a_x4_local"))
  out=UOp.placeholder((16,),dtypes.char,0)
  monkeypatch.setattr(NVRTCCompiler,"compile",lambda self,src:b"render-only")
  program=to_program(kernel(out),CUDARenderer(Target.parse("NV:CUDA:sm_120")))
  src=next(x.arg for x in program.src if x.op is Ops.SOURCE)
  assert "ldmatrix.sync.aligned.m8n8.x4.b16" in src
  assert "signed_char20480" not in src and "unsigned_int20480" not in src

def test_q6k_k64_fragment_gate_emits_native_x2():
  out=UOp.placeholder((64,),dtypes.uint32,0); source=UOp.placeholder((128,),dtypes.uint32,1)
  program=to_program(emit_q6k_k64_fragment_readback()(out,source),CUDARenderer(Target.parse("NV:CUDA:sm_120")))
  src=next(x.arg for x in program.src if x.op is Ops.SOURCE)
  ptx=NVRTCCompiler("sm_120",ptx=True,cache_key="q6k_k64_fragment_gate_v1").compile(src).decode()
  assert "ldmatrix.sync.aligned.m8n8.x2.b16" in src and "ldmatrix.sync.aligned.m8n8.x2" in ptx

def test_native_fragment_x4_renders_one_ldmatrix_and_compiles():
  out=UOp.placeholder((128,),dtypes.uint32,0); source=UOp.placeholder((128,),dtypes.uint32,1)
  program=to_program(emit_native_fragment_readback()(out,source),CUDARenderer(Target.parse("NV:CUDA:sm_120")))
  src=next(x.arg for x in program.src if x.op is Ops.SOURCE)
  ptx=NVRTCCompiler("sm_120",ptx=True,cache_key="native_fragment_x4_readback_v1").compile(src).decode()
  assert program.arg.local_size==(32,1,1)
  assert src.count("ldmatrix.sync.aligned.m8n8.x4.b16")==1
  assert "ldmatrix.sync.aligned.m8n8.x4" in ptx

def test_native_fragment_feeds_existing_signed_imma_abi():
  out=UOp.placeholder((128,),dtypes.int32,0); a=UOp.placeholder((128,),dtypes.uint32,1); b=UOp.placeholder((256,),dtypes.int8,2)
  program=to_program(emit_native_fragment_imma()(out,a,b),CUDARenderer(Target.parse("NV:CUDA:sm_120")))
  src=next(x.arg for x in program.src if x.op is Ops.SOURCE)
  ptx=NVRTCCompiler("sm_120",ptx=True,cache_key="native_fragment_x4_imma_v1").compile(src).decode()
  assert src.count("ldmatrix.sync.aligned.m8n8.x4.b16")==1
  assert ptx.count("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32")==1
