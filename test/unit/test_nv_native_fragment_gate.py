from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import Ops,UOp
from extra.llm_research.prefill.nv_native_fragment_gate import emit_native_fragment_imma,emit_native_fragment_readback

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
