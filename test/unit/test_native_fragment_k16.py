from tinygrad import dtypes
from tinygrad.codegen.late.native_fragment import native_fragment_x2
from tinygrad.codegen.opt.tc import cuda_81616_i8
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import Ops,UOp
from extra.llm_research.prefill.nv_native_fragment_k16_gate import kernel
def test_native_k16_descriptor():
  assert cuda_81616_i8[0].dims == (8,16,16)
  assert cuda_81616_i8[0].elements_per_thread == (8,4,2)
  assert native_fragment_x2
  assert cuda_81616_i8[0] not in __import__('tinygrad.codegen.opt.tc',fromlist=['cuda_sm89']).cuda_sm89
def test_native_k16_renders_and_compiles():
  o=UOp.placeholder((128,),dtypes.int32,0); a=UOp.placeholder((64,),dtypes.uint32,1); b=UOp.placeholder((128,),dtypes.int8,2)
  p=to_program(kernel(o,a,b),CUDARenderer(Target.parse('NV:CUDA:sm_120'))); s=next(x.arg for x in p.src if x.op is Ops.SOURCE)
  NVRTCCompiler('sm_120',ptx=True,cache_key='native_fragment_k16_test').compile(s)
  assert s.count('ldmatrix.sync.aligned.m8n8.x2.b16')==1
  assert 'mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32' in s
