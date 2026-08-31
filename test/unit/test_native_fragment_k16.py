import numpy as np
from tinygrad import Device,dtypes
from tinygrad.codegen.late.native_fragment import native_fragment_x2
from tinygrad.codegen.opt.tc import cuda_81616_i8
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.device import BufferSpec
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import Ops,UOp
from extra.llm_research.prefill.nv_native_fragment_k16_gate import kernel
def test_native_k16_descriptor():
  assert cuda_81616_i8[0].dims == (8,16,16)
  assert cuda_81616_i8[0].elements_per_thread == (8,4,4)
  assert native_fragment_x2
  assert cuda_81616_i8[0] not in __import__('tinygrad.codegen.opt.tc',fromlist=['cuda_sm89']).cuda_sm89
def test_native_k16_renders_and_compiles():
  o=UOp.placeholder((128,),dtypes.int32,0); a=UOp.placeholder((64,),dtypes.uint32,1); b=UOp.placeholder((128,),dtypes.int8,2)
  p=to_program(kernel(o,a,b),CUDARenderer(Target.parse('NV:CUDA:sm_120'))); s=next(x.arg for x in p.src if x.op is Ops.SOURCE)
  NVRTCCompiler('sm_120',ptx=True,cache_key='native_fragment_k16_test').compile(s)
  assert s.count('ldmatrix.sync.aligned.m8n8.x2.b16')==1
  assert 'mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32' in s
  rng=np.random.default_rng(20260831); av=rng.integers(-127,128,(16,16),dtype=np.int8); bv=rng.integers(-127,128,(16,8),dtype=np.int8)
  d=Device['NV']; alloc=lambda n:d.allocator._alloc(n,BufferSpec())
  ob,ab,bb=alloc(128*4),alloc(av.nbytes),alloc(bv.nbytes)
  d.allocator._copyin(ab,memoryview(av.tobytes())); d.allocator._copyin(bb,memoryview(bv.tobytes()))
  NVProgram(d,'nv_native_fragment_k16',NVRTCCompiler(d.arch,ptx=False,cache_key='native_fragment_k16_runtime').compile(s))(
    ob,ab,bb,global_size=(1,1,1),local_size=(32,1,1),wait=True)
  out=memoryview(bytearray(ob.size)); d.allocator._copyout(out,ob)
  assert np.array_equal(np.frombuffer(out,np.int32,count=128).reshape(16,8),av.astype(np.int32)@bv.astype(np.int32))
