import numpy as np
from tinygrad import Device,dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.device import BufferSpec
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import Ops,UOp
from extra.llm_research.prefill.nv_generated_q6k_streamk_slots import q6_streamk_slot_kernel

def test_streamk_slot_boundary_segment_runtime():
  ph=lambda n,dt,i: UOp.placeholder((n,),dt,i)
  ast=q6_streamk_slot_kernel(ph(16384,dtypes.float32,0),ph(1,dtypes.int32,1),ph(3,dtypes.int32,2),
    ph(128*2*105,dtypes.uint16,3),ph(2*256*512,dtypes.int8,4),ph(2*8*512,dtypes.float32,5),
    total_k_blocks=2,slots=1,max_segment_blocks=1)
      src=next(x.arg for x in to_program(ast,CUDARenderer(Target.parse('NV:CUDA:sm_120'))).src if x.op is Ops.SOURCE)
      assert 'float buf1[1]' not in src
  rng=np.random.default_rng(20260907); raw=rng.integers(0,256,(128,2,210),dtype=np.uint8)
  raw[:,:,208:210]=np.frombuffer(np.float16(.03125).tobytes(),np.uint8); blocks=raw.view(np.uint16).reshape(-1)
  b=rng.integers(-4,5,(2*256,512),dtype=np.int8); db=np.full((2,8,512),.0625,np.float32)
  q=np.empty((128,16,16),np.int8); r=raw[:,1]
  for g in range(16):
    h,p=g//8,g%8; qi=h*64+(p%4)*16; hi=h*32+(p%2)*16
    q[:,g]=((((r[:,qi:qi+16]>>(4 if p>=4 else 0))&15)|(((r[:,128+hi:128+hi+16]>>((p//2)*2))&3)<<4)).astype(np.int16)-32).astype(np.int8)
  ref=np.zeros((128,128),np.float32); sc=r[:,192:208].view(np.int8)
  for p in range(8):
    z0=q[:,2*p].astype(np.int32)@b[256+32*p:256+32*p+16,:128].astype(np.int32)
    z1=q[:,2*p+1].astype(np.int32)@b[256+32*p+16:256+32*p+32,:128].astype(np.int32)
    ref += (.03125*db[1,p,:128])[None,:]*(sc[:,2*p,None]*z0+sc[:,2*p+1,None]*z1)
  dev=Device['NV']; host=(np.empty(16384,np.float32),np.empty(1,np.int32),np.array([0,1,2],np.int32),blocks,b,db)
  bufs=[dev.allocator._alloc(x.nbytes,BufferSpec()) for x in host]
  for buf,x in zip(bufs[2:],host[2:]): dev.allocator._copyin(buf,memoryview(x.tobytes()))
  NVProgram(dev,'nv_generated_q6k_streamk_slots',NVRTCCompiler(dev.arch,ptx=False,cache_key='q6_streamk_slots_runtime').compile(src))(*bufs,global_size=(1,1,1),local_size=(256,1,1),wait=True)
  mv=memoryview(bytearray(bufs[0].size)); dev.allocator._copyout(mv,bufs[0]); got=np.frombuffer(mv,np.float32,count=16384).reshape(128,128).T
  mid=memoryview(bytearray(bufs[1].size)); dev.allocator._copyout(mid,bufs[1])
  assert np.array_equal(got,ref); assert np.frombuffer(mid,np.int32,count=1)[0] == 0
