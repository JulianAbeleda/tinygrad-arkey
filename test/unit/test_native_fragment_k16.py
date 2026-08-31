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
from tinygrad.dtype import AddrSpace
from extra.llm_research.prefill.nv_native_fragment_k16_gate import (kernel,q6_two_k16_kernel,q6_packed_two_k16_kernel,
  q6_packed_k256_kernel,q6_first_two_k16_numpy)
from extra.llm_research.prefill.nv_native_fragment_k16_gate import q6_packed_kblocks_kernel
from extra.llm_research.prefill.nv_native_fragment_k16_gate import q6_packed_cta_k512_kernel,q6_packed_cta_kernel
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

def test_reused_native_fragment_is_materialized_once():
  from tinygrad.codegen.late.native_fragment import _lower, NATIVE_FRAGMENT_TAG, NATIVE_FRAGMENT_X2
  class Provider:
    native_fragment_x2=staticmethod(lambda buffer,index: UOp(Ops.CUSTOMI,dtypes.uint32.vec(2),(buffer,index),arg='typed_fragment'))
  buf=UOp.placeholder((64,),dtypes.uint32,20,addrspace=AddrSpace.LOCAL)
  lowered=_lower(Provider(), native_fragment_x2(buf,UOp.const(dtypes.int,0)))
  assert lowered is not None and lowered.tag == (NATIVE_FRAGMENT_TAG, NATIVE_FRAGMENT_X2)

def test_q6_two_k16_scale_semantics():
  ph=lambda n,dt,i:UOp.placeholder((n,),dt,i)
  args=(ph(128,dtypes.float32,0),ph(128,dtypes.int32,1),ph(128,dtypes.int32,2),ph(256,dtypes.int8,3),ph(128,dtypes.int8,4),
        ph(1,dtypes.int8,5),ph(1,dtypes.float32,6),ph(256,dtypes.int8,7),ph(128,dtypes.int8,8),ph(1,dtypes.int8,9),ph(1,dtypes.float32,10))
  p=to_program(q6_two_k16_kernel(*args),CUDARenderer(Target.parse('NV:CUDA:sm_120'))); src=next(x.arg for x in p.src if x.op is Ops.SOURCE)
  assert src.count('mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32')==1 and src.count('__WMMA_8_16_16_signed_char_int(')==3
  rng=np.random.default_rng(20260901); a0=rng.integers(-32,32,(16,16),dtype=np.int8); a1=rng.integers(-32,32,(16,16),dtype=np.int8)
  b0=rng.integers(-127,128,(16,8),dtype=np.int8); b1=rng.integers(-127,128,(16,8),dtype=np.int8)
  sv0,sv1=np.array([-17],np.int8),np.array([23],np.int8); dv,dbv=np.array([.03125],np.float32),np.array([.0625],np.float32)
  host=(np.empty(128,np.float32),np.empty(128,np.int32),np.empty(128,np.int32),a0,b0,sv0,dv,a1,b1,sv1,dbv)
  dev=Device['NV']; bufs=[dev.allocator._alloc(x.nbytes,BufferSpec()) for x in host]
  for buf,x in zip(bufs[3:],host[3:]): dev.allocator._copyin(buf,memoryview(x.tobytes()))
  NVProgram(dev,'nv_native_fragment_q6_two_k16',NVRTCCompiler(dev.arch,ptx=False,cache_key='q6_two_k16_runtime').compile(src))(
    *bufs,global_size=(1,1,1),local_size=(32,1,1),wait=True)
  got=[]
  for buf,dt in zip(bufs[:3],(np.float32,np.int32,np.int32)):
    mv=memoryview(bytearray(buf.size)); dev.allocator._copyout(mv,buf); got.append(np.frombuffer(mv,dt,count=128).reshape(16,8))
  ref0=a0.astype(np.int32)@b0.astype(np.int32); ref1=a1.astype(np.int32)@b1.astype(np.int32)
  ref=(dv[0]*dbv[0])*(np.float32(sv0[0])*ref0.astype(np.float32)+np.float32(sv1[0])*ref1.astype(np.float32))
  assert np.array_equal(got[1],ref0) and np.array_equal(got[2],ref1) and np.array_equal(got[0],ref)

def test_q6_direct_packed_two_k16():
  ph=lambda n,dt,i:UOp.placeholder((n,),dt,i)
  args=(ph(128,dtypes.float32,0),ph(128,dtypes.int32,1),ph(128,dtypes.int32,2),ph(16*105,dtypes.uint16,3),
        ph(128,dtypes.int8,4),ph(128,dtypes.int8,5),ph(8,dtypes.float32,6))
  p=to_program(q6_packed_two_k16_kernel(*args),CUDARenderer(Target.parse('NV:CUDA:sm_120'))); src=next(x.arg for x in p.src if x.op is Ops.SOURCE)
  assert src.count('__WMMA_8_16_16_signed_char_int(')==3 and '1680' in src
  rng=np.random.default_rng(20260902); blocks=rng.integers(0,256,(16,210),dtype=np.uint8)
  blocks[:,208:210]=np.frombuffer(np.float16(.03125).tobytes(),np.uint8)
  b0=rng.integers(-127,128,(16,8),dtype=np.int8); b1=rng.integers(-127,128,(16,8),dtype=np.int8)
  q,scales,wd=q6_first_two_k16_numpy(blocks.tobytes()); ref0=q[:,0].astype(np.int32)@b0.astype(np.int32); ref1=q[:,1].astype(np.int32)@b1.astype(np.int32)
  db=np.array([.0625,.03125,.125,.015625,.25,.0078125,.5,.00390625],np.float32)
  ref=(wd[:,None]*db[None,:])*(scales[:,0,None].astype(np.float32)*ref0+scales[:,1,None].astype(np.float32)*ref1)
  dev=Device['NV']; host=(np.empty(128,np.float32),np.empty(128,np.int32),np.empty(128,np.int32),blocks,b0,b1,db)
  bufs=[dev.allocator._alloc(x.nbytes,BufferSpec()) for x in host]
  for buf,x in zip(bufs[3:],host[3:]): dev.allocator._copyin(buf,memoryview(x.tobytes()))
  NVProgram(dev,'nv_native_fragment_q6_packed_two_k16',NVRTCCompiler(dev.arch,ptx=False,cache_key='q6_packed_two_k16_runtime').compile(src))(
    *bufs,global_size=(1,1,1),local_size=(32,1,1),wait=True)
  got=[]
  for buf,dt in zip(bufs[:3],(np.float32,np.int32,np.int32)):
    mv=memoryview(bytearray(buf.size)); dev.allocator._copyout(mv,buf); got.append(np.frombuffer(mv,dt,count=128).reshape(16,8))
  assert np.array_equal(got[1],ref0) and np.array_equal(got[2],ref1) and np.array_equal(got[0],ref)

def test_q6_direct_packed_full_block():
  ph=lambda n,dt,i:UOp.placeholder((n,),dt,i)
  args=(ph(128,dtypes.float32,0),ph(128,dtypes.float32,1),ph(16*105,dtypes.uint16,2),ph(256*8,dtypes.int8,3),ph(8*8,dtypes.float32,4))
  p=to_program(q6_packed_k256_kernel(*args),CUDARenderer(Target.parse('NV:CUDA:sm_120'))); src=next(x.arg for x in p.src if x.op is Ops.SOURCE)
  assert src.count('__WMMA_8_16_16_signed_char_int(')==17
  rng=np.random.default_rng(20260903); blocks=rng.integers(0,256,(16,210),dtype=np.uint8)
  blocks[:,208:210]=np.frombuffer(np.float16(.03125).tobytes(),np.uint8); b=rng.integers(-8,9,(256,8),dtype=np.int8)
  db=np.full((8,8),.0625,np.float32); raw=blocks
  q=np.empty((16,16,16),np.int8)
  for g in range(16):
    half,pgrp=g//8,g%8; qi=half*64+(pgrp%4)*16; hi=half*32+(pgrp%2)*16
    q[:,g]=((((raw[:,qi:qi+16]>>(4 if pgrp>=4 else 0))&15)|
      (((raw[:,128+hi:128+hi+16]>>((pgrp//2)*2))&3)<<4)).astype(np.int16)-32).astype(np.int8)
  scales=raw[:,192:208].view(np.int8); wd=np.full((16,),.03125,np.float32); ref=np.zeros((16,8),np.float32)
  for pidx in range(8):
    z0=q[:,2*pidx].astype(np.int32)@b[32*pidx:32*pidx+16].astype(np.int32)
    z1=q[:,2*pidx+1].astype(np.int32)@b[32*pidx+16:32*pidx+32].astype(np.int32)
    ref += (wd[:,None]*db[pidx])*(scales[:,2*pidx,None].astype(np.float32)*z0+scales[:,2*pidx+1,None].astype(np.float32)*z1)
  dev=Device['NV']; host=(np.empty(128,np.float32),np.empty(128,np.float32),blocks,b,db)
  bufs=[dev.allocator._alloc(x.nbytes,BufferSpec()) for x in host]
  for buf,x in zip(bufs[2:],host[2:]): dev.allocator._copyin(buf,memoryview(x.tobytes()))
  NVProgram(dev,'nv_native_fragment_q6_packed_k256',NVRTCCompiler(dev.arch,ptx=False,cache_key='q6_packed_k256_runtime').compile(src))(
    *bufs,global_size=(1,1,1),local_size=(32,1,1),wait=True)
  mv=memoryview(bytearray(bufs[0].size)); dev.allocator._copyout(mv,bufs[0]); got=np.frombuffer(mv,np.float32,count=128).reshape(16,8)
  assert np.array_equal(got,ref)

def test_q6_looped_packed_k512():
  kb=2; ph=lambda n,dt,i:UOp.placeholder((n,),dt,i)
  args=(ph(128,dtypes.float32,0),ph(16*kb*105,dtypes.uint16,1),ph(kb*256*8,dtypes.int8,2),ph(kb*8*8,dtypes.float32,3))
  p=to_program(q6_packed_kblocks_kernel(kb)(*args),CUDARenderer(Target.parse('NV:CUDA:sm_120'))); src=next(x.arg for x in p.src if x.op is Ops.SOURCE)
  assert src.count('__WMMA_8_16_16_signed_char_int(')==17
  rng=np.random.default_rng(20260904); blocks=rng.integers(0,256,(16,kb,210),dtype=np.uint8)
  blocks[:,:,208:210]=np.frombuffer(np.float16(.03125).tobytes(),np.uint8); b=rng.integers(-8,9,(kb*256,8),dtype=np.int8)
  db=np.full((kb,8,8),.0625,np.float32); ref=np.zeros((16,8),np.float32)
  for bi in range(kb):
    raw=blocks[:,bi]; q=np.empty((16,16,16),np.int8)
    for g in range(16):
      half,pgrp=g//8,g%8; qi=half*64+(pgrp%4)*16; hi=half*32+(pgrp%2)*16
      q[:,g]=((((raw[:,qi:qi+16]>>(4 if pgrp>=4 else 0))&15)|
        (((raw[:,128+hi:128+hi+16]>>((pgrp//2)*2))&3)<<4)).astype(np.int16)-32).astype(np.int8)
    scales=raw[:,192:208].view(np.int8)
    for pidx in range(8):
      z0=q[:,2*pidx].astype(np.int32)@b[bi*256+32*pidx:bi*256+32*pidx+16].astype(np.int32)
      z1=q[:,2*pidx+1].astype(np.int32)@b[bi*256+32*pidx+16:bi*256+32*pidx+32].astype(np.int32)
      ref += (.03125*db[bi,pidx])*(scales[:,2*pidx,None].astype(np.float32)*z0+scales[:,2*pidx+1,None].astype(np.float32)*z1)
  dev=Device['NV']; host=(np.empty(128,np.float32),blocks,b,db); bufs=[dev.allocator._alloc(x.nbytes,BufferSpec()) for x in host]
  for buf,x in zip(bufs[1:],host[1:]): dev.allocator._copyin(buf,memoryview(x.tobytes()))
  NVProgram(dev,'nv_native_fragment_q6_packed_k512',NVRTCCompiler(dev.arch,ptx=False,cache_key='q6_packed_k512_runtime').compile(src))(
    *bufs,global_size=(1,1,1),local_size=(32,1,1),wait=True)
  mv=memoryview(bytearray(bufs[0].size)); dev.allocator._copyout(mv,bufs[0]); got=np.frombuffer(mv,np.float32,count=128).reshape(16,8)
  assert np.array_equal(got,ref)

def test_q6_cta_128x16x512():
  kb=2; ph=lambda n,dt,i:UOp.placeholder((n,),dt,i)
  args=(ph(128*16,dtypes.float32,0),ph(128*kb*105,dtypes.uint16,1),ph(kb*256*16,dtypes.int8,2),ph(kb*8*16,dtypes.float32,3))
  p=to_program(q6_packed_cta_k512_kernel(*args),CUDARenderer(Target.parse('NV:CUDA:sm_120'))); src=next(x.arg for x in p.src if x.op is Ops.SOURCE)
  assert src.count('__WMMA_8_16_16_signed_char_int(')==33
  rng=np.random.default_rng(20260905); blocks=rng.integers(0,256,(128,kb,210),dtype=np.uint8)
  blocks[:,:,208:210]=np.frombuffer(np.float16(.03125).tobytes(),np.uint8); b=rng.integers(-4,5,(kb*256,16),dtype=np.int8)
  db=np.full((kb,8,16),.0625,np.float32); ref=np.zeros((128,16),np.float32)
  for bi in range(kb):
    raw=blocks[:,bi]; q=np.empty((128,16,16),np.int8)
    for g in range(16):
      half,pgrp=g//8,g%8; qi=half*64+(pgrp%4)*16; hi=half*32+(pgrp%2)*16
      q[:,g]=((((raw[:,qi:qi+16]>>(4 if pgrp>=4 else 0))&15)|
        (((raw[:,128+hi:128+hi+16]>>((pgrp//2)*2))&3)<<4)).astype(np.int16)-32).astype(np.int8)
    scales=raw[:,192:208].view(np.int8)
    for pi in range(8):
      z0=q[:,2*pi].astype(np.int32)@b[bi*256+32*pi:bi*256+32*pi+16].astype(np.int32)
      z1=q[:,2*pi+1].astype(np.int32)@b[bi*256+32*pi+16:bi*256+32*pi+32].astype(np.int32)
      ref += (.03125*db[bi,pi])*(scales[:,2*pi,None].astype(np.float32)*z0+scales[:,2*pi+1,None].astype(np.float32)*z1)
  dev=Device['NV']; host=(np.empty(128*16,np.float32),blocks,b,db); bufs=[dev.allocator._alloc(x.nbytes,BufferSpec()) for x in host]
  for buf,x in zip(bufs[1:],host[1:]): dev.allocator._copyin(buf,memoryview(x.tobytes()))
  NVProgram(dev,'nv_native_fragment_q6_cta_128x16x512',NVRTCCompiler(dev.arch,ptx=False,cache_key='q6_cta_128x16x512').compile(src))(
    *bufs,global_size=(1,1,1),local_size=(256,1,1),wait=True)
  mv=memoryview(bytearray(bufs[0].size)); dev.allocator._copyout(mv,bufs[0]); got=np.frombuffer(mv,np.float32,count=128*16).reshape(128,16)
  assert np.array_equal(got,ref)

def test_q6_cta_nonzero_segment_numpy_exact():
  """A block-1 segment uses absolute packed/Q8/dB indexing and exact FP32 scaling."""
  rng=np.random.default_rng(20260906); total=2
  blocks=rng.integers(0,256,(128,total,210),dtype=np.uint8)
  blocks[:,:,208:210]=np.frombuffer(np.float16(.03125).tobytes(),np.uint8)
  b=rng.integers(-4,5,(total*256,16),dtype=np.int8); db=np.full((total,8,16),.0625,np.float32)
  def contribution(bi):
    raw=blocks[:,bi]; q=np.empty((128,16,16),np.int8)
    for g in range(16):
      half,pgrp=g//8,g%8; qi=half*64+(pgrp%4)*16; hi=half*32+(pgrp%2)*16
      q[:,g]=((((raw[:,qi:qi+16]>>(4 if pgrp>=4 else 0))&15)|
        (((raw[:,128+hi:128+hi+16]>>((pgrp//2)*2))&3)<<4)).astype(np.int16)-32).astype(np.int8)
    scales=raw[:,192:208].view(np.int8); out=np.zeros((128,16),np.float32)
    for pi in range(8):
      z0=q[:,2*pi].astype(np.int32)@b[bi*256+32*pi:bi*256+32*pi+16].astype(np.int32)
      z1=q[:,2*pi+1].astype(np.int32)@b[bi*256+32*pi+16:bi*256+32*pi+32].astype(np.int32)
      out += (.03125*db[bi,pi])*(scales[:,2*pi,None].astype(np.float32)*z0+
        scales[:,2*pi+1,None].astype(np.float32)*z1)
    return out
  full=contribution(0)+contribution(1)
  segment=contribution(1)
  assert np.array_equal(segment,full-contribution(0))

def test_q6_cta_accepts_runtime_uniform_segment_bounds():
  """The Stream-K owner loop supplies scalar K bounds at runtime, not Python constants."""
  ph=lambda n,dt,i:UOp.placeholder((n,),dt,i)
  bound=ph(1,dtypes.int32,4)[0].load()
  ast=q6_packed_cta_kernel(ph(128*128,dtypes.float32,0),ph(128*48*105,dtypes.uint16,1),
    ph(48*256*128,dtypes.int8,2),ph(48*8*128,dtypes.float32,3),48,col_groups=8,
    block_start=bound,segment_blocks=bound,total_k_blocks=48)
  src=next(x.arg for x in to_program(ast,CUDARenderer(Target.parse('NV:CUDA:sm_120'))).src if x.op is Ops.SOURCE)
  assert 'for (' in src and 'mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32' in src
