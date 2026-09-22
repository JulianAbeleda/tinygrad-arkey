"""Exact sm_120 m16n8k32 gate using ldmatrix.x4 for the 16x32-byte A tile."""
import numpy as np
from tinygrad import Device
from tinygrad.device import BufferSpec
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler

SRC=r'''
extern "C" __global__ void ldmatrix_i8(int*out,const signed char*a,const signed char*b){
  __shared__ __align__(16) int sa[16*8];int lane=threadIdx.x;
  for(int z=lane;z<128;z+=32)sa[z]=((const int*)a)[z];
  asm volatile("bar.sync 0, 32;":::"memory");
  int ar[4],br[2],cr[4]={0,0,0,0};
  int *p=sa+(lane&15)*8+(lane>>4)*4;
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
    :"=r"(ar[0]),"=r"(ar[1]),"=r"(ar[2]),"=r"(ar[3]):"l"(p));
  int lr=lane>>2,lc=lane&3;
  #pragma unroll
  for(int r=0;r<2;r++){unsigned v=0;
    #pragma unroll
    for(int q=0;q<4;q++){int k=4*(lc+4*r)+q,col=lr;v|=(unsigned)(unsigned char)b[k*8+col]<<(8*q);}br[r]=v;}
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
   "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
   :"+r"(cr[0]),"+r"(cr[1]),"+r"(cr[2]),"+r"(cr[3])
   :"r"(ar[0]),"r"(ar[1]),"r"(ar[2]),"r"(ar[3]),"r"(br[0]),"r"(br[1]));
  #pragma unroll
  for(int r=0;r<4;r++)out[(lr+8*(r>>1))*8+2*lc+(r&1)]=cr[r];
}'''

def run():
  dev=Device['NV'];rng=np.random.default_rng(20260828)
  a=rng.integers(-127,128,(16,32),dtype=np.int8);b=rng.integers(-127,128,(32,8),dtype=np.int8)
  alloc=lambda n:dev.allocator._alloc(n,BufferSpec())
  ab,bb,ob=alloc(a.nbytes),alloc(b.nbytes),alloc(16*8*4)
  dev.allocator._copyin(ab,memoryview(a.tobytes()));dev.allocator._copyin(bb,memoryview(b.tobytes()))
  lib=NVRTCCompiler(dev.arch,ptx=False,cache_key='nv_i8_ldmatrix_gate_v1').compile(SRC);p=NVProgram(dev,'ldmatrix_i8',lib)
  us=p(ob,ab,bb,global_size=(1,1,1),local_size=(32,1,1),wait=True)*1e6
  mv=memoryview(bytearray(ob.size));dev.allocator._copyout(mv,ob);got=np.frombuffer(mv,np.int32,count=128).reshape(16,8)
  ref=a.astype(np.int32)@b.astype(np.int32)
  print({'exact':bool(np.array_equal(got,ref)),'max_abs':int(np.abs(got-ref).max()),'us':us,
         'regs':p.regs_usage,'shared':p.shmem_usage})
  assert np.array_equal(got,ref)

if __name__=='__main__':run()
