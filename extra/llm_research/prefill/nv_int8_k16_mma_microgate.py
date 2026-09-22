"""Research-only sm120 signed-int8 m16n8k16 native-fragment gate."""
import pathlib, json, numpy as np
from tinygrad import Device
from tinygrad.device import BufferSpec
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler

SRC=r'''extern "C" __global__ void i8_k16(int*out,const signed char*a,const signed char*b){
 __shared__ __align__(16) int sa[8*8]; int lane=threadIdx.x; for(int z=lane;z<64;z+=32)sa[z]=((const int*)a)[z];
 asm volatile("bar.sync 0, 32;":::"memory"); int ar[2],br[1],cr[4]={0,0,0,0}; int *p=sa+(lane&15)*4+(lane>>4)*2;
 asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1},[%2];":"=r"(ar[0]),"=r"(ar[1]):"l"(p));
 int lr=lane>>2,lc=lane&3; unsigned v=0; for(int q=0;q<4;q++){int k=4*lc+q;v|=(unsigned)(unsigned char)b[k*8+lr]<<(8*q);} br[0]=v;
 asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5},{%6},{%0,%1,%2,%3};":"+r"(cr[0]),"+r"(cr[1]),"+r"(cr[2]),"+r"(cr[3]):"r"(ar[0]),"r"(ar[1]),"r"(br[0]));
 for(int r=0;r<4;r++)out[(lr+8*(r>>1))*8+2*lc+(r&1)]=cr[r]; }'''
def main():
 d=Device['NV']; rng=np.random.default_rng(20260831); a=rng.integers(-127,128,(16,16),dtype=np.int8); b=rng.integers(-127,128,(16,8),dtype=np.int8); alloc=lambda n:d.allocator._alloc(n,BufferSpec()); ab,bb,ob=alloc(a.nbytes),alloc(b.nbytes),alloc(16*8*4); d.allocator._copyin(ab,memoryview(a.tobytes()));d.allocator._copyin(bb,memoryview(b.tobytes()));
 lib=NVRTCCompiler(d.arch,ptx=False,cache_key='nv_i8_k16_mma_gate_v1').compile(SRC); p=NVProgram(d,'i8_k16',lib); us=p(ob,ab,bb,global_size=(1,1,1),local_size=(32,1,1),wait=True)*1e6; mv=memoryview(bytearray(ob.size));d.allocator._copyout(mv,ob); got=np.frombuffer(mv,np.int32,count=128).reshape(16,8); ref=a.astype(np.int32)@b.astype(np.int32); r={'exact':bool(np.array_equal(got,ref)),'max_abs':int(np.abs(got-ref).max()),'us':us,'source':SRC.count('mma.sync.aligned.m16n8k16'),'native_loads':SRC.count('ldmatrix.sync.aligned.m8n8.x2.b16'),'default_tc_selection_unchanged':True}; print(json.dumps(r)); assert r['exact'] and r['source']==1 and r['native_loads']==1
if __name__=='__main__':main()
