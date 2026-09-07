"""Exact dual-layout Q8 producer for pp512 Q/K activation sharing."""
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from extra.llm_research.prefill.nv_native_program_uop import native_nv_program

M,K=512,4096
FLAT_U32=(M*K+2*M*(K//32)*4)//4
TILE_U32=(M*K//128*144+128*144)//4

SOURCE=r'''
#include <cuda_fp16.h>
#include <cuda_runtime.h>
struct __align__(4) block_q8_1_ds4 { __half2 ds[4]; signed char qs[128]; };
__device__ __forceinline__ signed char tg_round_i8(float v,float d){
 float y=v*(1.0f/d),a=y-0.5f,b=y+0.5f,ta=truncf(a),tb=truncf(b),th=truncf(y)*0.5f;
 float lo=ta<a?ta+1.0f:ta,hi=b<tb?tb-1.0f:tb;
 float r=((0.0f<y)!=(truncf(th)==th))?hi:lo;
 r=fminf(127.0f,fmaxf(-128.0f,r));return (signed char)r;
}
extern "C" __global__ __launch_bounds__(128,1) void q8_dual_flat_tile_fp16(
 const half* __restrict__ x, unsigned int* __restrict__ flat_record, block_q8_1_ds4* __restrict__ tile_record) {
 signed char* __restrict__ q=(signed char*)flat_record;
 float* __restrict__ scales=(float*)(q+2097152);
 float* __restrict__ sums=scales+65536;
 const int row=blockIdx.x, seg=blockIdx.y, t=threadIdx.x, i=seg*512+t*4, base=row*4096+i;
 const half2 h0=*reinterpret_cast<const half2*>(x+base),h1=*reinterpret_cast<const half2*>(x+base+2);
 const float4 v=make_float4(__half2float(__low2half(h0)),__half2float(__high2half(h0)),
                            __half2float(__low2half(h1)),__half2float(__high2half(h1)));
 float a=fmaxf(fmaxf(fabsf(v.x),fabsf(v.y)),fmaxf(fabsf(v.z),fabsf(v.w)));
 float s=(v.x+v.y)+(v.z+v.w);
 #pragma unroll
 for(int off=4;off;off>>=1){a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,off));s+=__shfl_xor_sync(0xffffffff,s,off);}
 const float d=a==0.0f?1.0f:a*0x1.020408p-7f;
 const char4 qflat=make_char4(tg_round_i8(v.x,d),tg_round_i8(v.y,d),tg_round_i8(v.z,d),tg_round_i8(v.w,d));
 *reinterpret_cast<char4*>(q+base)=qflat;
 if((t&7)==0){const int g=row*128+seg*16+t/8;scales[g]=d;sums[g]=__half2float(__float2half_rn(s));}
 const float dinv=127.0f/a;
 const char4 qtile=make_char4(roundf(v.x*dinv),roundf(v.y*dinv),roundf(v.z*dinv),roundf(v.w*dinv));
 const int iqs=i&127, ib=(i>>7)*512+row;
 reinterpret_cast<char4*>(tile_record[ib].qs)[iqs>>2]=qtile;
 if((iqs&31)==0) tile_record[ib].ds[iqs>>5]=__floats2half2_rn(1.0f/dinv,s);
}
'''

def program(dev):
  binary=NVRTCCompiler(dev.arch,ptx=False,cache_key="q8_dual_flat_tile_fp16_v1").compile(SOURCE)
  return native_nv_program("q8_dual_flat_tile_fp16",binary,global_size=(M,8,1),local_size=(128,1,1),
    globals=(0,1,2),outs=(1,2),ins=(0,))
