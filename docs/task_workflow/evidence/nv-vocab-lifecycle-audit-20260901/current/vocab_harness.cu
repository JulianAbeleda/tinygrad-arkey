
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#ifndef INFINITY
#define INFINITY (__int_as_float(0x7f800000))
#endif
#ifndef NAN
#define NAN (__int_as_float(0x7fffffff))
#endif
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f=v; return u.t; }
struct __align__(8) half4 { half x, y, z, w; };
__device__ half4 make_half4(half x, half y, half z, half w) { half4 r={x,y,z,w}; return r; }
extern "C" __global__ void __launch_bounds__(128) q6k_vocab_four_warp_fp16_direct_151936_4096(float* data0_151936, unsigned short* data1_255252480, half* data2_4096) {
  int gidx0 = blockIdx.x; /* 151936 */
  int lidx0 = threadIdx.x; /* 128 */
  float buf0;
  buf0 = 0.0f;
  unsigned short cast0 = ((unsigned short)(((lidx0&1)<<3)));
  int alu1 = (lidx0>>4);
  for (int Ridx0 = 0; Ridx0 < 2; Ridx0++) {
    int alu2 = ((alu1*210)+(Ridx0*105)+(gidx0*1680));
    int alu3 = (alu2+((lidx0>>1)&7));
    unsigned short val0 = (*(data1_255252480+(alu3+8)));
    unsigned short val1 = (*(data1_255252480+(alu3+16)));
    unsigned short val2 = (*(data1_255252480+(alu3+24)));
    unsigned short val3 = (*(data1_255252480+(alu3+32)));
    unsigned short val4 = (*(data1_255252480+(alu3+40)));
    unsigned short val5 = (*(data1_255252480+(alu3+48)));
    unsigned short val6 = (*(data1_255252480+(alu3+56)));
    unsigned short val7 = (*(data1_255252480+(alu3+64)));
    unsigned short val8 = (*(data1_255252480+(alu3+72)));
    unsigned short val9 = (*(data1_255252480+(alu3+80)));
    unsigned short val10 = (*(data1_255252480+(alu3+88)));
    unsigned short val11 = (*(data1_255252480+alu3));
    unsigned short val12 = (*(data1_255252480+(alu2+96)));
    unsigned short val13 = (*(data1_255252480+(alu2+97)));
    unsigned short val14 = (*(data1_255252480+(alu2+98)));
    unsigned short val15 = (*(data1_255252480+(alu2+99)));
    unsigned short val16 = (*(data1_255252480+(alu2+100)));
    unsigned short val17 = (*(data1_255252480+(alu2+101)));
    unsigned short val18 = (*(data1_255252480+(alu2+102)));
    unsigned short val19 = (*(data1_255252480+(alu2+103)));
    unsigned short val20 = (*(data1_255252480+(alu2+104)));
    int alu4 = ((alu1<<9)+(Ridx0<<8)+(lidx0&15));
    half val21 = (*(data2_4096+(alu4+16)));
    half val22 = (*(data2_4096+(alu4+32)));
    half val23 = (*(data2_4096+(alu4+48)));
    half val24 = (*(data2_4096+(alu4+64)));
    half val25 = (*(data2_4096+(alu4+80)));
    half val26 = (*(data2_4096+(alu4+96)));
    half val27 = (*(data2_4096+(alu4+112)));
    half val28 = (*(data2_4096+(alu4+128)));
    half val29 = (*(data2_4096+(alu4+144)));
    half val30 = (*(data2_4096+(alu4+160)));
    half val31 = (*(data2_4096+(alu4+176)));
    half val32 = (*(data2_4096+(alu4+192)));
    half val33 = (*(data2_4096+(alu4+208)));
    half val34 = (*(data2_4096+(alu4+224)));
    half val35 = (*(data2_4096+(alu4+240)));
    half val36 = (*(data2_4096+alu4));
    int cast1 = ((int)(val12));
    int cast2 = ((int)(val13));
    int cast3 = ((int)(val14));
    int cast4 = ((int)(val15));
    int cast5 = ((int)(val16));
    int cast6 = ((int)(val17));
    int cast7 = ((int)(val18));
    int cast8 = ((int)(val19));
    float cast9 = ((float)(tg_bitcast<half>((unsigned short)(val20))));
    unsigned short alu5 = ((val8>>cast0)&((unsigned short)(255u)));
    unsigned short alu6 = ((val0>>cast0)&((unsigned short)(255u)));
    unsigned short alu7 = ((val7>>cast0)&((unsigned short)(255u)));
    unsigned short alu8 = ((val1>>cast0)&((unsigned short)(255u)));
    unsigned short alu9 = ((val2>>cast0)&((unsigned short)(255u)));
    unsigned short alu10 = ((val9>>cast0)&((unsigned short)(255u)));
    unsigned short alu11 = ((val3>>cast0)&((unsigned short)(255u)));
    unsigned short alu12 = ((val10>>cast0)&((unsigned short)(255u)));
    unsigned short alu13 = ((val4>>cast0)&((unsigned short)(255u)));
    unsigned short alu14 = ((val5>>cast0)&((unsigned short)(255u)));
    unsigned short alu15 = ((val6>>cast0)&((unsigned short)(255u)));
    unsigned short alu16 = ((val11>>cast0)&((unsigned short)(255u)));
    buf0 = (buf0+(cast9*(((float)((((alu16>>((unsigned short)(0u)))&((unsigned short)(15u)))|(((alu7>>((unsigned short)(0u)))&((unsigned short)(3u)))<<((unsigned short)(4u))))))+-32.0f)*((float)(tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast1>>0)&255)))))))*((float)(val36)))+(cast9*(((float)((((alu6>>((unsigned short)(0u)))&((unsigned short)(15u)))|(((alu5>>((unsigned short)(0u)))&((unsigned short)(3u)))<<((unsigned short)(4u))))))+-32.0f)*((float)(tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast1>>8)&255)))))))*((float)(val21)))+(cast9*(((float)((((alu8>>((unsigned short)(0u)))&((unsigned short)(15u)))|(((alu7>>((unsigned short)(2u)))&((unsigned short)(3u)))<<((unsigned short)(4u))))))+-32.0f)*((float)(tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast2>>0)&255)))))))*((float)(val22)))+(cast9*(((float)((((alu9>>((unsigned short)(0u)))&((unsigned short)(15u)))|(((alu5>>((unsigned short)(2u)))&((unsigned short)(3u)))<<((unsigned short)(4u))))))+-32.0f)*((float)(tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast2>>8)&255)))))))*((float)(val23)))+(cast9*(((float)((((alu16>>((unsigned short)(4u)))&((unsigned short)(15u)))|(((alu7>>((unsigned short)(4u)))&((unsigned short)(3u)))<<((unsigned short)(4u))))))+-32.0f)*((float)(tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast3>>0)&255)))))))*((float)(val24)))+(cast9*(((float)((((alu6>>((unsigned short)(4u)))&((unsigned short)(15u)))|(((alu5>>((unsigned short)(4u)))&((unsigned short)(3u)))<<((unsigned short)(4u))))))+-32.0f)*((float)(tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast3>>8)&255)))))))*((float)(val25)))+(cast9*(((float)((((alu8>>((unsigned short)(4u)))&((unsigned short)(15u)))|(((alu7>>((unsigned short)(6u)))&((unsigned short)(3u)))<<((unsigned short)(4u))))))+-32.0f)*((float)(tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast4>>0)&255)))))))*((float)(val26)))+(cast9*(((float)((((alu9>>((unsigned short)(4u)))&((unsigned short)(15u)))|(((alu5>>((unsigned short)(6u)))&((unsigned short)(3u)))<<((unsigned short)(4u))))))+-32.0f)*((float)(tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast4>>8)&255)))))))*((float)(val27)))+(cast9*(((float)((((alu11>>((unsigned short)(0u)))&((unsigned short)(15u)))|(((alu10>>((unsigned short)(0u)))&((unsigned short)(3u)))<<((unsigned short)(4u))))))+-32.0f)*((float)(tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast5>>0)&255)))))))*((float)(val28)))+(cast9*(((float)((((alu13>>((unsigned short)(0u)))&((unsigned short)(15u)))|(((alu12>>((unsigned short)(0u)))&((unsigned short)(3u)))<<((unsigned short)(4u))))))+-32.0f)*((float)(tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast5>>8)&255)))))))*((float)(val29)))+(cast9*(((float)((((alu14>>((unsigned short)(0u)))&((unsigned short)(15u)))|(((alu10>>((unsigned short)(2u)))&((unsigned short)(3u)))<<((unsigned short)(4u))))))+-32.0f)*((float)(tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast6>>0)&255)))))))*((float)(val30)))+(cast9*(((float)((((alu15>>((unsigned short)(0u)))&((unsigned short)(15u)))|(((alu12>>((unsigned short)(2u)))&((unsigned short)(3u)))<<((unsigned short)(4u))))))+-32.0f)*((float)(tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast6>>8)&255)))))))*((float)(val31)))+(cast9*(((float)((((alu11>>((unsigned short)(4u)))&((unsigned short)(15u)))|(((alu10>>((unsigned short)(4u)))&((unsigned short)(3u)))<<((unsigned short)(4u))))))+-32.0f)*((float)(tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast7>>0)&255)))))))*((float)(val32)))+(cast9*(((float)((((alu13>>((unsigned short)(4u)))&((unsigned short)(15u)))|(((alu12>>((unsigned short)(4u)))&((unsigned short)(3u)))<<((unsigned short)(4u))))))+-32.0f)*((float)(tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast7>>8)&255)))))))*((float)(val33)))+(cast9*(((float)((((alu14>>((unsigned short)(4u)))&((unsigned short)(15u)))|(((alu10>>((unsigned short)(6u)))&((unsigned short)(3u)))<<((unsigned short)(4u))))))+-32.0f)*((float)(tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast8>>0)&255)))))))*((float)(val34)))+(cast9*(((float)((((alu15>>((unsigned short)(4u)))&((unsigned short)(15u)))|(((alu12>>((unsigned short)(6u)))&((unsigned short)(3u)))<<((unsigned short)(4u))))))+-32.0f)*((float)(tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast8>>8)&255)))))))*((float)(val35))));
  }
  __shared__ __align__(16) float buf1[4];
  float buf2;
  float buf3;
  float buf4;
  float buf5;
  float buf6;
  buf2 = __shfl_xor_sync(0xffffffffu, buf0, 16);
  float alu20 = (buf0+buf2);
  buf3 = __shfl_xor_sync(0xffffffffu, alu20, 8);
  float alu22 = (alu20+buf3);
  buf4 = __shfl_xor_sync(0xffffffffu, alu22, 4);
  float alu24 = (alu22+buf4);
  buf5 = __shfl_xor_sync(0xffffffffu, alu24, 2);
  float alu26 = (alu24+buf5);
  buf6 = __shfl_xor_sync(0xffffffffu, alu26, 1);
  if (((lidx0&31)==0)) {
    *(buf1+(lidx0>>5)) = (alu26+buf6);
  }
  __syncthreads();
  float val37 = (*(buf1+0));
  float val38 = (*(buf1+1));
  float val39 = (*(buf1+2));
  float val40 = (*(buf1+3));
  if ((lidx0==0)) {
    *(data0_151936+gidx0) = (val37+val38+val39+val40);
  }
}
static void check(cudaError_t e,const char* what) { if(e!=cudaSuccess) { fprintf(stderr,"%s: %s\n",what,cudaGetErrorString(e)); exit(2); } }
int main(int argc,char** argv) {
  int passes=argc>1?atoi(argv[1]):25, reps=argc>2?atoi(argv[2]):5;
  float *out=nullptr; unsigned short* w=nullptr; half* x=nullptr;
  check(cudaMalloc(&out,151936*sizeof(float)),"out");
  check(cudaMalloc(&w,255252480*sizeof(unsigned short)),"w");
  check(cudaMalloc(&x,4096*sizeof(half)),"x");
  check(cudaMemset(out,0,151936*sizeof(float)),"zero out");
  check(cudaMemset(w,0,255252480*sizeof(unsigned short)),"zero w");
  check(cudaMemset(x,0,4096*sizeof(half)),"zero x");
  q6k_vocab_four_warp_fp16_direct_151936_4096<<<151936,128>>>(out,w,x); check(cudaDeviceSynchronize(),"warmup");
  for(int r=0;r<reps;r++) {
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); cudaEventRecord(s);
    for(int i=0;i<passes;i++) q6k_vocab_four_warp_fp16_direct_151936_4096<<<151936,128>>>(out,w,x);
    cudaEventRecord(e); check(cudaDeviceSynchronize(),"sync"); float ms=0; cudaEventElapsedTime(&ms,s,e);
    printf("rep=%d current=%.4f\n",r,ms*1000.0f/passes); cudaEventDestroy(s); cudaEventDestroy(e);
  }
  return 0;
}
