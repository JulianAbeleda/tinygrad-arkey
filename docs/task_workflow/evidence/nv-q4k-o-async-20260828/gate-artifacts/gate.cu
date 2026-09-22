
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#define ROWS 4096
#define K 4096
#define K_BLOCKS 16
#define WORDS 2359296
#define ROTATIONS 16
template<class T,class F>__device__ __forceinline__ T tg_bitcast(F v){union U{F f;T t;};U u;u.f=v;return u.t;}
struct __align__(8) half4{half x,y,z,w;};__device__ half4 make_half4(half x,half y,half z,half w){half4 r={x,y,z,w};return r;}
extern "C" __global__ void __launch_bounds__(32) q4k_g3_lanemap_gemv_vec_epi_resadd_4096_4096(float* data0_4096, unsigned int* data1_2359296, half* data2_4096, float* data3_4096) {
  int gidx0 = blockIdx.x; /* 4096 */
  int lidx0 = threadIdx.x; /* 32 */
  float buf0[1];
  (*(buf0+0)) = 0.0f;
  int alu1 = (lidx0>>3);
  int alu2 = (lidx0&7);
  for (int Ridx0 = 0; Ridx0 < 4; Ridx0++) {
    int alu3 = ((alu1*144)+(Ridx0*36)+(gidx0*576));
    int alu4 = (alu3+alu2);
    unsigned int val0 = (*(data1_2359296+(alu4+4)));
    unsigned int val1 = (*(data1_2359296+(alu4+12)));
    unsigned int val2 = (*(data1_2359296+(alu4+20)));
    unsigned int val3 = (*(data1_2359296+(alu4+28)));
    uint4 val4 = (*((uint4*)((data1_2359296+alu3))));
    int alu5 = ((alu1<<10)+(Ridx0<<8)+(alu2<<2));
    half4 val5 = (*((half4*)((data2_4096+(alu5+32)))));
    half4 val6 = (*((half4*)((data2_4096+(alu5+64)))));
    half4 val7 = (*((half4*)((data2_4096+(alu5+96)))));
    half4 val8 = (*((half4*)((data2_4096+(alu5+128)))));
    half4 val9 = (*((half4*)((data2_4096+(alu5+160)))));
    half4 val10 = (*((half4*)((data2_4096+(alu5+192)))));
    half4 val11 = (*((half4*)((data2_4096+(alu5+224)))));
    half4 val12 = (*((half4*)((data2_4096+alu5))));
    float cast0 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val4.x&65535u)))))));
    float cast1 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)(((val4.x>>16u)&65535u)))))));
    unsigned int alu6 = (val4.z>>0u);
    unsigned int alu7 = (val4.w>>0u);
    unsigned int alu8 = (val4.z>>8u);
    unsigned int alu9 = (val4.w>>8u);
    unsigned int alu10 = (val4.z>>16u);
    unsigned int alu11 = (val4.w>>16u);
    unsigned int alu12 = (val4.z>>24u);
    unsigned int alu13 = (val4.w>>24u);
    unsigned int alu14 = (val4.y>>0u);
    unsigned int alu15 = (val4.y>>8u);
    unsigned int alu16 = (val4.y>>16u);
    unsigned int alu17 = (val4.y>>24u);
    unsigned int alu18 = ((val0>>0u)&252645135u);
    unsigned int alu19 = ((val0>>4u)&252645135u);
    unsigned int alu20 = ((val1>>0u)&252645135u);
    unsigned int alu21 = ((val1>>4u)&252645135u);
    unsigned int alu22 = ((val2>>0u)&252645135u);
    unsigned int alu23 = ((val2>>4u)&252645135u);
    unsigned int alu24 = ((val3>>0u)&252645135u);
    unsigned int alu25 = ((val3>>4u)&252645135u);
    float alu26 = (cast0*((float)((alu14&63u))));
    float alu27 = (cast1*((float)((alu6&63u))));
    float alu28 = (cast0*((float)((alu15&63u))));
    float alu29 = (cast1*((float)((alu8&63u))));
    float alu30 = (cast0*((float)((alu16&63u))));
    float alu31 = (cast1*((float)((alu10&63u))));
    float alu32 = (cast0*((float)((alu17&63u))));
    float alu33 = (cast1*((float)((alu12&63u))));
    float alu34 = (cast0*((float)(((alu7&15u)|(((alu14&255u)>>6u)<<4u)))));
    float alu35 = (cast1*((float)((((alu7&255u)>>4u)|(((alu6&255u)>>6u)<<4u)))));
    float alu36 = (cast0*((float)(((alu9&15u)|(((alu15&255u)>>6u)<<4u)))));
    float alu37 = (cast1*((float)((((alu9&255u)>>4u)|(((alu8&255u)>>6u)<<4u)))));
    float alu38 = (cast0*((float)(((alu11&15u)|(((alu16&255u)>>6u)<<4u)))));
    float alu39 = (cast1*((float)((((alu11&255u)>>4u)|(((alu10&255u)>>6u)<<4u)))));
    float alu40 = (cast0*((float)(((alu13&15u)|(((alu17&255u)>>6u)<<4u)))));
    float alu41 = (cast1*((float)((((alu13&255u)>>4u)|(((alu12&255u)>>6u)<<4u)))));
    (*(buf0+0)) = ((*(buf0+0))+(((alu26*((float)(((alu18>>0u)&15u))))-alu27)*float(val12.x))+(((alu26*((float)(((alu18>>8u)&15u))))-alu27)*float(val12.y))+(((alu26*((float)(((alu18>>16u)&15u))))-alu27)*float(val12.z))+(((alu26*((float)(((alu18>>24u)&15u))))-alu27)*float(val12.w))+(((alu28*((float)(((alu19>>0u)&15u))))-alu29)*float(val5.x))+(((alu28*((float)(((alu19>>8u)&15u))))-alu29)*float(val5.y))+(((alu28*((float)(((alu19>>16u)&15u))))-alu29)*float(val5.z))+(((alu28*((float)(((alu19>>24u)&15u))))-alu29)*float(val5.w))+(((alu30*((float)(((alu20>>0u)&15u))))-alu31)*float(val6.x))+(((alu30*((float)(((alu20>>8u)&15u))))-alu31)*float(val6.y))+(((alu30*((float)(((alu20>>16u)&15u))))-alu31)*float(val6.z))+(((alu30*((float)(((alu20>>24u)&15u))))-alu31)*float(val6.w))+(((alu32*((float)(((alu21>>0u)&15u))))-alu33)*float(val7.x))+(((alu32*((float)(((alu21>>8u)&15u))))-alu33)*float(val7.y))+(((alu32*((float)(((alu21>>16u)&15u))))-alu33)*float(val7.z))+(((alu32*((float)(((alu21>>24u)&15u))))-alu33)*float(val7.w))+(((alu34*((float)(((alu22>>0u)&15u))))-alu35)*float(val8.x))+(((alu34*((float)(((alu22>>8u)&15u))))-alu35)*float(val8.y))+(((alu34*((float)(((alu22>>16u)&15u))))-alu35)*float(val8.z))+(((alu34*((float)(((alu22>>24u)&15u))))-alu35)*float(val8.w))+(((alu36*((float)(((alu23>>0u)&15u))))-alu37)*float(val9.x))+(((alu36*((float)(((alu23>>8u)&15u))))-alu37)*float(val9.y))+(((alu36*((float)(((alu23>>16u)&15u))))-alu37)*float(val9.z))+(((alu36*((float)(((alu23>>24u)&15u))))-alu37)*float(val9.w))+(((alu38*((float)(((alu24>>0u)&15u))))-alu39)*float(val10.x))+(((alu38*((float)(((alu24>>8u)&15u))))-alu39)*float(val10.y))+(((alu38*((float)(((alu24>>16u)&15u))))-alu39)*float(val10.z))+(((alu38*((float)(((alu24>>24u)&15u))))-alu39)*float(val10.w))+(((alu40*((float)(((alu25>>0u)&15u))))-alu41)*float(val11.x))+(((alu40*((float)(((alu25>>8u)&15u))))-alu41)*float(val11.y))+(((alu40*((float)(((alu25>>16u)&15u))))-alu41)*float(val11.z))+(((alu40*((float)(((alu25>>24u)&15u))))-alu41)*float(val11.w)));
  }
  float val13 = (*(data3_4096+gidx0));
  float buf1[1];
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, (*(buf0+0)), 16);
  float alu45 = ((*(buf0+0))+(*(buf1+0)));
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu45, 8);
  float alu47 = (alu45+(*(buf1+0)));
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu47, 4);
  float alu49 = (alu47+(*(buf1+0)));
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu49, 2);
  float alu51 = (alu49+(*(buf1+0)));
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu51, 1);
  *(data0_4096+gidx0) = (alu51+(*(buf1+0))+val13);
}
extern "C" __global__ void __launch_bounds__(32) q4k_g3_lanemap_gemv_vec_epi_resadd_async1_4096_4096(float* data0_4096, unsigned int* data1_2359296, half* data2_4096, float* data3_4096) {
  int gidx0 = blockIdx.x; /* 4096 */
  int lidx0 = threadIdx.x; /* 32 */
  __shared__ __align__(16) unsigned int qtile[144];
  float buf0[1];
  (*(buf0+0)) = 0.0f;
  int alu1 = (lidx0>>3);
  int alu2 = (lidx0&7);
  for (int Ridx0 = 0; Ridx0 < 4; Ridx0++) {
    {
      unsigned int sd = (unsigned int)__cvta_generic_to_shared(qtile+(alu1*36)+alu2*4);
      const unsigned int* sg = data1_2359296+((alu1*144)+(Ridx0*36)+(gidx0*576))+alu2*4;
      asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(sd), "l"(sg));
      if (alu2 == 0) {
        unsigned int td = (unsigned int)__cvta_generic_to_shared(qtile+(alu1*36)+32);
        const unsigned int* tg = data1_2359296+((alu1*144)+(Ridx0*36)+(gidx0*576))+32;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(td), "l"(tg));
      }
      asm volatile("cp.async.commit_group;");
    }
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    int alu3 = alu1*36;
    int alu4 = (alu3+alu2);
    unsigned int val0 = (*(qtile+(alu4+4)));
    unsigned int val1 = (*(qtile+(alu4+12)));
    unsigned int val2 = (*(qtile+(alu4+20)));
    unsigned int val3 = (*(qtile+(alu4+28)));
    uint4 val4 = (*((uint4*)((qtile+alu3))));
    int alu5 = ((alu1<<10)+(Ridx0<<8)+(alu2<<2));
    half4 val5 = (*((half4*)((data2_4096+(alu5+32)))));
    half4 val6 = (*((half4*)((data2_4096+(alu5+64)))));
    half4 val7 = (*((half4*)((data2_4096+(alu5+96)))));
    half4 val8 = (*((half4*)((data2_4096+(alu5+128)))));
    half4 val9 = (*((half4*)((data2_4096+(alu5+160)))));
    half4 val10 = (*((half4*)((data2_4096+(alu5+192)))));
    half4 val11 = (*((half4*)((data2_4096+(alu5+224)))));
    half4 val12 = (*((half4*)((data2_4096+alu5))));
    float cast0 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val4.x&65535u)))))));
    float cast1 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)(((val4.x>>16u)&65535u)))))));
    unsigned int alu6 = (val4.z>>0u);
    unsigned int alu7 = (val4.w>>0u);
    unsigned int alu8 = (val4.z>>8u);
    unsigned int alu9 = (val4.w>>8u);
    unsigned int alu10 = (val4.z>>16u);
    unsigned int alu11 = (val4.w>>16u);
    unsigned int alu12 = (val4.z>>24u);
    unsigned int alu13 = (val4.w>>24u);
    unsigned int alu14 = (val4.y>>0u);
    unsigned int alu15 = (val4.y>>8u);
    unsigned int alu16 = (val4.y>>16u);
    unsigned int alu17 = (val4.y>>24u);
    unsigned int alu18 = ((val0>>0u)&252645135u);
    unsigned int alu19 = ((val0>>4u)&252645135u);
    unsigned int alu20 = ((val1>>0u)&252645135u);
    unsigned int alu21 = ((val1>>4u)&252645135u);
    unsigned int alu22 = ((val2>>0u)&252645135u);
    unsigned int alu23 = ((val2>>4u)&252645135u);
    unsigned int alu24 = ((val3>>0u)&252645135u);
    unsigned int alu25 = ((val3>>4u)&252645135u);
    float alu26 = (cast0*((float)((alu14&63u))));
    float alu27 = (cast1*((float)((alu6&63u))));
    float alu28 = (cast0*((float)((alu15&63u))));
    float alu29 = (cast1*((float)((alu8&63u))));
    float alu30 = (cast0*((float)((alu16&63u))));
    float alu31 = (cast1*((float)((alu10&63u))));
    float alu32 = (cast0*((float)((alu17&63u))));
    float alu33 = (cast1*((float)((alu12&63u))));
    float alu34 = (cast0*((float)(((alu7&15u)|(((alu14&255u)>>6u)<<4u)))));
    float alu35 = (cast1*((float)((((alu7&255u)>>4u)|(((alu6&255u)>>6u)<<4u)))));
    float alu36 = (cast0*((float)(((alu9&15u)|(((alu15&255u)>>6u)<<4u)))));
    float alu37 = (cast1*((float)((((alu9&255u)>>4u)|(((alu8&255u)>>6u)<<4u)))));
    float alu38 = (cast0*((float)(((alu11&15u)|(((alu16&255u)>>6u)<<4u)))));
    float alu39 = (cast1*((float)((((alu11&255u)>>4u)|(((alu10&255u)>>6u)<<4u)))));
    float alu40 = (cast0*((float)(((alu13&15u)|(((alu17&255u)>>6u)<<4u)))));
    float alu41 = (cast1*((float)((((alu13&255u)>>4u)|(((alu12&255u)>>6u)<<4u)))));
    (*(buf0+0)) = ((*(buf0+0))+(((alu26*((float)(((alu18>>0u)&15u))))-alu27)*float(val12.x))+(((alu26*((float)(((alu18>>8u)&15u))))-alu27)*float(val12.y))+(((alu26*((float)(((alu18>>16u)&15u))))-alu27)*float(val12.z))+(((alu26*((float)(((alu18>>24u)&15u))))-alu27)*float(val12.w))+(((alu28*((float)(((alu19>>0u)&15u))))-alu29)*float(val5.x))+(((alu28*((float)(((alu19>>8u)&15u))))-alu29)*float(val5.y))+(((alu28*((float)(((alu19>>16u)&15u))))-alu29)*float(val5.z))+(((alu28*((float)(((alu19>>24u)&15u))))-alu29)*float(val5.w))+(((alu30*((float)(((alu20>>0u)&15u))))-alu31)*float(val6.x))+(((alu30*((float)(((alu20>>8u)&15u))))-alu31)*float(val6.y))+(((alu30*((float)(((alu20>>16u)&15u))))-alu31)*float(val6.z))+(((alu30*((float)(((alu20>>24u)&15u))))-alu31)*float(val6.w))+(((alu32*((float)(((alu21>>0u)&15u))))-alu33)*float(val7.x))+(((alu32*((float)(((alu21>>8u)&15u))))-alu33)*float(val7.y))+(((alu32*((float)(((alu21>>16u)&15u))))-alu33)*float(val7.z))+(((alu32*((float)(((alu21>>24u)&15u))))-alu33)*float(val7.w))+(((alu34*((float)(((alu22>>0u)&15u))))-alu35)*float(val8.x))+(((alu34*((float)(((alu22>>8u)&15u))))-alu35)*float(val8.y))+(((alu34*((float)(((alu22>>16u)&15u))))-alu35)*float(val8.z))+(((alu34*((float)(((alu22>>24u)&15u))))-alu35)*float(val8.w))+(((alu36*((float)(((alu23>>0u)&15u))))-alu37)*float(val9.x))+(((alu36*((float)(((alu23>>8u)&15u))))-alu37)*float(val9.y))+(((alu36*((float)(((alu23>>16u)&15u))))-alu37)*float(val9.z))+(((alu36*((float)(((alu23>>24u)&15u))))-alu37)*float(val9.w))+(((alu38*((float)(((alu24>>0u)&15u))))-alu39)*float(val10.x))+(((alu38*((float)(((alu24>>8u)&15u))))-alu39)*float(val10.y))+(((alu38*((float)(((alu24>>16u)&15u))))-alu39)*float(val10.z))+(((alu38*((float)(((alu24>>24u)&15u))))-alu39)*float(val10.w))+(((alu40*((float)(((alu25>>0u)&15u))))-alu41)*float(val11.x))+(((alu40*((float)(((alu25>>8u)&15u))))-alu41)*float(val11.y))+(((alu40*((float)(((alu25>>16u)&15u))))-alu41)*float(val11.z))+(((alu40*((float)(((alu25>>24u)&15u))))-alu41)*float(val11.w)));
    __syncthreads();
  }
  float val13 = (*(data3_4096+gidx0));
  float buf1[1];
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, (*(buf0+0)), 16);
  float alu45 = ((*(buf0+0))+(*(buf1+0)));
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu45, 8);
  float alu47 = (alu45+(*(buf1+0)));
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu47, 4);
  float alu49 = (alu47+(*(buf1+0)));
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu49, 2);
  float alu51 = (alu49+(*(buf1+0)));
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu51, 1);
  *(data0_4096+gidx0) = (alu51+(*(buf1+0))+val13);
}
extern "C" __global__ void __launch_bounds__(32) q4k_g3_lanemap_gemv_vec_epi_resadd_async2_4096_4096(float* data0_4096, unsigned int* data1_2359296, half* data2_4096, float* data3_4096) {
  int gidx0 = blockIdx.x; /* 4096 */
  int lidx0 = threadIdx.x; /* 32 */
  __shared__ __align__(16) unsigned int qtile[288];
  float buf0[1];
  (*(buf0+0)) = 0.0f;
  int alu1 = (lidx0>>3);
  int alu2 = (lidx0&7);
    {
      unsigned int sd = (unsigned int)__cvta_generic_to_shared(qtile+(alu1*36)+alu2*4);
      const unsigned int* sg = data1_2359296+((alu1*144)+(gidx0*576))+alu2*4;
      asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(sd), "l"(sg));
      if (alu2 == 0) {
        unsigned int td = (unsigned int)__cvta_generic_to_shared(qtile+(alu1*36)+32);
        const unsigned int* tg = data1_2359296+((alu1*144)+(gidx0*576))+32;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(td), "l"(tg));
      }
      asm volatile("cp.async.commit_group;");
    }
  asm volatile("cp.async.wait_group 0;");
  __syncthreads();
  for (int Ridx0 = 0; Ridx0 < 4; Ridx0++) {
    if (Ridx0 < 3) {
    {
      unsigned int sd = (unsigned int)__cvta_generic_to_shared(qtile+(((Ridx0+1)&1)*144+alu1*36)+alu2*4);
      const unsigned int* sg = data1_2359296+((alu1*144)+((Ridx0+1)*36)+(gidx0*576))+alu2*4;
      asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(sd), "l"(sg));
      if (alu2 == 0) {
        unsigned int td = (unsigned int)__cvta_generic_to_shared(qtile+(((Ridx0+1)&1)*144+alu1*36)+32);
        const unsigned int* tg = data1_2359296+((alu1*144)+((Ridx0+1)*36)+(gidx0*576))+32;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(td), "l"(tg));
      }
      asm volatile("cp.async.commit_group;");
    }
    }
    int alu3 = (Ridx0&1)*144+alu1*36;
    int alu4 = (alu3+alu2);
    unsigned int val0 = (*(qtile+(alu4+4)));
    unsigned int val1 = (*(qtile+(alu4+12)));
    unsigned int val2 = (*(qtile+(alu4+20)));
    unsigned int val3 = (*(qtile+(alu4+28)));
    uint4 val4 = (*((uint4*)((qtile+alu3))));
    int alu5 = ((alu1<<10)+(Ridx0<<8)+(alu2<<2));
    half4 val5 = (*((half4*)((data2_4096+(alu5+32)))));
    half4 val6 = (*((half4*)((data2_4096+(alu5+64)))));
    half4 val7 = (*((half4*)((data2_4096+(alu5+96)))));
    half4 val8 = (*((half4*)((data2_4096+(alu5+128)))));
    half4 val9 = (*((half4*)((data2_4096+(alu5+160)))));
    half4 val10 = (*((half4*)((data2_4096+(alu5+192)))));
    half4 val11 = (*((half4*)((data2_4096+(alu5+224)))));
    half4 val12 = (*((half4*)((data2_4096+alu5))));
    float cast0 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val4.x&65535u)))))));
    float cast1 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)(((val4.x>>16u)&65535u)))))));
    unsigned int alu6 = (val4.z>>0u);
    unsigned int alu7 = (val4.w>>0u);
    unsigned int alu8 = (val4.z>>8u);
    unsigned int alu9 = (val4.w>>8u);
    unsigned int alu10 = (val4.z>>16u);
    unsigned int alu11 = (val4.w>>16u);
    unsigned int alu12 = (val4.z>>24u);
    unsigned int alu13 = (val4.w>>24u);
    unsigned int alu14 = (val4.y>>0u);
    unsigned int alu15 = (val4.y>>8u);
    unsigned int alu16 = (val4.y>>16u);
    unsigned int alu17 = (val4.y>>24u);
    unsigned int alu18 = ((val0>>0u)&252645135u);
    unsigned int alu19 = ((val0>>4u)&252645135u);
    unsigned int alu20 = ((val1>>0u)&252645135u);
    unsigned int alu21 = ((val1>>4u)&252645135u);
    unsigned int alu22 = ((val2>>0u)&252645135u);
    unsigned int alu23 = ((val2>>4u)&252645135u);
    unsigned int alu24 = ((val3>>0u)&252645135u);
    unsigned int alu25 = ((val3>>4u)&252645135u);
    float alu26 = (cast0*((float)((alu14&63u))));
    float alu27 = (cast1*((float)((alu6&63u))));
    float alu28 = (cast0*((float)((alu15&63u))));
    float alu29 = (cast1*((float)((alu8&63u))));
    float alu30 = (cast0*((float)((alu16&63u))));
    float alu31 = (cast1*((float)((alu10&63u))));
    float alu32 = (cast0*((float)((alu17&63u))));
    float alu33 = (cast1*((float)((alu12&63u))));
    float alu34 = (cast0*((float)(((alu7&15u)|(((alu14&255u)>>6u)<<4u)))));
    float alu35 = (cast1*((float)((((alu7&255u)>>4u)|(((alu6&255u)>>6u)<<4u)))));
    float alu36 = (cast0*((float)(((alu9&15u)|(((alu15&255u)>>6u)<<4u)))));
    float alu37 = (cast1*((float)((((alu9&255u)>>4u)|(((alu8&255u)>>6u)<<4u)))));
    float alu38 = (cast0*((float)(((alu11&15u)|(((alu16&255u)>>6u)<<4u)))));
    float alu39 = (cast1*((float)((((alu11&255u)>>4u)|(((alu10&255u)>>6u)<<4u)))));
    float alu40 = (cast0*((float)(((alu13&15u)|(((alu17&255u)>>6u)<<4u)))));
    float alu41 = (cast1*((float)((((alu13&255u)>>4u)|(((alu12&255u)>>6u)<<4u)))));
    (*(buf0+0)) = ((*(buf0+0))+(((alu26*((float)(((alu18>>0u)&15u))))-alu27)*float(val12.x))+(((alu26*((float)(((alu18>>8u)&15u))))-alu27)*float(val12.y))+(((alu26*((float)(((alu18>>16u)&15u))))-alu27)*float(val12.z))+(((alu26*((float)(((alu18>>24u)&15u))))-alu27)*float(val12.w))+(((alu28*((float)(((alu19>>0u)&15u))))-alu29)*float(val5.x))+(((alu28*((float)(((alu19>>8u)&15u))))-alu29)*float(val5.y))+(((alu28*((float)(((alu19>>16u)&15u))))-alu29)*float(val5.z))+(((alu28*((float)(((alu19>>24u)&15u))))-alu29)*float(val5.w))+(((alu30*((float)(((alu20>>0u)&15u))))-alu31)*float(val6.x))+(((alu30*((float)(((alu20>>8u)&15u))))-alu31)*float(val6.y))+(((alu30*((float)(((alu20>>16u)&15u))))-alu31)*float(val6.z))+(((alu30*((float)(((alu20>>24u)&15u))))-alu31)*float(val6.w))+(((alu32*((float)(((alu21>>0u)&15u))))-alu33)*float(val7.x))+(((alu32*((float)(((alu21>>8u)&15u))))-alu33)*float(val7.y))+(((alu32*((float)(((alu21>>16u)&15u))))-alu33)*float(val7.z))+(((alu32*((float)(((alu21>>24u)&15u))))-alu33)*float(val7.w))+(((alu34*((float)(((alu22>>0u)&15u))))-alu35)*float(val8.x))+(((alu34*((float)(((alu22>>8u)&15u))))-alu35)*float(val8.y))+(((alu34*((float)(((alu22>>16u)&15u))))-alu35)*float(val8.z))+(((alu34*((float)(((alu22>>24u)&15u))))-alu35)*float(val8.w))+(((alu36*((float)(((alu23>>0u)&15u))))-alu37)*float(val9.x))+(((alu36*((float)(((alu23>>8u)&15u))))-alu37)*float(val9.y))+(((alu36*((float)(((alu23>>16u)&15u))))-alu37)*float(val9.z))+(((alu36*((float)(((alu23>>24u)&15u))))-alu37)*float(val9.w))+(((alu38*((float)(((alu24>>0u)&15u))))-alu39)*float(val10.x))+(((alu38*((float)(((alu24>>8u)&15u))))-alu39)*float(val10.y))+(((alu38*((float)(((alu24>>16u)&15u))))-alu39)*float(val10.z))+(((alu38*((float)(((alu24>>24u)&15u))))-alu39)*float(val10.w))+(((alu40*((float)(((alu25>>0u)&15u))))-alu41)*float(val11.x))+(((alu40*((float)(((alu25>>8u)&15u))))-alu41)*float(val11.y))+(((alu40*((float)(((alu25>>16u)&15u))))-alu41)*float(val11.z))+(((alu40*((float)(((alu25>>24u)&15u))))-alu41)*float(val11.w)));
    if (Ridx0 < 3) { asm volatile("cp.async.wait_group 0;"); __syncthreads(); }
  }
  float val13 = (*(data3_4096+gidx0));
  float buf1[1];
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, (*(buf0+0)), 16);
  float alu45 = ((*(buf0+0))+(*(buf1+0)));
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu45, 8);
  float alu47 = (alu45+(*(buf1+0)));
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu47, 4);
  float alu49 = (alu47+(*(buf1+0)));
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu49, 2);
  float alu51 = (alu49+(*(buf1+0)));
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu51, 1);
  *(data0_4096+gidx0) = (alu51+(*(buf1+0))+val13);
}
static uint32_t step(uint32_t&s){s=1664525u*s+1013904223u;return s;}
static void fill(uint32_t*w,half*x,float*r,int f){uint32_t s=0x1234567u^(uint32_t(f)*0x9e3779b9u);const uint16_t d[4]={0x2c00,0x3000,0x3400,0x3800},m[4]={0x2800,0x2c00,0x3000,0x3400};for(size_t b=0;b<(size_t)ROWS*K_BLOCKS;b++){size_t z=b*36;int p=(b+f)&3;w[z]=uint32_t(d[p])|(uint32_t(m[(p+1)&3])<<16);if(f==0){w[z+1]=step(s);w[z+2]=step(s);w[z+3]=step(s);for(int i=4;i<36;i++)w[z+i]=step(s);}else if(f==1){w[z+1]=0x01020304u^b;w[z+2]=0x10203040u+b;w[z+3]=0x3f2f1f0f;for(int i=4;i<36;i++)w[z+i]=0x01234567u*i+b*0x11111111u;}else{w[z+1]=0x3f3f3f3f;w[z+2]=0x15151515;w[z+3]=0x2a2a2a2a;for(int i=4;i<36;i++)w[z+i]=(i&1)?0xfedcba98:0x76543210;}}for(int i=0;i<K;i++){float v=f==0?float((int(step(s)>>16)%511)-255)/512.0f:f==1?float((i%257)-128)/256.0f:float((i%17)-8)/64.0f;x[i]=__float2half(v);}for(int i=0;i<ROWS;i++)r[i]=f==0?float((int(step(s)>>16)%255)-127)/1024.0f:f==1?float((i%31)-15)/128.0f:float((i%7)-3)/32.0f;}
static void launch(int a,float*out,uint32_t*w,half*x,float*r){if(a==0)q4k_g3_lanemap_gemv_vec_epi_resadd_4096_4096<<<ROWS,32>>>(out,w,x,r);else if(a==1)q4k_g3_lanemap_gemv_vec_epi_resadd_async1_4096_4096<<<ROWS,32>>>(out,w,x,r);else q4k_g3_lanemap_gemv_vec_epi_resadd_async2_4096_4096<<<ROWS,32>>>(out,w,x,r);}
static double hot(int a,float*out,uint32_t*w,half*x,float*r,int n,cudaEvent_t s,cudaEvent_t e){cudaEventRecord(s);for(int i=0;i<n;i++)launch(a,out,w,x,r);cudaEventRecord(e);cudaEventSynchronize(e);float ms;cudaEventElapsedTime(&ms,s,e);return ms*1000/n;}
static double cold(int a,float*out,uint32_t*ws,half*x,float*r,int n,cudaEvent_t s,cudaEvent_t e){double us=0;for(int i=0;i<n;i++){cudaEventRecord(s);launch(a,out,ws+(size_t)(i%ROTATIONS)*WORDS,x,r);cudaEventRecord(e);cudaEventSynchronize(e);float ms;cudaEventElapsedTime(&ms,s,e);us+=ms*1000;}return us/n;}
int main(int ac,char**av){int hp=ac>1?atoi(av[1]):300,cp=ac>2?atoi(av[2]):32,reps=ac>3?atoi(av[3]):9;bool profile=ac>1&&!strcmp(av[1],"profile");uint32_t*ws;half*x;float*r,*out[3];cudaMalloc(&ws,(size_t)ROTATIONS*WORDS*4);cudaMalloc(&x,K*2);cudaMalloc(&r,ROWS*4);for(int a=0;a<3;a++)cudaMalloc(&out[a],ROWS*4);uint32_t*hw=(uint32_t*)malloc((size_t)WORDS*4);half*hx=(half*)malloc(K*2);float*hr=(float*)malloc(ROWS*4);int exact=1;
for(int f=0;f<3;f++){fill(hw,hx,hr,f);cudaMemcpy(ws,hw,(size_t)WORDS*4,cudaMemcpyHostToDevice);cudaMemcpy(x,hx,K*2,cudaMemcpyHostToDevice);cudaMemcpy(r,hr,ROWS*4,cudaMemcpyHostToDevice);for(int a=0;a<3;a++)launch(a,out[a],ws,x,r);cudaDeviceSynchronize();float*h0=(float*)malloc(ROWS*4),*ha=(float*)malloc(ROWS*4);cudaMemcpy(h0,out[0],ROWS*4,cudaMemcpyDeviceToHost);for(int a=1;a<3;a++){cudaMemcpy(ha,out[a],ROWS*4,cudaMemcpyDeviceToHost);int mm=0,finite=1;double ma=0;for(int i=0;i<ROWS;i++){uint32_t u,v;memcpy(&u,h0+i,4);memcpy(&v,ha+i,4);mm+=u!=v;finite&=isfinite(h0[i])&&isfinite(ha[i]);ma=fmax(ma,fabs(double(h0[i])-double(ha[i])));}printf("fixture=%d arm=%d finite=%d mismatched_words=%d max_abs=%.9g\n",f,a,finite,mm,ma);exact&=finite&&mm==0;}free(h0);free(ha);}fill(hw,hx,hr,0);for(int i=0;i<ROTATIONS;i++)cudaMemcpy(ws+(size_t)i*WORDS,hw,(size_t)WORDS*4,cudaMemcpyHostToDevice);cudaMemcpy(x,hx,K*2,cudaMemcpyHostToDevice);cudaMemcpy(r,hr,ROWS*4,cudaMemcpyHostToDevice);free(hw);free(hx);free(hr);for(int i=0;i<20;i++)for(int a=0;a<3;a++)launch(a,out[a],ws,x,r);cudaDeviceSynchronize();if(profile){for(int a=0;a<3;a++)launch(a,out[a],ws,x,r);cudaDeviceSynchronize();return exact?0:5;}cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);for(int j=0;j<reps;j++)for(int q=0;q<3;q++){int a=(j&1)?2-q:q;printf("rep=%d arm=%d hot=%.6f cold=%.6f\n",j,a,hot(a,out[a],ws,x,r,hp,s,e),cold(a,out[a],ws,x,r,cp,s,e));}return exact?0:5;}
