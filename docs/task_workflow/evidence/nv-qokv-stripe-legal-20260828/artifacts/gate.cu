
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#define Q_ROWS 4096
#define KV_ROWS 1024
#define TOTAL_ROWS 6144
#define K 4096
#define Q_WORDS 2359296
#define KV_WORDS 589824
#define GROUP_WORDS 3538944
#define ROTATIONS 16
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f=v; return u.t; }
struct __align__(8) half4 { half x,y,z,w; };
__device__ half4 make_half4(half x,half y,half z,half w) { half4 r={x,y,z,w}; return r; }
extern "C" __global__ void __launch_bounds__(32) q4k_g3_lanemap_gemv_vec_4096_4096(float* data0_4096, unsigned int* data1_2359296, half* data2_4096) {
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
  *(data0_4096+gidx0) = (alu51+(*(buf1+0)));
}
extern "C" __global__ void __launch_bounds__(32) q4k_g3_lanemap_gemv_pair_vec_1024_4096(float* data0_1024, float* data1_1024, unsigned int* data2_589824, unsigned int* data3_589824, half* data4_4096) {
  int gidx0 = blockIdx.x; /* 1024 */
  int lidx0 = threadIdx.x; /* 32 */
  float buf0[1];
  float buf1[1];
  (*(buf0+0)) = 0.0f;
  (*(buf1+0)) = 0.0f;
  int alu2 = (lidx0>>3);
  int alu3 = (lidx0&7);
  for (int Ridx0 = 0; Ridx0 < 4; Ridx0++) {
    int alu4 = ((alu2*144)+(Ridx0*36)+(gidx0*576));
    int alu5 = (alu4+alu3);
    int alu6 = (alu5+4);
    unsigned int val0 = (*(data2_589824+alu6));
    int alu7 = (alu5+12);
    unsigned int val1 = (*(data2_589824+alu7));
    int alu8 = (alu5+20);
    unsigned int val2 = (*(data2_589824+alu8));
    int alu9 = (alu5+28);
    unsigned int val3 = (*(data2_589824+alu9));
    unsigned int val4 = (*(data3_589824+alu6));
    unsigned int val5 = (*(data3_589824+alu7));
    unsigned int val6 = (*(data3_589824+alu8));
    unsigned int val7 = (*(data3_589824+alu9));
    uint4 val8 = (*((uint4*)((data2_589824+alu4))));
    uint4 val9 = (*((uint4*)((data3_589824+alu4))));
    int alu10 = ((alu2<<10)+(Ridx0<<8)+(alu3<<2));
    half4 val10 = (*((half4*)((data4_4096+(alu10+32)))));
    half4 val11 = (*((half4*)((data4_4096+(alu10+64)))));
    half4 val12 = (*((half4*)((data4_4096+(alu10+96)))));
    half4 val13 = (*((half4*)((data4_4096+(alu10+128)))));
    half4 val14 = (*((half4*)((data4_4096+(alu10+160)))));
    half4 val15 = (*((half4*)((data4_4096+(alu10+192)))));
    half4 val16 = (*((half4*)((data4_4096+(alu10+224)))));
    half4 val17 = (*((half4*)((data4_4096+alu10))));
    float cast0 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val8.x&65535u)))))));
    float cast1 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)(((val8.x>>16u)&65535u)))))));
    unsigned int alu11 = (val8.z>>0u);
    unsigned int alu12 = (val8.w>>0u);
    unsigned int alu13 = (val8.z>>8u);
    unsigned int alu14 = (val8.w>>8u);
    unsigned int alu15 = (val8.z>>16u);
    unsigned int alu16 = (val8.w>>16u);
    unsigned int alu17 = (val8.z>>24u);
    unsigned int alu18 = (val8.w>>24u);
    unsigned int alu19 = (val8.y>>0u);
    unsigned int alu20 = (val8.y>>8u);
    unsigned int alu21 = (val8.y>>16u);
    unsigned int alu22 = (val8.y>>24u);
    unsigned int alu23 = ((val0>>0u)&252645135u);
    unsigned int alu24 = ((val0>>4u)&252645135u);
    unsigned int alu25 = ((val1>>0u)&252645135u);
    unsigned int alu26 = ((val1>>4u)&252645135u);
    unsigned int alu27 = ((val2>>0u)&252645135u);
    unsigned int alu28 = ((val2>>4u)&252645135u);
    unsigned int alu29 = ((val3>>0u)&252645135u);
    unsigned int alu30 = ((val3>>4u)&252645135u);
    float alu31 = (cast0*((float)((alu19&63u))));
    float alu32 = (cast1*((float)((alu11&63u))));
    float alu33 = float(val17.x);
    float alu34 = float(val17.y);
    float alu35 = float(val17.z);
    float alu36 = float(val17.w);
    float alu37 = (cast0*((float)((alu20&63u))));
    float alu38 = (cast1*((float)((alu13&63u))));
    float alu39 = float(val10.x);
    float alu40 = float(val10.y);
    float alu41 = float(val10.z);
    float alu42 = float(val10.w);
    float alu43 = (cast0*((float)((alu21&63u))));
    float alu44 = (cast1*((float)((alu15&63u))));
    float alu45 = float(val11.x);
    float alu46 = float(val11.y);
    float alu47 = float(val11.z);
    float alu48 = float(val11.w);
    float alu49 = (cast0*((float)((alu22&63u))));
    float alu50 = (cast1*((float)((alu17&63u))));
    float alu51 = float(val12.x);
    float alu52 = float(val12.y);
    float alu53 = float(val12.z);
    float alu54 = float(val12.w);
    float alu55 = (cast0*((float)(((alu12&15u)|(((alu19&255u)>>6u)<<4u)))));
    float alu56 = (cast1*((float)((((alu12&255u)>>4u)|(((alu11&255u)>>6u)<<4u)))));
    float alu57 = float(val13.x);
    float alu58 = float(val13.y);
    float alu59 = float(val13.z);
    float alu60 = float(val13.w);
    float alu61 = (cast0*((float)(((alu14&15u)|(((alu20&255u)>>6u)<<4u)))));
    float alu62 = (cast1*((float)((((alu14&255u)>>4u)|(((alu13&255u)>>6u)<<4u)))));
    float alu63 = float(val14.x);
    float alu64 = float(val14.y);
    float alu65 = float(val14.z);
    float alu66 = float(val14.w);
    float alu67 = (cast0*((float)(((alu16&15u)|(((alu21&255u)>>6u)<<4u)))));
    float alu68 = (cast1*((float)((((alu16&255u)>>4u)|(((alu15&255u)>>6u)<<4u)))));
    float alu69 = float(val15.x);
    float alu70 = float(val15.y);
    float alu71 = float(val15.z);
    float alu72 = float(val15.w);
    float alu73 = (cast0*((float)(((alu18&15u)|(((alu22&255u)>>6u)<<4u)))));
    float alu74 = (cast1*((float)((((alu18&255u)>>4u)|(((alu17&255u)>>6u)<<4u)))));
    float alu75 = float(val16.x);
    float alu76 = float(val16.y);
    float alu77 = float(val16.z);
    float alu78 = float(val16.w);
    (*(buf0+0)) = ((*(buf0+0))+(((alu31*((float)(((alu23>>0u)&15u))))-alu32)*alu33)+(((alu31*((float)(((alu23>>8u)&15u))))-alu32)*alu34)+(((alu31*((float)(((alu23>>16u)&15u))))-alu32)*alu35)+(((alu31*((float)(((alu23>>24u)&15u))))-alu32)*alu36)+(((alu37*((float)(((alu24>>0u)&15u))))-alu38)*alu39)+(((alu37*((float)(((alu24>>8u)&15u))))-alu38)*alu40)+(((alu37*((float)(((alu24>>16u)&15u))))-alu38)*alu41)+(((alu37*((float)(((alu24>>24u)&15u))))-alu38)*alu42)+(((alu43*((float)(((alu25>>0u)&15u))))-alu44)*alu45)+(((alu43*((float)(((alu25>>8u)&15u))))-alu44)*alu46)+(((alu43*((float)(((alu25>>16u)&15u))))-alu44)*alu47)+(((alu43*((float)(((alu25>>24u)&15u))))-alu44)*alu48)+(((alu49*((float)(((alu26>>0u)&15u))))-alu50)*alu51)+(((alu49*((float)(((alu26>>8u)&15u))))-alu50)*alu52)+(((alu49*((float)(((alu26>>16u)&15u))))-alu50)*alu53)+(((alu49*((float)(((alu26>>24u)&15u))))-alu50)*alu54)+(((alu55*((float)(((alu27>>0u)&15u))))-alu56)*alu57)+(((alu55*((float)(((alu27>>8u)&15u))))-alu56)*alu58)+(((alu55*((float)(((alu27>>16u)&15u))))-alu56)*alu59)+(((alu55*((float)(((alu27>>24u)&15u))))-alu56)*alu60)+(((alu61*((float)(((alu28>>0u)&15u))))-alu62)*alu63)+(((alu61*((float)(((alu28>>8u)&15u))))-alu62)*alu64)+(((alu61*((float)(((alu28>>16u)&15u))))-alu62)*alu65)+(((alu61*((float)(((alu28>>24u)&15u))))-alu62)*alu66)+(((alu67*((float)(((alu29>>0u)&15u))))-alu68)*alu69)+(((alu67*((float)(((alu29>>8u)&15u))))-alu68)*alu70)+(((alu67*((float)(((alu29>>16u)&15u))))-alu68)*alu71)+(((alu67*((float)(((alu29>>24u)&15u))))-alu68)*alu72)+(((alu73*((float)(((alu30>>0u)&15u))))-alu74)*alu75)+(((alu73*((float)(((alu30>>8u)&15u))))-alu74)*alu76)+(((alu73*((float)(((alu30>>16u)&15u))))-alu74)*alu77)+(((alu73*((float)(((alu30>>24u)&15u))))-alu74)*alu78));
    float cast2 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val9.x&65535u)))))));
    float cast3 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)(((val9.x>>16u)&65535u)))))));
    unsigned int alu80 = (val9.z>>0u);
    unsigned int alu81 = (val9.w>>0u);
    unsigned int alu82 = (val9.z>>8u);
    unsigned int alu83 = (val9.w>>8u);
    unsigned int alu84 = (val9.z>>16u);
    unsigned int alu85 = (val9.w>>16u);
    unsigned int alu86 = (val9.z>>24u);
    unsigned int alu87 = (val9.w>>24u);
    unsigned int alu88 = (val9.y>>0u);
    unsigned int alu89 = (val9.y>>8u);
    unsigned int alu90 = (val9.y>>16u);
    unsigned int alu91 = (val9.y>>24u);
    unsigned int alu92 = ((val4>>0u)&252645135u);
    unsigned int alu93 = ((val4>>4u)&252645135u);
    unsigned int alu94 = ((val5>>0u)&252645135u);
    unsigned int alu95 = ((val5>>4u)&252645135u);
    unsigned int alu96 = ((val6>>0u)&252645135u);
    unsigned int alu97 = ((val6>>4u)&252645135u);
    unsigned int alu98 = ((val7>>0u)&252645135u);
    unsigned int alu99 = ((val7>>4u)&252645135u);
    float alu100 = (cast2*((float)((alu88&63u))));
    float alu101 = (cast3*((float)((alu80&63u))));
    float alu102 = (cast2*((float)((alu89&63u))));
    float alu103 = (cast3*((float)((alu82&63u))));
    float alu104 = (cast2*((float)((alu90&63u))));
    float alu105 = (cast3*((float)((alu84&63u))));
    float alu106 = (cast2*((float)((alu91&63u))));
    float alu107 = (cast3*((float)((alu86&63u))));
    float alu108 = (cast2*((float)(((alu81&15u)|(((alu88&255u)>>6u)<<4u)))));
    float alu109 = (cast3*((float)((((alu81&255u)>>4u)|(((alu80&255u)>>6u)<<4u)))));
    float alu110 = (cast2*((float)(((alu83&15u)|(((alu89&255u)>>6u)<<4u)))));
    float alu111 = (cast3*((float)((((alu83&255u)>>4u)|(((alu82&255u)>>6u)<<4u)))));
    float alu112 = (cast2*((float)(((alu85&15u)|(((alu90&255u)>>6u)<<4u)))));
    float alu113 = (cast3*((float)((((alu85&255u)>>4u)|(((alu84&255u)>>6u)<<4u)))));
    float alu114 = (cast2*((float)(((alu87&15u)|(((alu91&255u)>>6u)<<4u)))));
    float alu115 = (cast3*((float)((((alu87&255u)>>4u)|(((alu86&255u)>>6u)<<4u)))));
    (*(buf1+0)) = ((*(buf1+0))+(((alu100*((float)(((alu92>>0u)&15u))))-alu101)*alu33)+(((alu100*((float)(((alu92>>8u)&15u))))-alu101)*alu34)+(((alu100*((float)(((alu92>>16u)&15u))))-alu101)*alu35)+(((alu100*((float)(((alu92>>24u)&15u))))-alu101)*alu36)+(((alu102*((float)(((alu93>>0u)&15u))))-alu103)*alu39)+(((alu102*((float)(((alu93>>8u)&15u))))-alu103)*alu40)+(((alu102*((float)(((alu93>>16u)&15u))))-alu103)*alu41)+(((alu102*((float)(((alu93>>24u)&15u))))-alu103)*alu42)+(((alu104*((float)(((alu94>>0u)&15u))))-alu105)*alu45)+(((alu104*((float)(((alu94>>8u)&15u))))-alu105)*alu46)+(((alu104*((float)(((alu94>>16u)&15u))))-alu105)*alu47)+(((alu104*((float)(((alu94>>24u)&15u))))-alu105)*alu48)+(((alu106*((float)(((alu95>>0u)&15u))))-alu107)*alu51)+(((alu106*((float)(((alu95>>8u)&15u))))-alu107)*alu52)+(((alu106*((float)(((alu95>>16u)&15u))))-alu107)*alu53)+(((alu106*((float)(((alu95>>24u)&15u))))-alu107)*alu54)+(((alu108*((float)(((alu96>>0u)&15u))))-alu109)*alu57)+(((alu108*((float)(((alu96>>8u)&15u))))-alu109)*alu58)+(((alu108*((float)(((alu96>>16u)&15u))))-alu109)*alu59)+(((alu108*((float)(((alu96>>24u)&15u))))-alu109)*alu60)+(((alu110*((float)(((alu97>>0u)&15u))))-alu111)*alu63)+(((alu110*((float)(((alu97>>8u)&15u))))-alu111)*alu64)+(((alu110*((float)(((alu97>>16u)&15u))))-alu111)*alu65)+(((alu110*((float)(((alu97>>24u)&15u))))-alu111)*alu66)+(((alu112*((float)(((alu98>>0u)&15u))))-alu113)*alu69)+(((alu112*((float)(((alu98>>8u)&15u))))-alu113)*alu70)+(((alu112*((float)(((alu98>>16u)&15u))))-alu113)*alu71)+(((alu112*((float)(((alu98>>24u)&15u))))-alu113)*alu72)+(((alu114*((float)(((alu99>>0u)&15u))))-alu115)*alu75)+(((alu114*((float)(((alu99>>8u)&15u))))-alu115)*alu76)+(((alu114*((float)(((alu99>>16u)&15u))))-alu115)*alu77)+(((alu114*((float)(((alu99>>24u)&15u))))-alu115)*alu78));
  }
  float buf2[1];
  float buf3[1];
  (*(buf2+0)) = __shfl_xor_sync(0xffffffffu, (*(buf0+0)), 16);
  float alu119 = ((*(buf0+0))+(*(buf2+0)));
  (*(buf2+0)) = __shfl_xor_sync(0xffffffffu, alu119, 8);
  float alu121 = (alu119+(*(buf2+0)));
  (*(buf2+0)) = __shfl_xor_sync(0xffffffffu, alu121, 4);
  float alu123 = (alu121+(*(buf2+0)));
  (*(buf2+0)) = __shfl_xor_sync(0xffffffffu, alu123, 2);
  float alu125 = (alu123+(*(buf2+0)));
  (*(buf2+0)) = __shfl_xor_sync(0xffffffffu, alu125, 1);
  (*(buf3+0)) = __shfl_xor_sync(0xffffffffu, (*(buf1+0)), 16);
  float alu128 = ((*(buf1+0))+(*(buf3+0)));
  (*(buf3+0)) = __shfl_xor_sync(0xffffffffu, alu128, 8);
  float alu130 = (alu128+(*(buf3+0)));
  (*(buf3+0)) = __shfl_xor_sync(0xffffffffu, alu130, 4);
  float alu132 = (alu130+(*(buf3+0)));
  (*(buf3+0)) = __shfl_xor_sync(0xffffffffu, alu132, 2);
  float alu134 = (alu132+(*(buf3+0)));
  (*(buf3+0)) = __shfl_xor_sync(0xffffffffu, alu134, 1);
  *(data0_1024+gidx0) = (alu125+(*(buf2+0)));
  *(data1_1024+gidx0) = (alu134+(*(buf3+0)));
}
extern "C" __global__ void __launch_bounds__(32) q4k_g3_lanemap_gemv_qkv_full_4096_1024_4096(float* data0_4096, float* data1_1024, float* data2_1024, unsigned int* data3_2359296, unsigned int* data4_1179648, half* data5_4096) {
  int gidx0 = blockIdx.x; /* 4096 */
  int lidx0 = threadIdx.x; /* 32 */
  float buf0[1];
  (*(buf0+0)) = 0.0f;
  int alu1 = (gidx0*576);
  int alu2 = (lidx0>>3);
  int alu3 = (alu2*144);
  int alu4 = (alu2<<10);
  int alu5 = (lidx0&7);
  int alu6 = (alu5<<2);
  for (int Ridx0 = 0; Ridx0 < 4; Ridx0++) {
    int alu7 = (alu3+(Ridx0*36)+alu1);
    int alu8 = (alu7+alu5);
    unsigned int val0 = (*(data3_2359296+(alu8+4)));
    unsigned int val1 = (*(data3_2359296+(alu8+12)));
    unsigned int val2 = (*(data3_2359296+(alu8+20)));
    unsigned int val3 = (*(data3_2359296+(alu8+28)));
    uint4 val4 = (*((uint4*)((data3_2359296+alu7))));
    int alu9 = (alu4+(Ridx0<<8)+alu6);
    half4 val5 = (*((half4*)((data5_4096+(alu9+32)))));
    half4 val6 = (*((half4*)((data5_4096+(alu9+64)))));
    half4 val7 = (*((half4*)((data5_4096+(alu9+96)))));
    half4 val8 = (*((half4*)((data5_4096+(alu9+128)))));
    half4 val9 = (*((half4*)((data5_4096+(alu9+160)))));
    half4 val10 = (*((half4*)((data5_4096+(alu9+192)))));
    half4 val11 = (*((half4*)((data5_4096+(alu9+224)))));
    half4 val12 = (*((half4*)((data5_4096+alu9))));
    float cast0 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val4.x&65535u)))))));
    float cast1 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)(((val4.x>>16u)&65535u)))))));
    unsigned int alu10 = (val4.z>>0u);
    unsigned int alu11 = (val4.w>>0u);
    unsigned int alu12 = (val4.z>>8u);
    unsigned int alu13 = (val4.w>>8u);
    unsigned int alu14 = (val4.z>>16u);
    unsigned int alu15 = (val4.w>>16u);
    unsigned int alu16 = (val4.z>>24u);
    unsigned int alu17 = (val4.w>>24u);
    unsigned int alu18 = (val4.y>>0u);
    unsigned int alu19 = (val4.y>>8u);
    unsigned int alu20 = (val4.y>>16u);
    unsigned int alu21 = (val4.y>>24u);
    unsigned int alu22 = ((val0>>0u)&252645135u);
    unsigned int alu23 = ((val0>>4u)&252645135u);
    unsigned int alu24 = ((val1>>0u)&252645135u);
    unsigned int alu25 = ((val1>>4u)&252645135u);
    unsigned int alu26 = ((val2>>0u)&252645135u);
    unsigned int alu27 = ((val2>>4u)&252645135u);
    unsigned int alu28 = ((val3>>0u)&252645135u);
    unsigned int alu29 = ((val3>>4u)&252645135u);
    float alu30 = (cast0*((float)((alu18&63u))));
    float alu31 = (cast1*((float)((alu10&63u))));
    float alu32 = (cast0*((float)((alu19&63u))));
    float alu33 = (cast1*((float)((alu12&63u))));
    float alu34 = (cast0*((float)((alu20&63u))));
    float alu35 = (cast1*((float)((alu14&63u))));
    float alu36 = (cast0*((float)((alu21&63u))));
    float alu37 = (cast1*((float)((alu16&63u))));
    float alu38 = (cast0*((float)(((alu11&15u)|(((alu18&255u)>>6u)<<4u)))));
    float alu39 = (cast1*((float)((((alu11&255u)>>4u)|(((alu10&255u)>>6u)<<4u)))));
    float alu40 = (cast0*((float)(((alu13&15u)|(((alu19&255u)>>6u)<<4u)))));
    float alu41 = (cast1*((float)((((alu13&255u)>>4u)|(((alu12&255u)>>6u)<<4u)))));
    float alu42 = (cast0*((float)(((alu15&15u)|(((alu20&255u)>>6u)<<4u)))));
    float alu43 = (cast1*((float)((((alu15&255u)>>4u)|(((alu14&255u)>>6u)<<4u)))));
    float alu44 = (cast0*((float)(((alu17&15u)|(((alu21&255u)>>6u)<<4u)))));
    float alu45 = (cast1*((float)((((alu17&255u)>>4u)|(((alu16&255u)>>6u)<<4u)))));
    (*(buf0+0)) = ((*(buf0+0))+(((alu30*((float)(((alu22>>0u)&15u))))-alu31)*float(val12.x))+(((alu30*((float)(((alu22>>8u)&15u))))-alu31)*float(val12.y))+(((alu30*((float)(((alu22>>16u)&15u))))-alu31)*float(val12.z))+(((alu30*((float)(((alu22>>24u)&15u))))-alu31)*float(val12.w))+(((alu32*((float)(((alu23>>0u)&15u))))-alu33)*float(val5.x))+(((alu32*((float)(((alu23>>8u)&15u))))-alu33)*float(val5.y))+(((alu32*((float)(((alu23>>16u)&15u))))-alu33)*float(val5.z))+(((alu32*((float)(((alu23>>24u)&15u))))-alu33)*float(val5.w))+(((alu34*((float)(((alu24>>0u)&15u))))-alu35)*float(val6.x))+(((alu34*((float)(((alu24>>8u)&15u))))-alu35)*float(val6.y))+(((alu34*((float)(((alu24>>16u)&15u))))-alu35)*float(val6.z))+(((alu34*((float)(((alu24>>24u)&15u))))-alu35)*float(val6.w))+(((alu36*((float)(((alu25>>0u)&15u))))-alu37)*float(val7.x))+(((alu36*((float)(((alu25>>8u)&15u))))-alu37)*float(val7.y))+(((alu36*((float)(((alu25>>16u)&15u))))-alu37)*float(val7.z))+(((alu36*((float)(((alu25>>24u)&15u))))-alu37)*float(val7.w))+(((alu38*((float)(((alu26>>0u)&15u))))-alu39)*float(val8.x))+(((alu38*((float)(((alu26>>8u)&15u))))-alu39)*float(val8.y))+(((alu38*((float)(((alu26>>16u)&15u))))-alu39)*float(val8.z))+(((alu38*((float)(((alu26>>24u)&15u))))-alu39)*float(val8.w))+(((alu40*((float)(((alu27>>0u)&15u))))-alu41)*float(val9.x))+(((alu40*((float)(((alu27>>8u)&15u))))-alu41)*float(val9.y))+(((alu40*((float)(((alu27>>16u)&15u))))-alu41)*float(val9.z))+(((alu40*((float)(((alu27>>24u)&15u))))-alu41)*float(val9.w))+(((alu42*((float)(((alu28>>0u)&15u))))-alu43)*float(val10.x))+(((alu42*((float)(((alu28>>8u)&15u))))-alu43)*float(val10.y))+(((alu42*((float)(((alu28>>16u)&15u))))-alu43)*float(val10.z))+(((alu42*((float)(((alu28>>24u)&15u))))-alu43)*float(val10.w))+(((alu44*((float)(((alu29>>0u)&15u))))-alu45)*float(val11.x))+(((alu44*((float)(((alu29>>8u)&15u))))-alu45)*float(val11.y))+(((alu44*((float)(((alu29>>16u)&15u))))-alu45)*float(val11.z))+(((alu44*((float)(((alu29>>24u)&15u))))-alu45)*float(val11.w)));
  }
  float buf1[1];
  float buf2[1];
  (*(buf2+0)) = __shfl_xor_sync(0xffffffffu, (*(buf0+0)), 16);
  float alu49 = ((*(buf0+0))+(*(buf2+0)));
  (*(buf2+0)) = __shfl_xor_sync(0xffffffffu, alu49, 8);
  float alu51 = (alu49+(*(buf2+0)));
  (*(buf2+0)) = __shfl_xor_sync(0xffffffffu, alu51, 4);
  float alu53 = (alu51+(*(buf2+0)));
  (*(buf2+0)) = __shfl_xor_sync(0xffffffffu, alu53, 2);
  float alu55 = (alu53+(*(buf2+0)));
  (*(buf2+0)) = __shfl_xor_sync(0xffffffffu, alu55, 1);
  *(data0_4096+gidx0) = (alu55+(*(buf2+0)));
  __syncthreads();
  if ((gidx0<2048)) {
    (*(buf1+0)) = 0.0f;
    for (int Ridx1 = 0; Ridx1 < 4; Ridx1++) {
      int alu61 = (alu3+(Ridx1*36)+alu1);
      int alu62 = (alu61+alu5);
      unsigned int val13 = (*(data4_1179648+(alu62+4)));
      unsigned int val14 = (*(data4_1179648+(alu62+12)));
      unsigned int val15 = (*(data4_1179648+(alu62+20)));
      unsigned int val16 = (*(data4_1179648+(alu62+28)));
      unsigned int val17 = (*(data4_1179648+(alu61+1)));
      unsigned int val18 = (*(data4_1179648+(alu61+2)));
      unsigned int val19 = (*(data4_1179648+(alu61+3)));
      unsigned int val20 = (*(data4_1179648+alu61));
      int alu63 = (alu4+(Ridx1<<8)+alu6);
      half4 val21 = (*((half4*)((data5_4096+(alu63+32)))));
      half4 val22 = (*((half4*)((data5_4096+(alu63+64)))));
      half4 val23 = (*((half4*)((data5_4096+(alu63+96)))));
      half4 val24 = (*((half4*)((data5_4096+(alu63+128)))));
      half4 val25 = (*((half4*)((data5_4096+(alu63+160)))));
      half4 val26 = (*((half4*)((data5_4096+(alu63+192)))));
      half4 val27 = (*((half4*)((data5_4096+(alu63+224)))));
      half4 val28 = (*((half4*)((data5_4096+alu63))));
      float cast2 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val20&65535u)))))));
      float cast3 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)(((val20>>16u)&65535u)))))));
      unsigned int alu64 = (val18>>0u);
      unsigned int alu65 = (val19>>0u);
      unsigned int alu66 = (val18>>8u);
      unsigned int alu67 = (val19>>8u);
      unsigned int alu68 = (val18>>16u);
      unsigned int alu69 = (val19>>16u);
      unsigned int alu70 = (val18>>24u);
      unsigned int alu71 = (val19>>24u);
      unsigned int alu72 = (val17>>0u);
      unsigned int alu73 = (val17>>8u);
      unsigned int alu74 = (val17>>16u);
      unsigned int alu75 = (val17>>24u);
      unsigned int alu76 = ((val13>>0u)&252645135u);
      unsigned int alu77 = ((val13>>4u)&252645135u);
      unsigned int alu78 = ((val14>>0u)&252645135u);
      unsigned int alu79 = ((val14>>4u)&252645135u);
      unsigned int alu80 = ((val15>>0u)&252645135u);
      unsigned int alu81 = ((val15>>4u)&252645135u);
      unsigned int alu82 = ((val16>>0u)&252645135u);
      unsigned int alu83 = ((val16>>4u)&252645135u);
      float alu84 = (cast2*((float)((alu72&63u))));
      float alu85 = (cast3*((float)((alu64&63u))));
      float alu86 = (cast2*((float)((alu73&63u))));
      float alu87 = (cast3*((float)((alu66&63u))));
      float alu88 = (cast2*((float)((alu74&63u))));
      float alu89 = (cast3*((float)((alu68&63u))));
      float alu90 = (cast2*((float)((alu75&63u))));
      float alu91 = (cast3*((float)((alu70&63u))));
      float alu92 = (cast2*((float)(((alu65&15u)|(((alu72&255u)>>6u)<<4u)))));
      float alu93 = (cast3*((float)((((alu65&255u)>>4u)|(((alu64&255u)>>6u)<<4u)))));
      float alu94 = (cast2*((float)(((alu67&15u)|(((alu73&255u)>>6u)<<4u)))));
      float alu95 = (cast3*((float)((((alu67&255u)>>4u)|(((alu66&255u)>>6u)<<4u)))));
      float alu96 = (cast2*((float)(((alu69&15u)|(((alu74&255u)>>6u)<<4u)))));
      float alu97 = (cast3*((float)((((alu69&255u)>>4u)|(((alu68&255u)>>6u)<<4u)))));
      float alu98 = (cast2*((float)(((alu71&15u)|(((alu75&255u)>>6u)<<4u)))));
      float alu99 = (cast3*((float)((((alu71&255u)>>4u)|(((alu70&255u)>>6u)<<4u)))));
      (*(buf1+0)) = ((*(buf1+0))+(((alu84*((float)(((alu76>>0u)&15u))))-alu85)*float(val28.x))+(((alu84*((float)(((alu76>>8u)&15u))))-alu85)*float(val28.y))+(((alu84*((float)(((alu76>>16u)&15u))))-alu85)*float(val28.z))+(((alu84*((float)(((alu76>>24u)&15u))))-alu85)*float(val28.w))+(((alu86*((float)(((alu77>>0u)&15u))))-alu87)*float(val21.x))+(((alu86*((float)(((alu77>>8u)&15u))))-alu87)*float(val21.y))+(((alu86*((float)(((alu77>>16u)&15u))))-alu87)*float(val21.z))+(((alu86*((float)(((alu77>>24u)&15u))))-alu87)*float(val21.w))+(((alu88*((float)(((alu78>>0u)&15u))))-alu89)*float(val22.x))+(((alu88*((float)(((alu78>>8u)&15u))))-alu89)*float(val22.y))+(((alu88*((float)(((alu78>>16u)&15u))))-alu89)*float(val22.z))+(((alu88*((float)(((alu78>>24u)&15u))))-alu89)*float(val22.w))+(((alu90*((float)(((alu79>>0u)&15u))))-alu91)*float(val23.x))+(((alu90*((float)(((alu79>>8u)&15u))))-alu91)*float(val23.y))+(((alu90*((float)(((alu79>>16u)&15u))))-alu91)*float(val23.z))+(((alu90*((float)(((alu79>>24u)&15u))))-alu91)*float(val23.w))+(((alu92*((float)(((alu80>>0u)&15u))))-alu93)*float(val24.x))+(((alu92*((float)(((alu80>>8u)&15u))))-alu93)*float(val24.y))+(((alu92*((float)(((alu80>>16u)&15u))))-alu93)*float(val24.z))+(((alu92*((float)(((alu80>>24u)&15u))))-alu93)*float(val24.w))+(((alu94*((float)(((alu81>>0u)&15u))))-alu95)*float(val25.x))+(((alu94*((float)(((alu81>>8u)&15u))))-alu95)*float(val25.y))+(((alu94*((float)(((alu81>>16u)&15u))))-alu95)*float(val25.z))+(((alu94*((float)(((alu81>>24u)&15u))))-alu95)*float(val25.w))+(((alu96*((float)(((alu82>>0u)&15u))))-alu97)*float(val26.x))+(((alu96*((float)(((alu82>>8u)&15u))))-alu97)*float(val26.y))+(((alu96*((float)(((alu82>>16u)&15u))))-alu97)*float(val26.z))+(((alu96*((float)(((alu82>>24u)&15u))))-alu97)*float(val26.w))+(((alu98*((float)(((alu83>>0u)&15u))))-alu99)*float(val27.x))+(((alu98*((float)(((alu83>>8u)&15u))))-alu99)*float(val27.y))+(((alu98*((float)(((alu83>>16u)&15u))))-alu99)*float(val27.z))+(((alu98*((float)(((alu83>>24u)&15u))))-alu99)*float(val27.w)));
    }
    float buf3[1];
    (*(buf3+0)) = __shfl_xor_sync(0xffffffffu, (*(buf1+0)), 16);
    float alu103 = ((*(buf1+0))+(*(buf3+0)));
    (*(buf3+0)) = __shfl_xor_sync(0xffffffffu, alu103, 8);
    float alu105 = (alu103+(*(buf3+0)));
    (*(buf3+0)) = __shfl_xor_sync(0xffffffffu, alu105, 4);
    float alu107 = (alu105+(*(buf3+0)));
    (*(buf3+0)) = __shfl_xor_sync(0xffffffffu, alu107, 2);
    float alu109 = (alu107+(*(buf3+0)));
    (*(buf3+0)) = __shfl_xor_sync(0xffffffffu, alu109, 1);
    bool alu111 = (gidx0<1024);
    int alu112 = (alu111?gidx0:0);
    int alu113 = (alu111?0:(gidx0+-1024));
    float alu114 = (alu109+(*(buf3+0)));
    if (alu111) {
      *(data1_1024+alu112) = alu114;
    }
    if ((1023<gidx0)) {
      *(data2_1024+alu113) = alu114;
    }
  }
}
extern "C" __global__ void __launch_bounds__(32) q4k_projection_group_one_task_phased_6144_4096(float* data0_6144, unsigned int* data1_3538944, half* data2_4096) {
  int gidx0 = blockIdx.x; /* 6144 */
  int lidx0 = threadIdx.x; /* 32 */
  float buf0[1];
  (*(buf0+0)) = 0.0f;
  int alu1 = (lidx0>>3);
  int alu2 = (lidx0&7);
  for (int Ridx0 = 0; Ridx0 < 4; Ridx0++) {
    int alu3 = ((alu1*144)+(Ridx0*36)+(gidx0*576));
    int alu4 = (alu3+alu2);
    unsigned int val0 = (*(data1_3538944+(alu4+4)));
    unsigned int val1 = (*(data1_3538944+(alu4+12)));
    unsigned int val2 = (*(data1_3538944+(alu4+20)));
    unsigned int val3 = (*(data1_3538944+(alu4+28)));
    uint4 val4 = (*((uint4*)((data1_3538944+alu3))));
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
  *(data0_6144+gidx0) = (alu51+(*(buf1+0)));
}
extern "C" __global__ void __launch_bounds__(32) q4k_projection_group_one_task_interleaved_6144_4096(float* data0_6144, unsigned int* data1_3538944, half* data2_4096) {
  int gidx0 = blockIdx.x; /* 6144 */
  int lidx0 = threadIdx.x; /* 32 */
  float buf0[1];
  (*(buf0+0)) = 0.0f;
  int alu1 = (gidx0/6);
  int alu2 = (gidx0%6);
  int alu3 = (lidx0>>3);
  int alu4 = ((alu2!=4)?(alu1+5120):(alu1+4096));
  int alu5 = ((alu2<4)?((alu1<<2)+alu2):alu4);
  int alu6 = (lidx0&7);
  for (int Ridx0 = 0; Ridx0 < 4; Ridx0++) {
    int alu7 = ((alu3*144)+(Ridx0*36)+(alu5*576));
    int alu8 = (alu7+alu6);
    unsigned int val0 = (*(data1_3538944+(alu8+4)));
    unsigned int val1 = (*(data1_3538944+(alu8+12)));
    unsigned int val2 = (*(data1_3538944+(alu8+20)));
    unsigned int val3 = (*(data1_3538944+(alu8+28)));
    uint4 val4 = (*((uint4*)((data1_3538944+alu7))));
    int alu9 = ((alu3<<10)+(Ridx0<<8)+(alu6<<2));
    half4 val5 = (*((half4*)((data2_4096+(alu9+32)))));
    half4 val6 = (*((half4*)((data2_4096+(alu9+64)))));
    half4 val7 = (*((half4*)((data2_4096+(alu9+96)))));
    half4 val8 = (*((half4*)((data2_4096+(alu9+128)))));
    half4 val9 = (*((half4*)((data2_4096+(alu9+160)))));
    half4 val10 = (*((half4*)((data2_4096+(alu9+192)))));
    half4 val11 = (*((half4*)((data2_4096+(alu9+224)))));
    half4 val12 = (*((half4*)((data2_4096+alu9))));
    float cast0 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val4.x&65535u)))))));
    float cast1 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)(((val4.x>>16u)&65535u)))))));
    unsigned int alu10 = (val4.z>>0u);
    unsigned int alu11 = (val4.w>>0u);
    unsigned int alu12 = (val4.z>>8u);
    unsigned int alu13 = (val4.w>>8u);
    unsigned int alu14 = (val4.z>>16u);
    unsigned int alu15 = (val4.w>>16u);
    unsigned int alu16 = (val4.z>>24u);
    unsigned int alu17 = (val4.w>>24u);
    unsigned int alu18 = (val4.y>>0u);
    unsigned int alu19 = (val4.y>>8u);
    unsigned int alu20 = (val4.y>>16u);
    unsigned int alu21 = (val4.y>>24u);
    unsigned int alu22 = ((val0>>0u)&252645135u);
    unsigned int alu23 = ((val0>>4u)&252645135u);
    unsigned int alu24 = ((val1>>0u)&252645135u);
    unsigned int alu25 = ((val1>>4u)&252645135u);
    unsigned int alu26 = ((val2>>0u)&252645135u);
    unsigned int alu27 = ((val2>>4u)&252645135u);
    unsigned int alu28 = ((val3>>0u)&252645135u);
    unsigned int alu29 = ((val3>>4u)&252645135u);
    float alu30 = (cast0*((float)((alu18&63u))));
    float alu31 = (cast1*((float)((alu10&63u))));
    float alu32 = (cast0*((float)((alu19&63u))));
    float alu33 = (cast1*((float)((alu12&63u))));
    float alu34 = (cast0*((float)((alu20&63u))));
    float alu35 = (cast1*((float)((alu14&63u))));
    float alu36 = (cast0*((float)((alu21&63u))));
    float alu37 = (cast1*((float)((alu16&63u))));
    float alu38 = (cast0*((float)(((alu11&15u)|(((alu18&255u)>>6u)<<4u)))));
    float alu39 = (cast1*((float)((((alu11&255u)>>4u)|(((alu10&255u)>>6u)<<4u)))));
    float alu40 = (cast0*((float)(((alu13&15u)|(((alu19&255u)>>6u)<<4u)))));
    float alu41 = (cast1*((float)((((alu13&255u)>>4u)|(((alu12&255u)>>6u)<<4u)))));
    float alu42 = (cast0*((float)(((alu15&15u)|(((alu20&255u)>>6u)<<4u)))));
    float alu43 = (cast1*((float)((((alu15&255u)>>4u)|(((alu14&255u)>>6u)<<4u)))));
    float alu44 = (cast0*((float)(((alu17&15u)|(((alu21&255u)>>6u)<<4u)))));
    float alu45 = (cast1*((float)((((alu17&255u)>>4u)|(((alu16&255u)>>6u)<<4u)))));
    (*(buf0+0)) = ((*(buf0+0))+(((alu30*((float)(((alu22>>0u)&15u))))-alu31)*float(val12.x))+(((alu30*((float)(((alu22>>8u)&15u))))-alu31)*float(val12.y))+(((alu30*((float)(((alu22>>16u)&15u))))-alu31)*float(val12.z))+(((alu30*((float)(((alu22>>24u)&15u))))-alu31)*float(val12.w))+(((alu32*((float)(((alu23>>0u)&15u))))-alu33)*float(val5.x))+(((alu32*((float)(((alu23>>8u)&15u))))-alu33)*float(val5.y))+(((alu32*((float)(((alu23>>16u)&15u))))-alu33)*float(val5.z))+(((alu32*((float)(((alu23>>24u)&15u))))-alu33)*float(val5.w))+(((alu34*((float)(((alu24>>0u)&15u))))-alu35)*float(val6.x))+(((alu34*((float)(((alu24>>8u)&15u))))-alu35)*float(val6.y))+(((alu34*((float)(((alu24>>16u)&15u))))-alu35)*float(val6.z))+(((alu34*((float)(((alu24>>24u)&15u))))-alu35)*float(val6.w))+(((alu36*((float)(((alu25>>0u)&15u))))-alu37)*float(val7.x))+(((alu36*((float)(((alu25>>8u)&15u))))-alu37)*float(val7.y))+(((alu36*((float)(((alu25>>16u)&15u))))-alu37)*float(val7.z))+(((alu36*((float)(((alu25>>24u)&15u))))-alu37)*float(val7.w))+(((alu38*((float)(((alu26>>0u)&15u))))-alu39)*float(val8.x))+(((alu38*((float)(((alu26>>8u)&15u))))-alu39)*float(val8.y))+(((alu38*((float)(((alu26>>16u)&15u))))-alu39)*float(val8.z))+(((alu38*((float)(((alu26>>24u)&15u))))-alu39)*float(val8.w))+(((alu40*((float)(((alu27>>0u)&15u))))-alu41)*float(val9.x))+(((alu40*((float)(((alu27>>8u)&15u))))-alu41)*float(val9.y))+(((alu40*((float)(((alu27>>16u)&15u))))-alu41)*float(val9.z))+(((alu40*((float)(((alu27>>24u)&15u))))-alu41)*float(val9.w))+(((alu42*((float)(((alu28>>0u)&15u))))-alu43)*float(val10.x))+(((alu42*((float)(((alu28>>8u)&15u))))-alu43)*float(val10.y))+(((alu42*((float)(((alu28>>16u)&15u))))-alu43)*float(val10.z))+(((alu42*((float)(((alu28>>24u)&15u))))-alu43)*float(val10.w))+(((alu44*((float)(((alu29>>0u)&15u))))-alu45)*float(val11.x))+(((alu44*((float)(((alu29>>8u)&15u))))-alu45)*float(val11.y))+(((alu44*((float)(((alu29>>16u)&15u))))-alu45)*float(val11.z))+(((alu44*((float)(((alu29>>24u)&15u))))-alu45)*float(val11.w)));
  }
  float buf1[1];
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, (*(buf0+0)), 16);
  float alu49 = ((*(buf0+0))+(*(buf1+0)));
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu49, 8);
  float alu51 = (alu49+(*(buf1+0)));
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu51, 4);
  float alu53 = (alu51+(*(buf1+0)));
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu53, 2);
  float alu55 = (alu53+(*(buf1+0)));
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu55, 1);
  *(data0_6144+alu5) = (alu55+(*(buf1+0)));
}
static void ck(cudaError_t e,const char* what) { if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",what,cudaGetErrorString(e));exit(2);} }
static unsigned int step(unsigned int& s) { s=1664525u*s+1013904223u; return s; }
// Produce structurally legal Q4_K blocks: finite positive fp16 d/dmin,
// packed 6-bit scales/mins, and arbitrary packed nibbles.
static void fill_legal(unsigned int* w,int fixture) {
  unsigned int state=0x1234567u ^ (unsigned(fixture)*0x9e3779b9u);
  const unsigned short dbits[4]={0x2c00u,0x3000u,0x3400u,0x3800u};
  const unsigned short mbits[4]={0x2800u,0x2c00u,0x3000u,0x3400u};
  for(size_t b=0;b<(size_t)TOTAL_ROWS*(K/256);b++) {
    size_t base=b*36; int p=int((b+fixture)&3);
    w[base]=unsigned(dbits[p]) | (unsigned(mbits[(p+1)&3])<<16);
    if(fixture==0) {
      w[base+1]=step(state); w[base+2]=step(state); w[base+3]=step(state);
      for(int i=4;i<36;i++) w[base+i]=step(state);
    } else if(fixture==1) {
      w[base+1]=0x01020304u^unsigned(b); w[base+2]=0x10203040u+unsigned(b); w[base+3]=0x3f2f1f0fu;
      for(int i=4;i<36;i++) w[base+i]=0x01234567u*unsigned(i)+unsigned(b)*0x11111111u;
    } else {
      w[base+1]=0x3f3f3f3fu; w[base+2]=0x15151515u; w[base+3]=0x2a2a2a2au;
      for(int i=4;i<36;i++) w[base+i]=(i&1)?0xfedcba98u:0x76543210u;
    }
  }
}
static void launch(int arm,float* out,unsigned int* group,half* x) {
  if(arm==0) {
    q4k_g3_lanemap_gemv_vec_4096_4096<<<Q_ROWS,32>>>(out,group,x);
    q4k_g3_lanemap_gemv_pair_vec_1024_4096<<<KV_ROWS,32>>>(out+Q_ROWS,out+Q_ROWS+KV_ROWS,group+Q_WORDS,group+Q_WORDS+KV_WORDS,x);
  } else if(arm==1) {
    q4k_g3_lanemap_gemv_qkv_full_4096_1024_4096<<<Q_ROWS,32>>>(out,out+Q_ROWS,out+Q_ROWS+KV_ROWS,group,group+Q_WORDS,x);
  } else if(arm==2) {
    q4k_projection_group_one_task_phased_6144_4096<<<TOTAL_ROWS,32>>>(out,group,x);
  } else {
    q4k_projection_group_one_task_interleaved_6144_4096<<<TOTAL_ROWS,32>>>(out,group,x);
  }
}
static double hot(int arm,float* out,unsigned int* group,half* x,int passes) {
  cudaEvent_t s,e; ck(cudaEventCreate(&s),"event"); ck(cudaEventCreate(&e),"event"); ck(cudaEventRecord(s),"record");
  for(int i=0;i<passes;i++) launch(arm,out,group,x);
  ck(cudaEventRecord(e),"record"); ck(cudaEventSynchronize(e),"sync"); float ms=0; ck(cudaEventElapsedTime(&ms,s,e),"elapsed");
  cudaEventDestroy(s); cudaEventDestroy(e); return ms*1000.0/passes;
}
static double rotated(int arm,float* out,unsigned int* groups,half* x,int passes) {
  cudaEvent_t s,e; ck(cudaEventCreate(&s),"event"); ck(cudaEventCreate(&e),"event"); double us=0;
  for(int i=0;i<passes;i++) { unsigned int* g=groups+(i%ROTATIONS)*GROUP_WORDS; ck(cudaEventRecord(s),"record");
    launch(arm,out,g,x); ck(cudaEventRecord(e),"record"); ck(cudaEventSynchronize(e),"sync"); float ms=0; ck(cudaEventElapsedTime(&ms,s,e),"elapsed"); us+=ms*1000.0; }
  cudaEventDestroy(s); cudaEventDestroy(e); return us/passes;
}
int main(int argc,char** argv) {
  int hot_passes=argc>1?atoi(argv[1]):500,cold_passes=argc>2?atoi(argv[2]):32,reps=argc>3?atoi(argv[3]):9;
  float *outs[4]; unsigned int* groups; half* x;
  for(int a=0;a<4;a++) ck(cudaMalloc(&outs[a],TOTAL_ROWS*sizeof(float)),"out");
  ck(cudaMalloc(&groups,(size_t)ROTATIONS*GROUP_WORDS*sizeof(unsigned int)),"groups"); ck(cudaMalloc(&x,K*sizeof(half)),"x");
  unsigned int* hw=(unsigned int*)malloc((size_t)ROTATIONS*GROUP_WORDS*sizeof(unsigned int)); half* hx=(half*)malloc(K*sizeof(half));
  for(int r=0;r<ROTATIONS;r++) fill_legal(hw+(size_t)r*GROUP_WORDS,r%3);
  for(int i=0;i<K;i++) hx[i]=__float2half(((i%257)-128)*0.03125f);
  ck(cudaMemcpy(groups,hw,(size_t)ROTATIONS*GROUP_WORDS*sizeof(unsigned int),cudaMemcpyHostToDevice),"weights");
  ck(cudaMemcpy(x,hx,K*sizeof(half),cudaMemcpyHostToDevice),"x"); free(hw); free(hx);
  for(int a=0;a<4;a++) launch(a,outs[a],groups,x); ck(cudaDeviceSynchronize(),"warmup");
  float *ref=(float*)malloc(TOTAL_ROWS*sizeof(float)),*got=(float*)malloc(TOTAL_ROWS*sizeof(float));
  ck(cudaMemcpy(ref,outs[0],TOTAL_ROWS*sizeof(float),cudaMemcpyDeviceToHost),"ref");
  int finite=1; for(int i=0;i<TOTAL_ROWS;i++) finite &= isfinite(ref[i]);
  printf("finite_ref=%d\n",finite);
  for(int a=1;a<4;a++) { ck(cudaMemcpy(got,outs[a],TOTAL_ROWS*sizeof(float),cudaMemcpyDeviceToHost),"got");
    int afinite=1; for(int i=0;i<TOTAL_ROWS;i++) afinite &= isfinite(got[i]);
    printf("bitwise_arm%d=%d finite_arm%d=%d\n",a,memcmp(ref,got,TOTAL_ROWS*sizeof(float))==0,a,afinite); } free(ref); free(got);
  for(int r=0;r<reps;r++) for(int a=0;a<4;a++) {
    double h=hot(a,outs[a],groups,x,hot_passes),c=rotated(a,outs[a],groups,x,cold_passes);
    printf("rep=%d arm=%d hot=%.6f cold=%.6f\n",r,a,h,c);
  }
  return 0;
}
