
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
#include <cuda_fp16.h>
struct __align__(64) float16 { float x, y, z, w, a, b, c, d, e, f, g, h, i, j, k, l; }; __device__ float16 make_float16(float x, float y, float z, float w, float a, float b, float c, float d, float e, float f, float g, float h, float i, float j, float k, float l) { float16 r={x, y, z, w, a, b, c, d, e, f, g, h, i, j, k, l}; return r; }
struct __align__(32) half16 { half x, y, z, w, a, b, c, d, e, f, g, h, i, j, k, l; }; __device__ half16 make_half16(half x, half y, half z, half w, half a, half b, half c, half d, half e, half f, half g, half h, half i, half j, half k, half l) { half16 r={x, y, z, w, a, b, c, d, e, f, g, h, i, j, k, l}; return r; }
struct __align__(32) float8 { float x, y, z, w, a, b, c, d; }; __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { float8 r={x, y, z, w, a, b, c, d}; return r; }
struct __align__(8) half4 { half x, y, z, w; }; __device__ half4 make_half4(half x, half y, half z, half w) { half4 r={x, y, z, w}; return r; }
extern "C" __global__ void __launch_bounds__(128) flash_vec_llama_score_pv_32_128_6_widekv16_vtail1_lastcta(float* data0_24960, float* data1_4096, unsigned int* data2_1048576, half* final_out, unsigned int* counters, unsigned int* ready, unsigned int* heads_done) {
  float buf0[16];
  for (int Lidx41 = 0; Lidx41 < 16; Lidx41++) {
    (*(buf0+Lidx41)) = 0.0f;
  }
  int gidx1 = blockIdx.y; /* 32 */
  int lidx0 = threadIdx.x; /* 32 */
  half buf1[16];
  int alu2 = (lidx0&7);
  int alu3 = (alu2<<3);
  for (int Lidx40_0 = 0; Lidx40_0 < 2; Lidx40_0++) {
    for (int Lidx40_1 = 0; Lidx40_1 < 8; Lidx40_1++) {
      float val0 = (*(data1_4096+(alu3+(Lidx40_0<<6)+Lidx40_1+(gidx1<<7))));
      (*(buf1+((Lidx40_0<<3)+Lidx40_1))) = ((half)(val0));
    }
  }
  int gidx0 = blockIdx.x; /* 6 */
  int lidx1 = threadIdx.y; /* 4 */
  float buf2[1];
  float buf3[8];
  float buf4[1];
  int alu7 = (lidx0>>3);
  int alu8 = ((gidx0<<7)+(lidx1<<5)+(alu7<<3));
  int alu9 = ((gidx0<<13)+(lidx1<<11)+(alu7<<9));
  int alu10 = ((gidx1>>2)<<16);
  int alu11 = (alu2<<2);
  for (int Ridx5 = 0; Ridx5 < 8; Ridx5++) {
    int alu12 = (alu9+(Ridx5<<6)+alu10+alu11);
    uint4 val1 = (*((uint4*)((data2_1048576+(alu12+32)))));
    uint4 val2 = (*((uint4*)((data2_1048576+alu12))));
    (*(buf4+0)) = 0.0f;
    (*(buf4+0)) = (((((((((*(buf4+0))) + float(make_half2((*(buf1+0)),(*(buf1+1))).x) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val2.x>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val2.x>>16u)))))).x) + float(make_half2((*(buf1+0)),(*(buf1+1))).y) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val2.x>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val2.x>>16u)))))).y)) + float(make_half2((*(buf1+2)),(*(buf1+3))).x) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val2.y>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val2.y>>16u)))))).x) + float(make_half2((*(buf1+2)),(*(buf1+3))).y) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val2.y>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val2.y>>16u)))))).y)) + float(make_half2((*(buf1+4)),(*(buf1+5))).x) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val2.z>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val2.z>>16u)))))).x) + float(make_half2((*(buf1+4)),(*(buf1+5))).y) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val2.z>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val2.z>>16u)))))).y)) + float(make_half2((*(buf1+6)),(*(buf1+7))).x) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val2.w>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val2.w>>16u)))))).x) + float(make_half2((*(buf1+6)),(*(buf1+7))).y) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val2.w>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val2.w>>16u)))))).y)) + float(make_half2((*(buf1+8)),(*(buf1+9))).x) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val1.x>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val1.x>>16u)))))).x) + float(make_half2((*(buf1+8)),(*(buf1+9))).y) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val1.x>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val1.x>>16u)))))).y)) + float(make_half2((*(buf1+10)),(*(buf1+11))).x) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val1.y>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val1.y>>16u)))))).x) + float(make_half2((*(buf1+10)),(*(buf1+11))).y) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val1.y>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val1.y>>16u)))))).y)) + float(make_half2((*(buf1+12)),(*(buf1+13))).x) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val1.z>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val1.z>>16u)))))).x) + float(make_half2((*(buf1+12)),(*(buf1+13))).y) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val1.z>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val1.z>>16u)))))).y)) + float(make_half2((*(buf1+14)),(*(buf1+15))).x) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val1.w>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val1.w>>16u)))))).x) + float(make_half2((*(buf1+14)),(*(buf1+15))).y) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val1.w>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val1.w>>16u)))))).y);
    (*(buf2+0)) = __shfl_xor_sync(0xffffffffu, (*(buf4+0)), 4);
    float alu16 = ((*(buf4+0))+(*(buf2+0)));
    (*(buf2+0)) = __shfl_xor_sync(0xffffffffu, alu16, 2);
    float alu18 = (alu16+(*(buf2+0)));
    (*(buf2+0)) = __shfl_xor_sync(0xffffffffu, alu18, 1);
    float alu20 = (((alu8+Ridx5)<513)?((alu18+(*(buf2+0)))*0.08838834764831843f):((float)(-INFINITY)));
    (*(buf3+Ridx5)) = alu20;
  }
  float buf5[1];
  (*(buf5+0)) = ((float)(-INFINITY));
  for (int Ridx7 = 0; Ridx7 < 8; Ridx7++) {
    float alu24 = (((*(buf5+0))<(*(buf3+Ridx7)))?(*(buf3+Ridx7)):(*(buf5+0)));
    (*(buf5+0)) = alu24;
  }
  int alu27 = (alu9+alu10+alu11);
  uint4 val3 = (*((uint4*)((data2_1048576+(alu27+524736)))));
  uint4 val4 = (*((uint4*)((data2_1048576+(alu27+524768)))));
  float buf6[1];
  float buf7[1];
  half buf8[16];
  (*(buf2+0)) = __shfl_xor_sync(0xffffffffu, (*(buf5+0)), 8);
  float alu29 = (((*(buf5+0))<(*(buf2+0)))?(*(buf2+0)):(*(buf5+0)));
  (*(buf2+0)) = __shfl_xor_sync(0xffffffffu, alu29, 16);
  (*(buf6+0)) = 0.0f;
  (*(buf7+0)) = ((float)(-INFINITY));
  (*(buf8+0)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val3.x>>0u)))));
  (*(buf8+1)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val3.x>>16u)))));
  (*(buf8+2)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val3.y>>0u)))));
  (*(buf8+3)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val3.y>>16u)))));
  (*(buf8+4)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val3.z>>0u)))));
  (*(buf8+5)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val3.z>>16u)))));
  (*(buf8+6)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val3.w>>0u)))));
  (*(buf8+7)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val3.w>>16u)))));
  (*(buf8+8)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val4.x>>0u)))));
  (*(buf8+9)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val4.x>>16u)))));
  (*(buf8+10)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val4.y>>0u)))));
  (*(buf8+11)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val4.y>>16u)))));
  (*(buf8+12)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val4.z>>0u)))));
  (*(buf8+13)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val4.z>>16u)))));
  (*(buf8+14)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val4.w>>0u)))));
  (*(buf8+15)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val4.w>>16u)))));
  float alu49 = ((alu29<(*(buf2+0)))?(*(buf2+0)):alu29);
  bool alu50 = (-1e+30f<alu49);
  float alu51 = (alu50?exp2((((*(buf7+0))-alu49)*1.4426950408889634f)):1.0f);
  for (int Lidx8 = 0; Lidx8 < 16; Lidx8++) {
    (*(buf0+Lidx8)) = ((*(buf0+Lidx8))*alu51);
  }
  uint4 val5 = (*((uint4*)((data2_1048576+(alu27+524288)))));
  uint4 val6 = (*((uint4*)((data2_1048576+(alu27+524320)))));
  uint4 val7 = (*((uint4*)((data2_1048576+(alu27+524352)))));
  uint4 val8 = (*((uint4*)((data2_1048576+(alu27+524384)))));
  uint4 val9 = (*((uint4*)((data2_1048576+(alu27+524416)))));
  uint4 val10 = (*((uint4*)((data2_1048576+(alu27+524448)))));
  uint4 val11 = (*((uint4*)((data2_1048576+(alu27+524480)))));
  uint4 val12 = (*((uint4*)((data2_1048576+(alu27+524512)))));
  uint4 val13 = (*((uint4*)((data2_1048576+(alu27+524544)))));
  uint4 val14 = (*((uint4*)((data2_1048576+(alu27+524576)))));
  uint4 val15 = (*((uint4*)((data2_1048576+(alu27+524608)))));
  uint4 val16 = (*((uint4*)((data2_1048576+(alu27+524640)))));
  uint4 val17 = (*((uint4*)((data2_1048576+(alu27+524672)))));
  uint4 val18 = (*((uint4*)((data2_1048576+(alu27+524704)))));
  float buf9[1];
  __shared__ __align__(16) float buf10[512];
  (*(buf6+0)) = ((*(buf6+0))*alu51);
  float alu55 = (alu50?alu49:(*(buf7+0)));
  (*(buf7+0)) = alu55;
  float alu57 = ((alu8<513)?exp2((((*(buf3+0))-alu55)*1.4426950408889634f)):0.0f);
  (*(buf0+0)) = ((*(buf0+0))+(alu57*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val5.x>>0u)))))))));
  (*(buf0+1)) = ((*(buf0+1))+(alu57*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val5.x>>16u)))))))));
  (*(buf0+2)) = ((*(buf0+2))+(alu57*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val5.y>>0u)))))))));
  (*(buf0+3)) = ((*(buf0+3))+(alu57*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val5.y>>16u)))))))));
  (*(buf0+4)) = ((*(buf0+4))+(alu57*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val5.z>>0u)))))))));
  (*(buf0+5)) = ((*(buf0+5))+(alu57*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val5.z>>16u)))))))));
  (*(buf0+6)) = ((*(buf0+6))+(alu57*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val5.w>>0u)))))))));
  (*(buf0+7)) = ((*(buf0+7))+(alu57*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val5.w>>16u)))))))));
  (*(buf0+8)) = ((*(buf0+8))+(alu57*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val6.x>>0u)))))))));
  (*(buf0+9)) = ((*(buf0+9))+(alu57*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val6.x>>16u)))))))));
  (*(buf0+10)) = ((*(buf0+10))+(alu57*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val6.y>>0u)))))))));
  (*(buf0+11)) = ((*(buf0+11))+(alu57*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val6.y>>16u)))))))));
  (*(buf0+12)) = ((*(buf0+12))+(alu57*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val6.z>>0u)))))))));
  (*(buf0+13)) = ((*(buf0+13))+(alu57*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val6.z>>16u)))))))));
  (*(buf0+14)) = ((*(buf0+14))+(alu57*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val6.w>>0u)))))))));
  (*(buf0+15)) = ((*(buf0+15))+(alu57*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val6.w>>16u)))))))));
  (*(buf6+0)) = ((*(buf6+0))+alu57);
  float alu75 = ((alu8<512)?exp2((((*(buf3+1))-alu55)*1.4426950408889634f)):0.0f);
  (*(buf0+0)) = ((*(buf0+0))+(alu75*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val7.x>>0u)))))))));
  (*(buf0+1)) = ((*(buf0+1))+(alu75*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val7.x>>16u)))))))));
  (*(buf0+2)) = ((*(buf0+2))+(alu75*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val7.y>>0u)))))))));
  (*(buf0+3)) = ((*(buf0+3))+(alu75*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val7.y>>16u)))))))));
  (*(buf0+4)) = ((*(buf0+4))+(alu75*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val7.z>>0u)))))))));
  (*(buf0+5)) = ((*(buf0+5))+(alu75*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val7.z>>16u)))))))));
  (*(buf0+6)) = ((*(buf0+6))+(alu75*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val7.w>>0u)))))))));
  (*(buf0+7)) = ((*(buf0+7))+(alu75*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val7.w>>16u)))))))));
  (*(buf0+8)) = ((*(buf0+8))+(alu75*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val8.x>>0u)))))))));
  (*(buf0+9)) = ((*(buf0+9))+(alu75*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val8.x>>16u)))))))));
  (*(buf0+10)) = ((*(buf0+10))+(alu75*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val8.y>>0u)))))))));
  (*(buf0+11)) = ((*(buf0+11))+(alu75*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val8.y>>16u)))))))));
  (*(buf0+12)) = ((*(buf0+12))+(alu75*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val8.z>>0u)))))))));
  (*(buf0+13)) = ((*(buf0+13))+(alu75*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val8.z>>16u)))))))));
  (*(buf0+14)) = ((*(buf0+14))+(alu75*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val8.w>>0u)))))))));
  (*(buf0+15)) = ((*(buf0+15))+(alu75*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val8.w>>16u)))))))));
  (*(buf6+0)) = ((*(buf6+0))+alu75);
  float alu93 = ((alu8<511)?exp2((((*(buf3+2))-alu55)*1.4426950408889634f)):0.0f);
  (*(buf0+0)) = ((*(buf0+0))+(alu93*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val9.x>>0u)))))))));
  (*(buf0+1)) = ((*(buf0+1))+(alu93*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val9.x>>16u)))))))));
  (*(buf0+2)) = ((*(buf0+2))+(alu93*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val9.y>>0u)))))))));
  (*(buf0+3)) = ((*(buf0+3))+(alu93*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val9.y>>16u)))))))));
  (*(buf0+4)) = ((*(buf0+4))+(alu93*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val9.z>>0u)))))))));
  (*(buf0+5)) = ((*(buf0+5))+(alu93*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val9.z>>16u)))))))));
  (*(buf0+6)) = ((*(buf0+6))+(alu93*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val9.w>>0u)))))))));
  (*(buf0+7)) = ((*(buf0+7))+(alu93*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val9.w>>16u)))))))));
  (*(buf0+8)) = ((*(buf0+8))+(alu93*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val10.x>>0u)))))))));
  (*(buf0+9)) = ((*(buf0+9))+(alu93*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val10.x>>16u)))))))));
  (*(buf0+10)) = ((*(buf0+10))+(alu93*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val10.y>>0u)))))))));
  (*(buf0+11)) = ((*(buf0+11))+(alu93*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val10.y>>16u)))))))));
  (*(buf0+12)) = ((*(buf0+12))+(alu93*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val10.z>>0u)))))))));
  (*(buf0+13)) = ((*(buf0+13))+(alu93*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val10.z>>16u)))))))));
  (*(buf0+14)) = ((*(buf0+14))+(alu93*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val10.w>>0u)))))))));
  (*(buf0+15)) = ((*(buf0+15))+(alu93*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val10.w>>16u)))))))));
  (*(buf6+0)) = ((*(buf6+0))+alu93);
  float alu111 = ((alu8<510)?exp2((((*(buf3+3))-alu55)*1.4426950408889634f)):0.0f);
  (*(buf0+0)) = ((*(buf0+0))+(alu111*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val11.x>>0u)))))))));
  (*(buf0+1)) = ((*(buf0+1))+(alu111*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val11.x>>16u)))))))));
  (*(buf0+2)) = ((*(buf0+2))+(alu111*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val11.y>>0u)))))))));
  (*(buf0+3)) = ((*(buf0+3))+(alu111*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val11.y>>16u)))))))));
  (*(buf0+4)) = ((*(buf0+4))+(alu111*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val11.z>>0u)))))))));
  (*(buf0+5)) = ((*(buf0+5))+(alu111*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val11.z>>16u)))))))));
  (*(buf0+6)) = ((*(buf0+6))+(alu111*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val11.w>>0u)))))))));
  (*(buf0+7)) = ((*(buf0+7))+(alu111*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val11.w>>16u)))))))));
  (*(buf0+8)) = ((*(buf0+8))+(alu111*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val12.x>>0u)))))))));
  (*(buf0+9)) = ((*(buf0+9))+(alu111*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val12.x>>16u)))))))));
  (*(buf0+10)) = ((*(buf0+10))+(alu111*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val12.y>>0u)))))))));
  (*(buf0+11)) = ((*(buf0+11))+(alu111*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val12.y>>16u)))))))));
  (*(buf0+12)) = ((*(buf0+12))+(alu111*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val12.z>>0u)))))))));
  (*(buf0+13)) = ((*(buf0+13))+(alu111*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val12.z>>16u)))))))));
  (*(buf0+14)) = ((*(buf0+14))+(alu111*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val12.w>>0u)))))))));
  (*(buf0+15)) = ((*(buf0+15))+(alu111*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val12.w>>16u)))))))));
  (*(buf6+0)) = ((*(buf6+0))+alu111);
  float alu129 = ((alu8<509)?exp2((((*(buf3+4))-alu55)*1.4426950408889634f)):0.0f);
  (*(buf0+0)) = ((*(buf0+0))+(alu129*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val13.x>>0u)))))))));
  (*(buf0+1)) = ((*(buf0+1))+(alu129*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val13.x>>16u)))))))));
  (*(buf0+2)) = ((*(buf0+2))+(alu129*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val13.y>>0u)))))))));
  (*(buf0+3)) = ((*(buf0+3))+(alu129*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val13.y>>16u)))))))));
  (*(buf0+4)) = ((*(buf0+4))+(alu129*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val13.z>>0u)))))))));
  (*(buf0+5)) = ((*(buf0+5))+(alu129*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val13.z>>16u)))))))));
  (*(buf0+6)) = ((*(buf0+6))+(alu129*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val13.w>>0u)))))))));
  (*(buf0+7)) = ((*(buf0+7))+(alu129*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val13.w>>16u)))))))));
  (*(buf0+8)) = ((*(buf0+8))+(alu129*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val14.x>>0u)))))))));
  (*(buf0+9)) = ((*(buf0+9))+(alu129*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val14.x>>16u)))))))));
  (*(buf0+10)) = ((*(buf0+10))+(alu129*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val14.y>>0u)))))))));
  (*(buf0+11)) = ((*(buf0+11))+(alu129*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val14.y>>16u)))))))));
  (*(buf0+12)) = ((*(buf0+12))+(alu129*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val14.z>>0u)))))))));
  (*(buf0+13)) = ((*(buf0+13))+(alu129*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val14.z>>16u)))))))));
  (*(buf0+14)) = ((*(buf0+14))+(alu129*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val14.w>>0u)))))))));
  (*(buf0+15)) = ((*(buf0+15))+(alu129*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val14.w>>16u)))))))));
  (*(buf6+0)) = ((*(buf6+0))+alu129);
  float alu147 = ((alu8<508)?exp2((((*(buf3+5))-alu55)*1.4426950408889634f)):0.0f);
  (*(buf0+0)) = ((*(buf0+0))+(alu147*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val15.x>>0u)))))))));
  (*(buf0+1)) = ((*(buf0+1))+(alu147*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val15.x>>16u)))))))));
  (*(buf0+2)) = ((*(buf0+2))+(alu147*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val15.y>>0u)))))))));
  (*(buf0+3)) = ((*(buf0+3))+(alu147*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val15.y>>16u)))))))));
  (*(buf0+4)) = ((*(buf0+4))+(alu147*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val15.z>>0u)))))))));
  (*(buf0+5)) = ((*(buf0+5))+(alu147*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val15.z>>16u)))))))));
  (*(buf0+6)) = ((*(buf0+6))+(alu147*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val15.w>>0u)))))))));
  (*(buf0+7)) = ((*(buf0+7))+(alu147*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val15.w>>16u)))))))));
  (*(buf0+8)) = ((*(buf0+8))+(alu147*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val16.x>>0u)))))))));
  (*(buf0+9)) = ((*(buf0+9))+(alu147*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val16.x>>16u)))))))));
  (*(buf0+10)) = ((*(buf0+10))+(alu147*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val16.y>>0u)))))))));
  (*(buf0+11)) = ((*(buf0+11))+(alu147*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val16.y>>16u)))))))));
  (*(buf0+12)) = ((*(buf0+12))+(alu147*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val16.z>>0u)))))))));
  (*(buf0+13)) = ((*(buf0+13))+(alu147*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val16.z>>16u)))))))));
  (*(buf0+14)) = ((*(buf0+14))+(alu147*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val16.w>>0u)))))))));
  (*(buf0+15)) = ((*(buf0+15))+(alu147*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val16.w>>16u)))))))));
  (*(buf6+0)) = ((*(buf6+0))+alu147);
  float alu165 = ((alu8<507)?exp2((((*(buf3+6))-alu55)*1.4426950408889634f)):0.0f);
  (*(buf0+0)) = ((*(buf0+0))+(alu165*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val17.x>>0u)))))))));
  (*(buf0+1)) = ((*(buf0+1))+(alu165*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val17.x>>16u)))))))));
  (*(buf0+2)) = ((*(buf0+2))+(alu165*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val17.y>>0u)))))))));
  (*(buf0+3)) = ((*(buf0+3))+(alu165*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val17.y>>16u)))))))));
  (*(buf0+4)) = ((*(buf0+4))+(alu165*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val17.z>>0u)))))))));
  (*(buf0+5)) = ((*(buf0+5))+(alu165*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val17.z>>16u)))))))));
  (*(buf0+6)) = ((*(buf0+6))+(alu165*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val17.w>>0u)))))))));
  (*(buf0+7)) = ((*(buf0+7))+(alu165*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val17.w>>16u)))))))));
  (*(buf0+8)) = ((*(buf0+8))+(alu165*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val18.x>>0u)))))))));
  (*(buf0+9)) = ((*(buf0+9))+(alu165*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val18.x>>16u)))))))));
  (*(buf0+10)) = ((*(buf0+10))+(alu165*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val18.y>>0u)))))))));
  (*(buf0+11)) = ((*(buf0+11))+(alu165*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val18.y>>16u)))))))));
  (*(buf0+12)) = ((*(buf0+12))+(alu165*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val18.z>>0u)))))))));
  (*(buf0+13)) = ((*(buf0+13))+(alu165*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val18.z>>16u)))))))));
  (*(buf0+14)) = ((*(buf0+14))+(alu165*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val18.w>>0u)))))))));
  (*(buf0+15)) = ((*(buf0+15))+(alu165*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val18.w>>16u)))))))));
  (*(buf6+0)) = ((*(buf6+0))+alu165);
  float alu183 = ((alu8<506)?exp2((((*(buf3+7))-alu55)*1.4426950408889634f)):0.0f);
  (*(buf0+0)) = ((*(buf0+0))+(alu183*((float)((*(buf8+0))))));
  (*(buf0+1)) = ((*(buf0+1))+(alu183*((float)((*(buf8+1))))));
  (*(buf0+2)) = ((*(buf0+2))+(alu183*((float)((*(buf8+2))))));
  (*(buf0+3)) = ((*(buf0+3))+(alu183*((float)((*(buf8+3))))));
  (*(buf0+4)) = ((*(buf0+4))+(alu183*((float)((*(buf8+4))))));
  (*(buf0+5)) = ((*(buf0+5))+(alu183*((float)((*(buf8+5))))));
  (*(buf0+6)) = ((*(buf0+6))+(alu183*((float)((*(buf8+6))))));
  (*(buf0+7)) = ((*(buf0+7))+(alu183*((float)((*(buf8+7))))));
  (*(buf0+8)) = ((*(buf0+8))+(alu183*((float)((*(buf8+8))))));
  (*(buf0+9)) = ((*(buf0+9))+(alu183*((float)((*(buf8+9))))));
  (*(buf0+10)) = ((*(buf0+10))+(alu183*((float)((*(buf8+10))))));
  (*(buf0+11)) = ((*(buf0+11))+(alu183*((float)((*(buf8+11))))));
  (*(buf0+12)) = ((*(buf0+12))+(alu183*((float)((*(buf8+12))))));
  (*(buf0+13)) = ((*(buf0+13))+(alu183*((float)((*(buf8+13))))));
  (*(buf0+14)) = ((*(buf0+14))+(alu183*((float)((*(buf8+14))))));
  (*(buf0+15)) = ((*(buf0+15))+(alu183*((float)((*(buf8+15))))));
  (*(buf6+0)) = ((*(buf6+0))+alu183);
  for (int Lidx13_0 = 0; Lidx13_0 < 2; Lidx13_0++) {
    for (int Lidx13_1 = 0; Lidx13_1 < 8; Lidx13_1++) {
      (*(buf9+0)) = __shfl_xor_sync(0xffffffffu, (*(buf0+((Lidx13_0<<3)+Lidx13_1))), 8);
      float alu202 = ((*(buf0+((Lidx13_0<<3)+Lidx13_1)))+(*(buf9+0)));
      (*(buf9+0)) = __shfl_xor_sync(0xffffffffu, alu202, 16);
      if ((lidx0<8)) {
        *(buf10+(alu3+(Lidx13_0<<6)+Lidx13_1+(lidx1<<7))) = (alu202+(*(buf9+0)));
      }
    }
  }
  float buf11[1];
  float buf12[1];
  __shared__ __align__(16) float buf13[4];
  __shared__ __align__(16) float buf14[4];
  float buf15[1];
  (*(buf11+0)) = __shfl_xor_sync(0xffffffffu, (*(buf6+0)), 8);
  float alu210 = ((*(buf6+0))+(*(buf11+0)));
  (*(buf11+0)) = __shfl_xor_sync(0xffffffffu, alu210, 16);
  (*(buf12+0)) = __shfl_xor_sync(0xffffffffu, (*(buf7+0)), 8);
  float alu213 = (((*(buf7+0))<(*(buf12+0)))?(*(buf12+0)):(*(buf7+0)));
  (*(buf12+0)) = __shfl_xor_sync(0xffffffffu, alu213, 16);
  bool alu215 = (lidx0==0);
  if (alu215) {
    *(buf13+lidx1) = (alu210+(*(buf11+0)));
  }
  float alu219 = ((alu213<(*(buf12+0)))?(*(buf12+0)):alu213);
  if (alu215) {
    *(buf14+lidx1) = alu219;
  }
  __syncthreads();
  (*(buf15+0)) = ((float)(-INFINITY));
  for (int Ridx14 = 0; Ridx14 < 4; Ridx14++) {
    float val19 = (*(buf14+Ridx14));
    float alu225 = (((*(buf15+0))<val19)?val19:(*(buf15+0)));
    (*(buf15+0)) = alu225;
  }
  float buf16[16];
  for (int Lidx15 = 0; Lidx15 < 16; Lidx15++) {
    (*(buf16+Lidx15)) = 0.0f;
  }
  float buf17[1];
  (*(buf17+0)) = 0.0f;
  for (int Ridx16 = 0; Ridx16 < 4; Ridx16++) {
    float val20 = (*(buf14+Ridx16));
    float alu231 = ((-1e+30f<(*(buf15+0)))?exp2(((val20-(*(buf15+0)))*1.4426950408889634f)):0.0f);
    for (int Lidx17_0 = 0; Lidx17_0 < 2; Lidx17_0++) {
      for (int Lidx17_1 = 0; Lidx17_1 < 8; Lidx17_1++) {
        float val21 = (*(buf10+(alu3+(Lidx17_0<<6)+Lidx17_1+(Ridx16<<7))));
        int alu232 = ((Lidx17_0<<3)+Lidx17_1);
        (*(buf16+alu232)) = ((*(buf16+alu232))+(alu231*val21));
      }
    }
    float val22 = (*(buf13+Ridx16));
    (*(buf17+0)) = ((*(buf17+0))+(alu231*val22));
  }
  int alu238 = ((gidx0*130)+(gidx1*780));
  for (int Lidx18_0 = 0; Lidx18_0 < 2; Lidx18_0++) {
    for (int Lidx18_1 = 0; Lidx18_1 < 8; Lidx18_1++) {
      if ((lidx1==0)) {
        *(data0_24960+(alu3+(Lidx18_0<<6)+Lidx18_1+alu238)) = (*(buf16+((Lidx18_0<<3)+Lidx18_1)));
      }
    }
  }
  if (alu215) {
    *(data0_24960+(alu238+128)) = (*(buf17+0));
  }
  if (alu215) {
    *(data0_24960+(alu238+129)) = (*(buf15+0));
  }

  __shared__ int tg_last;
  __syncthreads();
  if (lidx0 == 0 && lidx1 == 0) {
    __threadfence();
    tg_last = (atomicAdd(counters + gidx1, 1u) == 5u);
  }
  __syncthreads();
  if (tg_last) {
    int tg_lane = lidx1 * 32 + lidx0;
    float tg_max = -1e30f;
    #pragma unroll
    for (int tg_s=0; tg_s<6; tg_s++) tg_max = fmaxf(tg_max, data0_24960[(gidx1*6+tg_s)*130+129]);
    float tg_acc=0.0f, tg_den=0.0f;
    #pragma unroll
    for (int tg_s=0; tg_s<6; tg_s++) {
      float tg_w=exp2f((data0_24960[(gidx1*6+tg_s)*130+129]-tg_max)*1.4426950408889634f);
      tg_acc += tg_w * data0_24960[(gidx1*6+tg_s)*130+tg_lane];
      tg_den += tg_w * data0_24960[(gidx1*6+tg_s)*130+128];
    }
    final_out[gidx1*128+tg_lane] = (half)(tg_acc/tg_den);
    __syncthreads();
    if (tg_lane == 0) {
      counters[gidx1]=0u;
      __threadfence();
      if (atomicAdd(heads_done, 1u) == 31u) {
        heads_done[0]=0u;
        __threadfence();
        ready[0]=1u;
      }
    }
  }
}
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
}extern "C" __global__ void __launch_bounds__(128) q4k_o_bounded_persistent_w128(float* data0_4096, unsigned int* data1_2359296, half* data2_4096, float* data3_4096, unsigned int* data4_1) {
  int gidx0 = blockIdx.x; /* 128 */
  int lidx0 = threadIdx.x; /* 128 */
  float buf0[1];
  float buf1[1];
  (*(buf0+0)) = 0.0f;
  int alu1 = (lidx0&7);
  long cast0 = ((long)(([&]() { if (threadIdx.x == 0) while (*((volatile unsigned int*)(data4_1+0)) == 0u) __nanosleep(64); __syncthreads(); return 0u; })()));
  int alu2 = (lidx0>>3);
  for (int gidx18 = 0; gidx18 < 8; gidx18++) {
    for (int Ridx0 = 0; Ridx0 < 4; Ridx0++) {
      long alu3 = (((long)(((gidx0*2304)+(alu2*144)+(gidx18*294912))))+(cast0*576ll)+((long)((Ridx0*36))));
      unsigned int val0 = (*(data1_2359296+alu3));
      unsigned int val1 = (*(data1_2359296+(alu3+1ll)));
      unsigned int val2 = (*(data1_2359296+(alu3+2ll)));
      unsigned int val3 = (*(data1_2359296+(alu3+3ll)));
      long alu4 = (alu3+((long)(alu1)));
      unsigned int val4 = (*(data1_2359296+(alu4+4ll)));
      unsigned int val5 = (*(data1_2359296+(alu4+12ll)));
      unsigned int val6 = (*(data1_2359296+(alu4+20ll)));
      unsigned int val7 = (*(data1_2359296+(alu4+28ll)));
      int alu5 = (((alu2&3)<<10)+(Ridx0<<8)+(alu1<<2));
      half4 val8 = (*((half4*)((data2_4096+(alu5+32)))));
      half4 val9 = (*((half4*)((data2_4096+(alu5+64)))));
      half4 val10 = (*((half4*)((data2_4096+(alu5+96)))));
      half4 val11 = (*((half4*)((data2_4096+(alu5+128)))));
      half4 val12 = (*((half4*)((data2_4096+(alu5+160)))));
      half4 val13 = (*((half4*)((data2_4096+(alu5+192)))));
      half4 val14 = (*((half4*)((data2_4096+(alu5+224)))));
      half4 val15 = (*((half4*)((data2_4096+alu5))));
      float cast1 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val0&65535u)))))));
      float cast2 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)(((val0>>16u)&65535u)))))));
      unsigned int alu6 = (val2>>0u);
      unsigned int alu7 = (val3>>0u);
      unsigned int alu8 = (val2>>8u);
      unsigned int alu9 = (val3>>8u);
      unsigned int alu10 = (val2>>16u);
      unsigned int alu11 = (val3>>16u);
      unsigned int alu12 = (val2>>24u);
      unsigned int alu13 = (val3>>24u);
      unsigned int alu14 = (val1>>0u);
      unsigned int alu15 = (val1>>8u);
      unsigned int alu16 = (val1>>16u);
      unsigned int alu17 = (val1>>24u);
      unsigned int alu18 = ((val4>>0u)&252645135u);
      unsigned int alu19 = ((val4>>4u)&252645135u);
      unsigned int alu20 = ((val5>>0u)&252645135u);
      unsigned int alu21 = ((val5>>4u)&252645135u);
      unsigned int alu22 = ((val6>>0u)&252645135u);
      unsigned int alu23 = ((val6>>4u)&252645135u);
      unsigned int alu24 = ((val7>>0u)&252645135u);
      unsigned int alu25 = ((val7>>4u)&252645135u);
      float alu26 = (cast1*((float)((alu14&63u))));
      float alu27 = (cast2*((float)((alu6&63u))));
      float alu28 = (cast1*((float)((alu15&63u))));
      float alu29 = (cast2*((float)((alu8&63u))));
      float alu30 = (cast1*((float)((alu16&63u))));
      float alu31 = (cast2*((float)((alu10&63u))));
      float alu32 = (cast1*((float)((alu17&63u))));
      float alu33 = (cast2*((float)((alu12&63u))));
      float alu34 = (cast1*((float)(((alu7&15u)|(((alu14&255u)>>6u)<<4u)))));
      float alu35 = (cast2*((float)((((alu7&255u)>>4u)|(((alu6&255u)>>6u)<<4u)))));
      float alu36 = (cast1*((float)(((alu9&15u)|(((alu15&255u)>>6u)<<4u)))));
      float alu37 = (cast2*((float)((((alu9&255u)>>4u)|(((alu8&255u)>>6u)<<4u)))));
      float alu38 = (cast1*((float)(((alu11&15u)|(((alu16&255u)>>6u)<<4u)))));
      float alu39 = (cast2*((float)((((alu11&255u)>>4u)|(((alu10&255u)>>6u)<<4u)))));
      float alu40 = (cast1*((float)(((alu13&15u)|(((alu17&255u)>>6u)<<4u)))));
      float alu41 = (cast2*((float)((((alu13&255u)>>4u)|(((alu12&255u)>>6u)<<4u)))));
      (*(buf0+0)) = ((*(buf0+0))+(((alu26*((float)(((alu18>>0u)&15u))))-alu27)*float(val15.x))+(((alu26*((float)(((alu18>>8u)&15u))))-alu27)*float(val15.y))+(((alu26*((float)(((alu18>>16u)&15u))))-alu27)*float(val15.z))+(((alu26*((float)(((alu18>>24u)&15u))))-alu27)*float(val15.w))+(((alu28*((float)(((alu19>>0u)&15u))))-alu29)*float(val8.x))+(((alu28*((float)(((alu19>>8u)&15u))))-alu29)*float(val8.y))+(((alu28*((float)(((alu19>>16u)&15u))))-alu29)*float(val8.z))+(((alu28*((float)(((alu19>>24u)&15u))))-alu29)*float(val8.w))+(((alu30*((float)(((alu20>>0u)&15u))))-alu31)*float(val9.x))+(((alu30*((float)(((alu20>>8u)&15u))))-alu31)*float(val9.y))+(((alu30*((float)(((alu20>>16u)&15u))))-alu31)*float(val9.z))+(((alu30*((float)(((alu20>>24u)&15u))))-alu31)*float(val9.w))+(((alu32*((float)(((alu21>>0u)&15u))))-alu33)*float(val10.x))+(((alu32*((float)(((alu21>>8u)&15u))))-alu33)*float(val10.y))+(((alu32*((float)(((alu21>>16u)&15u))))-alu33)*float(val10.z))+(((alu32*((float)(((alu21>>24u)&15u))))-alu33)*float(val10.w))+(((alu34*((float)(((alu22>>0u)&15u))))-alu35)*float(val11.x))+(((alu34*((float)(((alu22>>8u)&15u))))-alu35)*float(val11.y))+(((alu34*((float)(((alu22>>16u)&15u))))-alu35)*float(val11.z))+(((alu34*((float)(((alu22>>24u)&15u))))-alu35)*float(val11.w))+(((alu36*((float)(((alu23>>0u)&15u))))-alu37)*float(val12.x))+(((alu36*((float)(((alu23>>8u)&15u))))-alu37)*float(val12.y))+(((alu36*((float)(((alu23>>16u)&15u))))-alu37)*float(val12.z))+(((alu36*((float)(((alu23>>24u)&15u))))-alu37)*float(val12.w))+(((alu38*((float)(((alu24>>0u)&15u))))-alu39)*float(val13.x))+(((alu38*((float)(((alu24>>8u)&15u))))-alu39)*float(val13.y))+(((alu38*((float)(((alu24>>16u)&15u))))-alu39)*float(val13.z))+(((alu38*((float)(((alu24>>24u)&15u))))-alu39)*float(val13.w))+(((alu40*((float)(((alu25>>0u)&15u))))-alu41)*float(val14.x))+(((alu40*((float)(((alu25>>8u)&15u))))-alu41)*float(val14.y))+(((alu40*((float)(((alu25>>16u)&15u))))-alu41)*float(val14.z))+(((alu40*((float)(((alu25>>24u)&15u))))-alu41)*float(val14.w)));
    }
    long alu44 = (cast0+((long)(((gidx0<<2)+(gidx18<<9)+(lidx0>>5)))));
    float val16 = (*(data3_4096+alu44));
    (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, (*(buf0+0)), 16);
    float alu46 = ((*(buf0+0))+(*(buf1+0)));
    (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu46, 8);
    float alu48 = (alu46+(*(buf1+0)));
    (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu48, 4);
    float alu50 = (alu48+(*(buf1+0)));
    (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu50, 2);
    float alu52 = (alu50+(*(buf1+0)));
    (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu52, 1);
    *(data0_4096+alu44) = (alu52+(*(buf1+0))+val16);
  }
}
#define ROWS 4096
#define O_WORDS 2359296
#define CACHE_WORDS 1048576
#define ROTATIONS 16
#define WORKERS 128
static void ck(cudaError_t e,const char*w){if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e));exit(2);}}
struct Ctx{cudaStream_t producer,consumer;cudaEvent_t start,done;};
static void init(Ctx&c){int least,greatest;ck(cudaDeviceGetStreamPriorityRange(&least,&greatest),"priority");
  ck(cudaStreamCreateWithPriority(&c.producer,cudaStreamNonBlocking,greatest),"producer-stream");
  ck(cudaStreamCreateWithPriority(&c.consumer,cudaStreamNonBlocking,least),"consumer-stream");
  ck(cudaEventCreate(&c.start),"start-event");ck(cudaEventCreate(&c.done),"done-event");}
static float elapsed(cudaEvent_t a,cudaEvent_t b){float ms;ck(cudaEventElapsedTime(&ms,a,b),"elapsed");return ms*1000.0f;}
static void prep(unsigned*ready,unsigned*heads,unsigned*ctr,cudaStream_t s){ck(cudaMemsetAsync(ready,0,4,s),"ready0");
  ck(cudaMemsetAsync(heads,0,4,s),"heads0");ck(cudaMemsetAsync(ctr,0,128,s),"ctr0");}
static float control(Ctx&c,float*out,half*att,float*partial,float*q,unsigned*cache,unsigned*w,float*res,unsigned*ctr,unsigned*ready,unsigned*heads){
  prep(ready,heads,ctr,c.producer);ck(cudaEventRecord(c.start,c.producer),"c-start");
  flash_vec_llama_score_pv_32_128_6_widekv16_vtail1_lastcta<<<dim3(6,32),dim3(32,4),0,c.producer>>>(partial,q,cache,att,ctr,ready,heads);
  q4k_g3_lanemap_gemv_vec_epi_resadd_4096_4096<<<ROWS,32,0,c.producer>>>(out,w,att,res);ck(cudaEventRecord(c.done,c.producer),"c-done");
  ck(cudaEventSynchronize(c.done),"c-sync");return elapsed(c.start,c.done);}
static float candidate(Ctx&c,float*out,half*att,float*partial,float*q,unsigned*cache,unsigned*w,float*res,unsigned*ctr,unsigned*ready,unsigned*heads){
  prep(ready,heads,ctr,c.producer);ck(cudaEventRecord(c.start,c.producer),"p-start");
  ck(cudaStreamWaitEvent(c.consumer,c.start),"consumer-start");
  q4k_o_bounded_persistent_w128<<<WORKERS,128,0,c.consumer>>>(out,w,att,res,ready);
  flash_vec_llama_score_pv_32_128_6_widekv16_vtail1_lastcta<<<dim3(6,32),dim3(32,4),0,c.producer>>>(partial,q,cache,att,ctr,ready,heads);
  ck(cudaEventRecord(c.done,c.consumer),"p-done");ck(cudaEventSynchronize(c.done),"p-sync");return elapsed(c.start,c.done);}
static float installed_ready(Ctx&c,float*out,half*x,unsigned*w,float*res,unsigned*ready){ck(cudaMemsetAsync(ready,1,4,c.producer),"r1");
  ck(cudaEventRecord(c.start,c.producer),"i-start");q4k_g3_lanemap_gemv_vec_epi_resadd_4096_4096<<<ROWS,32,0,c.producer>>>(out,w,x,res);
  ck(cudaEventRecord(c.done,c.producer),"i-done");ck(cudaEventSynchronize(c.done),"i-sync");return elapsed(c.start,c.done);}
static float persistent_ready(Ctx&c,float*out,half*x,unsigned*w,float*res,unsigned*ready){ck(cudaMemsetAsync(ready,1,4,c.producer),"r1");
  ck(cudaEventRecord(c.start,c.producer),"pr-start");q4k_o_bounded_persistent_w128<<<WORKERS,128,0,c.producer>>>(out,w,x,res,ready);
  ck(cudaEventRecord(c.done,c.producer),"pr-done");ck(cudaEventSynchronize(c.done),"pr-sync");return elapsed(c.start,c.done);}
static void legal_words(unsigned*w,size_t n,unsigned seed){for(size_t i=0;i<n;i++)w[i]=(unsigned)((i*2654435761u)^(seed*2246822519u)^0x9e3779b9u);
  for(size_t b=0;b<n;b+=36){w[b]=0x30003000u+(seed&3u);w[b+1]=0x10101010u;}}
int main(int ac,char**av){int hp=atoi(av[1]),cp=atoi(av[2]),reps=atoi(av[3]);Ctx c;init(c);
  float *a,*b,*pa,*pb,*q,*res;half *aa,*bb;unsigned *wa,*wb,*ka,*kb,*ca,*cb,*ra,*rb,*ha,*hb;
  ck(cudaMalloc(&a,ROWS*4),"a");ck(cudaMalloc(&b,ROWS*4),"b");ck(cudaMalloc(&pa,24960*4),"pa");ck(cudaMalloc(&pb,24960*4),"pb");
  ck(cudaMalloc(&aa,8192),"aa");ck(cudaMalloc(&bb,8192),"bb");ck(cudaMalloc(&q,16384),"q");ck(cudaMalloc(&res,ROWS*4),"res");
  ck(cudaMalloc(&wa,(size_t)ROTATIONS*O_WORDS*4),"wa");ck(cudaMalloc(&wb,(size_t)ROTATIONS*O_WORDS*4),"wb");
  ck(cudaMalloc(&ka,(size_t)ROTATIONS*CACHE_WORDS*4),"ka");ck(cudaMalloc(&kb,(size_t)ROTATIONS*CACHE_WORDS*4),"kb");
  ck(cudaMalloc(&ca,128),"ca");ck(cudaMalloc(&cb,128),"cb");ck(cudaMalloc(&ra,4),"ra");ck(cudaMalloc(&rb,4),"rb");
  ck(cudaMalloc(&ha,4),"ha");ck(cudaMalloc(&hb,4),"hb");
  std::vector<unsigned> hwa((size_t)ROTATIONS*O_WORDS),hwb((size_t)ROTATIONS*O_WORDS);for(int r=0;r<ROTATIONS;r++){
    legal_words(hwa.data()+(size_t)r*O_WORDS,O_WORDS,17+r);legal_words(hwb.data()+(size_t)r*O_WORDS,O_WORDS,17+r);}
  std::vector<float> hq(4096),hr(ROWS);for(int i=0;i<4096;i++)hq[i]=float((i*17%127)-63)/256.0f;
  for(int i=0;i<ROWS;i++)hr[i]=float((i%31)-15)/128.0f;
  ck(cudaMemcpy(wa,hwa.data(),hwa.size()*4,cudaMemcpyHostToDevice),"wa-copy");ck(cudaMemcpy(wb,hwb.data(),hwb.size()*4,cudaMemcpyHostToDevice),"wb-copy");
  ck(cudaMemcpy(q,hq.data(),16384,cudaMemcpyHostToDevice),"q-copy");ck(cudaMemcpy(res,hr.data(),ROWS*4,cudaMemcpyHostToDevice),"res-copy");
  ck(cudaMemset(ka,0x30,(size_t)ROTATIONS*CACHE_WORDS*4),"ka-init");ck(cudaMemset(kb,0x30,(size_t)ROTATIONS*CACHE_WORDS*4),"kb-init");
  bool exact=true,finite=true;std::vector<float> xa(ROWS),xb(ROWS);for(int rot: {0,5,11}){
    control(c,a,aa,pa,q,ka+(size_t)rot*CACHE_WORDS,wa+(size_t)rot*O_WORDS,res,ca,ra,ha);
    candidate(c,b,bb,pb,q,kb+(size_t)rot*CACHE_WORDS,wb+(size_t)rot*O_WORDS,res,cb,rb,hb);
    ck(cudaMemcpy(xa.data(),a,ROWS*4,cudaMemcpyDeviceToHost),"a-copy");ck(cudaMemcpy(xb.data(),b,ROWS*4,cudaMemcpyDeviceToHost),"b-copy");
    exact&=memcmp(xa.data(),xb.data(),ROWS*4)==0;for(int i=0;i<ROWS;i++)finite&=std::isfinite(xa[i])&&std::isfinite(xb[i]);}
  printf("validate exact=%d finite=%d\n",(int)exact,(int)finite);
  for(int r=0;r<reps;r++){double ih=0,ph=0,ic=0,pc0=0,ch=0,hh=0,cc=0,hc=0;
    for(int i=0;i<hp;i++){if((r+i)&1){ph+=persistent_ready(c,b,bb,wb,res,rb);ih+=installed_ready(c,a,aa,wa,res,ra);hh+=candidate(c,b,bb,pb,q,kb,wb,res,cb,rb,hb);ch+=control(c,a,aa,pa,q,ka,wa,res,ca,ra,ha);}
      else{ih+=installed_ready(c,a,aa,wa,res,ra);ph+=persistent_ready(c,b,bb,wb,res,rb);ch+=control(c,a,aa,pa,q,ka,wa,res,ca,ra,ha);hh+=candidate(c,b,bb,pb,q,kb,wb,res,cb,rb,hb);}}
    for(int i=0;i<cp;i++){int z=(r*cp+i)%ROTATIONS;if((r+i)&1){pc0+=persistent_ready(c,b,bb,wb+(size_t)z*O_WORDS,res,rb);ic+=installed_ready(c,a,aa,wa+(size_t)z*O_WORDS,res,ra);hc+=candidate(c,b,bb,pb,q,kb+(size_t)z*CACHE_WORDS,wb+(size_t)z*O_WORDS,res,cb,rb,hb);cc+=control(c,a,aa,pa,q,ka+(size_t)z*CACHE_WORDS,wa+(size_t)z*O_WORDS,res,ca,ra,ha);}
      else{ic+=installed_ready(c,a,aa,wa+(size_t)z*O_WORDS,res,ra);pc0+=persistent_ready(c,b,bb,wb+(size_t)z*O_WORDS,res,rb);cc+=control(c,a,aa,pa,q,ka+(size_t)z*CACHE_WORDS,wa+(size_t)z*O_WORDS,res,ca,ra,ha);hc+=candidate(c,b,bb,pb,q,kb+(size_t)z*CACHE_WORDS,wb+(size_t)z*O_WORDS,res,cb,rb,hb);}}
    printf("rep=%d installed_hot=%.6f persistent_hot=%.6f installed_cold=%.6f persistent_cold=%.6f control_span_hot=%.6f candidate_span_hot=%.6f control_span_cold=%.6f candidate_span_cold=%.6f\n",
      r,ih/hp,ph/hp,ic/cp,pc0/cp,ch/hp,hh/hp,cc/cp,hc/cp);}
  return exact&&finite?0:5;}
