#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
#include <cuda_fp16.h>
struct __align__(64) float16 { float x, y, z, w, a, b, c, d, e, f, g, h, i, j, k, l; }; __device__ float16 make_float16(float x, float y, float z, float w, float a, float b, float c, float d, float e, float f, float g, float h, float i, float j, float k, float l) { float16 r={x, y, z, w, a, b, c, d, e, f, g, h, i, j, k, l}; return r; }
struct __align__(32) half16 { half x, y, z, w, a, b, c, d, e, f, g, h, i, j, k, l; }; __device__ half16 make_half16(half x, half y, half z, half w, half a, half b, half c, half d, half e, half f, half g, half h, half i, half j, half k, half l) { half16 r={x, y, z, w, a, b, c, d, e, f, g, h, i, j, k, l}; return r; }
struct __align__(32) float8 { float x, y, z, w, a, b, c, d; }; __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { float8 r={x, y, z, w, a, b, c, d}; return r; }
struct __align__(8) half4 { half x, y, z, w; }; __device__ half4 make_half4(half x, half y, half z, half w) { half4 r={x, y, z, w}; return r; }
extern "C" __global__ void __launch_bounds__(128) flash_vec_llama_score_pv_32_128_6_widekv16_vtail1(float* data0_24960, float* data1_4096, unsigned int* data2_1048576) {
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
}
extern "C" __global__ void __launch_bounds__(128) flash_fused_gmax_combine_f16_32_128_s6_lw128(half* data0_4096, float* data1_24960) {
  int gidx0 = blockIdx.x; /* 32 */
  float buf0[1];
  (*(buf0+0)) = -1e+30f;
  int alu1 = (gidx0*780);
  for (int Ridx2 = 0; Ridx2 < 6; Ridx2++) {
    float val0 = (*(data1_24960+(alu1+(Ridx2*130)+129)));
    float alu2 = (((*(buf0+0))<val0)?val0:(*(buf0+0)));
    (*(buf0+0)) = alu2;
  }
  int lidx0 = threadIdx.x; /* 128 */
  bool alu5 = (lidx0<6);
  int alu6 = (alu5?lidx0:0);
  float val1 = (*(data1_24960+(alu1+(alu6*130)+129)));
  __shared__ __align__(16) float buf1[6];
  float buf2[1];
  float buf3[1];
  if (alu5) {
    *(buf1+alu6) = exp2(((val1-(*(buf0+0)))*1.4426950408889634f));
  }
  __syncthreads();
  (*(buf2+0)) = 0.0f;
  (*(buf3+0)) = 0.0f;
  for (int Ridx4 = 0; Ridx4 < 6; Ridx4++) {
    int alu13 = (alu1+(Ridx4*130));
    float val2 = (*(data1_24960+(lidx0+alu13)));
    float val3 = (*(data1_24960+(alu13+128)));
    float val4 = (*(buf1+Ridx4));
    (*(buf2+0)) = ((*(buf2+0))+(val4*val2));
    (*(buf3+0)) = ((*(buf3+0))+(val4*val3));
  }
  *(data0_4096+(lidx0+(gidx0<<7))) = ((half)(((*(buf2+0))*(1/(*(buf3+0))))));
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
}
extern "C" __global__ void __launch_bounds__(32) q4k_o_segment16_first(float* data0_65536, unsigned int* data1_2359296, half* data2_4096) {
  int gidx0 = blockIdx.x; /* 2048 */
  int lidx0 = threadIdx.x; /* 32 */
  float buf0[1];
  (*(buf0+0)) = 0.0f;
  int alu1 = (lidx0>>4);
  int alu2 = (lidx0&15);
  int alu3 = (alu2>>3);
  int alu4 = (alu2&7);
  for (int Ridx0 = 0; Ridx0 < 4; Ridx0++) {
    int alu5 = ((gidx0*1152)+(alu1*576)+(alu3*144)+(Ridx0*36));
    int alu6 = (alu5+alu4);
    unsigned int val0 = (*(data1_2359296+(alu6+4)));
    unsigned int val1 = (*(data1_2359296+(alu6+12)));
    unsigned int val2 = (*(data1_2359296+(alu6+20)));
    unsigned int val3 = (*(data1_2359296+(alu6+28)));
    uint4 val4 = (*((uint4*)((data1_2359296+alu5))));
    int alu7 = ((alu3<<10)+(Ridx0<<8)+(alu4<<2));
    half4 val5 = (*((half4*)((data2_4096+(alu7+32)))));
    half4 val6 = (*((half4*)((data2_4096+(alu7+64)))));
    half4 val7 = (*((half4*)((data2_4096+(alu7+96)))));
    half4 val8 = (*((half4*)((data2_4096+(alu7+128)))));
    half4 val9 = (*((half4*)((data2_4096+(alu7+160)))));
    half4 val10 = (*((half4*)((data2_4096+(alu7+192)))));
    half4 val11 = (*((half4*)((data2_4096+(alu7+224)))));
    half4 val12 = (*((half4*)((data2_4096+alu7))));
    float cast0 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val4.x&65535u)))))));
    float cast1 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)(((val4.x>>16u)&65535u)))))));
    unsigned int alu8 = (val4.z>>0u);
    unsigned int alu9 = (val4.w>>0u);
    unsigned int alu10 = (val4.z>>8u);
    unsigned int alu11 = (val4.w>>8u);
    unsigned int alu12 = (val4.z>>16u);
    unsigned int alu13 = (val4.w>>16u);
    unsigned int alu14 = (val4.z>>24u);
    unsigned int alu15 = (val4.w>>24u);
    unsigned int alu16 = (val4.y>>0u);
    unsigned int alu17 = (val4.y>>8u);
    unsigned int alu18 = (val4.y>>16u);
    unsigned int alu19 = (val4.y>>24u);
    unsigned int alu20 = ((val0>>0u)&252645135u);
    unsigned int alu21 = ((val0>>4u)&252645135u);
    unsigned int alu22 = ((val1>>0u)&252645135u);
    unsigned int alu23 = ((val1>>4u)&252645135u);
    unsigned int alu24 = ((val2>>0u)&252645135u);
    unsigned int alu25 = ((val2>>4u)&252645135u);
    unsigned int alu26 = ((val3>>0u)&252645135u);
    unsigned int alu27 = ((val3>>4u)&252645135u);
    float alu28 = (cast0*((float)((alu16&63u))));
    float alu29 = (cast1*((float)((alu8&63u))));
    float alu30 = (cast0*((float)((alu17&63u))));
    float alu31 = (cast1*((float)((alu10&63u))));
    float alu32 = (cast0*((float)((alu18&63u))));
    float alu33 = (cast1*((float)((alu12&63u))));
    float alu34 = (cast0*((float)((alu19&63u))));
    float alu35 = (cast1*((float)((alu14&63u))));
    float alu36 = (cast0*((float)(((alu9&15u)|(((alu16&255u)>>6u)<<4u)))));
    float alu37 = (cast1*((float)((((alu9&255u)>>4u)|(((alu8&255u)>>6u)<<4u)))));
    float alu38 = (cast0*((float)(((alu11&15u)|(((alu17&255u)>>6u)<<4u)))));
    float alu39 = (cast1*((float)((((alu11&255u)>>4u)|(((alu10&255u)>>6u)<<4u)))));
    float alu40 = (cast0*((float)(((alu13&15u)|(((alu18&255u)>>6u)<<4u)))));
    float alu41 = (cast1*((float)((((alu13&255u)>>4u)|(((alu12&255u)>>6u)<<4u)))));
    float alu42 = (cast0*((float)(((alu15&15u)|(((alu19&255u)>>6u)<<4u)))));
    float alu43 = (cast1*((float)((((alu15&255u)>>4u)|(((alu14&255u)>>6u)<<4u)))));
    (*(buf0+0)) = ((*(buf0+0))+(((alu28*((float)(((alu20>>0u)&15u))))-alu29)*float(val12.x))+(((alu28*((float)(((alu20>>8u)&15u))))-alu29)*float(val12.y))+(((alu28*((float)(((alu20>>16u)&15u))))-alu29)*float(val12.z))+(((alu28*((float)(((alu20>>24u)&15u))))-alu29)*float(val12.w))+(((alu30*((float)(((alu21>>0u)&15u))))-alu31)*float(val5.x))+(((alu30*((float)(((alu21>>8u)&15u))))-alu31)*float(val5.y))+(((alu30*((float)(((alu21>>16u)&15u))))-alu31)*float(val5.z))+(((alu30*((float)(((alu21>>24u)&15u))))-alu31)*float(val5.w))+(((alu32*((float)(((alu22>>0u)&15u))))-alu33)*float(val6.x))+(((alu32*((float)(((alu22>>8u)&15u))))-alu33)*float(val6.y))+(((alu32*((float)(((alu22>>16u)&15u))))-alu33)*float(val6.z))+(((alu32*((float)(((alu22>>24u)&15u))))-alu33)*float(val6.w))+(((alu34*((float)(((alu23>>0u)&15u))))-alu35)*float(val7.x))+(((alu34*((float)(((alu23>>8u)&15u))))-alu35)*float(val7.y))+(((alu34*((float)(((alu23>>16u)&15u))))-alu35)*float(val7.z))+(((alu34*((float)(((alu23>>24u)&15u))))-alu35)*float(val7.w))+(((alu36*((float)(((alu24>>0u)&15u))))-alu37)*float(val8.x))+(((alu36*((float)(((alu24>>8u)&15u))))-alu37)*float(val8.y))+(((alu36*((float)(((alu24>>16u)&15u))))-alu37)*float(val8.z))+(((alu36*((float)(((alu24>>24u)&15u))))-alu37)*float(val8.w))+(((alu38*((float)(((alu25>>0u)&15u))))-alu39)*float(val9.x))+(((alu38*((float)(((alu25>>8u)&15u))))-alu39)*float(val9.y))+(((alu38*((float)(((alu25>>16u)&15u))))-alu39)*float(val9.z))+(((alu38*((float)(((alu25>>24u)&15u))))-alu39)*float(val9.w))+(((alu40*((float)(((alu26>>0u)&15u))))-alu41)*float(val10.x))+(((alu40*((float)(((alu26>>8u)&15u))))-alu41)*float(val10.y))+(((alu40*((float)(((alu26>>16u)&15u))))-alu41)*float(val10.z))+(((alu40*((float)(((alu26>>24u)&15u))))-alu41)*float(val10.w))+(((alu42*((float)(((alu27>>0u)&15u))))-alu43)*float(val11.x))+(((alu42*((float)(((alu27>>8u)&15u))))-alu43)*float(val11.y))+(((alu42*((float)(((alu27>>16u)&15u))))-alu43)*float(val11.z))+(((alu42*((float)(((alu27>>24u)&15u))))-alu43)*float(val11.w)));
  }
  *(data0_65536+((gidx0<<5)+(alu1<<4)+alu2)) = (*(buf0+0));
}
extern "C" __global__ void __launch_bounds__(32) q4k_o_segment16_finish_epi_resadd(float* data0_4096, unsigned int* data1_2359296, half* data2_4096, float* data3_65536, float* data4_4096) {
  int gidx0 = blockIdx.x; /* 2048 */
  int lidx0 = threadIdx.x; /* 32 */
  float buf0[1];
  (*(buf0+0)) = 0.0f;
  int alu1 = (lidx0>>4);
  int alu2 = (lidx0&15);
  int alu3 = (alu2>>3);
  int alu4 = (alu2&7);
  for (int Ridx0 = 0; Ridx0 < 4; Ridx0++) {
    int alu5 = ((gidx0*1152)+(alu1*576)+(alu3*144)+(Ridx0*36));
    int alu6 = (alu5+alu4);
    unsigned int val0 = (*(data1_2359296+(alu6+292)));
    unsigned int val1 = (*(data1_2359296+(alu6+300)));
    unsigned int val2 = (*(data1_2359296+(alu6+308)));
    unsigned int val3 = (*(data1_2359296+(alu6+316)));
    uint4 val4 = (*((uint4*)((data1_2359296+(alu5+288)))));
    int alu7 = ((alu3<<10)+(Ridx0<<8)+(alu4<<2));
    half4 val5 = (*((half4*)((data2_4096+(alu7+2048)))));
    half4 val6 = (*((half4*)((data2_4096+(alu7+2080)))));
    half4 val7 = (*((half4*)((data2_4096+(alu7+2112)))));
    half4 val8 = (*((half4*)((data2_4096+(alu7+2144)))));
    half4 val9 = (*((half4*)((data2_4096+(alu7+2176)))));
    half4 val10 = (*((half4*)((data2_4096+(alu7+2208)))));
    half4 val11 = (*((half4*)((data2_4096+(alu7+2240)))));
    half4 val12 = (*((half4*)((data2_4096+(alu7+2272)))));
    float cast0 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val4.x&65535u)))))));
    float cast1 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)(((val4.x>>16u)&65535u)))))));
    unsigned int alu8 = (val4.z>>0u);
    unsigned int alu9 = (val4.w>>0u);
    unsigned int alu10 = (val4.z>>8u);
    unsigned int alu11 = (val4.w>>8u);
    unsigned int alu12 = (val4.z>>16u);
    unsigned int alu13 = (val4.w>>16u);
    unsigned int alu14 = (val4.z>>24u);
    unsigned int alu15 = (val4.w>>24u);
    unsigned int alu16 = (val4.y>>0u);
    unsigned int alu17 = (val4.y>>8u);
    unsigned int alu18 = (val4.y>>16u);
    unsigned int alu19 = (val4.y>>24u);
    unsigned int alu20 = ((val0>>0u)&252645135u);
    unsigned int alu21 = ((val0>>4u)&252645135u);
    unsigned int alu22 = ((val1>>0u)&252645135u);
    unsigned int alu23 = ((val1>>4u)&252645135u);
    unsigned int alu24 = ((val2>>0u)&252645135u);
    unsigned int alu25 = ((val2>>4u)&252645135u);
    unsigned int alu26 = ((val3>>0u)&252645135u);
    unsigned int alu27 = ((val3>>4u)&252645135u);
    float alu28 = (cast0*((float)((alu16&63u))));
    float alu29 = (cast1*((float)((alu8&63u))));
    float alu30 = (cast0*((float)((alu17&63u))));
    float alu31 = (cast1*((float)((alu10&63u))));
    float alu32 = (cast0*((float)((alu18&63u))));
    float alu33 = (cast1*((float)((alu12&63u))));
    float alu34 = (cast0*((float)((alu19&63u))));
    float alu35 = (cast1*((float)((alu14&63u))));
    float alu36 = (cast0*((float)(((alu9&15u)|(((alu16&255u)>>6u)<<4u)))));
    float alu37 = (cast1*((float)((((alu9&255u)>>4u)|(((alu8&255u)>>6u)<<4u)))));
    float alu38 = (cast0*((float)(((alu11&15u)|(((alu17&255u)>>6u)<<4u)))));
    float alu39 = (cast1*((float)((((alu11&255u)>>4u)|(((alu10&255u)>>6u)<<4u)))));
    float alu40 = (cast0*((float)(((alu13&15u)|(((alu18&255u)>>6u)<<4u)))));
    float alu41 = (cast1*((float)((((alu13&255u)>>4u)|(((alu12&255u)>>6u)<<4u)))));
    float alu42 = (cast0*((float)(((alu15&15u)|(((alu19&255u)>>6u)<<4u)))));
    float alu43 = (cast1*((float)((((alu15&255u)>>4u)|(((alu14&255u)>>6u)<<4u)))));
    (*(buf0+0)) = ((*(buf0+0))+(((alu28*((float)(((alu20>>0u)&15u))))-alu29)*float(val5.x))+(((alu28*((float)(((alu20>>8u)&15u))))-alu29)*float(val5.y))+(((alu28*((float)(((alu20>>16u)&15u))))-alu29)*float(val5.z))+(((alu28*((float)(((alu20>>24u)&15u))))-alu29)*float(val5.w))+(((alu30*((float)(((alu21>>0u)&15u))))-alu31)*float(val6.x))+(((alu30*((float)(((alu21>>8u)&15u))))-alu31)*float(val6.y))+(((alu30*((float)(((alu21>>16u)&15u))))-alu31)*float(val6.z))+(((alu30*((float)(((alu21>>24u)&15u))))-alu31)*float(val6.w))+(((alu32*((float)(((alu22>>0u)&15u))))-alu33)*float(val7.x))+(((alu32*((float)(((alu22>>8u)&15u))))-alu33)*float(val7.y))+(((alu32*((float)(((alu22>>16u)&15u))))-alu33)*float(val7.z))+(((alu32*((float)(((alu22>>24u)&15u))))-alu33)*float(val7.w))+(((alu34*((float)(((alu23>>0u)&15u))))-alu35)*float(val8.x))+(((alu34*((float)(((alu23>>8u)&15u))))-alu35)*float(val8.y))+(((alu34*((float)(((alu23>>16u)&15u))))-alu35)*float(val8.z))+(((alu34*((float)(((alu23>>24u)&15u))))-alu35)*float(val8.w))+(((alu36*((float)(((alu24>>0u)&15u))))-alu37)*float(val9.x))+(((alu36*((float)(((alu24>>8u)&15u))))-alu37)*float(val9.y))+(((alu36*((float)(((alu24>>16u)&15u))))-alu37)*float(val9.z))+(((alu36*((float)(((alu24>>24u)&15u))))-alu37)*float(val9.w))+(((alu38*((float)(((alu25>>0u)&15u))))-alu39)*float(val10.x))+(((alu38*((float)(((alu25>>8u)&15u))))-alu39)*float(val10.y))+(((alu38*((float)(((alu25>>16u)&15u))))-alu39)*float(val10.z))+(((alu38*((float)(((alu25>>24u)&15u))))-alu39)*float(val10.w))+(((alu40*((float)(((alu26>>0u)&15u))))-alu41)*float(val11.x))+(((alu40*((float)(((alu26>>8u)&15u))))-alu41)*float(val11.y))+(((alu40*((float)(((alu26>>16u)&15u))))-alu41)*float(val11.z))+(((alu40*((float)(((alu26>>24u)&15u))))-alu41)*float(val11.w))+(((alu42*((float)(((alu27>>0u)&15u))))-alu43)*float(val12.x))+(((alu42*((float)(((alu27>>8u)&15u))))-alu43)*float(val12.y))+(((alu42*((float)(((alu27>>16u)&15u))))-alu43)*float(val12.z))+(((alu42*((float)(((alu27>>24u)&15u))))-alu43)*float(val12.w)));
  }
  float val13 = (*(data3_65536+((gidx0<<5)+(alu1<<4)+alu2)));
  int alu46 = ((gidx0<<1)+alu1);
  float val14 = (*(data4_4096+alu46));
  float buf1[1];
  float alu47 = (val13+(*(buf0+0)));
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu47, 8);
  float alu49 = (alu47+(*(buf1+0)));
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu49, 4);
  float alu51 = (alu49+(*(buf1+0)));
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu51, 2);
  float alu53 = (alu51+(*(buf1+0)));
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu53, 1);
  if ((alu2==0)) {
    *(data0_4096+alu46) = (alu53+(*(buf1+0))+val14);
  }
}
extern "C" __global__ void __launch_bounds__(128) flash_vec_llama_score_pv_32_128_6_widekv16_vtail1_h16o0(float* data0_24960, float* data1_4096, unsigned int* data2_1048576) {
  float buf0[16];
  for (int Lidx41 = 0; Lidx41 < 16; Lidx41++) {
    (*(buf0+Lidx41)) = 0.0f;
  }
  int gidx1 = blockIdx.y + 0; /* 16 of 32 */
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
}
extern "C" __global__ void __launch_bounds__(128) flash_vec_llama_score_pv_32_128_6_widekv16_vtail1_h16o16(float* data0_24960, float* data1_4096, unsigned int* data2_1048576) {
  float buf0[16];
  for (int Lidx41 = 0; Lidx41 < 16; Lidx41++) {
    (*(buf0+Lidx41)) = 0.0f;
  }
  int gidx1 = blockIdx.y + 16; /* 16 of 32 */
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
}
extern "C" __global__ void __launch_bounds__(128) flash_fused_gmax_combine_f16_32_128_s6_lw128_h16o0(half* data0_4096, float* data1_24960) {
  int gidx0 = blockIdx.x + 0; /* 16 of 32 */
  float buf0[1];
  (*(buf0+0)) = -1e+30f;
  int alu1 = (gidx0*780);
  for (int Ridx2 = 0; Ridx2 < 6; Ridx2++) {
    float val0 = (*(data1_24960+(alu1+(Ridx2*130)+129)));
    float alu2 = (((*(buf0+0))<val0)?val0:(*(buf0+0)));
    (*(buf0+0)) = alu2;
  }
  int lidx0 = threadIdx.x; /* 128 */
  bool alu5 = (lidx0<6);
  int alu6 = (alu5?lidx0:0);
  float val1 = (*(data1_24960+(alu1+(alu6*130)+129)));
  __shared__ __align__(16) float buf1[6];
  float buf2[1];
  float buf3[1];
  if (alu5) {
    *(buf1+alu6) = exp2(((val1-(*(buf0+0)))*1.4426950408889634f));
  }
  __syncthreads();
  (*(buf2+0)) = 0.0f;
  (*(buf3+0)) = 0.0f;
  for (int Ridx4 = 0; Ridx4 < 6; Ridx4++) {
    int alu13 = (alu1+(Ridx4*130));
    float val2 = (*(data1_24960+(lidx0+alu13)));
    float val3 = (*(data1_24960+(alu13+128)));
    float val4 = (*(buf1+Ridx4));
    (*(buf2+0)) = ((*(buf2+0))+(val4*val2));
    (*(buf3+0)) = ((*(buf3+0))+(val4*val3));
  }
  *(data0_4096+(lidx0+(gidx0<<7))) = ((half)(((*(buf2+0))*(1/(*(buf3+0))))));
}
extern "C" __global__ void __launch_bounds__(128) flash_fused_gmax_combine_f16_32_128_s6_lw128_h16o16(half* data0_4096, float* data1_24960) {
  int gidx0 = blockIdx.x + 16; /* 16 of 32 */
  float buf0[1];
  (*(buf0+0)) = -1e+30f;
  int alu1 = (gidx0*780);
  for (int Ridx2 = 0; Ridx2 < 6; Ridx2++) {
    float val0 = (*(data1_24960+(alu1+(Ridx2*130)+129)));
    float alu2 = (((*(buf0+0))<val0)?val0:(*(buf0+0)));
    (*(buf0+0)) = alu2;
  }
  int lidx0 = threadIdx.x; /* 128 */
  bool alu5 = (lidx0<6);
  int alu6 = (alu5?lidx0:0);
  float val1 = (*(data1_24960+(alu1+(alu6*130)+129)));
  __shared__ __align__(16) float buf1[6];
  float buf2[1];
  float buf3[1];
  if (alu5) {
    *(buf1+alu6) = exp2(((val1-(*(buf0+0)))*1.4426950408889634f));
  }
  __syncthreads();
  (*(buf2+0)) = 0.0f;
  (*(buf3+0)) = 0.0f;
  for (int Ridx4 = 0; Ridx4 < 6; Ridx4++) {
    int alu13 = (alu1+(Ridx4*130));
    float val2 = (*(data1_24960+(lidx0+alu13)));
    float val3 = (*(data1_24960+(alu13+128)));
    float val4 = (*(buf1+Ridx4));
    (*(buf2+0)) = ((*(buf2+0))+(val4*val2));
    (*(buf3+0)) = ((*(buf3+0))+(val4*val3));
  }
  *(data0_4096+(lidx0+(gidx0<<7))) = ((half)(((*(buf2+0))*(1/(*(buf3+0))))));
}
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define HQ 32
#define HD 128
#define SPLITS 6
#define STRIDE 130
#define ROWS 4096
#define O_WORDS 2359296
#define ROTATIONS 16
#define CACHE_WORDS 1048576
static void ck(cudaError_t e,const char*w){if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e));exit(2);}}
struct Ctx { cudaStream_t hi,lo,o; cudaEvent_t start,done,r0,r1,f1s,f1e,o0s,o0e; };
static void init_ctx(Ctx&c){int least,greatest;ck(cudaDeviceGetStreamPriorityRange(&least,&greatest),"priority");
  ck(cudaStreamCreateWithPriority(&c.hi,cudaStreamNonBlocking,greatest),"hi");
  ck(cudaStreamCreateWithPriority(&c.lo,cudaStreamNonBlocking,least),"lo");
  ck(cudaStreamCreateWithFlags(&c.o,cudaStreamNonBlocking),"o");
  for(cudaEvent_t*e:{&c.start,&c.done,&c.r0,&c.r1,&c.f1s,&c.f1e,&c.o0s,&c.o0e})ck(cudaEventCreate(e),"event");}
static float elapsed(cudaEvent_t a,cudaEvent_t b){float ms;ck(cudaEventElapsedTime(&ms,a,b),"elapsed");return ms*1000.0f;}
static double control(Ctx&c,float*out,half*attn,float*partial,float*q,unsigned int*cache,unsigned int*w,float*res){
  ck(cudaEventRecord(c.start,c.o),"cstart");
  flash_vec_llama_score_pv_32_128_6_widekv16_vtail1<<<dim3(SPLITS,HQ,1),dim3(32,4,1),0,c.o>>>(partial,q,cache);
  flash_fused_gmax_combine_f16_32_128_s6_lw128<<<HQ,128,0,c.o>>>(attn,partial);
  q4k_g3_lanemap_gemv_vec_epi_resadd_4096_4096<<<ROWS,32,0,c.o>>>(out,w,attn,res);
  ck(cudaEventRecord(c.done,c.o),"cdone");ck(cudaEventSynchronize(c.done),"csync");return elapsed(c.start,c.done);
}
static double candidate(Ctx&c,float*out,half*attn,float*partial,float*q,unsigned int*cache,unsigned int*w,float*scratch,float*res,bool trace){
  ck(cudaEventRecord(c.start,c.o),"start");ck(cudaStreamWaitEvent(c.hi,c.start),"hi-start");ck(cudaStreamWaitEvent(c.lo,c.start),"lo-start");
  flash_vec_llama_score_pv_32_128_6_widekv16_vtail1_h16o0<<<dim3(SPLITS,16,1),dim3(32,4,1),0,c.hi>>>(partial,q,cache);
  flash_fused_gmax_combine_f16_32_128_s6_lw128_h16o0<<<16,128,0,c.hi>>>(attn,partial);ck(cudaEventRecord(c.r0,c.hi),"r0");
  ck(cudaEventRecord(c.f1s,c.lo),"f1s");flash_vec_llama_score_pv_32_128_6_widekv16_vtail1_h16o16<<<dim3(SPLITS,16,1),dim3(32,4,1),0,c.lo>>>(partial,q,cache);
  flash_fused_gmax_combine_f16_32_128_s6_lw128_h16o16<<<16,128,0,c.lo>>>(attn,partial);ck(cudaEventRecord(c.f1e,c.lo),"f1e");ck(cudaEventRecord(c.r1,c.lo),"r1");
  ck(cudaStreamWaitEvent(c.o,c.r0),"wait-r0");ck(cudaEventRecord(c.o0s,c.o),"o0s");
  q4k_o_segment16_first<<<ROWS/2,32,0,c.o>>>(scratch,w,attn);ck(cudaEventRecord(c.o0e,c.o),"o0e");
  ck(cudaStreamWaitEvent(c.o,c.r1),"wait-r1");
  q4k_o_segment16_finish_epi_resadd<<<ROWS/2,32,0,c.o>>>(out,w,attn,scratch,res);
  ck(cudaEventRecord(c.done,c.o),"done");ck(cudaEventSynchronize(c.done),"sync");
  if(trace){double f1a=elapsed(c.start,c.f1s),f1b=elapsed(c.start,c.f1e),oa=elapsed(c.start,c.o0s),ob=elapsed(c.start,c.o0e);
    double ov=std::max(0.0,std::min(f1b,ob)-std::max(f1a,oa));printf("timeline flash1_start=%.6f flash1_end=%.6f o0_start=%.6f o0_end=%.6f overlap=%.6f\n",f1a,f1b,oa,ob,ov);}
  return elapsed(c.start,c.done);
}
int main(int ac,char**av){int hp=atoi(av[1]),cp=atoi(av[2]),reps=atoi(av[3]);Ctx c;init_ctx(c);
  float *co,*so,*cp0,*sp,*q,*res,*scratch;half *ca,*sa;unsigned int *ws,*caches;unsigned char*evict;
  ck(cudaMalloc(&co,ROWS*4),"co");ck(cudaMalloc(&so,ROWS*4),"so");ck(cudaMalloc(&cp0,HQ*SPLITS*STRIDE*4),"cp");ck(cudaMalloc(&sp,HQ*SPLITS*STRIDE*4),"sp");
  ck(cudaMalloc(&ca,HQ*HD*2),"ca");ck(cudaMalloc(&sa,HQ*HD*2),"sa");ck(cudaMalloc(&q,HQ*HD*4),"q");ck(cudaMalloc(&res,ROWS*4),"res");
  ck(cudaMalloc(&scratch,ROWS*16*4),"scratch");ck(cudaMalloc(&ws,(size_t)ROTATIONS*O_WORDS*4),"ws");
  ck(cudaMalloc(&caches,(size_t)ROTATIONS*CACHE_WORDS*4),"cache");ck(cudaMalloc(&evict,128ull<<20),"evict");
  std::vector<float> hq(HQ*HD),hr(ROWS);for(size_t i=0;i<hq.size();i++)hq[i]=float((int(i*17+3)%127)-63)/256.0f;for(int i=0;i<ROWS;i++)hr[i]=float((i%31)-15)/128.0f;
  ck(cudaMemcpy(q,hq.data(),hq.size()*4,cudaMemcpyHostToDevice),"qcopy");ck(cudaMemcpy(res,hr.data(),hr.size()*4,cudaMemcpyHostToDevice),"rcopy");
  ck(cudaMemset(ws,7,(size_t)ROTATIONS*O_WORDS*4),"winit");ck(cudaMemset(caches,0x30,(size_t)ROTATIONS*CACHE_WORDS*4),"cinit");
  control(c,co,ca,cp0,q,caches,ws,res);candidate(c,so,sa,sp,q,caches,ws,scratch,res,false);
  std::vector<unsigned char>a(HQ*SPLITS*STRIDE*4),b(a.size());cudaMemcpy(a.data(),cp0,a.size(),cudaMemcpyDeviceToHost);cudaMemcpy(b.data(),sp,b.size(),cudaMemcpyDeviceToHost);printf("score_bitwise=%d\n",memcmp(a.data(),b.data(),a.size())==0);
  a.resize(HQ*HD*2);b.resize(a.size());cudaMemcpy(a.data(),ca,a.size(),cudaMemcpyDeviceToHost);cudaMemcpy(b.data(),sa,b.size(),cudaMemcpyDeviceToHost);printf("combine_bitwise=%d\n",memcmp(a.data(),b.data(),a.size())==0);
  a.resize(ROWS*4);b.resize(a.size());cudaMemcpy(a.data(),co,a.size(),cudaMemcpyDeviceToHost);cudaMemcpy(b.data(),so,b.size(),cudaMemcpyDeviceToHost);printf("final_bitwise=%d\n",memcmp(a.data(),b.data(),a.size())==0);
  for(int i=0;i<20;i++){control(c,co,ca,cp0,q,caches,ws,res);candidate(c,so,sa,sp,q,caches,ws,scratch,res,false);}
  candidate(c,so,sa,sp,q,caches,ws,scratch,res,true);
  for(int r=0;r<reps;r++)for(int oi=0;oi<2;oi++){int arm=(r&1)?1-oi:oi;double hot=0,cold=0;
    for(int i=0;i<hp;i++)hot+=(arm?candidate(c,so,sa,sp,q,caches,ws,scratch,res,false):control(c,co,ca,cp0,q,caches,ws,res));
    for(int i=0;i<cp;i++){ck(cudaMemset(evict,(r*cp+i)&255,128ull<<20),"evict");ck(cudaDeviceSynchronize(),"evictsync");int rot=(r*cp+i)%ROTATIONS;
      cold+=(arm?candidate(c,so,sa,sp,q,caches+(size_t)rot*CACHE_WORDS,ws+(size_t)rot*O_WORDS,scratch,res,false):control(c,co,ca,cp0,q,caches+(size_t)rot*CACHE_WORDS,ws+(size_t)rot*O_WORDS,res));}
    printf("rep=%d arm=%d hot=%.6f cold=%.6f\n",r,arm,hot/hp,cold/cp);
  }return 0;}
