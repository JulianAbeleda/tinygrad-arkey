
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
#include <cuda_fp16.h>
struct __align__(64) float16 { float x, y, z, w, a, b, c, d, e, f, g, h, i, j, k, l; }; __device__ float16 make_float16(float x, float y, float z, float w, float a, float b, float c, float d, float e, float f, float g, float h, float i, float j, float k, float l) { float16 r={x, y, z, w, a, b, c, d, e, f, g, h, i, j, k, l}; return r; }
struct __align__(32) half16 { half x, y, z, w, a, b, c, d, e, f, g, h, i, j, k, l; }; __device__ half16 make_half16(half x, half y, half z, half w, half a, half b, half c, half d, half e, half f, half g, half h, half i, half j, half k, half l) { half16 r={x, y, z, w, a, b, c, d, e, f, g, h, i, j, k, l}; return r; }
struct __align__(32) float8 { float x, y, z, w, a, b, c, d; }; __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { float8 r={x, y, z, w, a, b, c, d}; return r; }
extern "C" __global__ void __launch_bounds__(128, 1) flash_vec_llama_score_pv_32_128_6_widekv16(float* data0_24960, float* data1_4096, unsigned int* data2_1048576) {
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
    float alu20 = (((alu8+Ridx5)<512)?((alu18+(*(buf2+0)))*0.08838834764831843f):((float)(-INFINITY)));
    (*(buf3+Ridx5)) = alu20;
  }
  float buf5[1];
  (*(buf5+0)) = ((float)(-INFINITY));
  for (int Ridx7 = 0; Ridx7 < 8; Ridx7++) {
    float alu24 = (((*(buf5+0))<(*(buf3+Ridx7)))?(*(buf3+Ridx7)):(*(buf5+0)));
    (*(buf5+0)) = alu24;
  }
  float buf6[1];
  float buf7[1];
  (*(buf2+0)) = __shfl_xor_sync(0xffffffffu, (*(buf5+0)), 8);
  float alu28 = (((*(buf5+0))<(*(buf2+0)))?(*(buf2+0)):(*(buf5+0)));
  (*(buf2+0)) = __shfl_xor_sync(0xffffffffu, alu28, 16);
  (*(buf6+0)) = 0.0f;
  (*(buf7+0)) = ((float)(-INFINITY));
  float alu32 = ((alu28<(*(buf2+0)))?(*(buf2+0)):alu28);
  bool alu33 = (-1e+30f<alu32);
  float alu34 = (alu33?exp2((((*(buf7+0))-alu32)*1.4426950408889634f)):1.0f);
  for (int Lidx8 = 0; Lidx8 < 16; Lidx8++) {
    (*(buf0+Lidx8)) = ((*(buf0+Lidx8))*alu34);
  }
  (*(buf6+0)) = ((*(buf6+0))*alu34);
  float alu38 = (alu33?alu32:(*(buf7+0)));
  (*(buf7+0)) = alu38;
  for (int Ridx9 = 0; Ridx9 < 8; Ridx9++) {
    int alu40 = (alu9+(Ridx9<<6)+alu10+alu11);
    uint4 val3 = (*((uint4*)((data2_1048576+(alu40+524288)))));
    uint4 val4 = (*((uint4*)((data2_1048576+(alu40+524320)))));
    float alu41 = (((alu8+Ridx9)<512)?exp2((((*(buf3+Ridx9))-alu38)*1.4426950408889634f)):0.0f);
    (*(buf0+0)) = ((*(buf0+0))+(alu41*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val3.x>>0u)))))))));
    (*(buf0+1)) = ((*(buf0+1))+(alu41*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val3.x>>16u)))))))));
    (*(buf0+2)) = ((*(buf0+2))+(alu41*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val3.y>>0u)))))))));
    (*(buf0+3)) = ((*(buf0+3))+(alu41*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val3.y>>16u)))))))));
    (*(buf0+4)) = ((*(buf0+4))+(alu41*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val3.z>>0u)))))))));
    (*(buf0+5)) = ((*(buf0+5))+(alu41*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val3.z>>16u)))))))));
    (*(buf0+6)) = ((*(buf0+6))+(alu41*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val3.w>>0u)))))))));
    (*(buf0+7)) = ((*(buf0+7))+(alu41*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val3.w>>16u)))))))));
    (*(buf0+8)) = ((*(buf0+8))+(alu41*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val4.x>>0u)))))))));
    (*(buf0+9)) = ((*(buf0+9))+(alu41*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val4.x>>16u)))))))));
    (*(buf0+10)) = ((*(buf0+10))+(alu41*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val4.y>>0u)))))))));
    (*(buf0+11)) = ((*(buf0+11))+(alu41*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val4.y>>16u)))))))));
    (*(buf0+12)) = ((*(buf0+12))+(alu41*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val4.z>>0u)))))))));
    (*(buf0+13)) = ((*(buf0+13))+(alu41*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val4.z>>16u)))))))));
    (*(buf0+14)) = ((*(buf0+14))+(alu41*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val4.w>>0u)))))))));
    (*(buf0+15)) = ((*(buf0+15))+(alu41*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val4.w>>16u)))))))));
    (*(buf6+0)) = ((*(buf6+0))+alu41);
  }
  float buf8[1];
  __shared__ __align__(16) float buf9[512];
  for (int Lidx13_0 = 0; Lidx13_0 < 2; Lidx13_0++) {
    for (int Lidx13_1 = 0; Lidx13_1 < 8; Lidx13_1++) {
      (*(buf8+0)) = __shfl_xor_sync(0xffffffffu, (*(buf0+((Lidx13_0<<3)+Lidx13_1))), 8);
      float alu61 = ((*(buf0+((Lidx13_0<<3)+Lidx13_1)))+(*(buf8+0)));
      (*(buf8+0)) = __shfl_xor_sync(0xffffffffu, alu61, 16);
      if ((lidx0<8)) {
        *(buf9+(alu3+(Lidx13_0<<6)+Lidx13_1+(lidx1<<7))) = (alu61+(*(buf8+0)));
      }
    }
  }
  float buf10[1];
  float buf11[1];
  __shared__ __align__(16) float buf12[4];
  __shared__ __align__(16) float buf13[4];
  float buf14[1];
  (*(buf10+0)) = __shfl_xor_sync(0xffffffffu, (*(buf6+0)), 8);
  float alu69 = ((*(buf6+0))+(*(buf10+0)));
  (*(buf10+0)) = __shfl_xor_sync(0xffffffffu, alu69, 16);
  (*(buf11+0)) = __shfl_xor_sync(0xffffffffu, (*(buf7+0)), 8);
  float alu72 = (((*(buf7+0))<(*(buf11+0)))?(*(buf11+0)):(*(buf7+0)));
  (*(buf11+0)) = __shfl_xor_sync(0xffffffffu, alu72, 16);
  bool alu74 = (lidx0==0);
  if (alu74) {
    *(buf12+lidx1) = (alu69+(*(buf10+0)));
  }
  float alu78 = ((alu72<(*(buf11+0)))?(*(buf11+0)):alu72);
  if (alu74) {
    *(buf13+lidx1) = alu78;
  }
  __syncthreads();
  (*(buf14+0)) = ((float)(-INFINITY));
  for (int Ridx14 = 0; Ridx14 < 4; Ridx14++) {
    float val5 = (*(buf13+Ridx14));
    float alu84 = (((*(buf14+0))<val5)?val5:(*(buf14+0)));
    (*(buf14+0)) = alu84;
  }
  float buf15[16];
  for (int Lidx15 = 0; Lidx15 < 16; Lidx15++) {
    (*(buf15+Lidx15)) = 0.0f;
  }
  float buf16[1];
  (*(buf16+0)) = 0.0f;
  for (int Ridx16 = 0; Ridx16 < 4; Ridx16++) {
    float val6 = (*(buf13+Ridx16));
    float alu90 = ((-1e+30f<(*(buf14+0)))?exp2(((val6-(*(buf14+0)))*1.4426950408889634f)):0.0f);
    for (int Lidx17_0 = 0; Lidx17_0 < 2; Lidx17_0++) {
      for (int Lidx17_1 = 0; Lidx17_1 < 8; Lidx17_1++) {
        float val7 = (*(buf9+(alu3+(Lidx17_0<<6)+Lidx17_1+(Ridx16<<7))));
        int alu91 = ((Lidx17_0<<3)+Lidx17_1);
        (*(buf15+alu91)) = ((*(buf15+alu91))+(alu90*val7));
      }
    }
    float val8 = (*(buf12+Ridx16));
    (*(buf16+0)) = ((*(buf16+0))+(alu90*val8));
  }
  int alu97 = ((gidx0*130)+(gidx1*780));
  for (int Lidx18_0 = 0; Lidx18_0 < 2; Lidx18_0++) {
    for (int Lidx18_1 = 0; Lidx18_1 < 8; Lidx18_1++) {
      if ((lidx1==0)) {
        *(data0_24960+(alu3+(Lidx18_0<<6)+Lidx18_1+alu97)) = (*(buf15+((Lidx18_0<<3)+Lidx18_1)));
      }
    }
  }
  if (alu74) {
    *(data0_24960+(alu97+128)) = (*(buf16+0));
  }
  if (alu74) {
    *(data0_24960+(alu97+129)) = (*(buf14+0));
  }
}
extern "C" __global__ void __launch_bounds__(128, 1) flash_vec_llama_score_pv_32_128_6_widekv16_warpprob(float* data0_24960, float* data1_4096, unsigned int* data2_1048576) {
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
  float buf3[1];
  float buf4[1];
  int alu7 = (lidx0>>3);
  int alu8 = (alu7<<3);
  int alu9 = ((gidx0<<7)+(lidx1<<5)+alu8);
  int alu10 = ((gidx0<<13)+(lidx1<<11)+(alu7<<9));
  int alu11 = ((gidx1>>2)<<16);
  int alu12 = (alu2<<2);
  for (int Ridx5 = 0; Ridx5 < 8; Ridx5++) {
    int alu13 = (alu10+(Ridx5<<6)+alu11+alu12);
    uint4 val1 = (*((uint4*)((data2_1048576+(alu13+32)))));
    uint4 val2 = (*((uint4*)((data2_1048576+alu13))));
    (*(buf3+0)) = 0.0f;
    (*(buf3+0)) = (((((((((*(buf3+0))) + float(make_half2((*(buf1+0)),(*(buf1+1))).x) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val2.x>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val2.x>>16u)))))).x) + float(make_half2((*(buf1+0)),(*(buf1+1))).y) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val2.x>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val2.x>>16u)))))).y)) + float(make_half2((*(buf1+2)),(*(buf1+3))).x) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val2.y>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val2.y>>16u)))))).x) + float(make_half2((*(buf1+2)),(*(buf1+3))).y) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val2.y>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val2.y>>16u)))))).y)) + float(make_half2((*(buf1+4)),(*(buf1+5))).x) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val2.z>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val2.z>>16u)))))).x) + float(make_half2((*(buf1+4)),(*(buf1+5))).y) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val2.z>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val2.z>>16u)))))).y)) + float(make_half2((*(buf1+6)),(*(buf1+7))).x) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val2.w>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val2.w>>16u)))))).x) + float(make_half2((*(buf1+6)),(*(buf1+7))).y) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val2.w>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val2.w>>16u)))))).y)) + float(make_half2((*(buf1+8)),(*(buf1+9))).x) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val1.x>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val1.x>>16u)))))).x) + float(make_half2((*(buf1+8)),(*(buf1+9))).y) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val1.x>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val1.x>>16u)))))).y)) + float(make_half2((*(buf1+10)),(*(buf1+11))).x) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val1.y>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val1.y>>16u)))))).x) + float(make_half2((*(buf1+10)),(*(buf1+11))).y) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val1.y>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val1.y>>16u)))))).y)) + float(make_half2((*(buf1+12)),(*(buf1+13))).x) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val1.z>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val1.z>>16u)))))).x) + float(make_half2((*(buf1+12)),(*(buf1+13))).y) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val1.z>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val1.z>>16u)))))).y)) + float(make_half2((*(buf1+14)),(*(buf1+15))).x) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val1.w>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val1.w>>16u)))))).x) + float(make_half2((*(buf1+14)),(*(buf1+15))).y) * float(make_half2(tg_bitcast<half>((unsigned short)(((unsigned short)((val1.w>>0u))))),tg_bitcast<half>((unsigned short)(((unsigned short)((val1.w>>16u)))))).y);
    (*(buf2+0)) = __shfl_xor_sync(0xffffffffu, (*(buf3+0)), 4);
    float alu17 = ((*(buf3+0))+(*(buf2+0)));
    (*(buf2+0)) = __shfl_xor_sync(0xffffffffu, alu17, 2);
    float alu19 = (alu17+(*(buf2+0)));
    (*(buf2+0)) = __shfl_xor_sync(0xffffffffu, alu19, 1);
    float alu21 = (((alu9+Ridx5)<512)?((alu19+(*(buf2+0)))*0.08838834764831843f):((float)(-INFINITY)));
    if ((alu2==Ridx5)) {
      (*(buf4+0)) = alu21;
    }
  }
  float buf5[1];
  (*(buf5+0)) = ((float)(-INFINITY));
  for (int Ridx7 = 0; Ridx7 < 8; Ridx7++) {
    float alu27 = __shfl_sync(0xffffffffu, (*(buf4+0)), (((((unsigned int)((alu8+Ridx7)))<<2u)) >> 2));
    float alu28 = (((*(buf5+0))<alu27)?alu27:(*(buf5+0)));
    (*(buf5+0)) = alu28;
  }
  float buf6[1];
  float buf7[1];
  (*(buf2+0)) = __shfl_xor_sync(0xffffffffu, (*(buf5+0)), 8);
  float alu32 = (((*(buf5+0))<(*(buf2+0)))?(*(buf2+0)):(*(buf5+0)));
  (*(buf2+0)) = __shfl_xor_sync(0xffffffffu, alu32, 16);
  (*(buf6+0)) = 0.0f;
  (*(buf7+0)) = ((float)(-INFINITY));
  float alu36 = ((alu32<(*(buf2+0)))?(*(buf2+0)):alu32);
  bool alu37 = (-1e+30f<alu36);
  float alu38 = (alu37?exp2((((*(buf7+0))-alu36)*1.4426950408889634f)):1.0f);
  for (int Lidx8 = 0; Lidx8 < 16; Lidx8++) {
    (*(buf0+Lidx8)) = ((*(buf0+Lidx8))*alu38);
  }
  (*(buf6+0)) = ((*(buf6+0))*alu38);
  float alu42 = (alu37?alu36:(*(buf7+0)));
  float alu43 = (((alu9+alu2)<512)?exp2((((*(buf4+0))-alu42)*1.4426950408889634f)):0.0f);
  (*(buf7+0)) = alu42;
  for (int Ridx9 = 0; Ridx9 < 8; Ridx9++) {
    int alu45 = (alu10+(Ridx9<<6)+alu11+alu12);
    uint4 val3 = (*((uint4*)((data2_1048576+(alu45+524288)))));
    uint4 val4 = (*((uint4*)((data2_1048576+(alu45+524320)))));
    float alu46 = __shfl_sync(0xffffffffu, alu43, (((((unsigned int)((alu8+Ridx9)))<<2u)) >> 2));
    (*(buf0+0)) = ((*(buf0+0))+(alu46*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val3.x>>0u)))))))));
    (*(buf0+1)) = ((*(buf0+1))+(alu46*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val3.x>>16u)))))))));
    (*(buf0+2)) = ((*(buf0+2))+(alu46*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val3.y>>0u)))))))));
    (*(buf0+3)) = ((*(buf0+3))+(alu46*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val3.y>>16u)))))))));
    (*(buf0+4)) = ((*(buf0+4))+(alu46*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val3.z>>0u)))))))));
    (*(buf0+5)) = ((*(buf0+5))+(alu46*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val3.z>>16u)))))))));
    (*(buf0+6)) = ((*(buf0+6))+(alu46*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val3.w>>0u)))))))));
    (*(buf0+7)) = ((*(buf0+7))+(alu46*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val3.w>>16u)))))))));
    (*(buf0+8)) = ((*(buf0+8))+(alu46*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val4.x>>0u)))))))));
    (*(buf0+9)) = ((*(buf0+9))+(alu46*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val4.x>>16u)))))))));
    (*(buf0+10)) = ((*(buf0+10))+(alu46*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val4.y>>0u)))))))));
    (*(buf0+11)) = ((*(buf0+11))+(alu46*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val4.y>>16u)))))))));
    (*(buf0+12)) = ((*(buf0+12))+(alu46*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val4.z>>0u)))))))));
    (*(buf0+13)) = ((*(buf0+13))+(alu46*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val4.z>>16u)))))))));
    (*(buf0+14)) = ((*(buf0+14))+(alu46*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val4.w>>0u)))))))));
    (*(buf0+15)) = ((*(buf0+15))+(alu46*((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val4.w>>16u)))))))));
    (*(buf6+0)) = ((*(buf6+0))+alu46);
  }
  float buf8[1];
  __shared__ __align__(16) float buf9[512];
  for (int Lidx13_0 = 0; Lidx13_0 < 2; Lidx13_0++) {
    for (int Lidx13_1 = 0; Lidx13_1 < 8; Lidx13_1++) {
      (*(buf8+0)) = __shfl_xor_sync(0xffffffffu, (*(buf0+((Lidx13_0<<3)+Lidx13_1))), 8);
      float alu66 = ((*(buf0+((Lidx13_0<<3)+Lidx13_1)))+(*(buf8+0)));
      (*(buf8+0)) = __shfl_xor_sync(0xffffffffu, alu66, 16);
      if ((lidx0<8)) {
        *(buf9+(alu3+(Lidx13_0<<6)+Lidx13_1+(lidx1<<7))) = (alu66+(*(buf8+0)));
      }
    }
  }
  float buf10[1];
  float buf11[1];
  __shared__ __align__(16) float buf12[4];
  __shared__ __align__(16) float buf13[4];
  float buf14[1];
  (*(buf10+0)) = __shfl_xor_sync(0xffffffffu, (*(buf6+0)), 8);
  float alu74 = ((*(buf6+0))+(*(buf10+0)));
  (*(buf10+0)) = __shfl_xor_sync(0xffffffffu, alu74, 16);
  (*(buf11+0)) = __shfl_xor_sync(0xffffffffu, (*(buf7+0)), 8);
  float alu77 = (((*(buf7+0))<(*(buf11+0)))?(*(buf11+0)):(*(buf7+0)));
  (*(buf11+0)) = __shfl_xor_sync(0xffffffffu, alu77, 16);
  bool alu79 = (lidx0==0);
  if (alu79) {
    *(buf12+lidx1) = (alu74+(*(buf10+0)));
  }
  float alu83 = ((alu77<(*(buf11+0)))?(*(buf11+0)):alu77);
  if (alu79) {
    *(buf13+lidx1) = alu83;
  }
  __syncthreads();
  (*(buf14+0)) = ((float)(-INFINITY));
  for (int Ridx14 = 0; Ridx14 < 4; Ridx14++) {
    float val5 = (*(buf13+Ridx14));
    float alu89 = (((*(buf14+0))<val5)?val5:(*(buf14+0)));
    (*(buf14+0)) = alu89;
  }
  float buf15[16];
  for (int Lidx15 = 0; Lidx15 < 16; Lidx15++) {
    (*(buf15+Lidx15)) = 0.0f;
  }
  float buf16[1];
  (*(buf16+0)) = 0.0f;
  for (int Ridx16 = 0; Ridx16 < 4; Ridx16++) {
    float val6 = (*(buf13+Ridx16));
    float alu95 = ((-1e+30f<(*(buf14+0)))?exp2(((val6-(*(buf14+0)))*1.4426950408889634f)):0.0f);
    for (int Lidx17_0 = 0; Lidx17_0 < 2; Lidx17_0++) {
      for (int Lidx17_1 = 0; Lidx17_1 < 8; Lidx17_1++) {
        float val7 = (*(buf9+(alu3+(Lidx17_0<<6)+Lidx17_1+(Ridx16<<7))));
        int alu96 = ((Lidx17_0<<3)+Lidx17_1);
        (*(buf15+alu96)) = ((*(buf15+alu96))+(alu95*val7));
      }
    }
    float val8 = (*(buf12+Ridx16));
    (*(buf16+0)) = ((*(buf16+0))+(alu95*val8));
  }
  int alu102 = ((gidx0*130)+(gidx1*780));
  for (int Lidx18_0 = 0; Lidx18_0 < 2; Lidx18_0++) {
    for (int Lidx18_1 = 0; Lidx18_1 < 8; Lidx18_1++) {
      if ((lidx1==0)) {
        *(data0_24960+(alu3+(Lidx18_0<<6)+Lidx18_1+alu102)) = (*(buf15+((Lidx18_0<<3)+Lidx18_1)));
      }
    }
  }
  if (alu79) {
    *(data0_24960+(alu102+128)) = (*(buf16+0));
  }
  if (alu79) {
    *(data0_24960+(alu102+129)) = (*(buf14+0));
  }
}
static void ck(cudaError_t e,const char* w){if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e));exit(2);}}
static double run_control(float* o,float* q,unsigned int* c,int n){cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);for(int i=0;i<n;i++)flash_vec_llama_score_pv_32_128_6_widekv16<<<dim3(6,32,1),dim3(32,4,1)>>>(o,q,c);cudaEventRecord(b);ck(cudaEventSynchronize(b),"control");float ms;cudaEventElapsedTime(&ms,a,b);return 1000.0*ms/n;}
static double run_candidate(float* o,float* q,unsigned int* c,int n){cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);for(int i=0;i<n;i++)flash_vec_llama_score_pv_32_128_6_widekv16_warpprob<<<dim3(6,32,1),dim3(32,4,1)>>>(o,q,c);cudaEventRecord(b);ck(cudaEventSynchronize(b),"candidate");float ms;cudaEventElapsedTime(&ms,a,b);return 1000.0*ms/n;}
int main(int ac,char**av){int n=ac>1?atoi(av[1]):400,r=ac>2?atoi(av[2]):9;float *oc,*on,*q;unsigned int*c;ck(cudaMalloc(&oc,33280*4),"oc");ck(cudaMalloc(&on,24960*4),"on");ck(cudaMalloc(&q,4096*4),"q");ck(cudaMalloc(&c,1048576*4),"c");float*hq=(float*)malloc(4096*4);unsigned int*hc=(unsigned int*)malloc(1048576*4);for(int i=0;i<4096;i++)hq[i]=((i*17+3)%127-63)/256.0f;for(int i=0;i<1048576;i++)hc[i]=(i*2654435761u)^0x3c003c00u;cudaMemcpy(q,hq,4096*4,cudaMemcpyHostToDevice);cudaMemcpy(c,hc,1048576*4,cudaMemcpyHostToDevice);free(hq);free(hc);flash_vec_llama_score_pv_32_128_6_widekv16<<<dim3(6,32,1),dim3(32,4,1)>>>(oc,q,c);flash_vec_llama_score_pv_32_128_6_widekv16_warpprob<<<dim3(6,32,1),dim3(32,4,1)>>>(on,q,c);ck(cudaDeviceSynchronize(),"warm");if(1){float *a=(float*)malloc(24960*4),*b=(float*)malloc(24960*4);cudaMemcpy(a,oc,24960*4,cudaMemcpyDeviceToHost);cudaMemcpy(b,on,24960*4,cudaMemcpyDeviceToHost);int ne=0;float md=0;for(int i=0;i<24960;i++){unsigned int ua=*((unsigned int*)(a+i)),ub=*((unsigned int*)(b+i));if(ua!=ub)ne++;float d=fabsf(a[i]-b[i]);if(d>md)md=d;}printf("exact_mismatches=%d max_abs=%.9g\n",ne,md);free(a);free(b);}for(int i=0;i<r;i++)printf("rep=%d control=%.6f candidate=%.6f\n",i,run_control(oc,q,c,n),run_candidate(on,q,c,n));}
