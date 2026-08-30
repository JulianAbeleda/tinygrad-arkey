
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
extern "C" __global__ void __launch_bounds__(128, 1) flash_vec_llama_score_pv_32_128_6_widekv16_vtail8_vdimmajor(float* data0_24960, float* data1_4096, unsigned int* data2_1048576) {
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
  int alu27 = (alu9+alu10+alu11);
  uint4 val3 = (*((uint4*)((data2_1048576+(alu27+524288)))));
  uint4 val4 = (*((uint4*)((data2_1048576+(alu27+524320)))));
  uint4 val5 = (*((uint4*)((data2_1048576+(alu27+524352)))));
  uint4 val6 = (*((uint4*)((data2_1048576+(alu27+524384)))));
  uint4 val7 = (*((uint4*)((data2_1048576+(alu27+524416)))));
  uint4 val8 = (*((uint4*)((data2_1048576+(alu27+524448)))));
  uint4 val9 = (*((uint4*)((data2_1048576+(alu27+524480)))));
  uint4 val10 = (*((uint4*)((data2_1048576+(alu27+524512)))));
  uint4 val11 = (*((uint4*)((data2_1048576+(alu27+524544)))));
  uint4 val12 = (*((uint4*)((data2_1048576+(alu27+524576)))));
  uint4 val13 = (*((uint4*)((data2_1048576+(alu27+524608)))));
  uint4 val14 = (*((uint4*)((data2_1048576+(alu27+524640)))));
  uint4 val15 = (*((uint4*)((data2_1048576+(alu27+524672)))));
  uint4 val16 = (*((uint4*)((data2_1048576+(alu27+524704)))));
  uint4 val17 = (*((uint4*)((data2_1048576+(alu27+524736)))));
  uint4 val18 = (*((uint4*)((data2_1048576+(alu27+524768)))));
  float buf6[1];
  float buf7[1];
  half buf8[128];
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
  (*(buf8+16)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val5.x>>0u)))));
  (*(buf8+17)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val5.x>>16u)))));
  (*(buf8+18)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val5.y>>0u)))));
  (*(buf8+19)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val5.y>>16u)))));
  (*(buf8+20)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val5.z>>0u)))));
  (*(buf8+21)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val5.z>>16u)))));
  (*(buf8+22)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val5.w>>0u)))));
  (*(buf8+23)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val5.w>>16u)))));
  (*(buf8+24)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val6.x>>0u)))));
  (*(buf8+25)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val6.x>>16u)))));
  (*(buf8+26)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val6.y>>0u)))));
  (*(buf8+27)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val6.y>>16u)))));
  (*(buf8+28)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val6.z>>0u)))));
  (*(buf8+29)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val6.z>>16u)))));
  (*(buf8+30)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val6.w>>0u)))));
  (*(buf8+31)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val6.w>>16u)))));
  (*(buf8+32)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val7.x>>0u)))));
  (*(buf8+33)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val7.x>>16u)))));
  (*(buf8+34)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val7.y>>0u)))));
  (*(buf8+35)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val7.y>>16u)))));
  (*(buf8+36)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val7.z>>0u)))));
  (*(buf8+37)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val7.z>>16u)))));
  (*(buf8+38)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val7.w>>0u)))));
  (*(buf8+39)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val7.w>>16u)))));
  (*(buf8+40)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val8.x>>0u)))));
  (*(buf8+41)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val8.x>>16u)))));
  (*(buf8+42)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val8.y>>0u)))));
  (*(buf8+43)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val8.y>>16u)))));
  (*(buf8+44)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val8.z>>0u)))));
  (*(buf8+45)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val8.z>>16u)))));
  (*(buf8+46)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val8.w>>0u)))));
  (*(buf8+47)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val8.w>>16u)))));
  (*(buf8+48)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val9.x>>0u)))));
  (*(buf8+49)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val9.x>>16u)))));
  (*(buf8+50)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val9.y>>0u)))));
  (*(buf8+51)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val9.y>>16u)))));
  (*(buf8+52)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val9.z>>0u)))));
  (*(buf8+53)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val9.z>>16u)))));
  (*(buf8+54)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val9.w>>0u)))));
  (*(buf8+55)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val9.w>>16u)))));
  (*(buf8+56)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val10.x>>0u)))));
  (*(buf8+57)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val10.x>>16u)))));
  (*(buf8+58)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val10.y>>0u)))));
  (*(buf8+59)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val10.y>>16u)))));
  (*(buf8+60)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val10.z>>0u)))));
  (*(buf8+61)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val10.z>>16u)))));
  (*(buf8+62)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val10.w>>0u)))));
  (*(buf8+63)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val10.w>>16u)))));
  (*(buf8+64)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val11.x>>0u)))));
  (*(buf8+65)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val11.x>>16u)))));
  (*(buf8+66)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val11.y>>0u)))));
  (*(buf8+67)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val11.y>>16u)))));
  (*(buf8+68)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val11.z>>0u)))));
  (*(buf8+69)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val11.z>>16u)))));
  (*(buf8+70)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val11.w>>0u)))));
  (*(buf8+71)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val11.w>>16u)))));
  (*(buf8+72)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val12.x>>0u)))));
  (*(buf8+73)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val12.x>>16u)))));
  (*(buf8+74)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val12.y>>0u)))));
  (*(buf8+75)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val12.y>>16u)))));
  (*(buf8+76)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val12.z>>0u)))));
  (*(buf8+77)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val12.z>>16u)))));
  (*(buf8+78)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val12.w>>0u)))));
  (*(buf8+79)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val12.w>>16u)))));
  (*(buf8+80)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val13.x>>0u)))));
  (*(buf8+81)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val13.x>>16u)))));
  (*(buf8+82)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val13.y>>0u)))));
  (*(buf8+83)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val13.y>>16u)))));
  (*(buf8+84)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val13.z>>0u)))));
  (*(buf8+85)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val13.z>>16u)))));
  (*(buf8+86)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val13.w>>0u)))));
  (*(buf8+87)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val13.w>>16u)))));
  (*(buf8+88)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val14.x>>0u)))));
  (*(buf8+89)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val14.x>>16u)))));
  (*(buf8+90)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val14.y>>0u)))));
  (*(buf8+91)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val14.y>>16u)))));
  (*(buf8+92)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val14.z>>0u)))));
  (*(buf8+93)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val14.z>>16u)))));
  (*(buf8+94)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val14.w>>0u)))));
  (*(buf8+95)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val14.w>>16u)))));
  (*(buf8+96)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val15.x>>0u)))));
  (*(buf8+97)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val15.x>>16u)))));
  (*(buf8+98)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val15.y>>0u)))));
  (*(buf8+99)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val15.y>>16u)))));
  (*(buf8+100)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val15.z>>0u)))));
  (*(buf8+101)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val15.z>>16u)))));
  (*(buf8+102)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val15.w>>0u)))));
  (*(buf8+103)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val15.w>>16u)))));
  (*(buf8+104)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val16.x>>0u)))));
  (*(buf8+105)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val16.x>>16u)))));
  (*(buf8+106)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val16.y>>0u)))));
  (*(buf8+107)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val16.y>>16u)))));
  (*(buf8+108)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val16.z>>0u)))));
  (*(buf8+109)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val16.z>>16u)))));
  (*(buf8+110)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val16.w>>0u)))));
  (*(buf8+111)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val16.w>>16u)))));
  (*(buf8+112)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val17.x>>0u)))));
  (*(buf8+113)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val17.x>>16u)))));
  (*(buf8+114)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val17.y>>0u)))));
  (*(buf8+115)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val17.y>>16u)))));
  (*(buf8+116)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val17.z>>0u)))));
  (*(buf8+117)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val17.z>>16u)))));
  (*(buf8+118)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val17.w>>0u)))));
  (*(buf8+119)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val17.w>>16u)))));
  (*(buf8+120)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val18.x>>0u)))));
  (*(buf8+121)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val18.x>>16u)))));
  (*(buf8+122)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val18.y>>0u)))));
  (*(buf8+123)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val18.y>>16u)))));
  (*(buf8+124)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val18.z>>0u)))));
  (*(buf8+125)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val18.z>>16u)))));
  (*(buf8+126)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val18.w>>0u)))));
  (*(buf8+127)) = tg_bitcast<half>((unsigned short)(((unsigned short)((val18.w>>16u)))));
  float alu161 = ((alu29<(*(buf2+0)))?(*(buf2+0)):alu29);
  bool alu162 = (-1e+30f<alu161);
  float alu163 = (alu162?exp2((((*(buf7+0))-alu161)*1.4426950408889634f)):1.0f);
  for (int Lidx8 = 0; Lidx8 < 16; Lidx8++) {
    (*(buf0+Lidx8)) = ((*(buf0+Lidx8))*alu163);
  }
  float buf9[1];
  __shared__ __align__(16) float buf10[512];
  float buf11[8];
  float alu166 = (alu162?alu161:(*(buf7+0)));
  float alu167 = ((alu8<505)?exp2((((*(buf3+7))-alu166)*1.4426950408889634f)):0.0f);
  float alu168 = ((alu8<506)?exp2((((*(buf3+6))-alu166)*1.4426950408889634f)):0.0f);
  float alu169 = ((alu8<507)?exp2((((*(buf3+5))-alu166)*1.4426950408889634f)):0.0f);
  float alu170 = ((alu8<508)?exp2((((*(buf3+4))-alu166)*1.4426950408889634f)):0.0f);
  float alu171 = ((alu8<509)?exp2((((*(buf3+3))-alu166)*1.4426950408889634f)):0.0f);
  float alu172 = ((alu8<510)?exp2((((*(buf3+2))-alu166)*1.4426950408889634f)):0.0f);
  float alu173 = ((alu8<511)?exp2((((*(buf3+1))-alu166)*1.4426950408889634f)):0.0f);
  float alu174 = ((alu8<512)?exp2((((*(buf3+0))-alu166)*1.4426950408889634f)):0.0f);
  (*(buf11+0)) = alu174;
  (*(buf11+1)) = alu173;
  (*(buf11+2)) = alu172;
  (*(buf11+3)) = alu171;
  (*(buf11+4)) = alu170;
  (*(buf11+5)) = alu169;
  (*(buf11+6)) = alu168;
  (*(buf11+7)) = alu167;
  (*(buf6+0)) = ((*(buf6+0))*alu163);
  (*(buf7+0)) = alu166;
  (*(buf0+0)) = ((*(buf0+0))+((*(buf11+0))*((float)((*(buf8+0)))))+((*(buf11+1))*((float)((*(buf8+16)))))+((*(buf11+2))*((float)((*(buf8+32)))))+((*(buf11+3))*((float)((*(buf8+48)))))+((*(buf11+4))*((float)((*(buf8+64)))))+((*(buf11+5))*((float)((*(buf8+80)))))+((*(buf11+6))*((float)((*(buf8+96)))))+((*(buf11+7))*((float)((*(buf8+112))))));
  (*(buf0+1)) = ((*(buf0+1))+((*(buf11+0))*((float)((*(buf8+1)))))+((*(buf11+1))*((float)((*(buf8+17)))))+((*(buf11+2))*((float)((*(buf8+33)))))+((*(buf11+3))*((float)((*(buf8+49)))))+((*(buf11+4))*((float)((*(buf8+65)))))+((*(buf11+5))*((float)((*(buf8+81)))))+((*(buf11+6))*((float)((*(buf8+97)))))+((*(buf11+7))*((float)((*(buf8+113))))));
  (*(buf0+2)) = ((*(buf0+2))+((*(buf11+0))*((float)((*(buf8+2)))))+((*(buf11+1))*((float)((*(buf8+18)))))+((*(buf11+2))*((float)((*(buf8+34)))))+((*(buf11+3))*((float)((*(buf8+50)))))+((*(buf11+4))*((float)((*(buf8+66)))))+((*(buf11+5))*((float)((*(buf8+82)))))+((*(buf11+6))*((float)((*(buf8+98)))))+((*(buf11+7))*((float)((*(buf8+114))))));
  (*(buf0+3)) = ((*(buf0+3))+((*(buf11+0))*((float)((*(buf8+3)))))+((*(buf11+1))*((float)((*(buf8+19)))))+((*(buf11+2))*((float)((*(buf8+35)))))+((*(buf11+3))*((float)((*(buf8+51)))))+((*(buf11+4))*((float)((*(buf8+67)))))+((*(buf11+5))*((float)((*(buf8+83)))))+((*(buf11+6))*((float)((*(buf8+99)))))+((*(buf11+7))*((float)((*(buf8+115))))));
  (*(buf0+4)) = ((*(buf0+4))+((*(buf11+0))*((float)((*(buf8+4)))))+((*(buf11+1))*((float)((*(buf8+20)))))+((*(buf11+2))*((float)((*(buf8+36)))))+((*(buf11+3))*((float)((*(buf8+52)))))+((*(buf11+4))*((float)((*(buf8+68)))))+((*(buf11+5))*((float)((*(buf8+84)))))+((*(buf11+6))*((float)((*(buf8+100)))))+((*(buf11+7))*((float)((*(buf8+116))))));
  (*(buf0+5)) = ((*(buf0+5))+((*(buf11+0))*((float)((*(buf8+5)))))+((*(buf11+1))*((float)((*(buf8+21)))))+((*(buf11+2))*((float)((*(buf8+37)))))+((*(buf11+3))*((float)((*(buf8+53)))))+((*(buf11+4))*((float)((*(buf8+69)))))+((*(buf11+5))*((float)((*(buf8+85)))))+((*(buf11+6))*((float)((*(buf8+101)))))+((*(buf11+7))*((float)((*(buf8+117))))));
  (*(buf0+6)) = ((*(buf0+6))+((*(buf11+0))*((float)((*(buf8+6)))))+((*(buf11+1))*((float)((*(buf8+22)))))+((*(buf11+2))*((float)((*(buf8+38)))))+((*(buf11+3))*((float)((*(buf8+54)))))+((*(buf11+4))*((float)((*(buf8+70)))))+((*(buf11+5))*((float)((*(buf8+86)))))+((*(buf11+6))*((float)((*(buf8+102)))))+((*(buf11+7))*((float)((*(buf8+118))))));
  (*(buf0+7)) = ((*(buf0+7))+((*(buf11+0))*((float)((*(buf8+7)))))+((*(buf11+1))*((float)((*(buf8+23)))))+((*(buf11+2))*((float)((*(buf8+39)))))+((*(buf11+3))*((float)((*(buf8+55)))))+((*(buf11+4))*((float)((*(buf8+71)))))+((*(buf11+5))*((float)((*(buf8+87)))))+((*(buf11+6))*((float)((*(buf8+103)))))+((*(buf11+7))*((float)((*(buf8+119))))));
  (*(buf0+8)) = ((*(buf0+8))+((*(buf11+0))*((float)((*(buf8+8)))))+((*(buf11+1))*((float)((*(buf8+24)))))+((*(buf11+2))*((float)((*(buf8+40)))))+((*(buf11+3))*((float)((*(buf8+56)))))+((*(buf11+4))*((float)((*(buf8+72)))))+((*(buf11+5))*((float)((*(buf8+88)))))+((*(buf11+6))*((float)((*(buf8+104)))))+((*(buf11+7))*((float)((*(buf8+120))))));
  (*(buf0+9)) = ((*(buf0+9))+((*(buf11+0))*((float)((*(buf8+9)))))+((*(buf11+1))*((float)((*(buf8+25)))))+((*(buf11+2))*((float)((*(buf8+41)))))+((*(buf11+3))*((float)((*(buf8+57)))))+((*(buf11+4))*((float)((*(buf8+73)))))+((*(buf11+5))*((float)((*(buf8+89)))))+((*(buf11+6))*((float)((*(buf8+105)))))+((*(buf11+7))*((float)((*(buf8+121))))));
  (*(buf0+10)) = ((*(buf0+10))+((*(buf11+0))*((float)((*(buf8+10)))))+((*(buf11+1))*((float)((*(buf8+26)))))+((*(buf11+2))*((float)((*(buf8+42)))))+((*(buf11+3))*((float)((*(buf8+58)))))+((*(buf11+4))*((float)((*(buf8+74)))))+((*(buf11+5))*((float)((*(buf8+90)))))+((*(buf11+6))*((float)((*(buf8+106)))))+((*(buf11+7))*((float)((*(buf8+122))))));
  (*(buf0+11)) = ((*(buf0+11))+((*(buf11+0))*((float)((*(buf8+11)))))+((*(buf11+1))*((float)((*(buf8+27)))))+((*(buf11+2))*((float)((*(buf8+43)))))+((*(buf11+3))*((float)((*(buf8+59)))))+((*(buf11+4))*((float)((*(buf8+75)))))+((*(buf11+5))*((float)((*(buf8+91)))))+((*(buf11+6))*((float)((*(buf8+107)))))+((*(buf11+7))*((float)((*(buf8+123))))));
  (*(buf0+12)) = ((*(buf0+12))+((*(buf11+0))*((float)((*(buf8+12)))))+((*(buf11+1))*((float)((*(buf8+28)))))+((*(buf11+2))*((float)((*(buf8+44)))))+((*(buf11+3))*((float)((*(buf8+60)))))+((*(buf11+4))*((float)((*(buf8+76)))))+((*(buf11+5))*((float)((*(buf8+92)))))+((*(buf11+6))*((float)((*(buf8+108)))))+((*(buf11+7))*((float)((*(buf8+124))))));
  (*(buf0+13)) = ((*(buf0+13))+((*(buf11+0))*((float)((*(buf8+13)))))+((*(buf11+1))*((float)((*(buf8+29)))))+((*(buf11+2))*((float)((*(buf8+45)))))+((*(buf11+3))*((float)((*(buf8+61)))))+((*(buf11+4))*((float)((*(buf8+77)))))+((*(buf11+5))*((float)((*(buf8+93)))))+((*(buf11+6))*((float)((*(buf8+109)))))+((*(buf11+7))*((float)((*(buf8+125))))));
  (*(buf0+14)) = ((*(buf0+14))+((*(buf11+0))*((float)((*(buf8+14)))))+((*(buf11+1))*((float)((*(buf8+30)))))+((*(buf11+2))*((float)((*(buf8+46)))))+((*(buf11+3))*((float)((*(buf8+62)))))+((*(buf11+4))*((float)((*(buf8+78)))))+((*(buf11+5))*((float)((*(buf8+94)))))+((*(buf11+6))*((float)((*(buf8+110)))))+((*(buf11+7))*((float)((*(buf8+126))))));
  (*(buf0+15)) = ((*(buf0+15))+((*(buf11+0))*((float)((*(buf8+15)))))+((*(buf11+1))*((float)((*(buf8+31)))))+((*(buf11+2))*((float)((*(buf8+47)))))+((*(buf11+3))*((float)((*(buf8+63)))))+((*(buf11+4))*((float)((*(buf8+79)))))+((*(buf11+5))*((float)((*(buf8+95)))))+((*(buf11+6))*((float)((*(buf8+111)))))+((*(buf11+7))*((float)((*(buf8+127))))));
  (*(buf6+0)) = ((*(buf6+0))+(*(buf11+0)));
  (*(buf6+0)) = ((*(buf6+0))+(*(buf11+1)));
  (*(buf6+0)) = ((*(buf6+0))+(*(buf11+2)));
  (*(buf6+0)) = ((*(buf6+0))+(*(buf11+3)));
  (*(buf6+0)) = ((*(buf6+0))+(*(buf11+4)));
  (*(buf6+0)) = ((*(buf6+0))+(*(buf11+5)));
  (*(buf6+0)) = ((*(buf6+0))+(*(buf11+6)));
  (*(buf6+0)) = ((*(buf6+0))+(*(buf11+7)));
  for (int Lidx13_0 = 0; Lidx13_0 < 2; Lidx13_0++) {
    for (int Lidx13_1 = 0; Lidx13_1 < 8; Lidx13_1++) {
      (*(buf9+0)) = __shfl_xor_sync(0xffffffffu, (*(buf0+((Lidx13_0<<3)+Lidx13_1))), 8);
      float alu210 = ((*(buf0+((Lidx13_0<<3)+Lidx13_1)))+(*(buf9+0)));
      (*(buf9+0)) = __shfl_xor_sync(0xffffffffu, alu210, 16);
      if ((lidx0<8)) {
        *(buf10+(alu3+(Lidx13_0<<6)+Lidx13_1+(lidx1<<7))) = (alu210+(*(buf9+0)));
      }
    }
  }
  float buf12[1];
  float buf13[1];
  __shared__ __align__(16) float buf14[4];
  __shared__ __align__(16) float buf15[4];
  float buf16[1];
  (*(buf12+0)) = __shfl_xor_sync(0xffffffffu, (*(buf6+0)), 8);
  float alu218 = ((*(buf6+0))+(*(buf12+0)));
  (*(buf12+0)) = __shfl_xor_sync(0xffffffffu, alu218, 16);
  (*(buf13+0)) = __shfl_xor_sync(0xffffffffu, (*(buf7+0)), 8);
  float alu221 = (((*(buf7+0))<(*(buf13+0)))?(*(buf13+0)):(*(buf7+0)));
  (*(buf13+0)) = __shfl_xor_sync(0xffffffffu, alu221, 16);
  bool alu223 = (lidx0==0);
  if (alu223) {
    *(buf14+lidx1) = (alu218+(*(buf12+0)));
  }
  float alu227 = ((alu221<(*(buf13+0)))?(*(buf13+0)):alu221);
  if (alu223) {
    *(buf15+lidx1) = alu227;
  }
  __syncthreads();
  (*(buf16+0)) = ((float)(-INFINITY));
  for (int Ridx14 = 0; Ridx14 < 4; Ridx14++) {
    float val19 = (*(buf15+Ridx14));
    float alu233 = (((*(buf16+0))<val19)?val19:(*(buf16+0)));
    (*(buf16+0)) = alu233;
  }
  float buf17[16];
  for (int Lidx15 = 0; Lidx15 < 16; Lidx15++) {
    (*(buf17+Lidx15)) = 0.0f;
  }
  float buf18[1];
  (*(buf18+0)) = 0.0f;
  for (int Ridx16 = 0; Ridx16 < 4; Ridx16++) {
    float val20 = (*(buf15+Ridx16));
    float alu239 = ((-1e+30f<(*(buf16+0)))?exp2(((val20-(*(buf16+0)))*1.4426950408889634f)):0.0f);
    for (int Lidx17_0 = 0; Lidx17_0 < 2; Lidx17_0++) {
      for (int Lidx17_1 = 0; Lidx17_1 < 8; Lidx17_1++) {
        float val21 = (*(buf10+(alu3+(Lidx17_0<<6)+Lidx17_1+(Ridx16<<7))));
        int alu240 = ((Lidx17_0<<3)+Lidx17_1);
        (*(buf17+alu240)) = ((*(buf17+alu240))+(alu239*val21));
      }
    }
    float val22 = (*(buf14+Ridx16));
    (*(buf18+0)) = ((*(buf18+0))+(alu239*val22));
  }
  int alu246 = ((gidx0*130)+(gidx1*780));
  for (int Lidx18_0 = 0; Lidx18_0 < 2; Lidx18_0++) {
    for (int Lidx18_1 = 0; Lidx18_1 < 8; Lidx18_1++) {
      if ((lidx1==0)) {
        *(data0_24960+(alu3+(Lidx18_0<<6)+Lidx18_1+alu246)) = (*(buf17+((Lidx18_0<<3)+Lidx18_1)));
      }
    }
  }
  if (alu223) {
    *(data0_24960+(alu246+128)) = (*(buf18+0));
  }
  if (alu223) {
    *(data0_24960+(alu246+129)) = (*(buf16+0));
  }
}
static void ck(cudaError_t e,const char* w){if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e));exit(2);}}
static double run_control(float* o,float* q,unsigned int* c,int n){cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);for(int i=0;i<n;i++)flash_vec_llama_score_pv_32_128_6_widekv16<<<dim3(6,32,1),dim3(32,4,1)>>>(o,q,c);cudaEventRecord(b);ck(cudaEventSynchronize(b),"control");float ms;cudaEventElapsedTime(&ms,a,b);return 1000.0*ms/n;}
static double run_candidate(float* o,float* q,unsigned int* c,int n){cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);for(int i=0;i<n;i++)flash_vec_llama_score_pv_32_128_6_widekv16_vtail8_vdimmajor<<<dim3(6,32,1),dim3(32,4,1)>>>(o,q,c);cudaEventRecord(b);ck(cudaEventSynchronize(b),"candidate");float ms;cudaEventElapsedTime(&ms,a,b);return 1000.0*ms/n;}
int main(int ac,char**av){int n=ac>1?atoi(av[1]):400,r=ac>2?atoi(av[2]):9;float *oc,*on,*q;unsigned int*c;ck(cudaMalloc(&oc,33280*4),"oc");ck(cudaMalloc(&on,24960*4),"on");ck(cudaMalloc(&q,4096*4),"q");ck(cudaMalloc(&c,1048576*4),"c");float*hq=(float*)malloc(4096*4);unsigned int*hc=(unsigned int*)malloc(1048576*4);for(int i=0;i<4096;i++)hq[i]=((i*17+3)%127-63)/256.0f;for(int i=0;i<1048576;i++)hc[i]=(i*2654435761u)^0x3c003c00u;cudaMemcpy(q,hq,4096*4,cudaMemcpyHostToDevice);cudaMemcpy(c,hc,1048576*4,cudaMemcpyHostToDevice);free(hq);free(hc);flash_vec_llama_score_pv_32_128_6_widekv16<<<dim3(6,32,1),dim3(32,4,1)>>>(oc,q,c);flash_vec_llama_score_pv_32_128_6_widekv16_vtail8_vdimmajor<<<dim3(6,32,1),dim3(32,4,1)>>>(on,q,c);ck(cudaDeviceSynchronize(),"warm");if(1){float *a=(float*)malloc(24960*4),*b=(float*)malloc(24960*4);cudaMemcpy(a,oc,24960*4,cudaMemcpyDeviceToHost);cudaMemcpy(b,on,24960*4,cudaMemcpyDeviceToHost);int ne=0;float md=0;for(int i=0;i<24960;i++){unsigned int ua=*((unsigned int*)(a+i)),ub=*((unsigned int*)(b+i));if(ua!=ub)ne++;float d=fabsf(a[i]-b[i]);if(d>md)md=d;}printf("exact_mismatches=%d max_abs=%.9g\n",ne,md);free(a);free(b);}for(int i=0;i<r;i++)printf("rep=%d control=%.6f candidate=%.6f\n",i,run_control(oc,q,c,n),run_candidate(on,q,c,n));}
