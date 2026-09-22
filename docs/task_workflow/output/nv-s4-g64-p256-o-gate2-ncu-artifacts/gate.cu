
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
#define CONTROL_WORDS 2359296
#define CANDIDATE_WORDS 2228224
#define ROTATIONS 16
#define GUARD 32
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f=v; return u.t; }
struct __align__(8) half4 { half x,y,z,w; };
__device__ half4 make_half4(half x,half y,half z,half w) { half4 r={x,y,z,w}; return r; }

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
extern "C" __global__ void __launch_bounds__(32) s4_g64_p256_lanemap_gemv_vec_epi_resadd_4096_4096(float* data0_4096, unsigned int* data1_2228224, half* data2_4096, float* data3_4096) {
  int gidx0 = blockIdx.x; /* 4096 */
  int lidx0 = threadIdx.x; /* 32 */
  float buf0[1];
  (*(buf0+0)) = 0.0f;
  int alu1 = (lidx0>>3);
  int alu2 = (lidx0&7);
  for (int Ridx0 = 0; Ridx0 < 4; Ridx0++) {
    int alu3 = ((alu1*136)+(Ridx0*34)+(gidx0*544));
    int alu4 = (alu3+alu2);
    unsigned int val0 = (*(data1_2228224+(alu4+2)));
    unsigned int val1 = (*(data1_2228224+(alu4+10)));
    unsigned int val2 = (*(data1_2228224+(alu4+18)));
    unsigned int val3 = (*(data1_2228224+(alu4+26)));
    uint2 val4 = (*((uint2*)((data1_2228224+alu3))));
    int alu5 = ((alu1<<10)+(Ridx0<<8)+(alu2<<2));
    half4 val5 = (*((half4*)((data2_4096+(alu5+32)))));
    half4 val6 = (*((half4*)((data2_4096+(alu5+64)))));
    half4 val7 = (*((half4*)((data2_4096+(alu5+96)))));
    half4 val8 = (*((half4*)((data2_4096+(alu5+128)))));
    half4 val9 = (*((half4*)((data2_4096+(alu5+160)))));
    half4 val10 = (*((half4*)((data2_4096+(alu5+192)))));
    half4 val11 = (*((half4*)((data2_4096+(alu5+224)))));
    half4 val12 = (*((half4*)((data2_4096+alu5))));
    unsigned int alu6 = ((val0>>0u)&252645135u);
    unsigned int alu7 = ((alu6>>0u)&15u);
    int cast0 = ((int)(alu7));
    unsigned int alu8 = ((alu6>>8u)&15u);
    int cast1 = ((int)(alu8));
    unsigned int alu9 = ((alu6>>16u)&15u);
    int cast2 = ((int)(alu9));
    unsigned int alu10 = ((alu6>>24u)&15u);
    int cast3 = ((int)(alu10));
    unsigned int alu11 = ((val0>>4u)&252645135u);
    unsigned int alu12 = ((alu11>>0u)&15u);
    int cast4 = ((int)(alu12));
    unsigned int alu13 = ((alu11>>8u)&15u);
    int cast5 = ((int)(alu13));
    unsigned int alu14 = ((alu11>>16u)&15u);
    int cast6 = ((int)(alu14));
    unsigned int alu15 = ((alu11>>24u)&15u);
    int cast7 = ((int)(alu15));
    unsigned int alu16 = ((val1>>0u)&252645135u);
    unsigned int alu17 = ((alu16>>0u)&15u);
    int cast8 = ((int)(alu17));
    unsigned int alu18 = ((alu16>>8u)&15u);
    int cast9 = ((int)(alu18));
    unsigned int alu19 = ((alu16>>16u)&15u);
    int cast10 = ((int)(alu19));
    unsigned int alu20 = ((alu16>>24u)&15u);
    int cast11 = ((int)(alu20));
    unsigned int alu21 = ((val1>>4u)&252645135u);
    unsigned int alu22 = ((alu21>>0u)&15u);
    int cast12 = ((int)(alu22));
    unsigned int alu23 = ((alu21>>8u)&15u);
    int cast13 = ((int)(alu23));
    unsigned int alu24 = ((alu21>>16u)&15u);
    int cast14 = ((int)(alu24));
    unsigned int alu25 = ((alu21>>24u)&15u);
    int cast15 = ((int)(alu25));
    unsigned int alu26 = ((val2>>0u)&252645135u);
    unsigned int alu27 = ((alu26>>0u)&15u);
    int cast16 = ((int)(alu27));
    unsigned int alu28 = ((alu26>>8u)&15u);
    int cast17 = ((int)(alu28));
    unsigned int alu29 = ((alu26>>16u)&15u);
    int cast18 = ((int)(alu29));
    unsigned int alu30 = ((alu26>>24u)&15u);
    int cast19 = ((int)(alu30));
    unsigned int alu31 = ((val2>>4u)&252645135u);
    unsigned int alu32 = ((alu31>>0u)&15u);
    int cast20 = ((int)(alu32));
    unsigned int alu33 = ((alu31>>8u)&15u);
    int cast21 = ((int)(alu33));
    unsigned int alu34 = ((alu31>>16u)&15u);
    int cast22 = ((int)(alu34));
    unsigned int alu35 = ((alu31>>24u)&15u);
    int cast23 = ((int)(alu35));
    unsigned int alu36 = ((val3>>0u)&252645135u);
    unsigned int alu37 = ((alu36>>0u)&15u);
    int cast24 = ((int)(alu37));
    unsigned int alu38 = ((alu36>>8u)&15u);
    int cast25 = ((int)(alu38));
    unsigned int alu39 = ((alu36>>16u)&15u);
    int cast26 = ((int)(alu39));
    unsigned int alu40 = ((alu36>>24u)&15u);
    int cast27 = ((int)(alu40));
    unsigned int alu41 = ((val3>>4u)&252645135u);
    unsigned int alu42 = ((alu41>>0u)&15u);
    int cast28 = ((int)(alu42));
    unsigned int alu43 = ((alu41>>8u)&15u);
    int cast29 = ((int)(alu43));
    unsigned int alu44 = ((alu41>>16u)&15u);
    int cast30 = ((int)(alu44));
    unsigned int alu45 = ((alu41>>24u)&15u);
    int cast31 = ((int)(alu45));
    float cast32 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)(((val4.x>>0u)&65535u)))))));
    float cast33 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)(((val4.x>>16u)&65535u)))))));
    float cast34 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)(((val4.y>>0u)&65535u)))))));
    float cast35 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)(((val4.y>>16u)&65535u)))))));
    float alu46 = ((cast0<8)?((float)(alu7)):((float)((cast0+-16))));
    float alu47 = ((cast1<8)?((float)(alu8)):((float)((cast1+-16))));
    float alu48 = ((cast2<8)?((float)(alu9)):((float)((cast2+-16))));
    float alu49 = ((cast3<8)?((float)(alu10)):((float)((cast3+-16))));
    float alu50 = ((cast4<8)?((float)(alu12)):((float)((cast4+-16))));
    float alu51 = ((cast5<8)?((float)(alu13)):((float)((cast5+-16))));
    float alu52 = ((cast6<8)?((float)(alu14)):((float)((cast6+-16))));
    float alu53 = ((cast7<8)?((float)(alu15)):((float)((cast7+-16))));
    float alu54 = ((cast8<8)?((float)(alu17)):((float)((cast8+-16))));
    float alu55 = ((cast9<8)?((float)(alu18)):((float)((cast9+-16))));
    float alu56 = ((cast10<8)?((float)(alu19)):((float)((cast10+-16))));
    float alu57 = ((cast11<8)?((float)(alu20)):((float)((cast11+-16))));
    float alu58 = ((cast12<8)?((float)(alu22)):((float)((cast12+-16))));
    float alu59 = ((cast13<8)?((float)(alu23)):((float)((cast13+-16))));
    float alu60 = ((cast14<8)?((float)(alu24)):((float)((cast14+-16))));
    float alu61 = ((cast15<8)?((float)(alu25)):((float)((cast15+-16))));
    float alu62 = ((cast16<8)?((float)(alu27)):((float)((cast16+-16))));
    float alu63 = ((cast17<8)?((float)(alu28)):((float)((cast17+-16))));
    float alu64 = ((cast18<8)?((float)(alu29)):((float)((cast18+-16))));
    float alu65 = ((cast19<8)?((float)(alu30)):((float)((cast19+-16))));
    float alu66 = ((cast20<8)?((float)(alu32)):((float)((cast20+-16))));
    float alu67 = ((cast21<8)?((float)(alu33)):((float)((cast21+-16))));
    float alu68 = ((cast22<8)?((float)(alu34)):((float)((cast22+-16))));
    float alu69 = ((cast23<8)?((float)(alu35)):((float)((cast23+-16))));
    float alu70 = ((cast24<8)?((float)(alu37)):((float)((cast24+-16))));
    float alu71 = ((cast25<8)?((float)(alu38)):((float)((cast25+-16))));
    float alu72 = ((cast26<8)?((float)(alu39)):((float)((cast26+-16))));
    float alu73 = ((cast27<8)?((float)(alu40)):((float)((cast27+-16))));
    float alu74 = ((cast28<8)?((float)(alu42)):((float)((cast28+-16))));
    float alu75 = ((cast29<8)?((float)(alu43)):((float)((cast29+-16))));
    float alu76 = ((cast30<8)?((float)(alu44)):((float)((cast30+-16))));
    float alu77 = ((cast31<8)?((float)(alu45)):((float)((cast31+-16))));
    (*(buf0+0)) = ((*(buf0+0))+(cast32*alu46*float(val12.x))+(cast32*alu47*float(val12.y))+(cast32*alu48*float(val12.z))+(cast32*alu49*float(val12.w))+(cast32*alu50*float(val5.x))+(cast32*alu51*float(val5.y))+(cast32*alu52*float(val5.z))+(cast32*alu53*float(val5.w))+(cast33*alu54*float(val6.x))+(cast33*alu55*float(val6.y))+(cast33*alu56*float(val6.z))+(cast33*alu57*float(val6.w))+(cast33*alu58*float(val7.x))+(cast33*alu59*float(val7.y))+(cast33*alu60*float(val7.z))+(cast33*alu61*float(val7.w))+(cast34*alu62*float(val8.x))+(cast34*alu63*float(val8.y))+(cast34*alu64*float(val8.z))+(cast34*alu65*float(val8.w))+(cast34*alu66*float(val9.x))+(cast34*alu67*float(val9.y))+(cast34*alu68*float(val9.z))+(cast34*alu69*float(val9.w))+(cast35*alu70*float(val10.x))+(cast35*alu71*float(val10.y))+(cast35*alu72*float(val10.z))+(cast35*alu73*float(val10.w))+(cast35*alu74*float(val11.x))+(cast35*alu75*float(val11.y))+(cast35*alu76*float(val11.z))+(cast35*alu77*float(val11.w)));
  }
  float val13 = (*(data3_4096+gidx0));
  float buf1[1];
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, (*(buf0+0)), 16);
  float alu81 = ((*(buf0+0))+(*(buf1+0)));
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu81, 8);
  float alu83 = (alu81+(*(buf1+0)));
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu83, 4);
  float alu85 = (alu83+(*(buf1+0)));
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu85, 2);
  float alu87 = (alu85+(*(buf1+0)));
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu87, 1);
  *(data0_4096+gidx0) = (alu87+(*(buf1+0))+val13);
}

static void ck(cudaError_t e,const char* what) { if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",what,cudaGetErrorString(e));exit(2);} }
static uint32_t step(uint32_t& s) { s=1664525u*s+1013904223u; return s; }

static void fill_q4(uint32_t* w,int fixture) {
  uint32_t s=0x91427ab3u^(uint32_t(fixture)*0x9e3779b9u);
  const uint16_t hb[4]={0x2c00u,0x3000u,0x3400u,0x3800u};
  for(size_t b=0;b<(size_t)ROWS*K_BLOCKS;b++) { size_t z=b*36; int p=int((b+fixture)&3);
    w[z]=uint32_t(hb[p])|(uint32_t(hb[(p+1)&3])<<16);
    for(int i=1;i<36;i++) w[z+i]=step(s);
  }
}

// Legal S4_G64: finite non-negative FP16 scales and all signed nibble bit patterns.
static void fill_s4(uint32_t* w,half* x,float* residual,int fixture) {
  uint32_t s=0x53475032u^(uint32_t(fixture)*0x85ebca6bu);
  const uint16_t scales[8]={0x0000u,0x2800u,0x2c00u,0x3000u,0x3200u,0x3400u,0x3600u,0x3800u};
  for(size_t b=0;b<(size_t)ROWS*K_BLOCKS;b++) { size_t z=b*34;
    for(int i=0;i<2;i++) { uint16_t lo=scales[(b+i*2+fixture)%8],hi=scales[(b+i*2+1+fixture)%8]; w[z+i]=uint32_t(lo)|(uint32_t(hi)<<16); }
    for(int gp=0;gp<4;gp++) for(int wc=0;wc<8;wc++) {
      uint32_t q=fixture==0?step(s):(fixture==1?0xfedcba98u^(uint32_t(b)+uint32_t(gp*17+wc)):0x807f10e1u);
      w[z+2+gp*8+wc]=q;
    }
  }
  for(int i=0;i<K;i++) { float v=fixture==0?float(int(step(s)>>16)%511-255)/512.0f:
      fixture==1?float((i%257)-128)/256.0f:float((i%17)-8)/64.0f; x[i]=__float2half(v); }
  for(int i=0;i<ROWS;i++) residual[i]=fixture==0?float(int(step(s)>>16)%255-127)/1024.0f:
      fixture==1?float((i%31)-15)/128.0f:float((i%7)-3)/32.0f;
}

static float half_from_bits(uint16_t bits) { half h; memcpy(&h,&bits,2); return __half2float(h); }

// Independent scalar host oracle: direct logical weight indexing, double summation.
static double oracle_row(const uint32_t* w,const half* x,const float* residual,int row) {
  double sum=0.0;
  for(int blk=0;blk<K_BLOCKS;blk++) { const uint32_t* b=w+((size_t)row*K_BLOCKS+blk)*34;
    for(int g=0;g<8;g++) { int gp=g/2; uint16_t sb=uint16_t(b[gp/2]>>(16*(gp&1))); double sc=half_from_bits(sb);
      for(int j=0;j<32;j++) { int wc=j/4,nib=j&3; uint32_t qw=b[2+gp*8+wc]; int n=int((qw>>(4*(g&1)+8*nib))&15); if(n>=8)n-=16;
        sum += sc*double(n)*double(__half2float(x[blk*256+g*32+j]));
      }
    }
  }
  return sum+double(residual[row]);
}

static void launch(int arm,float* out,uint32_t* w,half* x,float* residual,cudaStream_t s=0) {
  if(arm==0) q4k_g3_lanemap_gemv_vec_epi_resadd_4096_4096<<<ROWS,32,0,s>>>(out,w,x,residual);
  else s4_g64_p256_lanemap_gemv_vec_epi_resadd_4096_4096<<<ROWS,32,0,s>>>(out,w,x,residual);
}
static double hot(int arm,float* out,uint32_t* w,half* x,float* r,int passes,cudaEvent_t a,cudaEvent_t b) {
  ck(cudaEventRecord(a),"hot start"); for(int i=0;i<passes;i++)launch(arm,out,w,x,r); ck(cudaEventRecord(b),"hot stop");ck(cudaEventSynchronize(b),"hot sync");
  float ms=0;ck(cudaEventElapsedTime(&ms,a,b),"hot elapsed");return ms*1000.0/passes;
}
static double rotated(int arm,float* out,uint32_t* ring,half* x,float* r,int passes,cudaEvent_t a,cudaEvent_t b) {
  size_t words=arm?CANDIDATE_WORDS:CONTROL_WORDS;double total=0;for(int i=0;i<passes;i++){uint32_t* w=ring+(size_t)(i%ROTATIONS)*words;ck(cudaEventRecord(a),"cold start");launch(arm,out,w,x,r);ck(cudaEventRecord(b),"cold stop");ck(cudaEventSynchronize(b),"cold sync");float ms=0;ck(cudaEventElapsedTime(&ms,a,b),"cold elapsed");total+=ms*1000.0;}return total/passes;
}

int main(int argc,char** argv) {
  int hp=argc>1?atoi(argv[1]):300,cp=argc>2?atoi(argv[2]):32,reps=argc>3?atoi(argv[3]):9; bool profile=argc>1&&!strcmp(argv[1],"profile");
  uint32_t *wc=nullptr,*ws=nullptr;half* x=nullptr;float *r=nullptr,*ocbase=nullptr,*osbase=nullptr;float *oc,*os;
  ck(cudaMalloc(&wc,(size_t)ROTATIONS*CONTROL_WORDS*4),"control weights");ck(cudaMalloc(&ws,(size_t)ROTATIONS*CANDIDATE_WORDS*4),"s4 weights");
  ck(cudaMalloc(&x,K*sizeof(half)),"x");ck(cudaMalloc(&r,ROWS*4),"residual");ck(cudaMalloc(&ocbase,(ROWS+2*GUARD)*4),"control output");ck(cudaMalloc(&osbase,(ROWS+2*GUARD)*4),"s4 output");oc=ocbase+GUARD;os=osbase+GUARD;
  uint32_t *hqc=(uint32_t*)malloc((size_t)CONTROL_WORDS*4),*hqs=(uint32_t*)malloc((size_t)CANDIDATE_WORDS*4),*check=(uint32_t*)malloc((size_t)CANDIDATE_WORDS*4);half* hx=(half*)malloc(K*2),*xcheck=(half*)malloc(K*2);float *hr=(float*)malloc(ROWS*4),*rcheck=(float*)malloc(ROWS*4),*hout=(float*)malloc((ROWS+2*GUARD)*4);
  if(!hqc||!hqs||!check||!hx||!xcheck||!hr||!rcheck||!hout){fprintf(stderr,"host allocation failed\n");return 3;}
  int all=1; const float sentinel=12345.25f;
  for(int fixture=0;fixture<3;fixture++) {fill_q4(hqc,fixture);fill_s4(hqs,hx,hr,fixture);ck(cudaMemcpy(wc,hqc,(size_t)CONTROL_WORDS*4,cudaMemcpyHostToDevice),"q4 fixture");ck(cudaMemcpy(ws,hqs,(size_t)CANDIDATE_WORDS*4,cudaMemcpyHostToDevice),"s4 fixture");ck(cudaMemcpy(x,hx,K*2,cudaMemcpyHostToDevice),"x fixture");ck(cudaMemcpy(r,hr,ROWS*4,cudaMemcpyHostToDevice),"r fixture");
    for(int i=0;i<ROWS+2*GUARD;i++)hout[i]=sentinel;ck(cudaMemcpy(ocbase,hout,(ROWS+2*GUARD)*4,cudaMemcpyHostToDevice),"q4 guards");ck(cudaMemcpy(osbase,hout,(ROWS+2*GUARD)*4,cudaMemcpyHostToDevice),"s4 guards");launch(0,oc,wc,x,r);launch(1,os,ws,x,r);ck(cudaDeviceSynchronize(),"fixture sync");
    ck(cudaMemcpy(hout,osbase,(ROWS+2*GUARD)*4,cudaMemcpyDeviceToHost),"s4 result");ck(cudaMemcpy(check,ws,(size_t)CANDIDATE_WORDS*4,cudaMemcpyDeviceToHost),"s4 readonly");ck(cudaMemcpy(xcheck,x,K*2,cudaMemcpyDeviceToHost),"x readonly");ck(cudaMemcpy(rcheck,r,ROWS*4,cudaMemcpyDeviceToHost),"r readonly");
    int guards=1,finite=1,bad=0;double maxabs=0,maxrel=0;for(int i=0;i<GUARD;i++)guards&=hout[i]==sentinel&&hout[GUARD+ROWS+i]==sentinel;for(int row=0;row<ROWS;row++){double ref=oracle_row(hqs,hx,hr,row),got=hout[GUARD+row],ae=fabs(got-ref),re=ae/fmax(1.0,fabs(ref));finite&=isfinite(got)&&isfinite(ref);bad+=!(ae<=0.02+2e-5*fabs(ref));maxabs=fmax(maxabs,ae);maxrel=fmax(maxrel,re);}int readonly=!memcmp(check,hqs,(size_t)CANDIDATE_WORDS*4)&&!memcmp(xcheck,hx,K*2)&&!memcmp(rcheck,hr,ROWS*4);
    printf("fixture=%d finite=%d guards=%d readonly=%d bad=%d max_abs=%.9g max_rel=%.9g\n",fixture,finite,guards,readonly,bad,maxabs,maxrel);all&=finite&&guards&&readonly&&bad==0;
  }
  fill_q4(hqc,0);fill_s4(hqs,hx,hr,0);for(int i=0;i<ROTATIONS;i++){ck(cudaMemcpy(wc+(size_t)i*CONTROL_WORDS,hqc,(size_t)CONTROL_WORDS*4,cudaMemcpyHostToDevice),"q4 rotation");ck(cudaMemcpy(ws+(size_t)i*CANDIDATE_WORDS,hqs,(size_t)CANDIDATE_WORDS*4,cudaMemcpyHostToDevice),"s4 rotation");}ck(cudaMemcpy(x,hx,K*2,cudaMemcpyHostToDevice),"timing x");ck(cudaMemcpy(r,hr,ROWS*4,cudaMemcpyHostToDevice),"timing residual");free(hqc);free(hqs);free(check);free(hx);free(xcheck);free(hr);free(rcheck);free(hout);
  for(int i=0;i<20;i++){launch(0,oc,wc,x,r);launch(1,os,ws,x,r);}ck(cudaDeviceSynchronize(),"warm sync");if(profile){launch(0,oc,wc,x,r);launch(1,os,ws,x,r);ck(cudaDeviceSynchronize(),"profile sync");return all?0:5;}
  cudaEvent_t a,b;ck(cudaEventCreate(&a),"event");ck(cudaEventCreate(&b),"event");for(int z=0;z<reps;z++){double ch,sh,cc,sc;if(!(z&1)){ch=hot(0,oc,wc,x,r,hp,a,b);sh=hot(1,os,ws,x,r,hp,a,b);cc=rotated(0,oc,wc,x,r,cp,a,b);sc=rotated(1,os,ws,x,r,cp,a,b);}else{sh=hot(1,os,ws,x,r,hp,a,b);ch=hot(0,oc,wc,x,r,hp,a,b);sc=rotated(1,os,ws,x,r,cp,a,b);cc=rotated(0,oc,wc,x,r,cp,a,b);}printf("rep=%d hot_control_us=%.6f hot_candidate_us=%.6f cold_control_us=%.6f cold_candidate_us=%.6f\n",z,ch,sh,cc,sc);}return all?0:5;
}
