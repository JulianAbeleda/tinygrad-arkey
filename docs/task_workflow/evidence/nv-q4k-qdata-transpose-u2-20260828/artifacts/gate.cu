
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
extern "C" __global__ void __launch_bounds__(32) q4k_g3_lanemap_gemv_qdata_t_u2_epi_resadd_4096_4096(float* data0_4096, unsigned int* data1_2359296, half* data2_4096, float* data3_4096) {
  int gidx0 = blockIdx.x; /* 4096 */
  int lidx0 = threadIdx.x; /* 32 */
  float buf0[1];
  (*(buf0+0)) = 0.0f;
  int alu1 = (lidx0>>3);
  int alu2 = ((lidx0&7)<<2);
  for (int Ridx0 = 0; Ridx0 < 4; Ridx0++) {
    int alu3 = ((alu1*144)+(Ridx0*36)+(gidx0*576));
    int alu4 = (alu3+alu2);
    uint2 val0 = (*((uint2*)((data1_2359296+(alu4+4)))));
    uint2 val1 = (*((uint2*)((data1_2359296+(alu4+6)))));
    uint4 val2 = (*((uint4*)((data1_2359296+alu3))));
    int alu5 = ((alu1<<10)+(Ridx0<<8)+alu2);
    half4 val3 = (*((half4*)((data2_4096+(alu5+32)))));
    half4 val4 = (*((half4*)((data2_4096+(alu5+64)))));
    half4 val5 = (*((half4*)((data2_4096+(alu5+96)))));
    half4 val6 = (*((half4*)((data2_4096+(alu5+128)))));
    half4 val7 = (*((half4*)((data2_4096+(alu5+160)))));
    half4 val8 = (*((half4*)((data2_4096+(alu5+192)))));
    half4 val9 = (*((half4*)((data2_4096+(alu5+224)))));
    half4 val10 = (*((half4*)((data2_4096+alu5))));
    float cast0 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val2.x&65535u)))))));
    float cast1 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)(((val2.x>>16u)&65535u)))))));
    unsigned int alu6 = (val2.z>>0u);
    unsigned int alu7 = (val2.w>>0u);
    unsigned int alu8 = (val2.z>>8u);
    unsigned int alu9 = (val2.w>>8u);
    unsigned int alu10 = (val2.z>>16u);
    unsigned int alu11 = (val2.w>>16u);
    unsigned int alu12 = (val2.z>>24u);
    unsigned int alu13 = (val2.w>>24u);
    unsigned int alu14 = (val2.y>>0u);
    unsigned int alu15 = (val2.y>>8u);
    unsigned int alu16 = (val2.y>>16u);
    unsigned int alu17 = (val2.y>>24u);
    unsigned int alu18 = ((val0.x>>0u)&252645135u);
    unsigned int alu19 = ((val0.x>>4u)&252645135u);
    unsigned int alu20 = ((val0.y>>0u)&252645135u);
    unsigned int alu21 = ((val0.y>>4u)&252645135u);
    unsigned int alu22 = ((val1.x>>0u)&252645135u);
    unsigned int alu23 = ((val1.x>>4u)&252645135u);
    unsigned int alu24 = ((val1.y>>0u)&252645135u);
    unsigned int alu25 = ((val1.y>>4u)&252645135u);
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
    (*(buf0+0)) = ((*(buf0+0))+(((alu26*((float)(((alu18>>0u)&15u))))-alu27)*float(val10.x))+(((alu26*((float)(((alu18>>8u)&15u))))-alu27)*float(val10.y))+(((alu26*((float)(((alu18>>16u)&15u))))-alu27)*float(val10.z))+(((alu26*((float)(((alu18>>24u)&15u))))-alu27)*float(val10.w))+(((alu28*((float)(((alu19>>0u)&15u))))-alu29)*float(val3.x))+(((alu28*((float)(((alu19>>8u)&15u))))-alu29)*float(val3.y))+(((alu28*((float)(((alu19>>16u)&15u))))-alu29)*float(val3.z))+(((alu28*((float)(((alu19>>24u)&15u))))-alu29)*float(val3.w))+(((alu30*((float)(((alu20>>0u)&15u))))-alu31)*float(val4.x))+(((alu30*((float)(((alu20>>8u)&15u))))-alu31)*float(val4.y))+(((alu30*((float)(((alu20>>16u)&15u))))-alu31)*float(val4.z))+(((alu30*((float)(((alu20>>24u)&15u))))-alu31)*float(val4.w))+(((alu32*((float)(((alu21>>0u)&15u))))-alu33)*float(val5.x))+(((alu32*((float)(((alu21>>8u)&15u))))-alu33)*float(val5.y))+(((alu32*((float)(((alu21>>16u)&15u))))-alu33)*float(val5.z))+(((alu32*((float)(((alu21>>24u)&15u))))-alu33)*float(val5.w))+(((alu34*((float)(((alu22>>0u)&15u))))-alu35)*float(val6.x))+(((alu34*((float)(((alu22>>8u)&15u))))-alu35)*float(val6.y))+(((alu34*((float)(((alu22>>16u)&15u))))-alu35)*float(val6.z))+(((alu34*((float)(((alu22>>24u)&15u))))-alu35)*float(val6.w))+(((alu36*((float)(((alu23>>0u)&15u))))-alu37)*float(val7.x))+(((alu36*((float)(((alu23>>8u)&15u))))-alu37)*float(val7.y))+(((alu36*((float)(((alu23>>16u)&15u))))-alu37)*float(val7.z))+(((alu36*((float)(((alu23>>24u)&15u))))-alu37)*float(val7.w))+(((alu38*((float)(((alu24>>0u)&15u))))-alu39)*float(val8.x))+(((alu38*((float)(((alu24>>8u)&15u))))-alu39)*float(val8.y))+(((alu38*((float)(((alu24>>16u)&15u))))-alu39)*float(val8.z))+(((alu38*((float)(((alu24>>24u)&15u))))-alu39)*float(val8.w))+(((alu40*((float)(((alu25>>0u)&15u))))-alu41)*float(val9.x))+(((alu40*((float)(((alu25>>8u)&15u))))-alu41)*float(val9.y))+(((alu40*((float)(((alu25>>16u)&15u))))-alu41)*float(val9.z))+(((alu40*((float)(((alu25>>24u)&15u))))-alu41)*float(val9.w)));
  }
  float val11 = (*(data3_4096+gidx0));
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
  *(data0_4096+gidx0) = (alu51+(*(buf1+0))+val11);
}

static void ck(cudaError_t e,const char* what) { if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",what,cudaGetErrorString(e));exit(2);} }
static uint32_t step(uint32_t& s) { s=1664525u*s+1013904223u; return s; }

static void transpose_blocks(const uint32_t* src,uint32_t* dst) {
  for(size_t b=0;b<(size_t)ROWS*K_BLOCKS;b++) {
    size_t base=b*36;
    for(int i=0;i<4;i++) dst[base+i]=src[base+i];
    for(int gp=0;gp<4;gp++) for(int wc=0;wc<8;wc++)
      dst[base+4+wc*4+gp]=src[base+4+gp*8+wc];
  }
}

// All fixtures use finite positive fp16 d/dmin metadata and legal packed Q/scales.
static void fill_fixture(uint32_t* w,half* x,float* residual,int fixture) {
  uint32_t state=0x1234567u ^ (uint32_t(fixture)*0x9e3779b9u);
  const uint16_t dbits[4]={0x2c00u,0x3000u,0x3400u,0x3800u};
  const uint16_t mbits[4]={0x2800u,0x2c00u,0x3000u,0x3400u};
  for(size_t b=0;b<(size_t)ROWS*K_BLOCKS;b++) {
    size_t base=b*36; int p=int((b+fixture)%4);
    w[base]=uint32_t(dbits[p]) | (uint32_t(mbits[(p+1)&3])<<16);
    if(fixture==0) {
      w[base+1]=step(state); w[base+2]=step(state); w[base+3]=step(state);
      for(int i=4;i<36;i++) w[base+i]=step(state);
    } else if(fixture==1) {
      w[base+1]=0x01020304u^(uint32_t)b; w[base+2]=0x10203040u+(uint32_t)b; w[base+3]=0x3f2f1f0fu;
      for(int i=4;i<36;i++) w[base+i]=0x01234567u*uint32_t(i)+(uint32_t)b*0x11111111u;
    } else {
      w[base+1]=0x3f3f3f3fu; w[base+2]=0x15151515u; w[base+3]=0x2a2a2a2au;
      for(int i=4;i<36;i++) w[base+i]=(i&1)?0xfedcba98u:0x76543210u;
    }
  }
  for(int i=0;i<K;i++) {
    float v=fixture==0?float((int(step(state)>>16)%511)-255)/512.0f:
            fixture==1?float((i%257)-128)/256.0f:float((i%17)-8)/64.0f;
    x[i]=__float2half(v);
  }
  for(int i=0;i<ROWS;i++) residual[i]=fixture==0?float((int(step(state)>>16)%255)-127)/1024.0f:
                                      fixture==1?float((i%31)-15)/128.0f:float((i%7)-3)/32.0f;
}

static void launch(int arm,float* out,uint32_t* w,half* x,float* residual,cudaStream_t s=0) {
  if(arm==0) q4k_g3_lanemap_gemv_vec_epi_resadd_4096_4096<<<ROWS,32,0,s>>>(out,w,x,residual);
  else q4k_g3_lanemap_gemv_qdata_t_u2_epi_resadd_4096_4096<<<ROWS,32,0,s>>>(out,w,x,residual);
}

static double hot(int arm,float* out,uint32_t* w,half* x,float* residual,int passes,cudaEvent_t start,cudaEvent_t stop) {
  ck(cudaEventRecord(start),"hot start");
  for(int i=0;i<passes;i++) launch(arm,out,w,x,residual);
  ck(cudaEventRecord(stop),"hot stop"); ck(cudaEventSynchronize(stop),"hot sync");
  float ms=0; ck(cudaEventElapsedTime(&ms,start,stop),"hot elapsed"); return ms*1000.0/passes;
}

static double rotated(int arm,float* out,uint32_t* rotations,half* x,float* residual,int passes,cudaEvent_t start,cudaEvent_t stop) {
  double total=0.0;
  for(int i=0;i<passes;i++) {
    uint32_t* w=rotations+(size_t)(i%ROTATIONS)*WORDS;
    ck(cudaEventRecord(start),"cold start"); launch(arm,out,w,x,residual); ck(cudaEventRecord(stop),"cold stop");
    ck(cudaEventSynchronize(stop),"cold sync"); float ms=0; ck(cudaEventElapsedTime(&ms,start,stop),"cold elapsed"); total+=ms*1000.0;
  }
  return total/passes;
}

int main(int argc,char** argv) {
  int hot_passes=argc>1?atoi(argv[1]):300,cold_passes=argc>2?atoi(argv[2]):32,reps=argc>3?atoi(argv[3]):9;
  bool profile=argc>1 && !strcmp(argv[1],"profile");
  uint32_t *wc=nullptr,*wt=nullptr; half* x=nullptr; float *residual=nullptr,*outc=nullptr,*outt=nullptr;
  ck(cudaMalloc(&wc,(size_t)ROTATIONS*WORDS*4),"control weights");
  ck(cudaMalloc(&wt,(size_t)ROTATIONS*WORDS*4),"candidate weights");
  ck(cudaMalloc(&x,K*sizeof(half)),"x"); ck(cudaMalloc(&residual,ROWS*sizeof(float)),"residual");
  ck(cudaMalloc(&outc,ROWS*sizeof(float)),"control output"); ck(cudaMalloc(&outt,ROWS*sizeof(float)),"candidate output");
  uint32_t* hw=(uint32_t*)malloc((size_t)WORDS*4),*ht=(uint32_t*)malloc((size_t)WORDS*4);
  half* hx=(half*)malloc(K*sizeof(half)); float* hr=(float*)malloc(ROWS*sizeof(float));
  if(!hw||!ht||!hx||!hr){fprintf(stderr,"host allocation failed\n");return 3;}

  int exact_all=1;
  for(int fixture=0;fixture<3;fixture++) {
    fill_fixture(hw,hx,hr,fixture); transpose_blocks(hw,ht);
    ck(cudaMemcpy(wc,hw,(size_t)WORDS*4,cudaMemcpyHostToDevice),"control fixture");
    ck(cudaMemcpy(wt,ht,(size_t)WORDS*4,cudaMemcpyHostToDevice),"candidate fixture");
    ck(cudaMemcpy(x,hx,K*sizeof(half),cudaMemcpyHostToDevice),"x fixture");
    ck(cudaMemcpy(residual,hr,ROWS*sizeof(float),cudaMemcpyHostToDevice),"residual fixture");
    launch(0,outc,wc,x,residual); launch(1,outt,wt,x,residual); ck(cudaDeviceSynchronize(),"fixture sync");
    float *hc=(float*)malloc(ROWS*4),*htout=(float*)malloc(ROWS*4);
    ck(cudaMemcpy(hc,outc,ROWS*4,cudaMemcpyDeviceToHost),"control result"); ck(cudaMemcpy(htout,outt,ROWS*4,cudaMemcpyDeviceToHost),"candidate result");
    int mismatch=0,finite=1; double max_abs=0.0;
    for(int i=0;i<ROWS;i++){uint32_t a,b;memcpy(&a,hc+i,4);memcpy(&b,htout+i,4);mismatch+=a!=b;finite&=isfinite(hc[i])&&isfinite(htout[i]);max_abs=fmax(max_abs,fabs(double(hc[i])-double(htout[i])));}
    printf("fixture=%d finite=%d mismatched_words=%d max_abs=%.9g\n",fixture,finite,mismatch,max_abs);
    exact_all &= finite && mismatch==0; free(hc); free(htout);
  }

  // Fixture zero is the timed production-shaped legal/random input.  Replicate
  // both physical layouts across a >L2 ring without including copies in timing.
  fill_fixture(hw,hx,hr,0); transpose_blocks(hw,ht);
  for(int r=0;r<ROTATIONS;r++) {
    ck(cudaMemcpy(wc+(size_t)r*WORDS,hw,(size_t)WORDS*4,cudaMemcpyHostToDevice),"control rotation");
    ck(cudaMemcpy(wt+(size_t)r*WORDS,ht,(size_t)WORDS*4,cudaMemcpyHostToDevice),"candidate rotation");
  }
  ck(cudaMemcpy(x,hx,K*sizeof(half),cudaMemcpyHostToDevice),"timing x"); ck(cudaMemcpy(residual,hr,ROWS*sizeof(float),cudaMemcpyHostToDevice),"timing residual");
  free(hw);free(ht);free(hx);free(hr);
  for(int i=0;i<20;i++){launch(0,outc,wc,x,residual);launch(1,outt,wt,x,residual);} ck(cudaDeviceSynchronize(),"warm sync");
  if(profile){launch(0,outc,wc,x,residual);launch(1,outt,wt,x,residual);ck(cudaDeviceSynchronize(),"profile sync");return exact_all?0:5;}
  cudaEvent_t start,stop;ck(cudaEventCreate(&start),"event");ck(cudaEventCreate(&stop),"event");
  for(int r=0;r<reps;r++) {
    double ch,th,cc,tc;
    if((r&1)==0) {ch=hot(0,outc,wc,x,residual,hot_passes,start,stop);th=hot(1,outt,wt,x,residual,hot_passes,start,stop);cc=rotated(0,outc,wc,x,residual,cold_passes,start,stop);tc=rotated(1,outt,wt,x,residual,cold_passes,start,stop);}
    else {th=hot(1,outt,wt,x,residual,hot_passes,start,stop);ch=hot(0,outc,wc,x,residual,hot_passes,start,stop);tc=rotated(1,outt,wt,x,residual,cold_passes,start,stop);cc=rotated(0,outc,wc,x,residual,cold_passes,start,stop);}
    printf("rep=%d hot_control_us=%.6f hot_candidate_us=%.6f cold_control_us=%.6f cold_candidate_us=%.6f\n",r,ch,th,cc,tc);
  }
  return exact_all?0:5;
}
