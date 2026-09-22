
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#define ROWS 12288
#define K 4096
#define K_BLOCKS 16
#define CONTROL_WORDS 7077888
#define CANDIDATE_WORDS 6684672
#define ROTATIONS 16
#define GUARD 32
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f=v; return u.t; }
struct __align__(8) half4 { half x,y,z,w; };
__device__ half4 make_half4(half x,half y,half z,half w) { half4 r={x,y,z,w}; return r; }

extern "C" __global__ void __launch_bounds__(32) q4k_g3_lanemap_gemv_vec_epi_resadd_12288_4096(float* data0_12288, unsigned int* data1_7077888, half* data2_4096, float* data3_12288) {
  int gidx0 = blockIdx.x; /* 12288 */
  int lidx0 = threadIdx.x; /* 32 */
  float buf0[1];
  (*(buf0+0)) = 0.0f;
  int alu1 = (lidx0>>3);
  int alu2 = (lidx0&7);
  for (int Ridx0 = 0; Ridx0 < 4; Ridx0++) {
    int alu3 = ((alu1*144)+(Ridx0*36)+(gidx0*576));
    int alu4 = (alu3+alu2);
    unsigned int val0 = (*(data1_7077888+(alu4+4)));
    unsigned int val1 = (*(data1_7077888+(alu4+12)));
    unsigned int val2 = (*(data1_7077888+(alu4+20)));
    unsigned int val3 = (*(data1_7077888+(alu4+28)));
    uint4 val4 = (*((uint4*)((data1_7077888+alu3))));
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
  float val13 = (*(data3_12288+gidx0));
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
  *(data0_12288+gidx0) = (alu51+(*(buf1+0))+val13);
}
extern "C" __global__ void __launch_bounds__(32) u4z8_g64_p256_lanemap_gemv_vec_epi_resadd_12288_4096(float* data0_12288, unsigned int* data1_6684672, half* data2_4096, float* data3_12288) {
  int gidx0 = blockIdx.x; /* 12288 */
  int lidx0 = threadIdx.x; /* 32 */
  float buf0[1];
  (*(buf0+0)) = 0.0f;
  int alu1 = (lidx0>>3);
  int alu2 = (lidx0&7);
  for (int Ridx0 = 0; Ridx0 < 4; Ridx0++) {
    int alu3 = ((alu1*136)+(Ridx0*34)+(gidx0*544));
    int alu4 = (alu3+alu2);
    unsigned int val0 = (*(data1_6684672+(alu4+2)));
    unsigned int val1 = (*(data1_6684672+(alu4+10)));
    unsigned int val2 = (*(data1_6684672+(alu4+18)));
    unsigned int val3 = (*(data1_6684672+(alu4+26)));
    uint2 val4 = (*((uint2*)((data1_6684672+alu3))));
    int alu5 = ((alu1<<10)+(Ridx0<<8)+(alu2<<2));
    half4 val5 = (*((half4*)((data2_4096+(alu5+32)))));
    half4 val6 = (*((half4*)((data2_4096+(alu5+64)))));
    half4 val7 = (*((half4*)((data2_4096+(alu5+96)))));
    half4 val8 = (*((half4*)((data2_4096+(alu5+128)))));
    half4 val9 = (*((half4*)((data2_4096+(alu5+160)))));
    half4 val10 = (*((half4*)((data2_4096+(alu5+192)))));
    half4 val11 = (*((half4*)((data2_4096+(alu5+224)))));
    half4 val12 = (*((half4*)((data2_4096+alu5))));
    float cast0 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)(((val4.x>>0u)&65535u)))))));
    float cast1 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)(((val4.x>>16u)&65535u)))))));
    float cast2 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)(((val4.y>>0u)&65535u)))))));
    float cast3 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)(((val4.y>>16u)&65535u)))))));
    unsigned int alu6 = ((val0>>0u)&252645135u);
    unsigned int alu7 = ((val0>>4u)&252645135u);
    unsigned int alu8 = ((val1>>0u)&252645135u);
    unsigned int alu9 = ((val1>>4u)&252645135u);
    unsigned int alu10 = ((val2>>0u)&252645135u);
    unsigned int alu11 = ((val2>>4u)&252645135u);
    unsigned int alu12 = ((val3>>0u)&252645135u);
    unsigned int alu13 = ((val3>>4u)&252645135u);
    (*(buf0+0)) = ((*(buf0+0))+(cast0*(((float)(((alu6>>0u)&15u)))+-8.0f)*float(val12.x))+(cast0*(((float)(((alu6>>8u)&15u)))+-8.0f)*float(val12.y))+(cast0*(((float)(((alu6>>16u)&15u)))+-8.0f)*float(val12.z))+(cast0*(((float)(((alu6>>24u)&15u)))+-8.0f)*float(val12.w))+(cast0*(((float)(((alu7>>0u)&15u)))+-8.0f)*float(val5.x))+(cast0*(((float)(((alu7>>8u)&15u)))+-8.0f)*float(val5.y))+(cast0*(((float)(((alu7>>16u)&15u)))+-8.0f)*float(val5.z))+(cast0*(((float)(((alu7>>24u)&15u)))+-8.0f)*float(val5.w))+(cast1*(((float)(((alu8>>0u)&15u)))+-8.0f)*float(val6.x))+(cast1*(((float)(((alu8>>8u)&15u)))+-8.0f)*float(val6.y))+(cast1*(((float)(((alu8>>16u)&15u)))+-8.0f)*float(val6.z))+(cast1*(((float)(((alu8>>24u)&15u)))+-8.0f)*float(val6.w))+(cast1*(((float)(((alu9>>0u)&15u)))+-8.0f)*float(val7.x))+(cast1*(((float)(((alu9>>8u)&15u)))+-8.0f)*float(val7.y))+(cast1*(((float)(((alu9>>16u)&15u)))+-8.0f)*float(val7.z))+(cast1*(((float)(((alu9>>24u)&15u)))+-8.0f)*float(val7.w))+(cast2*(((float)(((alu10>>0u)&15u)))+-8.0f)*float(val8.x))+(cast2*(((float)(((alu10>>8u)&15u)))+-8.0f)*float(val8.y))+(cast2*(((float)(((alu10>>16u)&15u)))+-8.0f)*float(val8.z))+(cast2*(((float)(((alu10>>24u)&15u)))+-8.0f)*float(val8.w))+(cast2*(((float)(((alu11>>0u)&15u)))+-8.0f)*float(val9.x))+(cast2*(((float)(((alu11>>8u)&15u)))+-8.0f)*float(val9.y))+(cast2*(((float)(((alu11>>16u)&15u)))+-8.0f)*float(val9.z))+(cast2*(((float)(((alu11>>24u)&15u)))+-8.0f)*float(val9.w))+(cast3*(((float)(((alu12>>0u)&15u)))+-8.0f)*float(val10.x))+(cast3*(((float)(((alu12>>8u)&15u)))+-8.0f)*float(val10.y))+(cast3*(((float)(((alu12>>16u)&15u)))+-8.0f)*float(val10.z))+(cast3*(((float)(((alu12>>24u)&15u)))+-8.0f)*float(val10.w))+(cast3*(((float)(((alu13>>0u)&15u)))+-8.0f)*float(val11.x))+(cast3*(((float)(((alu13>>8u)&15u)))+-8.0f)*float(val11.y))+(cast3*(((float)(((alu13>>16u)&15u)))+-8.0f)*float(val11.z))+(cast3*(((float)(((alu13>>24u)&15u)))+-8.0f)*float(val11.w)));
  }
  float val13 = (*(data3_12288+gidx0));
  float buf1[1];
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, (*(buf0+0)), 16);
  float alu17 = ((*(buf0+0))+(*(buf1+0)));
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu17, 8);
  float alu19 = (alu17+(*(buf1+0)));
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu19, 4);
  float alu21 = (alu19+(*(buf1+0)));
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu21, 2);
  float alu23 = (alu21+(*(buf1+0)));
  (*(buf1+0)) = __shfl_xor_sync(0xffffffffu, alu23, 1);
  *(data0_12288+gidx0) = (alu23+(*(buf1+0))+val13);
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

// Legal U4Z8_G64: finite non-negative FP16 scales and all offset-binary nibble patterns.
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
      for(int j=0;j<32;j++) { int wc=j/4,nib=j&3; uint32_t qw=b[2+gp*8+wc]; int n=int((qw>>(4*(g&1)+8*nib))&15)-8;
        sum += sc*double(n)*double(__half2float(x[blk*256+g*32+j]));
      }
    }
  }
  return sum+double(residual[row]);
}

static void launch(int arm,float* out,uint32_t* w,half* x,float* residual,cudaStream_t s=0) {
  if(arm==0) q4k_g3_lanemap_gemv_vec_epi_resadd_12288_4096<<<ROWS,32,0,s>>>(out,w,x,residual);
  else u4z8_g64_p256_lanemap_gemv_vec_epi_resadd_12288_4096<<<ROWS,32,0,s>>>(out,w,x,residual);
}
static double hot(int arm,float* out,uint32_t* w,half* x,float* r,int passes,cudaEvent_t a,cudaEvent_t b) {
  ck(cudaEventRecord(a),"hot start"); for(int i=0;i<passes;i++)launch(arm,out,w,x,r); ck(cudaEventRecord(b),"hot stop");ck(cudaEventSynchronize(b),"hot sync");
  float ms=0;ck(cudaEventElapsedTime(&ms,a,b),"hot elapsed");return ms*1000.0/passes;
}
static double rotated(int arm,float* out,uint32_t* ring,half* x,float* r,int passes,cudaEvent_t a,cudaEvent_t b) {
  size_t words=arm?CANDIDATE_WORDS:CONTROL_WORDS;double total=0;for(int i=0;i<passes;i++){uint32_t* w=ring+(size_t)(i%ROTATIONS)*words;ck(cudaEventRecord(a),"cold start");launch(arm,out,w,x,r);ck(cudaEventRecord(b),"cold stop");ck(cudaEventSynchronize(b),"cold sync");float ms=0;ck(cudaEventElapsedTime(&ms,a,b),"cold elapsed");total+=ms*1000.0;}return total/passes;
}
static double rotated_batch(int arm,float* out,uint32_t* ring,half* x,float* r,int passes,cudaEvent_t a,cudaEvent_t b) {
  size_t words=arm?CANDIDATE_WORDS:CONTROL_WORDS;ck(cudaEventRecord(a),"batch cold start");for(int i=0;i<passes;i++){uint32_t* w=ring+(size_t)(i%ROTATIONS)*words;launch(arm,out,w,x,r);}ck(cudaEventRecord(b),"batch cold stop");ck(cudaEventSynchronize(b),"batch cold sync");float ms=0;ck(cudaEventElapsedTime(&ms,a,b),"batch cold elapsed");return ms*1000.0/passes;
}

int main(int argc,char** argv) {
  int hp=argc>1?atoi(argv[1]):300,cp=argc>2?atoi(argv[2]):32,reps=argc>3?atoi(argv[3]):9; bool profile=argc>1&&!strcmp(argv[1],"profile"),batched=argc>4&&!strcmp(argv[4],"batch");
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
  cudaEvent_t a,b;ck(cudaEventCreate(&a),"event");ck(cudaEventCreate(&b),"event");for(int z=0;z<reps;z++){double ch,sh,cc,sc;if(!(z&1)){ch=hot(0,oc,wc,x,r,hp,a,b);sh=hot(1,os,ws,x,r,hp,a,b);cc=batched?rotated_batch(0,oc,wc,x,r,cp,a,b):rotated(0,oc,wc,x,r,cp,a,b);sc=batched?rotated_batch(1,os,ws,x,r,cp,a,b):rotated(1,os,ws,x,r,cp,a,b);}else{sh=hot(1,os,ws,x,r,hp,a,b);ch=hot(0,oc,wc,x,r,hp,a,b);sc=batched?rotated_batch(1,os,ws,x,r,cp,a,b):rotated(1,os,ws,x,r,cp,a,b);cc=batched?rotated_batch(0,oc,wc,x,r,cp,a,b):rotated(0,oc,wc,x,r,cp,a,b);}printf("rep=%d hot_control_us=%.6f hot_candidate_us=%.6f cold_control_us=%.6f cold_candidate_us=%.6f\n",z,ch,sh,cc,sc);}return all?0:5;
}
