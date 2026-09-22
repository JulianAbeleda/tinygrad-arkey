
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/atomic>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#define TOTAL_ROWS 6144
#define GROUP_WORDS 3538944
#define ROTATIONS 16
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f=v; return u.t; }
struct __align__(8) half4 { half x,y,z,w; };
__device__ half4 make_half4(half x,half y,half z,half w) { half4 r={x,y,z,w}; return r; }
__device__ __forceinline__ void q4k_row(float* data0_6144, unsigned int* data1_3538944, half* data2_4096, int gidx0, int lidx0) {
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

struct alignas(64) Control { unsigned epoch, next, done, abort, entered, exited; unsigned long long deadline; };

extern "C" __global__ __launch_bounds__(256) void persistent_qkv_live(float* out, unsigned int* words, half* x, Control* c, unsigned* gpu_complete) {
  const int lane=threadIdx.x&31;
  if(blockIdx.x==0&&threadIdx.x==0) cuda::atomic_ref<unsigned,cuda::thread_scope_system>(c->entered).store(1u,cuda::memory_order_release);
  unsigned epoch;
  do {
    epoch=cuda::atomic_ref<unsigned,cuda::thread_scope_system>(c->epoch).load(cuda::memory_order_acquire);
    if(cuda::atomic_ref<unsigned,cuda::thread_scope_system>(c->abort).load(cuda::memory_order_relaxed)) return;
    if(clock64()>c->deadline) { if(threadIdx.x==0) atomicExch_system(&c->abort,2u); return; }
    __nanosleep(64);
  } while(epoch==0);
  // Epoch publication is system-scoped, row assignment is deterministic and
  // GPU-local.  A system atomic per row is a PCIe-coherence benchmark, not a
  // persistent GEMV service design.
  const unsigned warp=blockIdx.x*(blockDim.x/32)+(threadIdx.x/32);
  const unsigned warps=gridDim.x*(blockDim.x/32);
  for(unsigned row=warp;row<TOTAL_ROWS;row+=warps) {
    q4k_row(out,words,x,(int)row,lane);
  }
  __syncthreads();
  if(threadIdx.x==0) {
    unsigned old=atomicAdd(gpu_complete,1u);
    if(old+1u==gridDim.x) { c->exited=gridDim.x; cuda::atomic_ref<unsigned,cuda::thread_scope_system>(c->done).store(epoch,cuda::memory_order_release); }
  }
}

extern "C" __global__ void __launch_bounds__(32) standalone_qkv(float* out,unsigned int* words,half* x) {
  q4k_row(out,words,x,(int)blockIdx.x,(int)threadIdx.x);
}

static void ck(cudaError_t e,const char* w){if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e));exit(2);}}
static void legal_words(unsigned* w,size_t n) {
  for(size_t i=0;i<n;i++) w[i]=(unsigned)((i*2654435761u)^0x9e3779b9u);
  for(size_t base=0;base<n;base+=36) { w[base]=0x30003000u; w[base+1]=0x10101010u; }
}
static double one_standalone(float* out,unsigned* words,half* x,cudaStream_t s) {
  cudaEvent_t a,b; ck(cudaEventCreate(&a),"event");ck(cudaEventCreate(&b),"event");
  ck(cudaEventRecord(a,s),"record"); standalone_qkv<<<TOTAL_ROWS,32,0,s>>>(out,words,x);ck(cudaEventRecord(b,s),"record");ck(cudaEventSynchronize(b),"sync");
  float ms;ck(cudaEventElapsedTime(&ms,a,b),"elapsed");cudaEventDestroy(a);cudaEventDestroy(b);return ms*1000.0;
}
static double one_persistent(float* out,unsigned* words,half* x,Control* c,unsigned* gpu_complete,int blocks,cudaStream_t s) {
  c->epoch=0;c->next=0;c->done=0;c->abort=0;c->entered=0;c->exited=0;
  int khz=0;ck(cudaDeviceGetAttribute(&khz,cudaDevAttrClockRate,0),"clock");
  c->deadline=(unsigned long long)khz*1000ull+100000000ull; // replaced in-kernel below from its own clock domain
  cudaEvent_t a,b;ck(cudaEventCreate(&a),"event");ck(cudaEventCreate(&b),"event");ck(cudaEventRecord(a,s),"record");
  // A zero deadline is not useful because clock64 is device-relative. Set a generous absolute deadline in a setup kernel-free way.
  c->deadline=~0ull;
  ck(cudaMemsetAsync(gpu_complete,0,sizeof(unsigned),s),"completion reset");
  persistent_qkv_live<<<blocks,256,0,s>>>(out,words,x,c,gpu_complete);
  auto until=std::chrono::steady_clock::now()+std::chrono::seconds(2);
  while(std::atomic_ref<unsigned>(c->entered).load(std::memory_order_acquire)==0 && std::chrono::steady_clock::now()<until) std::this_thread::yield();
  if(c->entered==0){c->abort=1;fprintf(stderr,"watchdog: resident CTAs never entered\n");exit(6);}
  auto service_start=std::chrono::steady_clock::now();
  std::atomic_ref<unsigned>(c->epoch).store(1u,std::memory_order_release);
  while(std::atomic_ref<unsigned>(c->done).load(std::memory_order_acquire)!=1u && !c->abort && std::chrono::steady_clock::now()<until) std::this_thread::yield();
  if(c->done!=1u){std::atomic_ref<unsigned>(c->abort).store(1u,std::memory_order_release);fprintf(stderr,"watchdog: done=%u abort=%u entered=%u exited=%u next=%u\n",c->done,c->abort,c->entered,c->exited,c->next);exit(7);}
  auto service_end=std::chrono::steady_clock::now();
  ck(cudaEventRecord(b,s),"record");ck(cudaEventSynchronize(b),"sync");cudaEventDestroy(a);cudaEventDestroy(b);
  return std::chrono::duration<double,std::micro>(service_end-service_start).count();
}
int main(int argc,char**argv){
  int reps=argc>1?atoi(argv[1]):9;cudaDeviceProp p;ck(cudaGetDeviceProperties(&p,0),"props");int blocks=p.multiProcessorCount-1;
  printf("sm=%d resident_blocks=%d warps=%d\n",p.multiProcessorCount,blocks,blocks*8);
  float *a,*b;unsigned* groups,*gpu_complete;half*x;Control*c;ck(cudaMalloc(&a,TOTAL_ROWS*4),"out");ck(cudaMalloc(&b,TOTAL_ROWS*4),"out");
  ck(cudaMalloc(&groups,(size_t)ROTATIONS*GROUP_WORDS*4),"groups");ck(cudaMalloc(&x,4096*2),"x");ck(cudaMalloc(&gpu_complete,sizeof(unsigned)),"gpu completion");ck(cudaHostAlloc(&c,sizeof(Control),cudaHostAllocMapped),"control");
  unsigned* hw=(unsigned*)malloc((size_t)ROTATIONS*GROUP_WORDS*4);half*hx=(half*)malloc(4096*2);legal_words(hw,(size_t)ROTATIONS*GROUP_WORDS);for(int i=0;i<4096;i++)hx[i]=__float2half(((i%257)-128)*.03125f);
  ck(cudaMemcpy(groups,hw,(size_t)ROTATIONS*GROUP_WORDS*4,cudaMemcpyHostToDevice),"weights");ck(cudaMemcpy(x,hx,4096*2,cudaMemcpyHostToDevice),"x");free(hw);free(hx);cudaStream_t s;ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking),"stream");
  standalone_qkv<<<TOTAL_ROWS,32,0,s>>>(a,groups,x);one_persistent(b,groups,x,c,gpu_complete,blocks,s);ck(cudaStreamSynchronize(s),"warm");
  float *ha=(float*)malloc(TOTAL_ROWS*4),*hb=(float*)malloc(TOTAL_ROWS*4);ck(cudaMemcpy(ha,a,TOTAL_ROWS*4,cudaMemcpyDeviceToHost),"ref");ck(cudaMemcpy(hb,b,TOTAL_ROWS*4,cudaMemcpyDeviceToHost),"got");
  int exact=memcmp(ha,hb,TOTAL_ROWS*4)==0,finite=1;for(int i=0;i<TOTAL_ROWS;i++)if(!isfinite(ha[i])||!isfinite(hb[i]))finite=0;printf("bitwise=%d finite=%d\n",exact,finite);free(ha);free(hb);
  for(int r=0;r<reps;r++){double sh=0,ph=0,sc=0,pc=0;for(int i=0;i<32;i++){sh+=one_standalone(a,groups,x,s);ph+=one_persistent(b,groups,x,c,gpu_complete,blocks,s);}for(int i=0;i<16;i++){unsigned* g=groups+(size_t)i*GROUP_WORDS;sc+=one_standalone(a,g,x,s);pc+=one_persistent(b,g,x,c,gpu_complete,blocks,s);}printf("rep=%d standalone_hot=%.6f persistent_hot=%.6f standalone_cold=%.6f persistent_cold=%.6f\n",r,sh/32,ph/32,sc/16,pc/16);}
  return exact&&finite?0:5;
}
