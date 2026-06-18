#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define QK_K 256
typedef struct { half d; half dmin; uint8_t scales[12]; uint8_t qs[128]; } block_q4_K;
__device__ __host__ static inline void get_scale_min(int j,const uint8_t*q,uint8_t*d,uint8_t*m){
  if(j<4){*d=q[j]&63;*m=q[j+4]&63;}else{*d=(q[j+4]&0xF)|((q[j-4]>>6)<<4);*m=(q[j+4]>>4)|((q[j]>>6)<<4);}
}
#define NW 4
#define WS 32
// W4A16: fp16 activation x int4 weight, dequant to fp, FMA. No q8 pack. (byte-identical to fp path)
__global__ void __launch_bounds__(NW*WS,1) w4a16_q4k(float* dst,const block_q4_K* x,const half* y,int ncols,int nrows){
  const int row=blockIdx.x; const int tid=threadIdx.y*WS+threadIdx.x; const int nbpr=ncols/QK_K;
  float acc=0.0f;
  for(int kb=tid/8; kb<nbpr; kb+=(NW*WS)/8){
    const block_q4_K* bx=x+row*nbpr+kb; const int sub=tid%8;
    uint8_t sc,mn; get_scale_min(sub,bx->scales,&sc,&mn);
    float dd=__half2float(bx->d)*sc, dm=__half2float(bx->dmin)*mn;
    const uint8_t* q=bx->qs+(sub/2)*32; const half* ya=y+kb*QK_K+sub*32;
    #pragma unroll
    for(int k=0;k<32;k++){
      int qb=q[k]; int nib=(sub&1)?((qb>>4)&0xF):(qb&0xF);
      float w=dd*nib-dm;
      acc+=w*__half2float(ya[k]);
    }
  }
  __shared__ float sm[NW];
  for(int o=WS/2;o>0;o>>=1) acc+=__shfl_xor(acc,o);
  if(threadIdx.x==0) sm[threadIdx.y]=acc;
  __syncthreads();
  if(tid==0){float s=0;for(int i=0;i<NW;i++)s+=sm[i];dst[row]=s;}
}
int main(){
  int ncols=4096,nrows=12288,nbpr=ncols/QK_K; size_t nx=(size_t)nrows*nbpr;
  std::vector<block_q4_K> hx(nx); std::vector<half> hy(ncols);
  srand(5);
  for(auto&b:hx){b.d=__float2half(0.01f);b.dmin=__float2half(0.005f);for(int i=0;i<12;i++)b.scales[i]=rand()&63;for(int i=0;i<128;i++)b.qs[i]=rand()&0xFF;}
  for(auto&v:hy) v=__float2half((rand()%200-100)/100.0f);
  block_q4_K*dx; half*dy; float*dd;
  hipMalloc(&dx,nx*sizeof(block_q4_K)); hipMalloc(&dy,ncols*sizeof(half)); hipMalloc(&dd,nrows*sizeof(float));
  hipMemcpy(dx,hx.data(),nx*sizeof(block_q4_K),hipMemcpyHostToDevice);
  hipMemcpy(dy,hy.data(),ncols*sizeof(half),hipMemcpyHostToDevice);
  dim3 grid(nrows),blk(WS,NW);
  for(int w=0;w<10;w++) w4a16_q4k<<<grid,blk>>>(dd,dx,dy,ncols,nrows);
  hipDeviceSynchronize();
  hipEvent_t s,e;hipEventCreate(&s);hipEventCreate(&e); int it=200; hipEventRecord(s);
  for(int i=0;i<it;i++) w4a16_q4k<<<grid,blk>>>(dd,dx,dy,ncols,nrows);
  hipEventRecord(e);hipEventSynchronize(e); float ms;hipEventElapsedTime(&ms,s,e); float us=ms*1000.0f/it;
  double gbs=nx*144.0/(us*1e-6)/1e9;
  printf("HANDWRITTEN W4A16 (fp16xint4, no q8): %.1f us  %.0f GB/s  (%.0f%% peak)\n",us,gbs,gbs/900.0*100);
  return 0;
}
