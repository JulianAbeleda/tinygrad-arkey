#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
// ---- Q4_K GGUF layout: 256 weights/block, 144 bytes ----
#define QK_K 256
typedef struct { half d; half dmin; uint8_t scales[12]; uint8_t qs[128]; } block_q4_K; // 144B
// q8_1: 32-elem blocks, int8 + fp16 (d, sum)
typedef struct { half d; half s; int8_t qs[32]; } block_q8_1; // 36B
// llama get_scale_min_k4: 6-bit scale + 6-bit min from the 12 packed bytes
__device__ __host__ static inline void get_scale_min(int j, const uint8_t*q, uint8_t*d, uint8_t*m){
  if(j<4){ *d=q[j]&63; *m=q[j+4]&63; }
  else { *d=(q[j+4]&0xF)|((q[j-4]>>6)<<4); *m=(q[j+4]>>4)|((q[j-0]>>6)<<4); }
}
__device__ static inline int sudot4(int a,int b){ return __builtin_amdgcn_sudot4(true,a,false,b,0,false); }
// ---- llama-style mul_mat_vec_q: 128 threads/row, K-blocks parallel, warp+shared reduce ----
#define NW 4
#define WS 32
__global__ void __launch_bounds__(NW*WS,1) mmvq_q4k(float* dst, const block_q4_K* x, const block_q8_1* y, int ncols, int nrows){
  const int row = blockIdx.x;                 // 1 row per block
  const int tid = threadIdx.y*WS + threadIdx.x;
  const int nbpr = ncols/QK_K;                // blocks per row (16)
  float tmp = 0.0f;
  // each thread strides over (block, inner) : 8 threads per block (QI4_K/VDR=4 groups *2)
  for(int kb = tid/8; kb < nbpr; kb += (NW*WS)/8){
    const block_q4_K* bx = x + row*nbpr + kb;
    const int sub = tid%8;                     // which 32-quant sub-chunk (0..7)
    const block_q8_1* by = y + kb*8 + sub;     // 8 q8_1 blocks per Q4_K block (256/32)
    uint8_t sc,mn; get_scale_min(sub, bx->scales, &sc, &mn);
    // 32 weights of this sub: qs nibbles. sub picks 32 of 256.
    const uint8_t* q = bx->qs + (sub/2)*32;    // each 32-byte half holds 2 sub-blocks (low/high nibble)
    int sumi=0, sumq=0;
    #pragma unroll
    for(int k=0;k<8;k++){                       // 32 weights = 8 int32 of q8
      int qb = ((const int*)q)[k];
      int nib = (sub&1) ? ((qb>>4)&0x0F0F0F0F) : (qb&0x0F0F0F0F);
      int u = ((const int*)by->qs)[k];
      sumi = __builtin_amdgcn_sudot4(false, nib, true, u, sumi, false); // q4(unsigned) . q8(signed)
      sumq = __builtin_amdgcn_sudot4(false, 0x01010101, true, u, sumq, false);
    }
    float d8 = __half2float(by->d);
    float dd = __half2float(bx->d), dm = __half2float(bx->dmin);
    tmp += d8*(dd*sc*sumi - dm*mn*sumq);
  }
  // reduce 128 -> 1
  __shared__ float sm[NW];
  for(int o=WS/2;o>0;o>>=1) tmp += __shfl_xor(tmp,o);
  if(threadIdx.x==0) sm[threadIdx.y]=tmp;
  __syncthreads();
  if(tid==0){ float s=0; for(int i=0;i<NW;i++) s+=sm[i]; dst[row]=s; }
}
int main(){
  int ncols=4096, nrows=12288; int nbpr=ncols/QK_K;
  size_t nx = (size_t)nrows*nbpr;
  std::vector<block_q4_K> hx(nx);
  std::vector<block_q8_1> hy(ncols/32);
  srand(5);
  for(auto&b:hx){ b.d=__float2half(0.01f); b.dmin=__float2half(0.005f); for(int i=0;i<12;i++)b.scales[i]=rand()&63; for(int i=0;i<128;i++)b.qs[i]=rand()&0xFF; }
  for(auto&b:hy){ b.d=__float2half(0.02f); b.s=__float2half(0.0f); for(int i=0;i<32;i++)b.qs[i]=(rand()%255)-127; }
  block_q4_K*dx; block_q8_1*dy; float*dd;
  hipMalloc(&dx,nx*sizeof(block_q4_K)); hipMalloc(&dy,hy.size()*sizeof(block_q8_1)); hipMalloc(&dd,nrows*sizeof(float));
  hipMemcpy(dx,hx.data(),nx*sizeof(block_q4_K),hipMemcpyHostToDevice);
  hipMemcpy(dy,hy.data(),hy.size()*sizeof(block_q8_1),hipMemcpyHostToDevice);
  dim3 grid(nrows), blk(WS,NW);
  for(int w=0;w<10;w++) mmvq_q4k<<<grid,blk>>>(dd,dx,dy,ncols,nrows);
  hipDeviceSynchronize();
  hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
  int iters=200; hipEventRecord(s);
  for(int i=0;i<iters;i++) mmvq_q4k<<<grid,blk>>>(dd,dx,dy,ncols,nrows);
  hipEventRecord(e); hipEventSynchronize(e);
  float ms; hipEventElapsedTime(&ms,s,e); float us=ms*1000.0f/iters;
  double bytes = nx*144.0;  // weight bytes
  double gbs = bytes/(us*1e-6)/1e9;
  printf("HANDWRITTEN Q4_K MMVQ (12288x4096): %.1f us  %.0f GB/s  (%.0f%% of 900 peak)\n", us, gbs, gbs/900.0*100);
  return 0;
}
