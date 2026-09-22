
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#define M 512
#define N 1024
#define K 4096
#define TM 128
#define TN 128
#define TILES ((M/TM)*(N/TN))
#define STREAM_BLOCKS 170
#define ELEMS (TM*TN)
static void ck(cudaError_t e,const char*w){if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e));exit(2);}}
extern "C" __global__ __launch_bounds__(256) void control32(const half* a,const half* b,float* out){
 int tile=blockIdx.x, mt=tile/(N/TN),nt=tile%(N/TN);
 for(int z=threadIdx.x;z<ELEMS;z+=blockDim.x){int mi=z/TN,ni=z%TN;float v=0;
  for(int k=0;k<K;k++)v=fmaf(__half2float(a[(mt*TM+mi)*K+k]),__half2float(b[(nt*TN+ni)*K+k]),v);
  out[(mt*TM+mi)*N+nt*TN+ni]=v;}}
extern "C" __global__ __launch_bounds__(256) void stream_main(const half* a,const half* b,float* partial){
 int tile=blockIdx.x%TILES, split=blockIdx.x/TILES, parts=(STREAM_BLOCKS-1-tile)/TILES+1;
 int k0=(K*split)/parts,k1=(K*(split+1))/parts,mt=tile/(N/TN),nt=tile%(N/TN);
 for(int z=threadIdx.x;z<ELEMS;z+=blockDim.x){int mi=z/TN,ni=z%TN;float v=0;
  for(int k=k0;k<k1;k++)v=fmaf(__half2float(a[(mt*TM+mi)*K+k]),__half2float(b[(nt*TN+ni)*K+k]),v);
  partial[blockIdx.x*ELEMS+z]=v;}}
extern "C" __global__ __launch_bounds__(256) void stream_fixup(const float* partial,float*out){
 int tile=blockIdx.x,mt=tile/(N/TN),nt=tile%(N/TN),parts=(STREAM_BLOCKS-1-tile)/TILES+1;
 for(int z=threadIdx.x;z<ELEMS;z+=blockDim.x){float v=0;for(int s=0;s<parts;s++)v+=partial[(tile+s*TILES)*ELEMS+z];
  int mi=z/TN,ni=z%TN;out[(mt*TM+mi)*N+nt*TN+ni]=v;}}
static float once(bool stream,const half*a,const half*b,float*p,float*out){cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);cudaEventRecord(s);
 if(stream){stream_main<<<STREAM_BLOCKS,256>>>(a,b,p);stream_fixup<<<TILES,256>>>(p,out);}else control32<<<TILES,256>>>(a,b,out);
 cudaEventRecord(e);ck(cudaEventSynchronize(e),"event");float ms;cudaEventElapsedTime(&ms,s,e);cudaEventDestroy(s);cudaEventDestroy(e);return ms*1000;}
int main(int ac,char**av){int reps=ac>1?atoi(av[1]):9;half *a,*b;float *p,*c,*o;ck(cudaMalloc(&a,M*K*2),"a");ck(cudaMalloc(&b,N*K*2),"b");
 ck(cudaMalloc(&p,(size_t)STREAM_BLOCKS*ELEMS*4),"p");ck(cudaMalloc(&c,M*N*4),"c");ck(cudaMalloc(&o,M*N*4),"o");
 unsigned short*ha=(unsigned short*)malloc(M*K*2),*hb=(unsigned short*)malloc(N*K*2);for(int i=0;i<M*K;i++)ha[i]=0x3400+(i%17);for(int i=0;i<N*K;i++)hb[i]=0x3000+(i%13);
 ck(cudaMemcpy(a,ha,M*K*2,cudaMemcpyHostToDevice),"ca");ck(cudaMemcpy(b,hb,N*K*2,cudaMemcpyHostToDevice),"cb");once(0,a,b,p,c);once(1,a,b,p,o);
 float *hc=(float*)malloc(M*N*4),*ho=(float*)malloc(M*N*4);ck(cudaMemcpy(hc,c,M*N*4,cudaMemcpyDeviceToHost),"oc");ck(cudaMemcpy(ho,o,M*N*4,cudaMemcpyDeviceToHost),"oo");double ma=0,me=0;int finite=1;for(int i=0;i<M*N;i++){double d=fabs((double)hc[i]-ho[i]);if(d>ma)ma=d;me+=d;if(!isfinite(ho[i]))finite=0;}me/=M*N;
 printf("correct finite=%d max_abs=%.9g mean_abs=%.9g tiles=%d stream_blocks=%d\n",finite,ma,me,TILES,STREAM_BLOCKS);
 for(int r=0;r<reps;r++){float x=once(0,a,b,p,c),y=once(1,a,b,p,o),z=once(0,a,b,p,c);printf("rep=%d control_a_us=%.3f stream_us=%.3f control_c_us=%.3f\n",r,x,y,z);}return 0;}
