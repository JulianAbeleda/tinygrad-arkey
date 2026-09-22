
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
constexpr int HEADS=32,ROWS=4096,THREADS=256;
static void ck(cudaError_t e,const char*w){if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e));exit(2);}}

__global__ void direct_exact(float*out,const float*part){
  for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<ROWS;row+=gridDim.x*blockDim.x){
    float acc=0.0f;
    #pragma unroll
    for(int h=0;h<HEADS;h++)acc=acc+part[h*ROWS+row];
    out[row]=acc;
  }
}
__global__ void write_scratch(float*scratch,const float*part){
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<HEADS*ROWS;i+=gridDim.x*blockDim.x)scratch[i]=part[i];
}
__global__ void combine_exact(float*out,const float*scratch){
  for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<ROWS;row+=gridDim.x*blockDim.x){
    float acc=0.0f;
    #pragma unroll
    for(int h=0;h<HEADS;h++)acc=acc+scratch[h*ROWS+row];
    out[row]=acc;
  }
}
__global__ void atomic_half(half*out,const float*part){
  const int h=blockIdx.x;
  for(int row=threadIdx.x;row<ROWS;row+=blockDim.x)atomicAdd(out+row,__float2half(part[h*ROWS+row]));
}

template<class F>static double timed(F launch,int iters,cudaStream_t s){cudaEvent_t a,b;ck(cudaEventCreate(&a),"ea");ck(cudaEventCreate(&b),"eb");for(int i=0;i<100;i++)launch();ck(cudaEventRecord(a,s),"ra");for(int i=0;i<iters;i++)launch();ck(cudaEventRecord(b,s),"rb");ck(cudaEventSynchronize(b),"sb");float ms;ck(cudaEventElapsedTime(&ms,a,b),"el");cudaEventDestroy(a);cudaEventDestroy(b);return ms*1000.0/iters;}

int main(int argc,char**argv){int reps=argc>1?atoi(argv[1]):9,iters=argc>2?atoi(argv[2]):1000;float *part,*scratch,*direct,*exact;half*atomic;
  ck(cudaMalloc(&part,(size_t)HEADS*ROWS*4),"part");ck(cudaMalloc(&scratch,(size_t)HEADS*ROWS*4),"scratch");ck(cudaMalloc(&direct,ROWS*4),"direct");ck(cudaMalloc(&exact,ROWS*4),"exact");ck(cudaMalloc(&atomic,ROWS*2),"atomic");
  std::vector<float>hp((size_t)HEADS*ROWS);for(int h=0;h<HEADS;h++)for(int r=0;r<ROWS;r++){int q=((h*131+r*17)%257)-128;hp[(size_t)h*ROWS+r]=(float)q*(h%3==0?0.0009765625f:0.00390625f);}
  ck(cudaMemcpy(part,hp.data(),hp.size()*4,cudaMemcpyHostToDevice),"part copy");cudaStream_t s;ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking),"stream");
  direct_exact<<<16,THREADS,0,s>>>(direct,part);write_scratch<<<170,THREADS,0,s>>>(scratch,part);combine_exact<<<16,THREADS,0,s>>>(exact,scratch);ck(cudaMemsetAsync(atomic,0,ROWS*2,s),"zero");atomic_half<<<HEADS,THREADS,0,s>>>(atomic,part);ck(cudaStreamSynchronize(s),"validate sync");
  std::vector<float>hd(ROWS),he(ROWS);std::vector<half>ha(ROWS);ck(cudaMemcpy(hd.data(),direct,ROWS*4,cudaMemcpyDeviceToHost),"direct copy");ck(cudaMemcpy(he.data(),exact,ROWS*4,cudaMemcpyDeviceToHost),"exact copy");ck(cudaMemcpy(ha.data(),atomic,ROWS*2,cudaMemcpyDeviceToHost),"atomic copy");
  int bitwise=memcmp(hd.data(),he.data(),ROWS*4)==0;double ss=0,se=0;float ma=0;for(int i=0;i<ROWS;i++){float av=__half2float(ha[i]),e=av-hd[i];ma=fmaxf(ma,fabsf(e));ss+=(double)e*e;se+=(double)hd[i]*hd[i];}printf("validate exact_bitwise=%d atomic_max_abs=%.9g atomic_rel_l2=%.9g\n",bitwise,ma,sqrt(ss/se));
  for(int r=0;r<reps;r++){double d=timed([&](){direct_exact<<<16,THREADS,0,s>>>(direct,part);},iters,s);double x=timed([&](){write_scratch<<<170,THREADS,0,s>>>(scratch,part);combine_exact<<<16,THREADS,0,s>>>(exact,scratch);},iters,s);double a=timed([&](){cudaMemsetAsync(atomic,0,ROWS*2,s);atomic_half<<<HEADS,THREADS,0,s>>>(atomic,part);},iters,s);printf("sample rep=%d direct_us=%.6f scratch_exact_us=%.6f atomic_half_us=%.6f\n",r,d,x,a);}
  return bitwise?0:5;
}
