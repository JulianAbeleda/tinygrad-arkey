
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
namespace cg = cooperative_groups;

constexpr int WIDTH=4096;
constexpr int THREADS=128;

static void ck(cudaError_t e,const char*w) {
  if(e!=cudaSuccess) { std::fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e)); std::exit(2); }
}

template<int CS> __global__ __cluster_dims__(CS,1,1)
void cluster_barrier(float*out) {
  cg::cluster_group cluster=cg::this_cluster();
  float v=(float)(blockIdx.x+threadIdx.x);
  cluster.sync();
  out[(size_t)blockIdx.x*blockDim.x+threadIdx.x]=v;
}

template<int CS> __global__ __cluster_dims__(CS,1,1)
void dsm_handoff(float*out,const float*in) {
  constexpr int SEG=WIDTH/CS;
  __shared__ float tile[SEG];
  cg::cluster_group cluster=cg::this_cluster();
  const int rank=cluster.block_rank();
  for(int i=threadIdx.x;i<SEG;i+=blockDim.x) tile[i]=in[rank*SEG+i]+(float)rank*0.25f;
  cluster.sync();
  float acc=0.0f;
  for(int i=threadIdx.x;i<WIDTH;i+=blockDim.x) {
    const int remote_rank=i/SEG, offset=i-remote_rank*SEG;
    float*remote=cluster.map_shared_rank(tile,remote_rank);
    acc+=remote[offset];
  }
  out[(size_t)blockIdx.x*blockDim.x+threadIdx.x]=acc;
  cluster.sync();
}

template<int CS> __global__ __cluster_dims__(CS,1,1)
void global_handoff(float*out,float*stage,const float*in) {
  constexpr int SEG=WIDTH/CS;
  cg::cluster_group cluster=cg::this_cluster();
  const int rank=cluster.block_rank(), cid=blockIdx.x/CS;
  float*base=stage+(size_t)cid*WIDTH;
  for(int i=threadIdx.x;i<SEG;i+=blockDim.x) base[rank*SEG+i]=in[rank*SEG+i]+(float)rank*0.25f;
  cluster.sync();
  float acc=0.0f;
  for(int i=threadIdx.x;i<WIDTH;i+=blockDim.x) acc+=base[i];
  out[(size_t)blockIdx.x*blockDim.x+threadIdx.x]=acc;
  cluster.sync();
}

template<int CS> __global__ __cluster_dims__(CS,1,1)
void direct_global(float*out,const float*in) {
  cg::cluster_group cluster=cg::this_cluster();
  float acc=0.0f;
  for(int i=threadIdx.x;i<WIDTH;i+=blockDim.x) acc+=in[i/ (WIDTH/CS) * (WIDTH/CS) + i%(WIDTH/CS)] + (float)(i/(WIDTH/CS))*0.25f;
  out[(size_t)blockIdx.x*blockDim.x+threadIdx.x]=acc;
  cluster.sync();
}

template<class F> static double timed(F launch,int iters,cudaStream_t stream) {
  cudaEvent_t a,b; ck(cudaEventCreate(&a),"event-a"); ck(cudaEventCreate(&b),"event-b");
  for(int i=0;i<100;i++) launch();
  ck(cudaEventRecord(a,stream),"record-a");
  for(int i=0;i<iters;i++) launch();
  ck(cudaEventRecord(b,stream),"record-b"); ck(cudaEventSynchronize(b),"sync-b");
  float ms=0; ck(cudaEventElapsedTime(&ms,a,b),"elapsed");
  cudaEventDestroy(a);cudaEventDestroy(b);return ms*1000.0/iters;
}

template<int CS> static void run_size(int clusters,int reps,int iters,float*out,float*stage,float*in,cudaStream_t stream) {
  const int blocks=clusters*CS;
  dsm_handoff<CS><<<blocks,THREADS,0,stream>>>(out,in); ck(cudaGetLastError(),"dsm warm");
  global_handoff<CS><<<blocks,THREADS,0,stream>>>(out,stage,in); ck(cudaGetLastError(),"global warm");
  direct_global<CS><<<blocks,THREADS,0,stream>>>(out,in); ck(cudaGetLastError(),"direct warm");
  ck(cudaStreamSynchronize(stream),"warm sync");
  for(int r=0;r<reps;r++) {
    const double barrier=timed([&](){cluster_barrier<CS><<<blocks,THREADS,0,stream>>>(out);},iters,stream);
    const double dsm=timed([&](){dsm_handoff<CS><<<blocks,THREADS,0,stream>>>(out,in);},iters,stream);
    const double global=timed([&](){global_handoff<CS><<<blocks,THREADS,0,stream>>>(out,stage,in);},iters,stream);
    const double direct=timed([&](){direct_global<CS><<<blocks,THREADS,0,stream>>>(out,in);},iters,stream);
    std::printf("sample cs=%d clusters=%d rep=%d barrier_us=%.6f dsm_us=%.6f global_us=%.6f direct_us=%.6f\n",
      CS,clusters,r,barrier,dsm,global,direct);
  }
}

int main(int argc,char**argv) {
  const int reps=argc>1?std::atoi(argv[1]):9, iters=argc>2?std::atoi(argv[2]):1000;
  int cluster_launch=0,sm_count=0;cudaDeviceProp p{};
  ck(cudaGetDeviceProperties(&p,0),"properties");
  ck(cudaDeviceGetAttribute(&cluster_launch,cudaDevAttrClusterLaunch,0),"cluster attr");
  ck(cudaDeviceGetAttribute(&sm_count,cudaDevAttrMultiProcessorCount,0),"sm attr");
  std::printf("device name=%s cc=%d.%d sm=%d cluster_launch=%d\n",p.name,p.major,p.minor,sm_count,cluster_launch);
  if(!cluster_launch) return 4;
  const int max_blocks=sm_count*4;
  float *out,*stage,*in;cudaStream_t stream;
  ck(cudaMalloc(&out,(size_t)max_blocks*THREADS*sizeof(float)),"out");
  ck(cudaMalloc(&stage,(size_t)max_blocks*WIDTH*sizeof(float)),"stage");
  ck(cudaMalloc(&in,WIDTH*sizeof(float)),"in");
  float h[WIDTH];for(int i=0;i<WIDTH;i++)h[i]=(float)((i%127)-63)*0.03125f;
  ck(cudaMemcpy(in,h,sizeof(h),cudaMemcpyHostToDevice),"input");
  ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"stream");
  run_size<2>((sm_count+1)/2,reps,iters,out,stage,in,stream);
  run_size<4>((sm_count+3)/4,reps,iters,out,stage,in,stream);
  run_size<8>((sm_count+7)/8,reps,iters,out,stage,in,stream);
  ck(cudaStreamSynchronize(stream),"final sync");
  return 0;
}
