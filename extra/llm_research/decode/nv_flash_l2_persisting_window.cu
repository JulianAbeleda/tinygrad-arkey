#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <vector>

__global__ void touch(const uint4 *p, size_t n, unsigned *sink) {
  unsigned v=0;
  for (size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x) {
    uint4 x=p[i]; v^=x.x^x.y^x.z^x.w;
  }
  if (v==0xdeadbeef) sink[blockIdx.x]=v;
}

static float sample(cudaStream_t s, const uint4 *foot,size_t fn,const uint4 *dist,size_t dn,unsigned *sink,bool persist,size_t win) {
  cudaCtxResetPersistingL2Cache();
  cudaStreamAttrValue attr{};
  if (persist) {
    attr.accessPolicyWindow.base_ptr=(void*)foot; attr.accessPolicyWindow.num_bytes=win;
    attr.accessPolicyWindow.hitRatio=1.0; attr.accessPolicyWindow.hitProp=cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp=cudaAccessPropertyNormal;
  }
  cudaStreamSetAttribute(s,cudaStreamAttributeAccessPolicyWindow,&attr);
  touch<<<512,256,0,s>>>(dist,dn,sink); touch<<<512,256,0,s>>>(foot,fn,sink); touch<<<512,256,0,s>>>(dist,dn,sink);
  attr={}; cudaStreamSetAttribute(s,cudaStreamAttributeAccessPolicyWindow,&attr);
  cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a,s);
  touch<<<512,256,0,s>>>(foot,fn,sink);cudaEventRecord(b,s);cudaEventSynchronize(b);
  float ms=0;cudaEventElapsedTime(&ms,a,b);cudaEventDestroy(a);cudaEventDestroy(b);return ms*1000.0f;
}

int main(int argc,char **argv) {
  int n=argc>1?atoi(argv[1]):32,warm=argc>2?atoi(argv[2]):6;cudaDeviceProp p{};cudaGetDeviceProperties(&p,0);
  size_t fb=72ull<<20,db=256ull<<20,win=std::min(fb,(size_t)p.accessPolicyMaxWindowSize);
  size_t reserve=std::min((size_t)p.persistingL2CacheMaxSize,win);cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,reserve);
  uint4 *foot,*dist;unsigned *sink;cudaMalloc(&foot,fb);cudaMalloc(&dist,db);cudaMalloc(&sink,512*4);cudaMemset(foot,0,fb);cudaMemset(dist,0,db);
  cudaStream_t s;cudaStreamCreate(&s);std::vector<float> ctl,cand;
  for(int i=0;i<n;i++){float x=sample(s,foot,fb/16,dist,db/16,sink,false,win);if(i>=warm)ctl.push_back(x);x=sample(s,foot,fb/16,dist,db/16,sink,true,win);if(i>=warm)cand.push_back(x);}
  auto med=[](std::vector<float> x){std::sort(x.begin(),x.end());return x[x.size()/2];};
  printf("{\"device\":\"%s\",\"l2_bytes\":%d,\"persisting_max_bytes\":%d,\"window_max_bytes\":%d,\"footprint_bytes\":%zu,\"window_bytes\":%zu,\"control_median_us\":%.6f,\"candidate_median_us\":%.6f}\n",
    p.name,p.l2CacheSize,p.persistingL2CacheMaxSize,p.accessPolicyMaxWindowSize,fb,win,med(ctl),med(cand));
}
