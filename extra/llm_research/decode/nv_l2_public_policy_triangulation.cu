#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

__global__ void touch_normal(const uint4 *p,size_t n,unsigned *sink) {
  unsigned v=0;for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){uint4 x=p[i];v^=x.x^x.y^x.z^x.w;}
  if(v==0xdeadbeefu)sink[blockIdx.x]=v;
}
__global__ void touch_descriptor(const uint4 *p,size_t n,unsigned *sink,unsigned long long policy) {
  unsigned v=0;for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){uint4 x;
    asm volatile("ld.global.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;":"=r"(x.x),"=r"(x.y),"=r"(x.z),"=r"(x.w):"l"(p+i),"l"(policy));v^=x.x^x.y^x.z^x.w;}
  if(v==0xdeadbeefu)sink[blockIdx.x]=v;
}
static unsigned long long interleave_desc(float ratio) {
  unsigned num=(unsigned)((ratio-1.1920928955078125e-7f)*16.0f);return ((unsigned long long)num<<52)|(0ull<<56)|(2ull<<57)|(2ull<<59);
}
static unsigned long long range_desc(const void *ptr,uint32_t primary,uint32_t total) {
  unsigned lg=0;for(uint32_t x=total-1;x;x>>=1)lg++;unsigned benum=lg>19?lg-19:0,blog=12+benum,bsize=1u<<blog;
  unsigned start=(unsigned)((uintptr_t)ptr>>blog),end=(unsigned)(((uintptr_t)ptr+primary+bsize-1)>>blog),count=std::clamp(end-start,1u,127u);
  return ((unsigned long long)count<<37)|((unsigned long long)start<<44)|((unsigned long long)benum<<52)|(0ull<<56)|(2ull<<57)|(3ull<<59);
}
static float one(cudaStream_t s,const uint4 *foot,size_t fn,const uint4 *dist,size_t dn,unsigned *sink,const std::string &arm,size_t reserve,size_t win,float ratio) {
  cudaCtxResetPersistingL2Cache();cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,reserve);cudaStreamAttrValue attr{};
  bool apw=arm=="apw",desc=arm=="interleave"||arm=="range";
  if(apw){attr.accessPolicyWindow.base_ptr=(void*)foot;attr.accessPolicyWindow.num_bytes=win;attr.accessPolicyWindow.hitRatio=ratio;attr.accessPolicyWindow.hitProp=cudaAccessPropertyPersisting;attr.accessPolicyWindow.missProp=cudaAccessPropertyNormal;cudaStreamSetAttribute(s,cudaStreamAttributeAccessPolicyWindow,&attr);}
  touch_normal<<<512,256,0,s>>>(dist,dn,sink);
  if(desc){unsigned long long d=arm=="range"?range_desc(foot,(uint32_t)(win*ratio),(uint32_t)win):interleave_desc(ratio);touch_descriptor<<<512,256,0,s>>>(foot,fn,sink,d);}
  else touch_normal<<<512,256,0,s>>>(foot,fn,sink);
  touch_normal<<<512,256,0,s>>>(dist,dn,sink);attr={};cudaStreamSetAttribute(s,cudaStreamAttributeAccessPolicyWindow,&attr);
  cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a,s);touch_normal<<<512,256,0,s>>>(foot,fn,sink);cudaEventRecord(b,s);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);cudaEventDestroy(a);cudaEventDestroy(b);return ms*1000;
}
static float med(std::vector<float> x){std::sort(x.begin(),x.end());return x[x.size()/2];}
int main(int argc,char **argv){int n=argc>1?atoi(argv[1]):32,warm=argc>2?atoi(argv[2]):6;cudaDeviceProp p{};cudaGetDeviceProperties(&p,0);
  size_t fb=72ull<<20,db=256ull<<20,win=std::min(fb,(size_t)p.accessPolicyMaxWindowSize),reserve=std::min(win,(size_t)p.persistingL2CacheMaxSize);uint4 *foot,*dist;unsigned *sink;cudaMalloc(&foot,fb);cudaMalloc(&dist,db);cudaMalloc(&sink,2048);cudaMemset(foot,0,fb);cudaMemset(dist,0,db);cudaStream_t s;cudaStreamCreate(&s);
  const char *arms[]={"control","reserve","apw_noreserve","apw","interleave_noreserve","interleave"};float ratios[]={1,1,1,1,1,1};
  printf("{\"device\":\"%s\",\"l2_bytes\":%d,\"reserve_bytes\":%zu,\"window_bytes\":%zu,\"rows\":[",p.name,p.l2CacheSize,reserve,win);
  for(int a=0;a<6;a++){std::vector<float>x;std::string mode=arms[a];size_t r=(mode=="reserve"||mode=="apw"||mode=="interleave")?reserve:0;
    if(mode=="apw_noreserve")mode="apw";if(mode=="interleave_noreserve")mode="interleave";
    for(int i=0;i<n;i++){float v=one(s,foot,fb/16,dist,db/16,sink,mode,r,win,ratios[a]);if(i>=warm)x.push_back(v);}printf("%s{\"arm\":\"%s\",\"ratio\":%.3f,\"median_us\":%.6f}",a?",":"",arms[a],ratios[a],med(x));}
  printf("]}\n");}
