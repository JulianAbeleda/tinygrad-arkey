#define FLASH_ATTN_AVAILABLE
#include "/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>

template <class T> static std::vector<T> load(const char *path, size_t count) {
  std::vector<T> out(count); std::ifstream f(path, std::ios::binary);
  if (!f.read(reinterpret_cast<char *>(out.data()), count*sizeof(T))) { std::fprintf(stderr,"read failed: %s\n",path); std::exit(2); }
  return out;
}

#ifndef PARTS
#define PARTS 1
#endif
int main() {
  constexpr int D=128,T=512,H=32,HK=8,PB=PARTS;
  const std::string root="/tmp/nv-cleanroom-flash-fixture/";
  auto q=load<float>((root+"q.bin").c_str(),size_t(H)*T*D);
  auto k=load<half>((root+"k.bin").c_str(),size_t(HK)*T*D);
  auto v=load<half>((root+"v.bin").c_str(),size_t(HK)*T*D);
  auto expected=load<float>((root+"expected.bin").c_str(),size_t(H)*T*D);
  std::vector<half> mask(size_t(T)*T);
  for (int iq=0;iq<T;iq++) for (int ik=0;ik<T;ik++) mask[size_t(iq)*T+ik]=__float2half(ik<=iq ? 0.0f : -INFINITY);
  float *dq,*dout,*dtmp; float2 *dmeta; half *dk,*dv,*dmask;
  cudaMalloc(&dq,q.size()*sizeof(float)); cudaMalloc(&dk,k.size()*sizeof(half)); cudaMalloc(&dv,v.size()*sizeof(half));
  cudaMalloc(&dmask,mask.size()*sizeof(half)); cudaMalloc(&dout,(size_t(H)*T*D+16)*sizeof(float));
  cudaMalloc(&dtmp,size_t(PB)*H*T*D*sizeof(float)); cudaMalloc(&dmeta,size_t(PB)*H*T*sizeof(float2));
  cudaMemcpy(dq,q.data(),q.size()*sizeof(float),cudaMemcpyHostToDevice); cudaMemcpy(dk,k.data(),k.size()*sizeof(half),cudaMemcpyHostToDevice);
  cudaMemcpy(dv,v.data(),v.size()*sizeof(half),cudaMemcpyHostToDevice); cudaMemcpy(dmask,mask.data(),mask.size()*sizeof(half),cudaMemcpyHostToDevice);
  cudaMemset(dout,0,(size_t(H)*T*D+16)*sizeof(float));
  const uint3 ne01=init_fastdiv_values(T);
  float *primary=PB==1?dout:dtmp; float2 *meta=PB==1?nullptr:dmeta;
  auto launch=[&](){ flash_attn_ext_vec<128,2,GGML_TYPE_F16,GGML_TYPE_F16,false><<<dim3(256,PB,32),dim3(32,4,1)>>>(
    reinterpret_cast<const char *>(dq),reinterpret_cast<const char *>(dk),reinterpret_cast<const char *>(dv),
    reinterpret_cast<const char *>(dmask),nullptr,nullptr,primary,meta,
    1.0f/std::sqrt(float(D)),0.0f,1.0f,1.0f,32u,0.0f,
    D,ne01,H,1,D*int(sizeof(float)),T*D*int(sizeof(float)),H*T*D*int(sizeof(float)),
    D,T,HK,1,D*int(sizeof(half)),T*D*int(sizeof(half)),int64_t(HK)*T*D*int(sizeof(half)),
    D*int(sizeof(half)),T*D*int(sizeof(half)),int64_t(HK)*T*D*int(sizeof(half)),
    T,1,1,T*int(sizeof(half)),T*T*int(sizeof(half)),int64_t(T)*T*int(sizeof(half)) );
    if constexpr(PB>1) flash_attn_combine_results<D><<<dim3(T,H,1),dim3(D,1,1),PB*sizeof(float2)>>>(dtmp,dmeta,dout,PB); };
  launch(); auto err=cudaDeviceSynchronize(); if(err!=cudaSuccess){std::fprintf(stderr,"launch: %s\n",cudaGetErrorString(err));return 3;}
  cudaEvent_t a,b; cudaEventCreate(&a);cudaEventCreate(&b); for(int i=0;i<5;i++)launch(); cudaEventRecord(a);for(int i=0;i<20;i++)launch();cudaEventRecord(b);cudaEventSynchronize(b);
  float ms;cudaEventElapsedTime(&ms,a,b);std::vector<float> out(size_t(H)*T*D+16);cudaMemcpy(out.data(),dout,out.size()*sizeof(float),cudaMemcpyDeviceToHost);
  double max_abs=0,mean_abs=0;size_t mismatch=0;
  for(int h=0;h<H;h++)for(int t=0;t<T;t++)for(int d=0;d<D;d++){
    float got=out[(size_t(t)*H+h)*D+d], ref=expected[(size_t(h)*T+t)*D+d], delta=std::fabs(got-ref);
    max_abs=std::max(max_abs,double(delta));mean_abs+=delta;mismatch+=!(delta<=2e-4f+2e-4f*std::fabs(ref));
  }
  mean_abs/=size_t(H)*T*D;bool canary=true;for(size_t i=size_t(H)*T*D;i<out.size();i++)canary&=out[i]==0.0f;
  std::printf("launch_us=%.3f max_abs=%.9g mean_abs=%.9g mismatches=%zu canary=%s\n",ms*1000/20,max_abs,mean_abs,mismatch,canary?"pass":"FAIL");
  return mismatch==0&&canary?0:4;
}
