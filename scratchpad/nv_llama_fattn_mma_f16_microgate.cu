#define FLASH_ATTN_AVAILABLE
#include "/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/common.cuh"
#undef GGML_CUDA_USE_PDL
#undef ggml_cuda_pdl_sync
#undef ggml_cuda_pdl_lc
#define ggml_cuda_pdl_sync() do {} while (0)
#define ggml_cuda_pdl_lc() do {} while (0)
#include "/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh"
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

int main() {
  constexpr int D=128,T=512,H=32,HK=8,NBLOCKS=256,NCOLS1=16,NCOLS2=4;
  const std::string root="/tmp/nv-cleanroom-flash-fixture/";
  auto q=load<float>((root+"q.bin").c_str(),size_t(H)*T*D);
  auto k=load<half>((root+"k.bin").c_str(),size_t(HK)*T*D);
  auto v=load<half>((root+"v.bin").c_str(),size_t(HK)*T*D);
  auto expected=load<float>((root+"expected.bin").c_str(),size_t(H)*T*D);
  std::vector<half> mask(size_t(T)*T);
  for (int iq=0;iq<T;iq++) for (int ik=0;ik<T;ik++) mask[size_t(iq)*T+ik]=__float2half(ik<=iq ? 0.0f : -65504.0f);
  float *dq,*dout; float2 *dmeta; half *dk,*dv,*dmask;
  cudaMalloc(&dq,q.size()*sizeof(float)); cudaMalloc(&dk,k.size()*sizeof(half)); cudaMalloc(&dv,v.size()*sizeof(half));
  cudaMalloc(&dmask,mask.size()*sizeof(half)); cudaMalloc(&dout,(size_t(H)*T*D+16)*sizeof(float));
  cudaMalloc(&dmeta,size_t(NBLOCKS)*(NCOLS1*NCOLS2)*(2+D/2)*sizeof(float2));
  cudaMemset(dmeta,0,size_t(NBLOCKS)*(NCOLS1*NCOLS2)*(2+D/2)*sizeof(float2));
  cudaMemcpy(dq,q.data(),q.size()*sizeof(float),cudaMemcpyHostToDevice); cudaMemcpy(dk,k.data(),k.size()*sizeof(half),cudaMemcpyHostToDevice);
  cudaMemcpy(dv,v.data(),v.size()*sizeof(half),cudaMemcpyHostToDevice); cudaMemcpy(dmask,mask.data(),mask.size()*sizeof(half),cudaMemcpyHostToDevice);
  cudaMemset(dout,0,(size_t(H)*T*D+16)*sizeof(float));
  const uint3 ne01=init_fastdiv_values(T);
  cudaFuncSetAttribute(flash_attn_ext_f16<128,128,16,4,false,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,37120);
  const uint3 fd0=init_fastdiv_values(1024),fd1=init_fastdiv_values(128),fd2=init_fastdiv_values(128),fd3=init_fastdiv_values(4);
  auto main_kernel=flash_attn_ext_f16<128,128,16,4,false,false>;
  auto fixup_kernel=flash_attn_stream_k_fixup_general<D,NCOLS1,NCOLS2>;
  cudaLaunchAttribute main_attr{},fixup_attr{}; main_attr.id=fixup_attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;
  main_attr.val.programmaticStreamSerializationAllowed=fixup_attr.val.programmaticStreamSerializationAllowed=1;
  cudaLaunchConfig_t main_cfg{},fixup_cfg{};
  main_cfg.gridDim=dim3(NBLOCKS,1,1);main_cfg.blockDim=dim3(32,4,1);main_cfg.dynamicSmemBytes=37120;main_cfg.attrs=&main_attr;main_cfg.numAttrs=1;
  fixup_cfg.gridDim=dim3(NBLOCKS,NCOLS1,NCOLS2);fixup_cfg.blockDim=dim3(D,1,1);fixup_cfg.attrs=&fixup_attr;fixup_cfg.numAttrs=1;
  auto launch=[&](){ cudaLaunchKernelEx(&main_cfg,main_kernel,
    reinterpret_cast<const char *>(dq),reinterpret_cast<const char *>(dk),reinterpret_cast<const char *>(dv),
    reinterpret_cast<const char *>(dmask),nullptr,nullptr,dout,dmeta,
    1.0f/std::sqrt(float(D)),0.0f,1.0f,1.0f,32u,0.0f,
    D,ne01,H,1,D*int(sizeof(float)),T*D*int(sizeof(float)),H*T*D*int(sizeof(float)),
    D,T,HK,1,D*int(sizeof(half)),T*D*int(sizeof(half)),int64_t(HK)*T*D*int(sizeof(half)),
    D*int(sizeof(half)),T*D*int(sizeof(half)),int64_t(HK)*T*D*int(sizeof(half)),
    T,1,1,T*int(sizeof(half)),T*T*int(sizeof(half)),int64_t(T)*T*int(sizeof(half)) );
    if constexpr(NBLOCKS!=256) cudaLaunchKernelEx(&fixup_cfg,fixup_kernel,dout,dmeta,T,H,4,1024,fd0,fd1,fd2,fd3); };
  launch(); auto err=cudaDeviceSynchronize(); if(err!=cudaSuccess){std::fprintf(stderr,"launch: %s\n",cudaGetErrorString(err));return 3;}
  cudaEvent_t a,b; cudaEventCreate(&a);cudaEventCreate(&b); for(int i=0;i<5;i++)launch(); cudaEventRecord(a);for(int i=0;i<20;i++)launch();cudaEventRecord(b);cudaEventSynchronize(b);
  float ms;cudaEventElapsedTime(&ms,a,b);std::vector<float> out(size_t(H)*T*D+16);cudaMemcpy(out.data(),dout,out.size()*sizeof(float),cudaMemcpyDeviceToHost);
  double max_abs=0,mean_abs=0,l2=0,l2ref=0;size_t mismatch=0,m02=0,m05=0,m10=0;
  for(int h=0;h<H;h++)for(int t=0;t<T;t++)for(int d=0;d<D;d++){
    float got=out[(size_t(t)*H+h)*D+d], ref=expected[(size_t(h)*T+t)*D+d], delta=std::fabs(got-ref);
    max_abs=std::max(max_abs,double(delta));mean_abs+=delta;l2+=double(got-ref)*(got-ref);l2ref+=double(ref)*ref;
    mismatch+=!(delta<=2e-4f+2e-4f*std::fabs(ref));m02+=delta>0.02f;m05+=delta>0.05f;m10+=delta>0.10f;
  }
  mean_abs/=size_t(H)*T*D;bool canary=true;for(size_t i=size_t(H)*T*D;i<out.size();i++)canary&=out[i]==0.0f;
  std::printf("launch_us=%.3f max_abs=%.9g mean_abs=%.9g rel_l2=%.9g strict=%zu gt02=%zu gt05=%zu gt10=%zu canary=%s\n",ms*1000/20,max_abs,mean_abs,std::sqrt(l2/l2ref),mismatch,m02,m05,m10,canary?"pass":"FAIL");
  return canary?0:4;
}
