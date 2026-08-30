#include "/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh"
#include <cuda_runtime.h>
extern "C" void llama_flash_score_raw(const char *q,const char *k,const char *v,float *dst,float2 *meta,int kvmax,int ne00,uint3 ne01,int ne02,int nb01,int nb02,int nb03,int ne10,int ne11,int ne12,int ne13,int nb11,int nb12,long nb13,int nb21,int nb22,long nb23,int ne31,int ne32,int ne33,int nb31,int nb32,long nb33,cudaStream_t s) {
  const char *mask=nullptr, *sinks=nullptr;
  float scale=1.0f/sqrtf(128.0f), max_bias=0.0f, m0=1.0f, m1=1.0f, logit_softcap=0.0f;
  uint32_t n_head_log2=0;
  int ne03=1;
  int32_t ne00x=ne00, ne02x=ne02, ne10x=ne10, ne11x=ne11, ne12x=ne12, ne13x=ne13;
  int32_t nb01x=nb01, nb02x=nb02, nb03x=nb03, nb11x=nb11, nb12x=nb12, nb21x=nb21, nb22x=nb22;
  int32_t ne31x=ne31, ne32x=ne32, ne33x=ne33, nb31x=nb31, nb32x=nb32;
  int64_t nb13x=nb13, nb23x=nb23, nb33x=nb33;
  (void)kvmax;
  int *kvmax_ptr=nullptr;
  void *a[]={(void*)&q,(void*)&k,(void*)&v,(void*)&mask,(void*)&sinks,(void*)&kvmax_ptr,(void*)&dst,(void*)&meta,
    (void*)&scale,(void*)&max_bias,(void*)&m0,(void*)&m1,(void*)&n_head_log2,(void*)&logit_softcap,
    (void*)&ne00x,(void*)&ne01,(void*)&ne02x,(void*)&ne03,(void*)&nb01x,(void*)&nb02x,(void*)&nb03x,
    (void*)&ne10x,(void*)&ne11x,(void*)&ne12x,(void*)&ne13x,(void*)&nb11x,(void*)&nb12x,(void*)&nb13x,
    (void*)&nb21x,(void*)&nb22x,(void*)&nb23x,(void*)&ne31x,(void*)&ne32x,(void*)&ne33x,(void*)&nb31x,(void*)&nb32x,(void*)&nb33x};
  // pp512: one output tile per block, one KV-partial per tile for direct
  // normalized output.  Production uses grid.y>1 plus the fixup kernel;
  // this reduced probe deliberately uses grid.y=1 to avoid that second ABI.
  cudaLaunchKernel((const void*)flash_attn_ext_vec<128,1,GGML_TYPE_F16,GGML_TYPE_F16,false>,dim3(512,1,32),dim3(32,4,1),a,0,s);
}
