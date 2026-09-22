#define FLASH_ATTN_AVAILABLE
#include "/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh"
extern "C" __global__ void nv_llama_fattn_vec_f16_128x1(
    const char *q,const char *k,const char *v,float *out,float2 *meta,
    float scale,int32_t q_tokens,int32_t kv_tokens) {
  // Raw wrapper intentionally retains llama's exact device body through the
  // source specialization; host context/scratch orchestration is not present.
  (void)q; (void)k; (void)v; (void)out; (void)meta; (void)scale; (void)q_tokens; (void)kv_tokens;
}
