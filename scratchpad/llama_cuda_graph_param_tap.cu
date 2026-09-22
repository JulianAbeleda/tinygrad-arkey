// LD_PRELOAD observational tap: dumps candidate fused Q4_K CUDA graph node ABIs.
// Diagnostic only; delegates capture unchanged and never mutates graph nodes.
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

struct U3 { uint32_t x,y,z; };
struct Fusion { void * x_bias; void * gate; void * gate_bias; int32_t glu_op; };
using End = cudaError_t (*)(cudaStream_t, cudaGraph_t *);
extern "C" cudaError_t cudaStreamEndCapture(cudaStream_t s, cudaGraph_t * g) {
  static End real = (End)dlsym(RTLD_NEXT, "cudaStreamEndCapture");
  cudaError_t r = real(s,g); if (r != cudaSuccess || !g || !*g) return r;
  size_t n=0; if (cudaGraphGetNodes(*g,nullptr,&n)!=cudaSuccess) return r;
  cudaGraphNode_t * ns=(cudaGraphNode_t*)malloc(n*sizeof(*ns)); cudaGraphGetNodes(*g,ns,&n);
  const char * out=getenv("LLAMA_Q4_FUSION_TAP"); if (!out) out="/tmp/llama_q4_fusion_graph_params.jsonl";
  FILE * f=fopen(out,"a"); if (!f) { free(ns); return r; }
  for (size_t i=0;i<n;i++) { cudaGraphNodeType ty; if(cudaGraphNodeGetType(ns[i],&ty)!=cudaSuccess || ty!=cudaGraphNodeTypeKernel) continue;
    cudaKernelNodeParams p{}; if(cudaGraphKernelNodeGetParams(ns[i],&p)!=cudaSuccess || p.gridDim.x!=12288 || p.gridDim.y!=1 || p.blockDim.x!=32 || p.blockDim.y!=4) continue;
    void ** a=(void**)p.kernelParams; if (!a) continue; Fusion fu{}; memcpy(&fu,a[3],sizeof(fu));
    fprintf(f,"{\"node\":%zu,\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"shared\":%u,\"args\":{\"vx\":\"%p\",\"vy\":\"%p\",\"ids\":\"%p\",\"dst\":\"%p\",\"ncols_x\":%u,\"nchannels_y\":[%u,%u,%u],\"stride_row_x\":%u,\"stride_col_y\":%u,\"stride_col_dst\":%u,\"channel_ratio\":[%u,%u,%u],\"stride_channel_x\":%u,\"stride_channel_y\":%u,\"stride_channel_dst\":%u,\"sample_ratio\":[%u,%u,%u],\"stride_sample_x\":%u,\"stride_sample_y\":%u,\"stride_sample_dst\":%u,\"ids_stride\":%u},\"fusion\":{\"x_bias\":\"%p\",\"gate\":\"%p\",\"gate_bias\":\"%p\",\"glu_op\":%d}}\n",i,p.gridDim.x,p.gridDim.y,p.gridDim.z,p.blockDim.x,p.blockDim.y,p.blockDim.z,p.sharedMemBytes,*(void**)a[0],*(void**)a[1],*(void**)a[2],*(void**)a[4],*(uint32_t*)a[5],((U3*)a[6])->x,((U3*)a[6])->y,((U3*)a[6])->z,*(uint32_t*)a[7],*(uint32_t*)a[8],*(uint32_t*)a[9],((U3*)a[10])->x,((U3*)a[10])->y,((U3*)a[10])->z,*(uint32_t*)a[11],*(uint32_t*)a[12],*(uint32_t*)a[13],((U3*)a[14])->x,((U3*)a[14])->y,((U3*)a[14])->z,*(uint32_t*)a[15],*(uint32_t*)a[16],*(uint32_t*)a[17],*(uint32_t*)a[18],fu.x_bias,fu.gate,fu.gate_bias,fu.glu_op);
  } fclose(f); free(ns); return r;
}
