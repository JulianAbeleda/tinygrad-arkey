#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstring>

struct U3 { uint32_t x, y, z; };
struct Fusion { void *x_bias, *gate, *gate_bias; int32_t glu_op; };
using End = cudaError_t (*)(cudaStream_t, cudaGraph_t *);

extern "C" cudaError_t cudaStreamEndCapture(cudaStream_t s, cudaGraph_t *g) {
  static End real = (End)dlsym(RTLD_NEXT, "cudaStreamEndCapture");
  cudaError_t r = real(s, g);
  if (r != cudaSuccess || !g || !*g) return r;
  size_t n = 0; if (cudaGraphGetNodes(*g, nullptr, &n) != cudaSuccess) return r;
  cudaGraphNode_t *nodes = new cudaGraphNode_t[n]; cudaGraphGetNodes(*g, nodes, &n);
  const char *path = getenv("LLAMA_VOCAB_TAP"); if (!path) path = "/tmp/llama_vocab_graph.jsonl";
  FILE *f = fopen(path, "a"); if (!f) { delete[] nodes; return r; }
  for (size_t i = 0; i < n; i++) {
    cudaGraphNodeType ty; if (cudaGraphNodeGetType(nodes[i], &ty) != cudaSuccess || ty != cudaGraphNodeTypeKernel) continue;
    cudaKernelNodeParams p{}; if (cudaGraphKernelNodeGetParams(nodes[i], &p) != cudaSuccess) continue;
    if (p.gridDim.x != 151936 || p.blockDim.x != 32 || p.blockDim.y != 4 || !p.kernelParams) continue;
    void **a = (void **)p.kernelParams; Fusion fu{}; memcpy(&fu, a[3], sizeof(fu));
    fprintf(f, "{\"node\":%zu,\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"shared\":%u,\"args\":[\n", i, p.gridDim.x,p.gridDim.y,p.gridDim.z,p.blockDim.x,p.blockDim.y,p.blockDim.z,p.sharedMemBytes);
    for (int j=0; j<19; j++) fprintf(f, "  {\"index\":%d,\"slot\":\"%p\",\"value\":\"%p\"}%s\n", j, a[j], *(void**)a[j], j==18?"":",");
    fprintf(f, "],\"fusion\":{\"x_bias\":\"%p\",\"gate\":\"%p\",\"gate_bias\":\"%p\",\"glu_op\":%d}}\n", fu.x_bias,fu.gate,fu.gate_bias,fu.glu_op);
  }
  fclose(f); delete[] nodes; return r;
}
