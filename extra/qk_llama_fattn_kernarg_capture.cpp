// Route B B1.1 capture — LD_PRELOAD shim to capture llama.cpp's flash_attn_tile / combine / mask_to_KV_max
// dispatches from a real ggml-hip decode. ggml launches these via the chevron `kernel<<<grid,block,shmem,stream>>>`
// (common.cuh:1639), which under HIP lowers to hipLaunchKernel(func, grid, block, void** args, shmem, stream) where
// `func` is the host shadow registered by __hipRegisterFunction(hostFn, deviceName,...). So we hook both:
//   __hipRegisterFunction -> map hostFn -> mangled deviceName
//   hipLaunchKernel        -> identify by name; dump grid/block/shmem + the per-arg VALUE bytes (args[i] -> *size_i)
// The exact 464-byte kernarg is reconstructed on the replay side from the .co's amdhsa.kernels arg-offset metadata;
// here we only record the raw arg values (in declaration order) + geometry. One capture per kernel symbol.
//
// build: g++ -std=c++17 -D__HIP_PLATFORM_AMD__=1 -shared -fPIC -I/opt/rocm-7.2.4/include \
//            extra/qk_llama_fattn_kernarg_capture.cpp -ldl -o /tmp/qk_llama_kacap.so
// run:   LD_PRELOAD=/tmp/qk_llama_kacap.so QK_KACAP=bench/qk-llama-hcq-tile/capture.jsonl <llama-bench ... -fa 1 ...>
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

typedef hipError_t (*launch_t)(const void*, dim3, dim3, void**, size_t, hipStream_t);
typedef void (*reg_t)(void**, const void*, char*, const char*, unsigned, void*, void*, void*, void*, int*);
static launch_t real_launch = nullptr;
static reg_t real_reg = nullptr;
// construct-on-first-use (heap): __hipRegisterFunction is called from librocblas/libggml STATIC INITIALIZERS during
// _dl_init, which can run BEFORE this shim's own global constructors -> a plain static std::map would be touched
// uninitialized and segfault. Function-local statics are lazily, safely constructed on first call.
static std::map<const void*, std::string>& g_names() { static auto* m = new std::map<const void*, std::string>(); return *m; }
static std::map<std::string, bool>& g_done() { static auto* m = new std::map<std::string, bool>(); return *m; }

// per-arg byte sizes in declaration order (AMD HSA passes one void* per source arg via hipLaunchKernel's args[]).
// flash_attn_tile<DKQ,DV,ncols1,ncols2,softcap>(8 ptrs, then scalars/dims) -- fattn-tile.cuh:788-811.
static const std::vector<int> TILE_ARGSZ = {
  8,8,8,8,8,8,8,8,            // Q,K,V,mask,sinks,KV_max,dst,dst_meta (pointers)
  4,4,4,4,4,4,                // float scale,max_bias,m0,m1; uint32 n_head_log2; float logit_softcap
  4,12,4,4,4,4,4,             // int32 ne00; uint3 ne01; int32 ne02,ne03,nb01,nb02,nb03
  4,4,4,4,4,4,8,              // int32 ne10,ne11,ne12,ne13,nb11,nb12; int64 nb13
  4,4,8,                      // int32 nb21,nb22; int64 nb23
  4,4,4,4,4,8                 // int32 ne31,ne32,ne33,nb31,nb32; int64 nb33
};
// flash_attn_combine_results<DV>(const float* VKQ_parts, const float2* VKQ_meta, float* dst, int parallel_blocks)
static const std::vector<int> COMBINE_ARGSZ = {8,8,8,4};
// flash_attn_mask_to_KV_max<ncols1>(const half2* mask, int* KV_max, int ne30, int s31) -- approx; capture generically
static const std::vector<int> KVMAX_ARGSZ = {8,8,4,4};

static const char* kernel_class(const std::string& sym) {
  if (sym.find("flash_attn_tile") != std::string::npos) return "tile";
  if (sym.find("flash_attn_combine_results") != std::string::npos) return "combine";
  if (sym.find("flash_attn_mask_to_KV_max") != std::string::npos) return "kv_max";
  return nullptr;
}

static int g_reg_count = 0, g_launch_count = 0;
extern "C" void __hipRegisterFunction(void** modules, const void* hostFunction, char* deviceFunction,
                                      const char* deviceName, unsigned threadLimit, void* tid, void* bid,
                                      void* blockDim, void* gridDim, int* wSize) {
  if (!real_reg) real_reg = (reg_t)dlsym(RTLD_NEXT, "__hipRegisterFunction");
  if (hostFunction && deviceName) {
    g_names()[hostFunction] = deviceName;
    if (++g_reg_count <= 3 || strstr(deviceName, "flash_attn"))
      fprintf(stderr, "[KACAP-REG #%d] %.70s\n", g_reg_count, deviceName);
  }
  real_reg(modules, hostFunction, deviceFunction, deviceName, threadLimit, tid, bid, blockDim, gridDim, wSize);
}

extern "C" hipError_t hipLaunchKernel(const void* func, dim3 grid, dim3 block, void** args, size_t shmem,
                                      hipStream_t stream) {
  if (!real_launch) real_launch = (launch_t)dlsym(RTLD_NEXT, "hipLaunchKernel");
  if (++g_launch_count <= 5) {
    auto dbg = g_names().find(func);
    fprintf(stderr, "[KACAP-LAUNCH #%d] name=%s\n", g_launch_count,
            dbg!=g_names().end()? dbg->second.c_str() : "UNREGISTERED");
  }
  auto it = g_names().find(func);
  if (it != g_names().end() && args) {
    const char* cls = kernel_class(it->second);
    if (cls) {
      const std::vector<int>* szs = strcmp(cls,"tile")==0 ? &TILE_ARGSZ
                                  : strcmp(cls,"combine")==0 ? &COMBINE_ARGSZ : &KVMAX_ARGSZ;
      // key per distinct KV length so warmup (small KV) does NOT block the real deep-context decode. tile: ne11 is
      // arg[22]; combine: key on parallel_blocks (arg[3]). One capture per (class, kv/pb).
      int keyv = 0;
      if (strcmp(cls,"tile")==0 && args[22]) keyv = *(const int*)args[22];
      else if (strcmp(cls,"combine")==0 && args[3]) keyv = *(const int*)args[3];
      char key[64]; snprintf(key, sizeof key, "%s_%d", cls, keyv);
      if (g_done().count(key)) return real_launch(func, grid, block, args, shmem, stream);
      g_done()[key] = true;
      const char* outp = getenv("QK_KACAP"); std::string path = outp ? outp : "/tmp/qk_llama_kacap.jsonl";
      FILE* fp = fopen(path.c_str(), "a");
      fprintf(fp, "{\"class\":\"%s\",\"symbol\":\"%s\",\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"shmem\":%zu,"
                  "\"nargs\":%zu,\"arg_sizes\":[", cls, it->second.c_str(), grid.x,grid.y,grid.z,
              block.x,block.y,block.z, shmem, szs->size());
      for (size_t i=0;i<szs->size();i++) fprintf(fp, "%s%d", i?",":"", (*szs)[i]);
      fprintf(fp, "],\"args\":[");
      for (size_t i=0;i<szs->size();i++) {
        const uint8_t* p = (const uint8_t*)args[i];   // args[i] points to the i-th arg value
        fprintf(fp, "%s[", i?",":"");
        for (int b=0;b<(*szs)[i];b++) fprintf(fp, "%s%u", b?",":"", p?p[b]:0);
        fprintf(fp, "]");
      }
      fprintf(fp, "]}\n"); fclose(fp);
      fprintf(stderr, "[LLAMA-KACAP] %s sym=%.60s grid(%u,%u,%u) block(%u,%u,%u) shmem=%zu\n",
              cls, it->second.c_str(), grid.x,grid.y,grid.z, block.x,block.y,block.z, shmem);
    }
  }
  return real_launch(func, grid, block, args, shmem, stream);
}
