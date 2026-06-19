// TPE-3/5 (capture) — LD_PRELOAD shim: intercept hipModuleGetFunction (to map hipFunction_t -> kernel symbol) and
// hipExtModuleLaunchKernel (to dump the exact kernarg bytes + launch geometry + symbol rocBLAS builds for each prefill
// GEMM role). Separate HIP-only process (Lane A bars in-process HIP, not a separate capture). Roles keyed by the
// kernarg-embedded sizes: SizesFree(off88: M,N,batch) + SizesSum(off100: K).
//   g++ -std=c++17 -D__HIP_PLATFORM_AMD__=1 -shared -fPIC -I/opt/rocm-7.2.4/include extra/qk_tensile_kernarg_capture.cpp \
//       -ldl -o /tmp/qk_kacap.so
//   LD_PRELOAD=/tmp/qk_kacap.so QK_KACAP=/tmp/kernarg_all.json ROCBLAS_TENSILE_LIBPATH=... LD_LIBRARY_PATH=... /tmp/qk_ceiling
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <map>
#include <string>

#define HIP_LAUNCH_PARAM_BUFFER_POINTER ((void*)0x01)
#define HIP_LAUNCH_PARAM_BUFFER_SIZE    ((void*)0x02)
#define HIP_LAUNCH_PARAM_END            ((void*)0x03)

typedef hipError_t (*ext_t)(hipFunction_t, uint32_t,uint32_t,uint32_t, uint32_t,uint32_t,uint32_t,
                            size_t, hipStream_t, void**, void**, hipEvent_t, hipEvent_t, uint32_t);
typedef hipError_t (*getfn_t)(hipFunction_t*, hipModule_t, const char*);
static ext_t real_ext = nullptr;
static getfn_t real_getfn = nullptr;
static std::map<void*, std::string> g_names;          // hipFunction_t -> symbol
static std::map<std::string, bool> g_done;            // role -> captured

extern "C" hipError_t hipModuleGetFunction(hipFunction_t* f, hipModule_t m, const char* name) {
  if (!real_getfn) real_getfn = (getfn_t)dlsym(RTLD_NEXT, "hipModuleGetFunction");
  hipError_t e = real_getfn(f, m, name);
  if (e == hipSuccess && f && name) g_names[(void*)*f] = name;
  return e;
}

static const char* role_of(int32_t M, int32_t N, int32_t K) {
  if (M==512 && N==12288 && K==4096)  return "ffn_gate_up";
  if (M==512 && N==4096  && K==12288) return "ffn_down";
  if (M==512 && N==4096  && K==4096)  return "attn_q_o";
  if (M==512 && N==1024  && K==4096)  return "attn_k_v";
  return nullptr;
}

extern "C" hipError_t hipExtModuleLaunchKernel(hipFunction_t f, uint32_t gx,uint32_t gy,uint32_t gz,
    uint32_t lx,uint32_t ly,uint32_t lz, size_t shmem, hipStream_t stream, void** kparams, void** extra,
    hipEvent_t se, hipEvent_t ee, uint32_t flags) {
  if (!real_ext) real_ext = (ext_t)dlsym(RTLD_NEXT, "hipExtModuleLaunchKernel");
  void* karg=nullptr; size_t ksz=0;
  if (extra) for (int i=0; extra[i]!=HIP_LAUNCH_PARAM_END && i<8; i+=2) {
    if (extra[i]==HIP_LAUNCH_PARAM_BUFFER_POINTER) karg=extra[i+1];
    else if (extra[i]==HIP_LAUNCH_PARAM_BUFFER_SIZE) ksz=*(size_t*)extra[i+1];
  }
  if (karg && ksz>=104) {
    uint8_t buf[256]; size_t n = ksz>256?256:ksz; memcpy(buf, karg, n);
    int32_t M=*(int32_t*)(buf+88), N=*(int32_t*)(buf+92), K=*(int32_t*)(buf+100);
    const char* role = role_of(M,N,K);
    if (role && !g_done[role]) {
      g_done[role] = true;
      auto it = g_names.find((void*)f); std::string sym = (it!=g_names.end()) ? it->second : "UNKNOWN";
      const char* outp = getenv("QK_KACAP"); std::string path = outp?outp:"/tmp/kernarg_all.json";
      // append one JSON object per line (JSONL); the python side parses all lines
      FILE* fp = fopen(path.c_str(), "a");
      fprintf(fp, "{\"role\":\"%s\",\"kernel_symbol\":\"%s\",\"global\":[%u,%u,%u],\"local\":[%u,%u,%u],"
                  "\"num_workgroups\":[%u,%u,%u],\"kernarg_size\":%zu,\"M\":%d,\"N\":%d,\"K\":%d,\"kernarg_bytes\":[",
              role, sym.c_str(), gx,gy,gz, lx,ly,lz, lx?gx/lx:0,ly?gy/ly:0,lz?gz/lz:0, ksz, M,N,K);
      for (size_t i=0;i<n;i++) fprintf(fp, "%s%u", i?",":"", buf[i]);
      fprintf(fp, "]}\n"); fclose(fp);
      fprintf(stderr, "[KACAP] %s: sym=%.40s global(%u,%u,%u) local(%u,%u,%u) ksz=%zu\n",
              role, sym.c_str(), gx,gy,gz, lx,ly,lz, ksz);
    }
  }
  return real_ext(f,gx,gy,gz,lx,ly,lz,shmem,stream,kparams,extra,se,ee,flags);
}
