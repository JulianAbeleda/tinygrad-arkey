// TPE-3 (capture) — LD_PRELOAD shim: intercept hipExtModuleLaunchKernel, dump the exact kernarg bytes + launch
// geometry rocBLAS builds for the ffn_gate/up Tensile GEMM. Separate HIP-only process (Lane A bars in-process HIP,
// not a separate capture). Identifies ffn_gate/up by the kernarg-embedded sizes (SizesSum K==4096, SizesFree M==512,N==12288).
//   g++ -std=c++17 -D__HIP_PLATFORM_AMD__=1 -shared -fPIC -I/opt/rocm-7.2.4/include extra/qk_tensile_kernarg_capture.cpp \
//       -ldl -o /tmp/qk_kacap.so
//   LD_PRELOAD=/tmp/qk_kacap.so QK_KACAP=/tmp/kernarg.json ROCBLAS_TENSILE_LIBPATH=... LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib /tmp/qk_ceiling
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>

#define HIP_LAUNCH_PARAM_BUFFER_POINTER ((void*)0x01)
#define HIP_LAUNCH_PARAM_BUFFER_SIZE    ((void*)0x02)
#define HIP_LAUNCH_PARAM_END            ((void*)0x03)

typedef hipError_t (*ext_t)(hipFunction_t, uint32_t,uint32_t,uint32_t, uint32_t,uint32_t,uint32_t,
                            size_t, hipStream_t, void**, void**, hipEvent_t, hipEvent_t, uint32_t);
static ext_t real_ext = nullptr;
static int captured = 0;

extern "C" hipError_t hipExtModuleLaunchKernel(hipFunction_t f, uint32_t gx,uint32_t gy,uint32_t gz,
    uint32_t lx,uint32_t ly,uint32_t lz, size_t shmem, hipStream_t stream, void** kparams, void** extra,
    hipEvent_t se, hipEvent_t ee, uint32_t flags) {
  if (!real_ext) real_ext = (ext_t)dlsym(RTLD_NEXT, "hipExtModuleLaunchKernel");
  // recover kernarg pointer+size from the `extra` config array
  void* karg=nullptr; size_t ksz=0;
  if (extra) for (int i=0; extra[i]!=HIP_LAUNCH_PARAM_END && i<8; i+=2) {
    if (extra[i]==HIP_LAUNCH_PARAM_BUFFER_POINTER) karg=extra[i+1];
    else if (extra[i]==HIP_LAUNCH_PARAM_BUFFER_SIZE) ksz=*(size_t*)extra[i+1];
  }
  // kernarg lives in device-visible (host-accessible) memory for Tensile; peek the bytes
  if (!captured && karg && ksz>=104) {
    // read SizesFree (off 88: 3x i32 = M,N,batch) and SizesSum (off 100: i32 = K)
    uint8_t buf[256]; size_t n = ksz>256?256:ksz; memcpy(buf, karg, n);
    int32_t M=*(int32_t*)(buf+88), N=*(int32_t*)(buf+92), K=*(int32_t*)(buf+100);
    if (M==512 && N==12288 && K==4096) {                 // ffn_gate/up
      const char* outp = getenv("QK_KACAP"); FILE* fp = fopen(outp?outp:"/tmp/kernarg.json","w");
      fprintf(fp, "{\n  \"kind\":\"hipExtModuleLaunchKernel\",\n");
      fprintf(fp, "  \"global\":[%u,%u,%u], \"local\":[%u,%u,%u], \"shmem\":%zu, \"kernarg_size\":%zu,\n",
              gx,gy,gz,lx,ly,lz,shmem,ksz);
      fprintf(fp, "  \"num_workgroups\":[%u,%u,%u],\n", lx?gx/lx:0, ly?gy/ly:0, lz?gz/lz:0);
      fprintf(fp, "  \"kernarg_bytes\":[");
      for (size_t i=0;i<n;i++) fprintf(fp, "%s%u", i?",":"", buf[i]);
      fprintf(fp, "]\n}\n"); fclose(fp);
      fprintf(stderr, "[KACAP] captured ffn_gate/up: global(%u,%u,%u) local(%u,%u,%u) ksz=%zu -> %s\n",
              gx,gy,gz,lx,ly,lz,ksz, outp?outp:"/tmp/kernarg.json");
      captured = 1;
    }
  }
  return real_ext(f,gx,gy,gz,lx,ly,lz,shmem,stream,kparams,extra,se,ee,flags);
}
