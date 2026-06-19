// Decode MMVQ P2 capture shim.
//
// Intercepts HIP module launches in a separate llama.cpp HIP-only process and
// dumps Q4_K/Q6_K mul_mat_vec_q launch geometry + raw kernarg bytes.
//
// build:
//   g++ -std=c++17 -D__HIP_PLATFORM_AMD__=1 -shared -fPIC -I/opt/rocm-7.2.4/include \
//     extra/qk_decode_mmvq_kernarg_capture.cpp -ldl -o /tmp/qk_decode_mmvq_cap.so
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <map>
#include <set>
#include <string>

#define HIP_LAUNCH_PARAM_BUFFER_POINTER ((void*)0x01)
#define HIP_LAUNCH_PARAM_BUFFER_SIZE    ((void*)0x02)
#define HIP_LAUNCH_PARAM_END            ((void*)0x03)

typedef hipError_t (*getfn_t)(hipFunction_t*, hipModule_t, const char*);
typedef hipError_t (*ext_t)(hipFunction_t, uint32_t,uint32_t,uint32_t, uint32_t,uint32_t,uint32_t,
                            size_t, hipStream_t, void**, void**, hipEvent_t, hipEvent_t, uint32_t);
typedef hipError_t (*mod_t)(hipFunction_t, uint32_t,uint32_t,uint32_t, uint32_t,uint32_t,uint32_t,
                            uint32_t, hipStream_t, void**, void**);
typedef hipError_t (*launch_t)(const void*, dim3, dim3, void**, size_t, hipStream_t);

static getfn_t real_getfn = nullptr;
static ext_t real_ext = nullptr;
static mod_t real_mod = nullptr;
static launch_t real_launch = nullptr;
static launch_t real_launch_spt = nullptr;
static int g_trace_count = 0;
static std::map<void*, std::string> g_names;
static std::set<std::string> g_seen;

extern "C" hipError_t hipModuleGetFunction(hipFunction_t* f, hipModule_t m, const char* name) {
  if (!real_getfn) real_getfn = (getfn_t)dlsym(RTLD_NEXT, "hipModuleGetFunction");
  hipError_t e = real_getfn(f, m, name);
  if (e == hipSuccess && f && name) g_names[(void*)*f] = name;
  return e;
}

static bool is_target(const std::string &sym) {
  return sym.find("mul_mat_vec_q") != std::string::npos &&
         (sym.find("ggml_type12") != std::string::npos || sym.find("ggml_type14") != std::string::npos);
}

static std::string symbol_from_lowbits(const void *fn) {
  uintptr_t lo = ((uintptr_t)fn) & 0xfffff;
  switch (lo) {
    case 0xd44b8: return "_ZL13mul_mat_vec_qIL9ggml_type12ELi1ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_typeIjLj3EEjjjS8_jjjS8_jjjj";
    case 0xd44a8: return "_ZL13mul_mat_vec_qIL9ggml_type12ELi1ELb0ELb1EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_typeIjLj3EEjjjS8_jjjS8_jjjj";
    case 0xd44b0: return "_ZL13mul_mat_vec_qIL9ggml_type12ELi1ELb1ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_typeIjLj3EEjjjS8_jjjS8_jjjj";
    case 0xd44a0: return "_ZL13mul_mat_vec_qIL9ggml_type12ELi1ELb1ELb1EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_typeIjLj3EEjjjS8_jjjS8_jjjj";
    case 0xd44c0: return "_ZL13mul_mat_vec_qIL9ggml_type12ELi2ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_typeIjLj3EEjjjS8_jjjS8_jjjj";
    case 0xd44c8: return "_ZL13mul_mat_vec_qIL9ggml_type12ELi3ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_typeIjLj3EEjjjS8_jjjS8_jjjj";
    case 0xd44d0: return "_ZL13mul_mat_vec_qIL9ggml_type12ELi4ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_typeIjLj3EEjjjS8_jjjS8_jjjj";
    case 0xd44d8: return "_ZL13mul_mat_vec_qIL9ggml_type12ELi5ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_typeIjLj3EEjjjS8_jjjS8_jjjj";
    case 0xd44e0: return "_ZL13mul_mat_vec_qIL9ggml_type12ELi6ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_typeIjLj3EEjjjS8_jjjS8_jjjj";
    case 0xd44e8: return "_ZL13mul_mat_vec_qIL9ggml_type12ELi7ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_typeIjLj3EEjjjS8_jjjS8_jjjj";
    case 0xd44f0: return "_ZL13mul_mat_vec_qIL9ggml_type12ELi8ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_typeIjLj3EEjjjS8_jjjS8_jjjj";
    case 0xd4578: return "_ZL13mul_mat_vec_qIL9ggml_type14ELi1ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_typeIjLj3EEjjjS8_jjjS8_jjjj";
    case 0xd4568: return "_ZL13mul_mat_vec_qIL9ggml_type14ELi1ELb0ELb1EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_typeIjLj3EEjjjS8_jjjS8_jjjj";
    case 0xd4570: return "_ZL13mul_mat_vec_qIL9ggml_type14ELi1ELb1ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_typeIjLj3EEjjjS8_jjjS8_jjjj";
    case 0xd4560: return "_ZL13mul_mat_vec_qIL9ggml_type14ELi1ELb1ELb1EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_typeIjLj3EEjjjS8_jjjS8_jjjj";
    case 0xd4580: return "_ZL13mul_mat_vec_qIL9ggml_type14ELi2ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_typeIjLj3EEjjjS8_jjjS8_jjjj";
    case 0xd4588: return "_ZL13mul_mat_vec_qIL9ggml_type14ELi3ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_typeIjLj3EEjjjS8_jjjS8_jjjj";
    case 0xd4590: return "_ZL13mul_mat_vec_qIL9ggml_type14ELi4ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_typeIjLj3EEjjjS8_jjjS8_jjjj";
    case 0xd4598: return "_ZL13mul_mat_vec_qIL9ggml_type14ELi5ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_typeIjLj3EEjjjS8_jjjS8_jjjj";
    case 0xd45a0: return "_ZL13mul_mat_vec_qIL9ggml_type14ELi6ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_typeIjLj3EEjjjS8_jjjS8_jjjj";
    case 0xd45a8: return "_ZL13mul_mat_vec_qIL9ggml_type14ELi7ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_typeIjLj3EEjjjS8_jjjS8_jjjj";
    case 0xd45b0: return "_ZL13mul_mat_vec_qIL9ggml_type14ELi8ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_typeIjLj3EEjjjS8_jjjS8_jjjj";
    default: return "UNKNOWN";
  }
}

static const char *type_name(const std::string &sym) {
  if (sym.find("ggml_type12") != std::string::npos) return "Q4_K";
  if (sym.find("ggml_type14") != std::string::npos) return "Q6_K";
  return "unknown";
}

static void dump_launch(hipFunction_t f, uint32_t gx,uint32_t gy,uint32_t gz, uint32_t lx,uint32_t ly,uint32_t lz,
                        size_t shmem, void** extra, const char *api) {
  auto it = g_names.find((void*)f);
  std::string sym = (it != g_names.end()) ? it->second : "UNKNOWN";
  if (!is_target(sym)) return;

  void *karg = nullptr;
  size_t ksz = 0;
  if (extra) {
    for (int i = 0; extra[i] != HIP_LAUNCH_PARAM_END && i < 16; i += 2) {
      if (extra[i] == HIP_LAUNCH_PARAM_BUFFER_POINTER) karg = extra[i+1];
      else if (extra[i] == HIP_LAUNCH_PARAM_BUFFER_SIZE) ksz = *(size_t*)extra[i+1];
    }
  }
  if (!karg || ksz == 0) return;

  std::string key = sym + "|" + std::to_string(gx) + "," + std::to_string(gy) + "," + std::to_string(gz) + "|" +
                    std::to_string(lx) + "," + std::to_string(ly) + "," + std::to_string(lz);
  if (g_seen.count(key)) return;
  g_seen.insert(key);

  uint8_t buf[256];
  size_t n = ksz > sizeof(buf) ? sizeof(buf) : ksz;
  memcpy(buf, karg, n);

  const char *outp = getenv("QK_MMVQ_KACAP");
  std::string path = outp ? outp : "/tmp/qk_decode_mmvq_kernarg.jsonl";
  FILE *fp = fopen(path.c_str(), "a");
  if (!fp) return;
  fprintf(fp, "{\"api\":\"%s\",\"type\":\"%s\",\"kernel_symbol\":\"%s\","
              "\"global\":[%u,%u,%u],\"local\":[%u,%u,%u],\"shared_mem\":%zu,"
              "\"num_workgroups\":[%u,%u,%u],\"kernarg_size\":%zu,\"captured_bytes\":%zu,\"kernarg_bytes\":[",
          api, type_name(sym), sym.c_str(), gx,gy,gz, lx,ly,lz, shmem,
          lx ? gx/lx : 0, ly ? gy/ly : 0, lz ? gz/lz : 0, ksz, n);
  for (size_t i = 0; i < n; i++) fprintf(fp, "%s%u", i ? "," : "", buf[i]);
  fprintf(fp, "]}\n");
  fclose(fp);
  fprintf(stderr, "[MMVQCAP] %s %s global(%u,%u,%u) local(%u,%u,%u) shmem=%zu ksz=%zu\n",
          type_name(sym), sym.substr(0, 80).c_str(), gx,gy,gz, lx,ly,lz, shmem, ksz);
}

static void dump_direct(const void *fn, dim3 grid, dim3 block, void **args, size_t shmem) {
  Dl_info info;
  std::string sym = "UNKNOWN";
  if (dladdr(fn, &info) && info.dli_sname) sym = info.dli_sname;
  if (sym == "UNKNOWN") sym = symbol_from_lowbits(fn);
  if (getenv("QK_MMVQ_TRACE_ALL") && g_trace_count++ < 80) {
    const char *outp = getenv("QK_MMVQ_TRACE_ALL");
    FILE *fp = fopen(outp, "a");
    if (fp) {
      fprintf(fp, "{\"fn\":\"%p\",\"symbol\":\"%s\",\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"shmem\":%zu}\n",
              fn, sym.c_str(), grid.x,grid.y,grid.z, block.x,block.y,block.z, shmem);
      fclose(fp);
    }
  }
  if (!is_target(sym)) return;
  if (!args) return;

  const int offsets[19] = {0,8,16,24,56,64,68,80,84,88,92,104,108,112,116,128,132,136,140};
  const int sizes[19]   = {8,8,8,32,8,4,12,4,4,4,12,4,4,4,12,4,4,4,4};
  uint8_t buf[144];
  memset(buf, 0, sizeof(buf));
  for (int i = 0; i < 19; i++) {
    if (!args[i]) return;
    memcpy(buf + offsets[i], args[i], sizes[i]);
  }

  std::string key = sym + "|" + std::to_string(grid.x) + "," + std::to_string(grid.y) + "," + std::to_string(grid.z) + "|" +
                    std::to_string(block.x) + "," + std::to_string(block.y) + "," + std::to_string(block.z);
  if (g_seen.count(key)) return;
  g_seen.insert(key);

  const char *outp = getenv("QK_MMVQ_KACAP");
  std::string path = outp ? outp : "/tmp/qk_decode_mmvq_kernarg.jsonl";
  FILE *fp = fopen(path.c_str(), "a");
  if (!fp) return;
  fprintf(fp, "{\"api\":\"hipLaunchKernel\",\"type\":\"%s\",\"kernel_symbol\":\"%s\","
              "\"global\":[%u,%u,%u],\"local\":[%u,%u,%u],\"shared_mem\":%zu,"
              "\"num_workgroups\":[%u,%u,%u],\"kernarg_size\":144,\"captured_bytes\":144,\"kernarg_bytes\":[",
          type_name(sym), sym.c_str(),
          grid.x * block.x, grid.y * block.y, grid.z * block.z, block.x, block.y, block.z, shmem,
          grid.x, grid.y, grid.z);
  for (size_t i = 0; i < sizeof(buf); i++) fprintf(fp, "%s%u", i ? "," : "", buf[i]);
  fprintf(fp, "]}\n");
  fclose(fp);
  fprintf(stderr, "[MMVQCAP] %s direct %.80s grid(%u,%u,%u) block(%u,%u,%u) shmem=%zu\n",
          type_name(sym), sym.substr(0, 80).c_str(), grid.x,grid.y,grid.z, block.x,block.y,block.z, shmem);
}

extern "C" hipError_t hipExtModuleLaunchKernel(hipFunction_t f, uint32_t gx,uint32_t gy,uint32_t gz,
    uint32_t lx,uint32_t ly,uint32_t lz, size_t shmem, hipStream_t stream, void** kparams, void** extra,
    hipEvent_t se, hipEvent_t ee, uint32_t flags) {
  if (!real_ext) real_ext = (ext_t)dlsym(RTLD_NEXT, "hipExtModuleLaunchKernel");
  dump_launch(f, gx,gy,gz, lx,ly,lz, shmem, extra, "hipExtModuleLaunchKernel");
  return real_ext(f,gx,gy,gz,lx,ly,lz,shmem,stream,kparams,extra,se,ee,flags);
}

extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t f, uint32_t gx,uint32_t gy,uint32_t gz,
    uint32_t lx,uint32_t ly,uint32_t lz, uint32_t shmem, hipStream_t stream, void** kparams, void** extra) {
  if (!real_mod) real_mod = (mod_t)dlsym(RTLD_NEXT, "hipModuleLaunchKernel");
  dump_launch(f, gx,gy,gz, lx,ly,lz, shmem, extra, "hipModuleLaunchKernel");
  return real_mod(f,gx,gy,gz,lx,ly,lz,shmem,stream,kparams,extra);
}

extern "C" hipError_t hipLaunchKernel(const void *function_address, dim3 numBlocks, dim3 dimBlocks,
    void **args, size_t sharedMemBytes, hipStream_t stream) {
  if (!real_launch) real_launch = (launch_t)dlsym(RTLD_NEXT, "hipLaunchKernel");
  dump_direct(function_address, numBlocks, dimBlocks, args, sharedMemBytes);
  return real_launch(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream);
}

extern "C" hipError_t hipLaunchKernel_spt(const void *function_address, dim3 numBlocks, dim3 dimBlocks,
    void **args, size_t sharedMemBytes, hipStream_t stream) {
  if (!real_launch_spt) real_launch_spt = (launch_t)dlsym(RTLD_NEXT, "hipLaunchKernel_spt");
  dump_direct(function_address, numBlocks, dimBlocks, args, sharedMemBytes);
  return real_launch_spt(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream);
}
