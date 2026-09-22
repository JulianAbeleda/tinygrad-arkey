// Cumulative llama production-stream conditioning for flash_attn_ext_vec.
//
// The Flash target is compiled from llama's own current template so its launch
// path matches the retained ~4 us standalone authority. Gate/up, down, Q, K,
// and V use the exact current libggml-cuda MMVQ cubin supplied on the CLI.
// Small quant/norm/completion hops retain production-sized buffer traffic.

#include "ggml-cuda/fattn-vec.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct FusionArgs { void * x_bias; void * gate; void * gate_bias; int32_t glu_op; };

static const char * MMVQ_Q4 =
  "_Z13mul_mat_vec_qIL9ggml_type12ELi1ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj";
static const char * MMVQ_Q4_FUSED =
  "_Z13mul_mat_vec_qIL9ggml_type12ELi1ELb1ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj";
static const char * MMVQ_Q6 =
  "_Z13mul_mat_vec_qIL9ggml_type14ELi1ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj";
static const char * MMVQ_Q6_FUSED =
  "_Z13mul_mat_vec_qIL9ggml_type14ELi1ELb1ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj";

static void check_cu(CUresult result, const char * what) {
  if (result == CUDA_SUCCESS) return;
  const char * text = nullptr; cuGetErrorString(result, &text);
  fprintf(stderr, "%s: %s\n", what, text ? text : "unknown CUresult"); exit(1);
}
static void check_rt(cudaError_t result, const char * what) {
  if (result == cudaSuccess) return;
  fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(result)); exit(1);
}

static uint3 fastdiv(uint32_t divisor) {
  uint32_t level = 0; while (level < 32 && (1u << level) < divisor) level++;
  uint32_t multiplier = (uint32_t)(((1ull << 32) * ((1ull << level) - divisor)) / divisor + 1);
  return make_uint3(multiplier, level, divisor);
}

struct Allocation {
  void * ptr = nullptr; size_t size = 0;
  Allocation() = default;
  explicit Allocation(size_t bytes) : size(bytes) {
    check_rt(cudaMalloc(&ptr, size), "cudaMalloc"); check_rt(cudaMemset(ptr, 0, size), "cudaMemset");
  }
  ~Allocation() { if (ptr) cudaFree(ptr); }
  Allocation(const Allocation &) = delete; Allocation & operator=(const Allocation &) = delete;
};

struct Harness {
  static constexpr int D = 128, HQ = 32, HKV = 8, MAX_TC = 768, TC = 768, SPLITS = 6;
  CUmodule module{}; CUfunction q4{}, q4_fused{}, q6{}, q6_fused{};
  Allocation w_gate_up{2ull*12288*(4096/256)*144};
  Allocation w_down{4096ull*(12288/256)*210};
  Allocation w_q{4096ull*(4096/256)*144};
  Allocation w_k{1024ull*(4096/256)*210};
  Allocation w_v{1024ull*(4096/256)*144};
  Allocation q8_4096{(4096/32)*36ull}, q8_12288{(12288/32)*36ull};
  Allocation gate_out{12288*4ull}, down_out{4096*4ull};
  Allocation q_raw{4096*4ull}, q_ready{4096*4ull}, k_raw{1024*4ull}, v_raw{1024*4ull};
  Allocation k_cache{HKV*MAX_TC*D*2ull}, v_cache{HKV*MAX_TC*D*2ull};
  Allocation flash_dst{HQ*SPLITS*D*4ull}, flash_meta{HQ*SPLITS*8ull};

  explicit Harness(const char * cubin) {
    check_rt(cudaFree(nullptr), "CUDA context initialization");
    check_cu(cuModuleLoad(&module, cubin), "cuModuleLoad MMVQ");
    check_cu(cuModuleGetFunction(&q4, module, MMVQ_Q4), "Q4 function");
    check_cu(cuModuleGetFunction(&q4_fused, module, MMVQ_Q4_FUSED), "Q4 fused function");
    check_cu(cuModuleGetFunction(&q6, module, MMVQ_Q6), "Q6 function");
    check_cu(cuModuleGetFunction(&q6_fused, module, MMVQ_Q6_FUSED), "Q6 fused function");
  }
  ~Harness() { if (module) cuModuleUnload(module); }

  void mmq(CUfunction fn, void * weight, void * activation, void * out, int rows, int k,
           void * gate = nullptr, void * bias = nullptr) {
    void * vx = weight; void * vy = activation; int32_t * ids = nullptr;
    FusionArgs fusion{bias, gate, nullptr, 2}; float * dst = (float *)out;
    uint32_t k32 = k, row_blocks = k/256, q8_blocks = k/32, rows32 = rows, zero32 = 0;
    uint3 zero = make_uint3(0, 0, 0), one = fastdiv(1);
    uint32_t nrb = rows*row_blocks, qb = q8_blocks;
    void * args[] = {&vx, &vy, &ids, &fusion, &dst, &k32, &zero, &row_blocks, &q8_blocks, &rows32, &one,
      &nrb, &qb, &rows32, &one, &nrb, &qb, &rows32, &zero32};
    check_cu(cuLaunchKernel(fn, rows, 1, 1, 32, 4, 1, 0, nullptr, args, nullptr), "MMVQ launch");
  }

  void flash() {
    constexpr int ncols = 1; constexpr ggml_type type_K = GGML_TYPE_F16, type_V = GGML_TYPE_F16;
    constexpr bool softcap = false;
    dim3 grid(1, SPLITS, HQ), block(32, 4, 1); const uint3 ne01 = init_fastdiv_values(1);
    const float scale = 1.0f/sqrtf((float)D);
    flash_attn_ext_vec<D, ncols, type_K, type_V, softcap><<<grid, block>>>(
      (const char *)q_ready.ptr, (const char *)k_cache.ptr, (const char *)v_cache.ptr,
      nullptr, nullptr, nullptr, (float *)flash_dst.ptr, (float2 *)flash_meta.ptr,
      scale, 0.0f, 1.0f, 1.0f, 32, 0.0f,
      D, ne01, HQ, 1,
      D*sizeof(float), D*sizeof(float), HQ*D*sizeof(float),
      D, TC, HKV, 1,
      D*sizeof(__half), MAX_TC*D*sizeof(__half), HKV*MAX_TC*D*sizeof(__half),
      D*sizeof(__half), MAX_TC*D*sizeof(__half), HKV*MAX_TC*D*sizeof(__half),
      0, 0, 0, 0, 0, 0);
    check_rt(cudaGetLastError(), "Flash launch");
  }

  void gate() {
    check_rt(cudaMemsetAsync(q8_4096.ptr, 0, q8_4096.size), "gate quant touch");
    char * second = (char *)w_gate_up.ptr + 12288ull*(4096/256)*144;
    mmq(q4_fused, w_gate_up.ptr, q8_4096.ptr, gate_out.ptr, 12288, 4096, second);
  }
  void down() {
    check_rt(cudaMemsetAsync(q8_12288.ptr, 0, q8_12288.size), "down quant touch");
    mmq(q6_fused, w_down.ptr, q8_12288.ptr, down_out.ptr, 4096, 12288, nullptr, down_out.ptr);
  }
  void attn_input() { check_rt(cudaMemsetAsync(q8_4096.ptr, 0, q8_4096.size), "attention input touch"); }
  void q_projection() { mmq(q4, w_q.ptr, q8_4096.ptr, q_raw.ptr, 4096, 4096); }
  void q_completion() { check_rt(cudaMemcpyAsync(q_ready.ptr, q_raw.ptr, q_ready.size, cudaMemcpyDeviceToDevice), "Q completion"); }
  void k_projection() {
    check_rt(cudaMemsetAsync(q8_4096.ptr, 0, q8_4096.size), "K quant touch");
    mmq(q6, w_k.ptr, q8_4096.ptr, k_raw.ptr, 1024, 4096);
  }
  void v_projection() {
    check_rt(cudaMemsetAsync(q8_4096.ptr, 0, q8_4096.size), "V quant touch");
    mmq(q4, w_v.ptr, q8_4096.ptr, v_raw.ptr, 1024, 4096);
  }
  void kv_completion() {
    check_rt(cudaMemcpyAsync(k_cache.ptr, k_raw.ptr, HKV*D*sizeof(__half), cudaMemcpyDeviceToDevice), "K store");
    check_rt(cudaMemcpyAsync(v_cache.ptr, v_raw.ptr, HKV*D*sizeof(__half), cudaMemcpyDeviceToDevice), "V store");
  }
  void prefix(int count) {
    if (count-- <= 0) return; gate();
    if (count-- <= 0) return; down();
    if (count-- <= 0) return; attn_input();
    if (count-- <= 0) return; q_projection();
    if (count-- <= 0) return; q_completion();
    if (count-- <= 0) return; k_projection();
    if (count-- <= 0) return; v_projection();
    if (count-- <= 0) return; kv_completion();
  }
};

static int prefix_count(const std::string & arm) {
  const char * names[] = {"hot", "gate", "ffn", "attn_input", "through_q", "through_qdone", "through_k", "through_v", "full_entry"};
  for (int i = 0; i < 9; i++) if (arm == names[i]) return i;
  fprintf(stderr, "unknown arm: %s\n", arm.c_str()); exit(2);
}

int main(int argc, char ** argv) {
  const char * cubin = nullptr; std::string arm = "hot"; int n = 48, warmup = 8;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--cubin")) cubin = argv[++i];
    else if (!strcmp(argv[i], "--arm")) arm = argv[++i];
    else if (!strcmp(argv[i], "--n")) n = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--warmup")) warmup = atoi(argv[++i]);
  }
  if (!cubin || n <= warmup) { fprintf(stderr, "usage: %s --cubin PATH --arm ARM --n N --warmup N\n", argv[0]); return 2; }
  Harness h(cubin); const int count = prefix_count(arm);
  std::vector<cudaEvent_t> starts(n), ends(n); std::vector<float> samples(n);
  for (int i = 0; i < n; i++) { check_rt(cudaEventCreate(&starts[i]), "start event"); check_rt(cudaEventCreate(&ends[i]), "end event"); }
  for (int i = 0; i < n; i++) {
    h.flash(); h.prefix(count); check_rt(cudaEventRecord(starts[i]), "record start"); h.flash(); check_rt(cudaEventRecord(ends[i]), "record end");
  }
  check_rt(cudaEventSynchronize(ends.back()), "event sync");
  for (int i = 0; i < n; i++) { float ms = 0; check_rt(cudaEventElapsedTime(&ms, starts[i], ends[i]), "elapsed"); samples[i] = ms*1000.0f; }
  for (auto x : starts) cudaEventDestroy(x); for (auto x : ends) cudaEventDestroy(x);
  std::vector<float> retained(samples.begin() + warmup, samples.end()), sorted = retained;
  std::sort(sorted.begin(), sorted.end());
  float median = sorted.size()%2 ? sorted[sorted.size()/2] : 0.5f*(sorted[sorted.size()/2-1] + sorted[sorted.size()/2]);
  double mean = 0; for (float x : retained) mean += x; mean /= retained.size();
  printf("arm=%s median_us=%.3f mean_us=%.3f min_us=%.3f max_us=%.3f samples_us=",
         arm.c_str(), median, mean, sorted.front(), sorted.back());
  for (size_t i = 0; i < retained.size(); i++) printf("%s%.3f", i ? "," : "", retained[i]);
  printf("\n"); return 0;
}
