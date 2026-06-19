#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <rocblas/rocblas.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr int WARMUP_ITERS = 10;
constexpr int TIMED_ITERS = 30;
constexpr double WMMA_PEAK_TFLOPS = 122.0;

struct Shape {
  const char *name;
  int m;
  int n;
  int k;
  double tinygrad_tflops;
};

struct BenchResult {
  bool ok = false;
  std::string status;
  float ms = 0.0f;
  double tflops = 0.0;
  double peak_pct = 0.0;
  int heuristic_count = -1;
};

const char *hipblas_status_name(hipblasStatus_t status) {
  switch (status) {
    case HIPBLAS_STATUS_SUCCESS: return "HIPBLAS_STATUS_SUCCESS";
    case HIPBLAS_STATUS_NOT_INITIALIZED: return "HIPBLAS_STATUS_NOT_INITIALIZED";
    case HIPBLAS_STATUS_ALLOC_FAILED: return "HIPBLAS_STATUS_ALLOC_FAILED";
    case HIPBLAS_STATUS_INVALID_VALUE: return "HIPBLAS_STATUS_INVALID_VALUE";
    case HIPBLAS_STATUS_MAPPING_ERROR: return "HIPBLAS_STATUS_MAPPING_ERROR";
    case HIPBLAS_STATUS_EXECUTION_FAILED: return "HIPBLAS_STATUS_EXECUTION_FAILED";
    case HIPBLAS_STATUS_INTERNAL_ERROR: return "HIPBLAS_STATUS_INTERNAL_ERROR";
    case HIPBLAS_STATUS_NOT_SUPPORTED: return "HIPBLAS_STATUS_NOT_SUPPORTED";
    case HIPBLAS_STATUS_ARCH_MISMATCH: return "HIPBLAS_STATUS_ARCH_MISMATCH";
    case HIPBLAS_STATUS_HANDLE_IS_NULLPTR: return "HIPBLAS_STATUS_HANDLE_IS_NULLPTR";
    case HIPBLAS_STATUS_INVALID_ENUM: return "HIPBLAS_STATUS_INVALID_ENUM";
    case HIPBLAS_STATUS_UNKNOWN: return "HIPBLAS_STATUS_UNKNOWN";
  }
  return "HIPBLAS_STATUS_UNRECOGNIZED";
}

const char *rocblas_status_name(rocblas_status status) {
  switch (status) {
    case rocblas_status_success: return "rocblas_status_success";
    case rocblas_status_invalid_handle: return "rocblas_status_invalid_handle";
    case rocblas_status_not_implemented: return "rocblas_status_not_implemented";
    case rocblas_status_invalid_pointer: return "rocblas_status_invalid_pointer";
    case rocblas_status_invalid_size: return "rocblas_status_invalid_size";
    case rocblas_status_memory_error: return "rocblas_status_memory_error";
    case rocblas_status_internal_error: return "rocblas_status_internal_error";
    case rocblas_status_perf_degraded: return "rocblas_status_perf_degraded";
    case rocblas_status_size_query_mismatch: return "rocblas_status_size_query_mismatch";
    case rocblas_status_size_increased: return "rocblas_status_size_increased";
    case rocblas_status_size_unchanged: return "rocblas_status_size_unchanged";
    case rocblas_status_invalid_value: return "rocblas_status_invalid_value";
    case rocblas_status_continue: return "rocblas_status_continue";
    case rocblas_status_check_numerics_fail: return "rocblas_status_check_numerics_fail";
    case rocblas_status_excluded_from_build: return "rocblas_status_excluded_from_build";
    case rocblas_status_arch_mismatch: return "rocblas_status_arch_mismatch";
  }
  return "rocblas_status_unrecognized";
}

void check_hip(hipError_t err, const char *what) {
  if (err != hipSuccess) {
    std::fprintf(stderr, "HIP error at %s: %s\n", what, hipGetErrorString(err));
    std::exit(2);
  }
}

void check_rocblas(rocblas_status status, const char *what) {
  if (status != rocblas_status_success) {
    std::fprintf(stderr, "rocBLAS error at %s: %s\n", what, rocblas_status_name(status));
    std::exit(3);
  }
}

void check_hipblas(hipblasStatus_t status, const char *what) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "hipBLASLt error at %s: %s\n", what, hipblas_status_name(status));
    std::exit(4);
  }
}

double shape_flops(const Shape &s) {
  return 2.0 * static_cast<double>(s.m) * static_cast<double>(s.n) * static_cast<double>(s.k);
}

void fill_result(const Shape &s, float total_ms, BenchResult &r) {
  r.ok = true;
  r.status = "ok";
  r.ms = total_ms / static_cast<float>(TIMED_ITERS);
  r.tflops = shape_flops(s) / (static_cast<double>(r.ms) * 1.0e9);
  r.peak_pct = 100.0 * r.tflops / WMMA_PEAK_TFLOPS;
}

BenchResult bench_rocblas(rocblas_handle handle, hipStream_t stream, const Shape &s, void *a, void *b, void *c) {
  BenchResult result;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto call = [&]() {
    return rocblas_gemm_ex(handle,
                           rocblas_operation_none,
                           rocblas_operation_none,
                           s.m,
                           s.n,
                           s.k,
                           &alpha,
                           a,
                           rocblas_datatype_f16_r,
                           s.m,
                           b,
                           rocblas_datatype_f16_r,
                           s.k,
                           &beta,
                           c,
                           rocblas_datatype_f16_r,
                           s.m,
                           c,
                           rocblas_datatype_f16_r,
                           s.m,
                           rocblas_datatype_f32_r,
                           rocblas_gemm_algo_standard,
                           0,
                           0);
  };
  rocblas_status status = call();
  if (status != rocblas_status_success) {
    result.status = rocblas_status_name(status);
    return result;
  }
  check_hip(hipStreamSynchronize(stream), "rocblas initial sync");
  for (int i = 0; i < WARMUP_ITERS; i++) check_rocblas(call(), "rocblas warmup");
  check_hip(hipStreamSynchronize(stream), "rocblas warmup sync");

  hipEvent_t start, stop;
  check_hip(hipEventCreate(&start), "rocblas event start");
  check_hip(hipEventCreate(&stop), "rocblas event stop");
  check_hip(hipEventRecord(start, stream), "rocblas event record start");
  for (int i = 0; i < TIMED_ITERS; i++) check_rocblas(call(), "rocblas timed");
  check_hip(hipEventRecord(stop, stream), "rocblas event record stop");
  check_hip(hipEventSynchronize(stop), "rocblas event sync");
  float total_ms = 0.0f;
  check_hip(hipEventElapsedTime(&total_ms, start, stop), "rocblas elapsed");
  check_hip(hipEventDestroy(start), "rocblas event destroy start");
  check_hip(hipEventDestroy(stop), "rocblas event destroy stop");
  fill_result(s, total_ms, result);
  return result;
}

BenchResult bench_hipblaslt(hipblasLtHandle_t handle, hipStream_t stream, const Shape &s, void *a, void *b, void *c, void *workspace, size_t workspace_size) {
  BenchResult result;
  hipblasLtMatmulDesc_t matmul = nullptr;
  hipblasLtMatrixLayout_t adesc = nullptr, bdesc = nullptr, cdesc = nullptr;
  hipblasLtMatmulPreference_t pref = nullptr;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  hipblasStatus_t status = hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    result.status = hipblas_status_name(status);
    return result;
  }
  hipblasOperation_t op = HIPBLAS_OP_N;
  check_hipblas(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op)), "hipblaslt transa");
  check_hipblas(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &op, sizeof(op)), "hipblaslt transb");
  check_hipblas(hipblasLtMatrixLayoutCreate(&adesc, HIP_R_16F, s.m, s.k, s.m), "hipblaslt A layout");
  check_hipblas(hipblasLtMatrixLayoutCreate(&bdesc, HIP_R_16F, s.k, s.n, s.k), "hipblaslt B layout");
  check_hipblas(hipblasLtMatrixLayoutCreate(&cdesc, HIP_R_16F, s.m, s.n, s.m), "hipblaslt C layout");
  check_hipblas(hipblasLtMatmulPreferenceCreate(&pref), "hipblaslt pref");
  uint64_t max_workspace = workspace_size;
  check_hipblas(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace, sizeof(max_workspace)), "hipblaslt workspace pref");

  constexpr int REQUESTED_ALGOS = 32;
  std::vector<hipblasLtMatmulHeuristicResult_t> heuristics(REQUESTED_ALGOS);
  int returned = 0;
  status = hipblasLtMatmulAlgoGetHeuristic(handle, matmul, adesc, bdesc, cdesc, cdesc, pref, REQUESTED_ALGOS, heuristics.data(), &returned);
  result.heuristic_count = returned;
  if (status != HIPBLAS_STATUS_SUCCESS || returned <= 0) {
    result.status = (status == HIPBLAS_STATUS_SUCCESS) ? "no_heuristic" : hipblas_status_name(status);
    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtMatrixLayoutDestroy(cdesc);
    hipblasLtMatrixLayoutDestroy(bdesc);
    hipblasLtMatrixLayoutDestroy(adesc);
    hipblasLtMatmulDescDestroy(matmul);
    return result;
  }

  auto call = [&]() {
    return hipblasLtMatmul(handle,
                           matmul,
                           &alpha,
                           a,
                           adesc,
                           b,
                           bdesc,
                           &beta,
                           c,
                           cdesc,
                           c,
                           cdesc,
                           &heuristics[0].algo,
                           workspace,
                           workspace_size,
                           stream);
  };
  status = call();
  if (status != HIPBLAS_STATUS_SUCCESS) {
    result.status = hipblas_status_name(status);
  } else {
    check_hip(hipStreamSynchronize(stream), "hipblaslt initial sync");
    for (int i = 0; i < WARMUP_ITERS; i++) check_hipblas(call(), "hipblaslt warmup");
    check_hip(hipStreamSynchronize(stream), "hipblaslt warmup sync");

    hipEvent_t start, stop;
    check_hip(hipEventCreate(&start), "hipblaslt event start");
    check_hip(hipEventCreate(&stop), "hipblaslt event stop");
    check_hip(hipEventRecord(start, stream), "hipblaslt event record start");
    for (int i = 0; i < TIMED_ITERS; i++) check_hipblas(call(), "hipblaslt timed");
    check_hip(hipEventRecord(stop, stream), "hipblaslt event record stop");
    check_hip(hipEventSynchronize(stop), "hipblaslt event sync");
    float total_ms = 0.0f;
    check_hip(hipEventElapsedTime(&total_ms, start, stop), "hipblaslt elapsed");
    check_hip(hipEventDestroy(start), "hipblaslt event destroy start");
    check_hip(hipEventDestroy(stop), "hipblaslt event destroy stop");
    fill_result(s, total_ms, result);
  }

  hipblasLtMatmulPreferenceDestroy(pref);
  hipblasLtMatrixLayoutDestroy(cdesc);
  hipblasLtMatrixLayoutDestroy(bdesc);
  hipblasLtMatrixLayoutDestroy(adesc);
  hipblasLtMatmulDescDestroy(matmul);
  return result;
}

void print_result(const BenchResult &r, bool last) {
  std::printf("      \"ok\": %s,\n", r.ok ? "true" : "false");
  std::printf("      \"status\": \"%s\",\n", r.status.c_str());
  if (r.heuristic_count >= 0) std::printf("      \"heuristic_count\": %d,\n", r.heuristic_count);
  std::printf("      \"ms\": %.6f,\n", r.ms);
  std::printf("      \"tflops\": %.6f,\n", r.tflops);
  std::printf("      \"peak_pct\": %.3f\n", r.peak_pct);
  std::printf("    }%s\n", last ? "" : ",");
}

} // namespace

int main() {
  std::vector<Shape> shapes = {
      {"ffn_gate_up", 512, 12288, 4096, 40.8},
      {"ffn_down", 512, 4096, 12288, 40.8},
      {"attn_q_o", 512, 4096, 4096, 40.8},
      {"attn_k_v", 512, 1024, 4096, 40.8},
  };

  int device = 0;
  hipDeviceProp_t prop;
  check_hip(hipSetDevice(device), "set device");
  check_hip(hipGetDeviceProperties(&prop, device), "device properties");

  hipStream_t stream;
  check_hip(hipStreamCreate(&stream), "stream create");
  rocblas_handle roc_handle;
  check_rocblas(rocblas_create_handle(&roc_handle), "rocblas handle");
  check_rocblas(rocblas_set_stream(roc_handle, stream), "rocblas stream");
  hipblasLtHandle_t lt_handle;
  check_hipblas(hipblasLtCreate(&lt_handle), "hipblaslt handle");

  constexpr size_t WORKSPACE_SIZE = 64ull * 1024ull * 1024ull;
  void *workspace = nullptr;
  check_hip(hipMalloc(&workspace, WORKSPACE_SIZE), "workspace malloc");

  std::printf("{\n");
  std::printf("  \"schema\": \"qk_prefill_external_blas_ceiling_v1\",\n");
  std::printf("  \"device\": \"%s\",\n", prop.name);
  std::printf("  \"warmup_iters\": %d,\n", WARMUP_ITERS);
  std::printf("  \"timed_iters\": %d,\n", TIMED_ITERS);
  std::printf("  \"wmma_peak_tflops_assumed\": %.1f,\n", WMMA_PEAK_TFLOPS);
  std::printf("  \"shapes\": [\n");

  for (size_t i = 0; i < shapes.size(); i++) {
    const Shape &s = shapes[i];
    size_t a_bytes = static_cast<size_t>(s.m) * static_cast<size_t>(s.k) * sizeof(_Float16);
    size_t b_bytes = static_cast<size_t>(s.k) * static_cast<size_t>(s.n) * sizeof(_Float16);
    size_t c_bytes = static_cast<size_t>(s.m) * static_cast<size_t>(s.n) * sizeof(_Float16);
    void *a = nullptr, *b = nullptr, *c = nullptr;
    check_hip(hipMalloc(&a, a_bytes), "A malloc");
    check_hip(hipMalloc(&b, b_bytes), "B malloc");
    check_hip(hipMalloc(&c, c_bytes), "C malloc");
    check_hip(hipMemsetAsync(a, 0x01, a_bytes, stream), "A memset");
    check_hip(hipMemsetAsync(b, 0x02, b_bytes, stream), "B memset");
    check_hip(hipMemsetAsync(c, 0x00, c_bytes, stream), "C memset");
    check_hip(hipStreamSynchronize(stream), "shape memset sync");

    BenchResult roc = bench_rocblas(roc_handle, stream, s, a, b, c);
    BenchResult lt = bench_hipblaslt(lt_handle, stream, s, a, b, c, workspace, WORKSPACE_SIZE);

    std::printf("    {\n");
    std::printf("      \"name\": \"%s\",\n", s.name);
    std::printf("      \"m\": %d,\n", s.m);
    std::printf("      \"n\": %d,\n", s.n);
    std::printf("      \"k\": %d,\n", s.k);
    std::printf("      \"tinygrad_tflops_reference\": %.3f,\n", s.tinygrad_tflops);
    std::printf("      \"libraries\": {\n");
    std::printf("    \"rocblas\": {\n");
    print_result(roc, false);
    std::printf("    \"hipblaslt\": {\n");
    print_result(lt, true);
    std::printf("      }\n");
    std::printf("    }%s\n", (i + 1 == shapes.size()) ? "" : ",");

    check_hip(hipFree(c), "C free");
    check_hip(hipFree(b), "B free");
    check_hip(hipFree(a), "A free");
  }

  std::printf("  ]\n");
  std::printf("}\n");

  check_hip(hipFree(workspace), "workspace free");
  check_hipblas(hipblasLtDestroy(lt_handle), "hipblaslt destroy");
  check_rocblas(rocblas_destroy_handle(roc_handle), "rocblas destroy");
  check_hip(hipStreamDestroy(stream), "stream destroy");
  return 0;
}
