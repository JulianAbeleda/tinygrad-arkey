// cuda_stream_overlap_probe.cu - E3 decisive-experiment CUDA leg
//
// Standalone CUDA-runtime probe (no tinygrad) mirroring the native multi-GPFIFO
// probe (extra/llm_research/decode/nv_multi_queue_probe.py) through the CUDA
// runtime, per section 3 of docs/task_workflow/input/nv-decode-parity-e1e3-
// measurement-scope-20260803.md: on this exact device/driver, do two or three
// independent CUDA streams co-schedule independent kernels (span < node-sum)?
// Same span/node-sum criterion and numeric checks as the native probe.
//
// Launches --kernels independent kernels per stream on --streams 1 (serial
// calibration; span must equal node-sum), 2, or 3 independent
// cudaStreamNonBlocking streams. No dependencies between streams, no graph
// capture (no cudaStreamBeginCapture): plain stream concurrency. Flavors:
//   a. elementwise: n floats, y[i] = x[i] * 1.000001f + (float)i (memory-bound)
//   b. partial-SM elementwise: the same kernel, grid divided by --grid-div
//      (2/4/8 partial-SM, default 1 = full grid)
//   c. optional compute-heavy MxMxM matmul via --matmul M (default 0 = off)
// Timing: CUDA events around every kernel (start/end) plus wall events around
// the whole batch; overlap = (node-sum - span) / node-sum. Host-side sampled
// numeric checks (every 100000th element), like the native probe's allclose.
//
//   nvcc -O3 -arch=sm_120 -o cuda_stream_overlap_probe cuda_stream_overlap_probe.cu
//   ./cuda_stream_overlap_probe --streams 1 --json serial.json    # calibration
//   ./cuda_stream_overlap_probe --streams 2 --json two_stream.json
//   ./cuda_stream_overlap_probe --streams 2 --grid-div 8          # partial-SM
//   ./cuda_stream_overlap_probe --streams 3 --grid-div 4
//   ./cuda_stream_overlap_probe --streams 2 --matmul 2048         # compute-heavy

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#ifndef TPB
#define TPB 256
#endif

#ifndef MM_TILE
#define MM_TILE 32
#endif

#define MAX_STREAMS 3
#define SAMPLE_STRIDE 100000

#define CUDA_CHECK(x) do { \
  cudaError_t err_ = (x); \
  if (err_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n", __FILE__, __LINE__, \
            cudaGetErrorString(err_), (int)err_); \
    exit(1); \
  } \
} while (0)

struct Experiment {
  const char* name;
  const char* check;
  int streams, kernels;
  long long grid;
  double* dur_us;            // [streams * kernels], stream-major
  double* stream_span_us;    // per-stream wall0 -> wall1
  double* stream_sum_us;     // per-stream node-sum
  double span_us, node_sum_us, wall_us, overlap;
  int numeric_ok, pass;
  double max_err;
  long long samples, check_stride;
};

__global__ __launch_bounds__(TPB) void ew_kernel(const float* __restrict__ x,
                                                 float* __restrict__ y, long long n) {
  long long stride = (long long)gridDim.x * blockDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    y[i] = x[i] * 1.000001f + (float)i;
  }
}

__global__ __launch_bounds__(MM_TILE * MM_TILE) void sgemm_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ c, int M, int K, int Nn) {
  __shared__ float as[MM_TILE][MM_TILE];
  __shared__ float bs[MM_TILE][MM_TILE];
  int row = blockIdx.y * MM_TILE + threadIdx.y;
  int col = blockIdx.x * MM_TILE + threadIdx.x;
  float acc = 0.0f;
  for (int t = 0; t < (K + MM_TILE - 1) / MM_TILE; ++t) {
    int ka = t * MM_TILE + threadIdx.x;
    int kb = t * MM_TILE + threadIdx.y;
    as[threadIdx.y][threadIdx.x] = (row < M && ka < K) ? a[(long long)row * K + ka] : 0.0f;
    bs[threadIdx.y][threadIdx.x] = (col < Nn && kb < K) ? b[(long long)kb * Nn + col] : 0.0f;
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < MM_TILE; ++j) acc += as[threadIdx.y][j] * bs[j][threadIdx.x];
    __syncthreads();
  }
  if (row < M && col < Nn) c[(long long)row * Nn + col] = acc;
}

static void elapsed_us(cudaEvent_t a, cudaEvent_t b, double* us) {
  float ms = 0.0f;
  cudaError_t err = cudaEventElapsedTime(&ms, a, b);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaEventElapsedTime failed: %s\n", cudaGetErrorString(err));
    exit(1);
  }
  *us = (double)ms * 1000.0;
}

typedef void (*launch_fn)(int k, int s, void* ctx);

// Enqueue the whole batch (kernels x streams) with per-kernel start/end events
// and per-stream wall events, then reduce to durations, node-sum, span, wall.
// The wall0 records are enqueued back to back before any kernel, so treating
// each stream's own wall0 as its reference zero is valid per cudaEventElapsedTime
// (same-stream pairs only) and the cross-stream offsets are at most a few us.
static void timed_batch(cudaStream_t* st, int streams, int kernels,
                        launch_fn launch, void* ctx, Experiment* out) {
  cudaEvent_t wall0[MAX_STREAMS], wall1[MAX_STREAMS];
  cudaEvent_t* start = new cudaEvent_t[streams * kernels];
  cudaEvent_t* end = new cudaEvent_t[streams * kernels];
  for (int s = 0; s < streams; s++) {
    CUDA_CHECK(cudaEventCreate(&wall0[s]));
    CUDA_CHECK(cudaEventCreate(&wall1[s]));
  }
  for (int i = 0; i < streams * kernels; i++) {
    CUDA_CHECK(cudaEventCreate(&start[i]));
    CUDA_CHECK(cudaEventCreate(&end[i]));
  }
  for (int s = 0; s < streams; s++) CUDA_CHECK(cudaEventRecord(wall0[s], st[s]));
  for (int k = 0; k < kernels; k++)
    for (int s = 0; s < streams; s++) {
      int i = s * kernels + k;
      CUDA_CHECK(cudaEventRecord(start[i], st[s]));
      launch(k, s, ctx);
      CUDA_CHECK(cudaEventRecord(end[i], st[s]));
    }
  for (int s = 0; s < streams; s++) CUDA_CHECK(cudaEventRecord(wall1[s], st[s]));
  CUDA_CHECK(cudaDeviceSynchronize());

  out->dur_us = new double[streams * kernels];
  out->stream_span_us = new double[streams];
  out->stream_sum_us = new double[streams];
  double node_sum = 0.0;
  out->wall_us = 0.0;
  for (int s = 0; s < streams; s++) {
    double ssum = 0.0;
    for (int k = 0; k < kernels; k++) {
      int i = s * kernels + k;
      elapsed_us(start[i], end[i], &out->dur_us[i]);
      ssum += out->dur_us[i];
    }
    out->stream_sum_us[s] = ssum;
    node_sum += ssum;
    elapsed_us(wall0[s], wall1[s], &out->stream_span_us[s]);
    if (out->stream_span_us[s] > out->wall_us) out->wall_us = out->stream_span_us[s];
  }
  out->node_sum_us = node_sum;

  // span = max end - min start across all kernels, per-stream wall0 references.
  double rmin = 1e30, rmax = -1e30;
  for (int s = 0; s < streams; s++)
    for (int k = 0; k < kernels; k++) {
      double rs, re;
      elapsed_us(wall0[s], start[s * kernels + k], &rs);
      elapsed_us(wall0[s], end[s * kernels + k], &re);
      if (rs < rmin) rmin = rs;
      if (re > rmax) rmax = re;
    }
  out->span_us = rmax - rmin;
  out->overlap = out->node_sum_us > 0.0 ? (out->node_sum_us - out->span_us) / out->node_sum_us : 0.0;

  for (int i = 0; i < streams * kernels; i++) {
    CUDA_CHECK(cudaEventDestroy(start[i]));
    CUDA_CHECK(cudaEventDestroy(end[i]));
  }
  for (int s = 0; s < streams; s++) {
    CUDA_CHECK(cudaEventDestroy(wall0[s]));
    CUDA_CHECK(cudaEventDestroy(wall1[s]));
  }
  delete[] start;
  delete[] end;
}

struct EwCtx {
  cudaStream_t* st;
  float** d_x;
  float** d_y;
  long long n, grid;
};

static void ew_launch(int k, int s, void* vctx) {
  (void)k;
  EwCtx* c = (EwCtx*)vctx;
  ew_kernel<<<(unsigned)c->grid, TPB, 0, c->st[s]>>>(c->d_x[s], c->d_y[s], c->n);
}

static void run_ew_experiment(int streams, int kernels, long long n, int grid_div,
                              const char* name, const char* check, Experiment* out) {
  out->name = name;
  out->check = check;
  out->streams = streams;
  out->kernels = kernels;
  out->check_stride = SAMPLE_STRIDE;

  // Per-stream x filled with a distinct constant so streams are independent;
  // y[i] = x[i] * 1.000001f + (float)i pins the host-side numeric check.
  float** h_x = new float*[streams];
  float** h_y = new float*[streams];
  float** d_x = new float*[streams];
  float** d_y = new float*[streams];
  for (int s = 0; s < streams; s++) {
    h_x[s] = new float[n];
    for (long long i = 0; i < n; i++) h_x[s][i] = (float)(s + 1);
    h_y[s] = new float[n];
    CUDA_CHECK(cudaMalloc(&d_x[s], (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y[s], (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x[s], h_x[s], (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
  }

  cudaStream_t st[MAX_STREAMS];
  for (int s = 0; s < streams; s++)
    CUDA_CHECK(cudaStreamCreateWithFlags(&st[s], cudaStreamNonBlocking));

  long long grid = (n + TPB - 1) / TPB / grid_div;
  if (grid < 1) grid = 1;
  out->grid = grid;

  // Warmup batch (module load, clock ramp, TLB); not timed.
  for (int k = 0; k < kernels; k++)
    for (int s = 0; s < streams; s++)
      ew_kernel<<<(unsigned)grid, TPB, 0, st[s]>>>(d_x[s], d_y[s], n);
  CUDA_CHECK(cudaDeviceSynchronize());

  EwCtx ctx = { st, d_x, d_y, n, grid };
  timed_batch(st, streams, kernels, ew_launch, &ctx, out);

  // Host-side sampled numeric check (every 100000th element), the native
  // probe's allclose criterion in spirit. (float)i is exact for these even
  // samples; tolerance covers float rounding of the multiply and add.
  double max_err = 0.0;
  long long samples = 0;
  int ok = 1;
  for (int s = 0; s < streams; s++) {
    CUDA_CHECK(cudaMemcpy(h_y[s], d_y[s], (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
    for (long long i = 0; i < n; i += SAMPLE_STRIDE) {
      double expected = (double)h_x[s][i] * 1.000001 + (double)i;
      double err = fabs((double)h_y[s][i] - expected);
      if (err > max_err) max_err = err;
      if (err > 1e-4 * fabs(expected) + 4.0) ok = 0;
      samples++;
    }
  }
  out->numeric_ok = ok;
  out->max_err = max_err;
  out->samples = samples;

  // Native thresholds: serial calibration passes at ~0 overlap (span ==
  // node-sum), concurrent modes need >= 5% overlap.
  out->pass = (streams == 1 ? out->overlap < 0.05 : out->overlap >= 0.05) && ok;

  for (int s = 0; s < streams; s++) {
    CUDA_CHECK(cudaStreamDestroy(st[s]));
    CUDA_CHECK(cudaFree(d_x[s]));
    CUDA_CHECK(cudaFree(d_y[s]));
    delete[] h_x[s];
    delete[] h_y[s];
  }
  delete[] h_x; delete[] h_y; delete[] d_x; delete[] d_y;
}

struct MmCtx {
  cudaStream_t* st;
  float** d_a;
  float** d_b;
  float** d_c;
  int M;
  dim3 gridm, block;
};

static void mm_launch(int k, int s, void* vctx) {
  (void)k;
  MmCtx* c = (MmCtx*)vctx;
  sgemm_kernel<<<c->gridm, c->block, 0, c->st[s]>>>(c->d_a[s], c->d_b[s], c->d_c[s], c->M, c->M, c->M);
}

static void run_mm_experiment(int streams, int kernels, int M,
                              const char* name, const char* check, Experiment* out) {
  out->name = name;
  out->check = check;
  out->streams = streams;
  out->kernels = kernels;
  int rowstep = M / 16 < 1 ? 1 : M / 16;
  out->check_stride = rowstep;

  size_t mbytes = (size_t)M * M * sizeof(float);
  float** h_a = new float*[streams];
  float** h_b = new float*[streams];
  float** h_c = new float*[streams];
  float** d_a = new float*[streams];
  float** d_b = new float*[streams];
  float** d_c = new float*[streams];
  for (int s = 0; s < streams; s++) {
    h_a[s] = new float[(size_t)M * M];
    h_b[s] = new float[(size_t)M * M];
    h_c[s] = new float[(size_t)M * M];
    for (long long i = 0; i < (long long)M * M; i++) { h_a[s][i] = 0.0f; h_b[s][i] = 0.0f; }
    // A = B = sqrt(M)*I pins C = M*I, the native probe's matmul check.
    for (int i = 0; i < M; i++) {
      h_a[s][i * M + i] = sqrtf((float)M);
      h_b[s][i * M + i] = sqrtf((float)M);
    }
    CUDA_CHECK(cudaMalloc(&d_a[s], mbytes));
    CUDA_CHECK(cudaMalloc(&d_b[s], mbytes));
    CUDA_CHECK(cudaMalloc(&d_c[s], mbytes));
    CUDA_CHECK(cudaMemcpy(d_a[s], h_a[s], mbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b[s], h_b[s], mbytes, cudaMemcpyHostToDevice));
  }

  cudaStream_t st[MAX_STREAMS];
  for (int s = 0; s < streams; s++)
    CUDA_CHECK(cudaStreamCreateWithFlags(&st[s], cudaStreamNonBlocking));

  dim3 block(MM_TILE, MM_TILE);
  unsigned nt = (unsigned)((M + MM_TILE - 1) / MM_TILE);
  dim3 gridm(nt, nt);
  out->grid = (long long)nt * nt;

  // Warmup batch (module load, clock ramp); not timed.
  for (int k = 0; k < kernels; k++)
    for (int s = 0; s < streams; s++)
      sgemm_kernel<<<gridm, block, 0, st[s]>>>(d_a[s], d_b[s], d_c[s], M, M, M);
  CUDA_CHECK(cudaDeviceSynchronize());

  MmCtx ctx = { st, d_a, d_b, d_c, M, gridm, block };
  timed_batch(st, streams, kernels, mm_launch, &ctx, out);

  // Sampled numeric check: C = M*I, diagonal M, off-diagonal 0.
  double max_err = 0.0;
  long long samples = 0;
  int ok = 1;
  double tol = 1e-2 * (double)M + 1.0;
  for (int s = 0; s < streams; s++) {
    CUDA_CHECK(cudaMemcpy(h_c[s], d_c[s], mbytes, cudaMemcpyDeviceToHost));
    for (int r = 0; r < M; r += rowstep) {
      double errd = fabs((double)h_c[s][r * M + r] - (double)M);
      if (errd > max_err) max_err = errd;
      if (errd > tol) ok = 0;
      samples++;
      if (M > 1) {
        int col = (r + 1) % M;
        double erro = fabs((double)h_c[s][r * M + col]);
        if (erro > max_err) max_err = erro;
        if (erro > tol) ok = 0;
        samples++;
      }
    }
  }
  out->numeric_ok = ok;
  out->max_err = max_err;
  out->samples = samples;

  out->pass = (streams == 1 ? out->overlap < 0.05 : out->overlap >= 0.05) && ok;

  for (int s = 0; s < streams; s++) {
    CUDA_CHECK(cudaStreamDestroy(st[s]));
    CUDA_CHECK(cudaFree(d_a[s]));
    CUDA_CHECK(cudaFree(d_b[s]));
    CUDA_CHECK(cudaFree(d_c[s]));
    delete[] h_a[s];
    delete[] h_b[s];
    delete[] h_c[s];
  }
  delete[] h_a; delete[] h_b; delete[] h_c; delete[] d_a; delete[] d_b; delete[] d_c;
}

static const char* arg_val(int argc, char** argv, int* i, const char* opt) {
  if (*i + 1 >= argc) {
    fprintf(stderr, "missing value for %s\n", opt);
    exit(2);
  }
  return argv[++(*i)];
}

static void usage(FILE* f, const char* prog) {
  fprintf(f, "usage: %s [--streams N] [--n N] [--kernels N] [--grid-div N] [--matmul M] [--json FILE]\n"
             "  --streams  N  1 = serial calibration (span == node-sum), 2 or 3 = independent\n"
             "                cudaStreamNonBlocking streams (default 2)\n"
             "  --n        N  elementwise vector length, floats (default 33554432)\n"
             "  --kernels  N  independent kernels per stream (default 8)\n"
             "  --grid-div N  divide the elementwise grid by N: 1 = full grid, 2/4/8 = partial-SM\n"
             "                (default 1)\n"
             "  --matmul   M  run the compute-heavy MxMxM matmul flavor instead (default 0 = off;\n"
             "                e.g. --matmul 2048)\n"
             "  --json  FILE  write the measurement record to FILE (also printed to stdout)\n",
             prog);
}

static void write_json(FILE* f, const cudaDeviceProp* prop, int driver, int streams,
                       int kernels, long long n, int grid_div, int matmul,
                       const Experiment* e, const char* status, const char* verdict) {
  fprintf(f, "{\n");
  fprintf(f, "  \"schema\": \"tinygrad.cuda_stream_overlap_probe.v1\",\n");
  fprintf(f, "  \"leg\": \"E3\",\n");
  fprintf(f, "  \"device\": {\"name\": \"%s\", \"compute_capability\": \"%d.%d\", \"cuda_driver_version\": %d},\n",
          prop->name, prop->major, prop->minor, driver);
  fprintf(f, "  \"config\": {\"streams\": %d, \"kernels_per_stream\": %d, \"n\": %lld, "
             "\"grid_div\": %d, \"matmul\": %d, \"threads_per_block\": %d},\n",
          streams, kernels, n, grid_div, matmul, TPB);
  fprintf(f, "  \"verdict\": \"%s\",\n", verdict);
  fprintf(f, "  \"experiments\": [\n");
  fprintf(f, "    {\n");
  fprintf(f, "      \"name\": \"%s\",\n", e->name);
  fprintf(f, "      \"status\": \"%s\",\n", status);
  fprintf(f, "      \"check\": \"%s\",\n", e->check);
  fprintf(f, "      \"grid\": %lld,\n", e->grid);
  fprintf(f, "      \"dur_us\": [");
  for (int i = 0; i < streams * kernels; i++) fprintf(f, "%s%g", i ? ", " : "", e->dur_us[i]);
  fprintf(f, "],\n");
  fprintf(f, "      \"stream_span_us\": [");
  for (int s = 0; s < streams; s++) fprintf(f, "%s%g", s ? ", " : "", e->stream_span_us[s]);
  fprintf(f, "],\n");
  fprintf(f, "      \"stream_sum_us\": [");
  for (int s = 0; s < streams; s++) fprintf(f, "%s%g", s ? ", " : "", e->stream_sum_us[s]);
  fprintf(f, "],\n");
  fprintf(f, "      \"span_us\": %g,\n", e->span_us);
  fprintf(f, "      \"node_sum_us\": %g,\n", e->node_sum_us);
  fprintf(f, "      \"wall_us\": %g,\n", e->wall_us);
  fprintf(f, "      \"overlap\": %g,\n", e->overlap);
  fprintf(f, "      \"numeric_ok\": %s,\n", e->numeric_ok ? "true" : "false");
  fprintf(f, "      \"max_err\": %g,\n", e->max_err);
  fprintf(f, "      \"samples_checked\": %lld,\n", e->samples);
  fprintf(f, "      \"check_stride\": %lld\n", e->check_stride);
  fprintf(f, "    }\n");
  fprintf(f, "  ]\n");
  fprintf(f, "}\n");
}

int main(int argc, char** argv) {
  int streams = 2, kernels = 8, grid_div = 1, matmul = 0;
  long long n = 1LL << 25;
  const char* json_path = NULL;

  for (int i = 1; i < argc; i++) {
    const char* a = argv[i];
    if (!strcmp(a, "--help") || !strcmp(a, "-h")) {
      usage(stdout, argv[0]);
      return 0;
    } else if (!strcmp(a, "--streams")) streams = atoi(arg_val(argc, argv, &i, a));
    else if (!strcmp(a, "--n")) n = atoll(arg_val(argc, argv, &i, a));
    else if (!strcmp(a, "--kernels")) kernels = atoi(arg_val(argc, argv, &i, a));
    else if (!strcmp(a, "--grid-div")) grid_div = atoi(arg_val(argc, argv, &i, a));
    else if (!strcmp(a, "--matmul")) matmul = atoi(arg_val(argc, argv, &i, a));
    else if (!strcmp(a, "--json")) json_path = arg_val(argc, argv, &i, a);
    else {
      fprintf(stderr, "unknown option: %s\n", a);
      usage(stderr, argv[0]);
      return 2;
    }
  }

  if (streams < 1 || streams > 3) {
    fprintf(stderr, "--streams must be 1, 2, or 3 (got %d)\n", streams);
    return 2;
  }
  if (kernels < 1) { fprintf(stderr, "--kernels must be >= 1\n"); return 2; }
  if (grid_div < 1) { fprintf(stderr, "--grid-div must be >= 1\n"); return 2; }
  if (n < 1) { fprintf(stderr, "--n must be >= 1\n"); return 2; }
  if (matmul < 0) { fprintf(stderr, "--matmul must be >= 0\n"); return 2; }
  if (matmul > 0 && grid_div > 1)
    fprintf(stderr, "note: --grid-div %d ignored for the --matmul flavor\n", grid_div);

  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int driver = 0;
  CUDA_CHECK(cudaDriverGetVersion(&driver));
  fprintf(stderr, "device ready: %s (sm_%d%d), CUDA driver %d, streams=%d kernels=%d "
                  "n=%lld grid_div=%d matmul=%d\n",
          prop.name, prop.major, prop.minor, driver, streams, kernels, n, grid_div, matmul);

  char name[96], check[160];
  const char* flavor = matmul > 0 ? "matmul" : (grid_div > 1 ? "elementwise-partial-sm" : "elementwise");
  if (streams == 1) snprintf(name, sizeof(name), "E2-cuda-serial-%s", flavor);
  else if (streams == 2) snprintf(name, sizeof(name), "E3-cuda-2stream-%s", flavor);
  else snprintf(name, sizeof(name), "E4-cuda-3stream-%s", flavor);
  if (matmul > 0)
    snprintf(check, sizeof(check), "%d-stream %dx%dx%d matmul overlap + correctness", streams, matmul, matmul, matmul);
  else if (grid_div > 1)
    snprintf(check, sizeof(check), "%d-stream elementwise overlap, grid/%d partial-SM", streams, grid_div);
  else
    snprintf(check, sizeof(check), "%d-stream elementwise overlap, full grid", streams);

  Experiment e;
  memset(&e, 0, sizeof(e));
  if (matmul > 0) run_mm_experiment(streams, kernels, matmul, name, check, &e);
  else run_ew_experiment(streams, kernels, n, grid_div, name, check, &e);

  fprintf(stderr, "%s span=%.1fus sum=%.1fus overlap=%.1f%% wall=%.1fus stream_spans=[",
          e.name, e.span_us, e.node_sum_us, e.overlap * 100.0, e.wall_us);
  for (int s = 0; s < streams; s++) fprintf(stderr, "%s%.1f", s ? "," : "", e.stream_span_us[s]);
  fprintf(stderr, "] numeric_ok=%d max_err=%.3e samples=%lld\n",
          e.numeric_ok, e.max_err, e.samples);

  const char* status = e.pass ? "pass" : "FAIL";
  const char* verdict = e.pass ? "PASS" : "FAIL";
  write_json(stdout, &prop, driver, streams, kernels, n, grid_div, matmul, &e, status, verdict);
  if (json_path) {
    FILE* jf = fopen(json_path, "w");
    if (!jf) { fprintf(stderr, "cannot write %s\n", json_path); return 2; }
    write_json(jf, &prop, driver, streams, kernels, n, grid_div, matmul, &e, status, verdict);
    fclose(jf);
  }
  return e.pass ? 0 : 1;
}
