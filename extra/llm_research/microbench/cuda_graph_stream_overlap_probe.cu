// cuda_graph_stream_overlap_probe.cu - B1 CUDA graph multi-stream overlap probe
//
// Standalone CUDA-runtime probe (no tinygrad) implementing experiment B1 of
// docs/task_workflow/input/nv-decode-overlap-route-b-implementation-scope-
// 20260804.md: on this exact device/driver, does a CUDA graph captured across
// 2-3 non-blocking streams via llama.cpp's event fork/join mechanism replay
// with span < node-sum (co-scheduling of independent graph nodes), while the
// same kernels in a single-stream graph replay serialized? Same span/node-sum/
// overlap criterion and numeric checks as the E3 probe
// (extra/llm_research/microbench/cuda_stream_overlap_probe.cu), whose kernels,
// event timing, and JSON conventions this file reuses. The E3 probe file is
// not modified.
//
// Arms (--arm, default ALL):
//   A. capture-based elementwise: --kernels independent kernels per stream on
//      1/2/3 non-blocking streams captured into one graph (event fork/join,
//      stream-0 kernels directly on the capture stream)
//   B. programmatic control: 16 kernels (8 x-half + 8 y-half) in a
//      single-stream programmatic graph, chained within each half, no edges
//      between halves (expects serialized replay)
//   C. capture-based matmul MxMxM (M=2048) on 2 streams, --kernels per stream
//
// Timing: CUDA events around every kernel (start/end) plus wall events around
// every graph replay; overlap = (node-sum - span_median) / node-sum. Host-side
// sampled numeric checks (every 100000th element, same tolerance as E3) plus a
// FNV-1a 64-bit hash over the sampled expected-vs-actual double bit patterns.
//
//   nvcc -O3 -arch=sm_120 -o cuda_graph_stream_overlap_probe cuda_graph_stream_overlap_probe.cu
//   ./cuda_graph_stream_overlap_probe --arm ALL
//   ./cuda_graph_stream_overlap_probe --arm A --streams 2 --reps 10 --json a2.json
//   ./cuda_graph_stream_overlap_probe --arm B
//   ./cuda_graph_stream_overlap_probe --arm C --streams 2

#include <cuda_runtime.h>
#include <cctype>
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

// Non-fatal variant for graph event nodes: replay-time semantics for events
// captured into a graph are driver-dependent, so a failure here must fall back
// to a native reference measurement rather than abort the probe.
static int try_elapsed_us(cudaEvent_t a, cudaEvent_t b, double* us) {
  float ms = 0.0f;
  cudaError_t err = cudaEventElapsedTime(&ms, a, b);
  if (err != cudaSuccess) {
    fprintf(stderr, "note: cudaEventElapsedTime failed (%s); node-sum will use the native reference\n",
            cudaGetErrorString(err));
    return -1;
  }
  *us = (double)ms * 1000.0;
  return 0;
}

typedef void (*launch_fn)(int k, int s, cudaStream_t stream, void* ctx);

struct ArmResult {
  const char* name;
  const char* method;          // "capture" | "programmatic"
  const char* capture_mode;    // "thread_local" | "global" | "none"
  const char* node_sum_source; // "graph_event_nodes" | "native_reference"
  int streams, kernels_per_stream, reps;
  long long n;
  int matmul;
  long long graph_node_count, graph_exec_count;
  double* per_replay_spans_us; // [reps]
  double span_us;              // median over replays
  double node_sum_us;
  double overlap;
  int overlap_valid;           // false when node_sum came from the native reference
  int numeric_ok;
  double max_err;
  long long samples_checked;
  unsigned long long hash;
};

struct EwStreamCtx {  // arm A: per-stream buffer pairs
  float** d_x;
  float** d_y;
  long long n;
  unsigned grid;
};

static void ew_launch_stream(int k, int s, cudaStream_t stream, void* vctx) {
  (void)k;
  EwStreamCtx* c = (EwStreamCtx*)vctx;
  ew_kernel<<<c->grid, TPB, 0, stream>>>(c->d_x[s], c->d_y[s], c->n);
}

struct EwKernCtx {  // arm B: per-kernel buffer pairs
  float** d_x;
  float** d_y;
  long long n;
  unsigned grid;
};

static void ew_launch_kern(int k, int s, cudaStream_t stream, void* vctx) {
  (void)s;
  EwKernCtx* c = (EwKernCtx*)vctx;
  ew_kernel<<<c->grid, TPB, 0, stream>>>(c->d_x[k], c->d_y[k], c->n);
}

struct MmCtx {  // arm C: per-stream matmul buffers
  float** d_a;
  float** d_b;
  float** d_c;
  int M;
  dim3 gridm, block;
};

static void mm_launch(int k, int s, cudaStream_t stream, void* vctx) {
  (void)k;
  MmCtx* c = (MmCtx*)vctx;
  sgemm_kernel<<<c->gridm, c->block, 0, stream>>>(c->d_a[s], c->d_b[s], c->d_c[s], c->M, c->M, c->M);
}

// FNV-1a 64-bit over the double bit patterns of the sampled expected-vs-actual
// pairs, in check order, so the record reproduces the exact numeric state.
static unsigned long long fnv1a64_init(void) { return 0xcbf29ce484222325ULL; }

static void fnv1a64_update(unsigned long long* h, const void* data, size_t len) {
  const unsigned char* p = (const unsigned char*)data;
  for (size_t i = 0; i < len; i++) {
    *h ^= p[i];
    *h *= 0x100000001b3ULL;
  }
}

static void fnv1a64_double(unsigned long long* h, double v) {
  fnv1a64_update(h, &v, sizeof(v));
}

static double median_d(const double* a, int n) {
  double* tmp = new double[n];
  memcpy(tmp, a, (size_t)n * sizeof(double));
  for (int i = 1; i < n; i++) {
    double v = tmp[i];
    int j = i - 1;
    while (j >= 0 && tmp[j] > v) { tmp[j + 1] = tmp[j]; j--; }
    tmp[j + 1] = v;
  }
  double med = (n % 2) ? tmp[n / 2] : 0.5 * (tmp[n / 2 - 1] + tmp[n / 2]);
  delete[] tmp;
  return med;
}

// Node-sum of the identical kernel batch launched on plain (non-captured)
// streams with regular per-kernel events; the fallback when replay-time
// durations cannot be read from the graph's event nodes.
static double native_node_sum(int streams, int kernels, cudaStream_t* st,
                              launch_fn launch, void* ctx) {
  int total = streams * kernels;
  cudaEvent_t* start = new cudaEvent_t[total];
  cudaEvent_t* end = new cudaEvent_t[total];
  for (int i = 0; i < total; i++) {
    CUDA_CHECK(cudaEventCreate(&start[i]));
    CUDA_CHECK(cudaEventCreate(&end[i]));
  }
  for (int k = 0; k < kernels; k++)
    for (int s = 0; s < streams; s++) {
      int i = s * kernels + k;
      CUDA_CHECK(cudaEventRecord(start[i], st[s]));
      launch(k, s, st[s], ctx);
      CUDA_CHECK(cudaEventRecord(end[i], st[s]));
    }
  CUDA_CHECK(cudaDeviceSynchronize());
  double sum = 0.0;
  for (int i = 0; i < total; i++) {
    double us;
    elapsed_us(start[i], end[i], &us);
    sum += us;
  }
  for (int i = 0; i < total; i++) {
    CUDA_CHECK(cudaEventDestroy(start[i]));
    CUDA_CHECK(cudaEventDestroy(end[i]));
  }
  delete[] start;
  delete[] end;
  return sum;
}

// Capture-based arm (A and C): llama.cpp's event fork/join capture, replayed
// --reps times. Per-kernel start/end events are recorded INSIDE the capture so
// they become graph event nodes; per-replay span comes from wall events
// recorded around each cuGraphLaunch outside the capture.
static void run_capture_arm(const char* name, int streams, int kernels, int reps,
                            launch_fn launch, void* ctx, cudaStream_t* warm_st,
                            ArmResult* out) {
  out->name = name;
  out->method = "capture";
  out->streams = streams;
  out->kernels_per_stream = kernels;
  out->graph_exec_count = 1;
  out->reps = reps;

  cudaStream_t cap;
  CUDA_CHECK(cudaStreamCreate(&cap));  // the launch/capture stream (normal stream)

  cudaEvent_t fork[MAX_STREAMS], join[MAX_STREAMS];
  int total = streams * kernels;
  cudaEvent_t* start = new cudaEvent_t[total];
  cudaEvent_t* end = new cudaEvent_t[total];
  for (int i = 0; i < total; i++) {
    CUDA_CHECK(cudaEventCreate(&start[i]));
    CUDA_CHECK(cudaEventCreate(&end[i]));
  }
  for (int s = 1; s < streams; s++) {
    CUDA_CHECK(cudaEventCreate(&fork[s]));
    CUDA_CHECK(cudaEventCreate(&join[s]));
  }
  cudaEvent_t* wall0 = new cudaEvent_t[reps];
  cudaEvent_t* wall1 = new cudaEvent_t[reps];
  for (int r = 0; r < reps; r++) {
    CUDA_CHECK(cudaEventCreate(&wall0[r]));
    CUDA_CHECK(cudaEventCreate(&wall1[r]));
  }

  // Warmup batch (module load, clock ramp); not captured, not timed.
  for (int k = 0; k < kernels; k++)
    for (int s = 0; s < streams; s++)
      launch(k, s, warm_st[s], ctx);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaError_t cerr = cudaStreamBeginCapture(cap, cudaStreamCaptureModeThreadLocal);
  const char* cap_mode = "thread_local";
  if (cerr != cudaSuccess) {
    fprintf(stderr, "note: ThreadLocal capture rejected (%s), using Global\n",
            cudaGetErrorString(cerr));
    CUDA_CHECK(cudaStreamBeginCapture(cap, cudaStreamCaptureModeGlobal));
    cap_mode = "global";
  }
  out->capture_mode = cap_mode;

  // Forks: all secondary streams wait a fork event recorded on the capture
  // stream, then their kernels, then join events recorded back onto the
  // capture stream (llama's multi-stream graph capture mechanism).
  for (int s = 1; s < streams; s++) {
    CUDA_CHECK(cudaEventRecord(fork[s], cap));
    CUDA_CHECK(cudaStreamWaitEvent(warm_st[s], fork[s], 0));
  }
  for (int s = 1; s < streams; s++)
    for (int k = 0; k < kernels; k++) {
      int i = s * kernels + k;
      CUDA_CHECK(cudaEventRecord(start[i], warm_st[s]));
      launch(k, s, warm_st[s], ctx);
      CUDA_CHECK(cudaEventRecord(end[i], warm_st[s]));
    }
  for (int k = 0; k < kernels; k++) {  // stream 0 directly on the capture stream
    CUDA_CHECK(cudaEventRecord(start[k], cap));
    launch(k, 0, cap, ctx);
    CUDA_CHECK(cudaEventRecord(end[k], cap));
  }
  for (int s = 1; s < streams; s++) {
    CUDA_CHECK(cudaEventRecord(join[s], warm_st[s]));
    CUDA_CHECK(cudaStreamWaitEvent(cap, join[s], 0));
  }

  cudaGraph_t graph;
  CUDA_CHECK(cudaStreamEndCapture(cap, &graph));
  size_t num_nodes = 0;
  CUDA_CHECK(cudaGraphGetNodes(graph, NULL, &num_nodes));
  out->graph_node_count = (long long)num_nodes;

  cudaGraphExec_t graph_exec;
  CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));

  // Replay the SAME graph_exec --reps times; per-replay span via wall events.
  double* spans = new double[reps];
  for (int r = 0; r < reps; r++) {
    CUDA_CHECK(cudaEventRecord(wall0[r], cap));
    CUDA_CHECK(cudaGraphLaunch(graph_exec, cap));
    CUDA_CHECK(cudaEventRecord(wall1[r], cap));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  for (int r = 0; r < reps; r++) elapsed_us(wall0[r], wall1[r], &spans[r]);
  out->per_replay_spans_us = spans;
  out->span_us = median_d(spans, reps);

  // Per-node durations from the graph event nodes (most recent replay).
  out->node_sum_source = "graph_event_nodes";
  int elapsed_ok = 1;
  double* durs = new double[total]();
  for (int i = 0; i < total; i++)
    if (try_elapsed_us(start[i], end[i], &durs[i]) != 0) elapsed_ok = 0;
  int all_zero = 1;
  double node_sum = 0.0;
  if (elapsed_ok) {
    for (int i = 0; i < total; i++) {
      node_sum += durs[i];
      if (durs[i] != 0.0) all_zero = 0;
    }
  }
  if (!elapsed_ok || all_zero) {
    node_sum = native_node_sum(streams, kernels, warm_st, launch, ctx);
    out->node_sum_source = "native_reference";
  }
  out->node_sum_us = node_sum;
  out->overlap_valid = (out->node_sum_source[0] == 'g');
  out->overlap = node_sum > 0.0 ? (node_sum - out->span_us) / node_sum : 0.0;

  CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  for (int i = 0; i < total; i++) {
    CUDA_CHECK(cudaEventDestroy(start[i]));
    CUDA_CHECK(cudaEventDestroy(end[i]));
  }
  for (int s = 1; s < streams; s++) {
    CUDA_CHECK(cudaEventDestroy(fork[s]));
    CUDA_CHECK(cudaEventDestroy(join[s]));
  }
  for (int r = 0; r < reps; r++) {
    CUDA_CHECK(cudaEventDestroy(wall0[r]));
    CUDA_CHECK(cudaEventDestroy(wall1[r]));
  }
  CUDA_CHECK(cudaStreamDestroy(cap));
  delete[] start;
  delete[] end;
  delete[] wall0;
  delete[] wall1;
  delete[] durs;
}

// Programmatic control arm (B): 16 kernels (8 x-half + 8 y-half) added with
// cudaGraphAddKernelNode, chained within each half, no edges between halves.
// Event record nodes around each kernel give replay-time per-node durations.
static void run_programmatic_arm(const char* name, int kernels, int reps, long long n,
                                 float** d_x, float** d_y, ArmResult* out) {
  out->name = name;
  out->method = "programmatic";
  out->streams = 1;  // programmatic graphs have no streams; single logical stream
  out->kernels_per_stream = kernels;
  out->capture_mode = "none";
  out->graph_exec_count = 1;
  out->reps = reps;

  cudaGraph_t graph;
  CUDA_CHECK(cudaGraphCreate(&graph, 0));
  cudaGraphNode_t knode[16], estart[16], eend[16];
  cudaEvent_t start[16], end[16];
  for (int k = 0; k < kernels; k++) {
    CUDA_CHECK(cudaEventCreate(&start[k]));
    CUDA_CHECK(cudaEventCreate(&end[k]));
  }

  cudaStream_t exec_st;
  CUDA_CHECK(cudaStreamCreate(&exec_st));

  long long nval = n;
  unsigned grid = (unsigned)((n + TPB - 1) / TPB);
  EwKernCtx ctx = { d_x, d_y, n, grid };

  // Warmup batch (module load, clock ramp); not part of the graph.
  for (int k = 0; k < kernels; k++) ew_launch_kern(k, 0, exec_st, &ctx);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (int k = 0; k < kernels; k++) {
    void* args[3] = { (void*)&d_x[k], (void*)&d_y[k], (void*)&nval };
    cudaKernelNodeParams p = {};
    p.func = (void*)ew_kernel;
    p.gridDim = dim3(grid);
    p.blockDim = dim3(TPB);
    p.sharedMemBytes = 0;
    p.kernelParams = args;
    p.extra = NULL;
    CUDA_CHECK(cudaGraphAddKernelNode(&knode[k], graph, NULL, 0, &p));
    CUDA_CHECK(cudaGraphAddEventRecordNode(&estart[k], graph, NULL, 0, start[k]));
    CUDA_CHECK(cudaGraphAddEventRecordNode(&eend[k], graph, NULL, 0, end[k]));
  }
  // Event ordering so per-node durations cover exactly one kernel...
  for (int k = 0; k < kernels; k++) {
    CUDA_CHECK(cudaGraphAddDependencies(graph, &estart[k], &knode[k], NULL, 1));
    CUDA_CHECK(cudaGraphAddDependencies(graph, &knode[k], &eend[k], NULL, 1));
  }
  // ...and the within-half chains (k depends on k-1), with no edges between
  // halves: estart[k] records only after kernel k-1 completes.
  int half = kernels / 2;
  for (int k = 1; k < half; k++) {
    cudaGraphNode_t from[2] = { knode[k - 1], knode[k - 1] };
    cudaGraphNode_t to[2] = { knode[k], estart[k] };
    CUDA_CHECK(cudaGraphAddDependencies(graph, from, to, NULL, 2));
  }
  for (int k = half + 1; k < kernels; k++) {
    cudaGraphNode_t from[2] = { knode[k - 1], knode[k - 1] };
    cudaGraphNode_t to[2] = { knode[k], estart[k] };
    CUDA_CHECK(cudaGraphAddDependencies(graph, from, to, NULL, 2));
  }

  size_t num_nodes = 0;
  CUDA_CHECK(cudaGraphGetNodes(graph, NULL, &num_nodes));
  out->graph_node_count = (long long)num_nodes;  // 16 kernels + 32 event records

  cudaGraphExec_t graph_exec;
  CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));

  cudaEvent_t* wall0 = new cudaEvent_t[reps];
  cudaEvent_t* wall1 = new cudaEvent_t[reps];
  for (int r = 0; r < reps; r++) {
    CUDA_CHECK(cudaEventCreate(&wall0[r]));
    CUDA_CHECK(cudaEventCreate(&wall1[r]));
  }

  double* spans = new double[reps];
  for (int r = 0; r < reps; r++) {
    CUDA_CHECK(cudaEventRecord(wall0[r], exec_st));
    CUDA_CHECK(cudaGraphLaunch(graph_exec, exec_st));
    CUDA_CHECK(cudaEventRecord(wall1[r], exec_st));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  for (int r = 0; r < reps; r++) elapsed_us(wall0[r], wall1[r], &spans[r]);
  out->per_replay_spans_us = spans;
  out->span_us = median_d(spans, reps);

  out->node_sum_source = "graph_event_nodes";
  int elapsed_ok = 1;
  double node_sum = 0.0;
  int all_zero = 1;
  for (int k = 0; k < kernels; k++) {
    double us = 0.0;
    if (try_elapsed_us(start[k], end[k], &us) != 0) elapsed_ok = 0;
    node_sum += us;
    if (us != 0.0) all_zero = 0;
  }
  if (!elapsed_ok || all_zero) {
    node_sum = native_node_sum(1, kernels, &exec_st, ew_launch_kern, &ctx);
    out->node_sum_source = "native_reference";
  }
  out->node_sum_us = node_sum;
  out->overlap_valid = (out->node_sum_source[0] == 'g');
  out->overlap = node_sum > 0.0 ? (node_sum - out->span_us) / node_sum : 0.0;

  CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  for (int k = 0; k < kernels; k++) {
    CUDA_CHECK(cudaEventDestroy(start[k]));
    CUDA_CHECK(cudaEventDestroy(end[k]));
  }
  for (int r = 0; r < reps; r++) {
    CUDA_CHECK(cudaEventDestroy(wall0[r]));
    CUDA_CHECK(cudaEventDestroy(wall1[r]));
  }
  CUDA_CHECK(cudaStreamDestroy(exec_st));
  delete[] wall0;
  delete[] wall1;
}

// Sampled numeric check + FNV-1a hash for elementwise arms. Pair p (stream for
// A, kernel for B) has x filled with the constant (p+1); tolerance as in E3.
static void verify_ew(ArmResult* out, float** d_x, float** d_y, int pairs, long long n) {
  float* h_x = new float[n];
  float* h_y = new float[n];
  unsigned long long h = fnv1a64_init();
  double max_err = 0.0;
  long long samples = 0;
  int ok = 1;
  long long stride = n >= (long long)SAMPLE_STRIDE * 16 ? SAMPLE_STRIDE : (n + 15) / 16;
  for (int p = 0; p < pairs; p++) {
    for (long long i = 0; i < n; i++) h_x[i] = (float)(p + 1);
    CUDA_CHECK(cudaMemcpy(h_y, d_y[p], (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
    for (long long i = 0; i < n; i += stride) {
      double expected = (double)h_x[i] * 1.000001 + (double)i;
      double actual = (double)h_y[i];
      double err = fabs(actual - expected);
      if (err > max_err) max_err = err;
      if (err > 1e-4 * fabs(expected) + 4.0) ok = 0;
      fnv1a64_double(&h, expected);
      fnv1a64_double(&h, actual);
      samples++;
    }
  }
  delete[] h_x;
  delete[] h_y;
  out->numeric_ok = ok;
  out->max_err = max_err;
  out->samples_checked = samples;
  out->hash = h;
}

// Sampled numeric check + FNV-1a hash for the matmul arm: A = B = sqrt(M)*I
// pins C = M*I (diagonal M, off-diagonal 0), E3's mm convention.
static void verify_mm(ArmResult* out, float** d_c, int streams, int M) {
  int rowstep = M / 16 < 1 ? 1 : M / 16;
  size_t mbytes = (size_t)M * M * sizeof(float);
  float* h_c = new float[(size_t)M * M];
  double tol = 1e-2 * (double)M + 1.0;
  unsigned long long h = fnv1a64_init();
  double max_err = 0.0;
  long long samples = 0;
  int ok = 1;
  for (int s = 0; s < streams; s++) {
    CUDA_CHECK(cudaMemcpy(h_c, d_c[s], mbytes, cudaMemcpyDeviceToHost));
    for (int r = 0; r < M; r += rowstep) {
      double expected_d = (double)M;
      double actual_d = (double)h_c[r * M + r];
      double errd = fabs(actual_d - expected_d);
      if (errd > max_err) max_err = errd;
      if (errd > tol) ok = 0;
      fnv1a64_double(&h, expected_d);
      fnv1a64_double(&h, actual_d);
      samples++;
      if (M > 1) {
        int col = (r + 1) % M;
        double expected_o = 0.0;
        double actual_o = (double)h_c[r * M + col];
        double erro = fabs(actual_o - expected_o);
        if (erro > max_err) max_err = erro;
        if (erro > tol) ok = 0;
        fnv1a64_double(&h, expected_o);
        fnv1a64_double(&h, actual_o);
        samples++;
      }
    }
  }
  delete[] h_c;
  out->numeric_ok = ok;
  out->max_err = max_err;
  out->samples_checked = samples;
  out->hash = h;
}

static const char* arg_val(int argc, char** argv, int* i, const char* opt) {
  if (*i + 1 >= argc) {
    fprintf(stderr, "missing value for %s\n", opt);
    exit(2);
  }
  return argv[++(*i)];
}

static void usage(FILE* f, const char* prog) {
  fprintf(f, "usage: %s [--arm A|B|C|ALL] [--streams N] [--reps N] [--n N] [--kernels N] [--json FILE]\n"
             "  --arm      A|B|C|ALL  which probe arms to run (default ALL):\n"
             "                A = capture-based elementwise on 1/2/3 streams (decisive)\n"
             "                B = programmatic 16-node single-stream graph control\n"
             "                C = capture-based MxMxM matmul on 2 streams\n"
             "  --streams  N  arm A: 1, 2, or 3, or 0 = run all of 1/2/3 (default 0);\n"
             "                arm C: must be 2\n"
             "  --reps     N  graph replays per arm (default 5)\n"
             "  --n        N  elementwise vector length in floats (default 33554432)\n"
             "  --kernels  N  kernels per stream for arms A and C (default 8; arm B fixed at 16)\n"
             "  --json  FILE  write the measurement record to FILE (also printed to stdout)\n"
             "  --help        print this usage and exit 0 before any CUDA call\n",
             prog);
}

static void write_json(FILE* f, const cudaDeviceProp* prop, int driver,
                       const char* arm_sel, int streams_opt, int reps, long long n, int kernels,
                       const ArmResult* arms, int n_arms) {
  fprintf(f, "{\n");
  fprintf(f, "  \"schema\": \"tinygrad.cuda_graph_stream_overlap_probe.v1\",\n");
  fprintf(f, "  \"leg\": \"B1\",\n");
  fprintf(f, "  \"device\": {\"name\": \"%s\", \"compute_capability\": \"%d.%d\", \"cuda_driver_version\": %d},\n",
          prop->name, prop->major, prop->minor, driver);
  fprintf(f, "  \"config\": {\"arm\": \"%s\", \"streams\": %d, \"reps\": %d, \"n\": %lld, \"kernels_per_stream\": %d},\n",
          arm_sel, streams_opt, reps, n, kernels);
  fprintf(f, "  \"arms\": [\n");
  for (int i = 0; i < n_arms; i++) {
    const ArmResult* r = &arms[i];
    fprintf(f, "    {\n");
    fprintf(f, "      \"name\": \"%s\",\n", r->name);
    fprintf(f, "      \"method\": \"%s\",\n", r->method);
    fprintf(f, "      \"streams\": %d,\n", r->streams);
    fprintf(f, "      \"kernels_per_stream\": %d,\n", r->kernels_per_stream);
    fprintf(f, "      \"n\": %lld,\n", r->n);
    fprintf(f, "      \"matmul\": %d,\n", r->matmul);
    fprintf(f, "      \"graph_node_count\": %lld,\n", r->graph_node_count);
    fprintf(f, "      \"graph_exec_count\": %lld,\n", r->graph_exec_count);
    fprintf(f, "      \"capture_mode\": \"%s\",\n", r->capture_mode);
    fprintf(f, "      \"node_sum_source\": \"%s\",\n", r->node_sum_source);
    fprintf(f, "      \"per_replay_spans_us\": [");
    for (int q = 0; q < r->reps; q++) fprintf(f, "%s%g", q ? ", " : "", r->per_replay_spans_us[q]);
    fprintf(f, "],\n");
    fprintf(f, "      \"span_us\": %g,\n", r->span_us);
    fprintf(f, "      \"node_sum_us\": %g,\n", r->node_sum_us);
    fprintf(f, "      \"overlap\": %g,\n", r->overlap);
    fprintf(f, "      \"overlap_valid\": %s,\n", r->overlap_valid ? "true" : "false");
    fprintf(f, "      \"numeric_ok\": %s,\n", r->numeric_ok ? "true" : "false");
    fprintf(f, "      \"max_err\": %g,\n", r->max_err);
    fprintf(f, "      \"samples_checked\": %lld,\n", r->samples_checked);
    fprintf(f, "      \"hash\": \"0x%016llx\"\n", r->hash);
    fprintf(f, "    }%s\n", (i + 1 < n_arms) ? "," : "");
  }
  fprintf(f, "  ]\n");
  fprintf(f, "}\n");
}

int main(int argc, char** argv) {
  const char* arm_sel = "ALL";
  int streams_opt = 0, reps = 5, kernels = 8;
  long long n = 1LL << 25;
  const char* json_path = NULL;

  for (int i = 1; i < argc; i++) {
    const char* a = argv[i];
    if (!strcmp(a, "--help") || !strcmp(a, "-h")) {
      usage(stdout, argv[0]);
      return 0;  // host-side smoke check: exits before any CUDA call
    } else if (!strcmp(a, "--arm")) arm_sel = arg_val(argc, argv, &i, a);
    else if (!strcmp(a, "--streams")) streams_opt = atoi(arg_val(argc, argv, &i, a));
    else if (!strcmp(a, "--reps")) reps = atoi(arg_val(argc, argv, &i, a));
    else if (!strcmp(a, "--n")) n = atoll(arg_val(argc, argv, &i, a));
    else if (!strcmp(a, "--kernels")) kernels = atoi(arg_val(argc, argv, &i, a));
    else if (!strcmp(a, "--json")) json_path = arg_val(argc, argv, &i, a);
    else {
      fprintf(stderr, "unknown option: %s\n", a);
      usage(stderr, argv[0]);
      return 2;
    }
  }

  char sel[8];
  snprintf(sel, sizeof(sel), "%s", arm_sel);
  for (char* p = sel; *p; p++) *p = (char)tolower((unsigned char)*p);
  int do_a = !strcmp(sel, "a") || !strcmp(sel, "all");
  int do_b = !strcmp(sel, "b") || !strcmp(sel, "all");
  int do_c = !strcmp(sel, "c") || !strcmp(sel, "all");
  if (!do_a && !do_b && !do_c) {
    fprintf(stderr, "--arm must be A, B, C, or ALL (got %s)\n", arm_sel);
    return 2;
  }
  if (streams_opt < 0 || streams_opt > 3) {
    fprintf(stderr, "--streams must be 0, 1, 2, or 3 (got %d)\n", streams_opt);
    return 2;
  }
  if (reps < 1) { fprintf(stderr, "--reps must be >= 1\n"); return 2; }
  if (kernels < 1) { fprintf(stderr, "--kernels must be >= 1\n"); return 2; }
  if (n < 1) { fprintf(stderr, "--n must be >= 1\n"); return 2; }
  if (do_c && streams_opt != 0 && streams_opt != 2) {
    fprintf(stderr, "arm C requires --streams 2 (got %d)\n", streams_opt);
    return 2;
  }
  int c_streams = (streams_opt == 0) ? 2 : streams_opt;

  int a_streams[MAX_STREAMS], n_a_streams = 0;
  if (streams_opt > 0) { a_streams[0] = streams_opt; n_a_streams = 1; }
  else { a_streams[0] = 1; a_streams[1] = 2; a_streams[2] = 3; n_a_streams = 3; }

  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int driver = 0;
  CUDA_CHECK(cudaDriverGetVersion(&driver));
  fprintf(stderr, "device ready: %s (sm_%d%d), CUDA driver %d, arm=%s streams=%d reps=%d "
                  "n=%lld kernels=%d\n",
          prop.name, prop.major, prop.minor, driver, arm_sel, streams_opt, reps, n, kernels);

  int max_arms = n_a_streams + 1 + 1;
  ArmResult* arms = new ArmResult[max_arms];
  memset(arms, 0, (size_t)max_arms * sizeof(ArmResult));
  int n_arms = 0;

  if (do_a) {
    for (int t = 0; t < n_a_streams; t++) {
      int s = a_streams[t];
      char name[64];
      snprintf(name, sizeof(name), "A-%dstream-elementwise", s);
      float** d_x = new float*[s];
      float** d_y = new float*[s];
      float* h = new float[n];
      for (int q = 0; q < s; q++) {
        CUDA_CHECK(cudaMalloc(&d_x[q], (size_t)n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y[q], (size_t)n * sizeof(float)));
        for (long long i = 0; i < n; i++) h[i] = (float)(q + 1);  // x constant per stream, like E3
        CUDA_CHECK(cudaMemcpy(d_x[q], h, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_y[q], 0, (size_t)n * sizeof(float)));
      }
      delete[] h;
      cudaStream_t* warm_st = new cudaStream_t[s];
      for (int q = 0; q < s; q++)
        CUDA_CHECK(cudaStreamCreateWithFlags(&warm_st[q], cudaStreamNonBlocking));
      EwStreamCtx ctx = { d_x, d_y, n, (unsigned)((n + TPB - 1) / TPB) };
      ArmResult* r = &arms[n_arms++];
      r->n = n;
      r->matmul = 0;
      run_capture_arm(name, s, kernels, reps, ew_launch_stream, &ctx, warm_st, r);
      verify_ew(r, d_x, d_y, s, n);
      for (int q = 0; q < s; q++) {
        CUDA_CHECK(cudaStreamDestroy(warm_st[q]));
        CUDA_CHECK(cudaFree(d_x[q]));
        CUDA_CHECK(cudaFree(d_y[q]));
      }
      delete[] warm_st;
      delete[] d_x;
      delete[] d_y;
    }
  }

  if (do_b) {
    const int bkernels = 16;  // 8 x-half + 8 y-half, fixed by design
    float** d_x = new float*[bkernels];
    float** d_y = new float*[bkernels];
    float* h = new float[n];
    for (int k = 0; k < bkernels; k++) {
      CUDA_CHECK(cudaMalloc(&d_x[k], (size_t)n * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_y[k], (size_t)n * sizeof(float)));
      for (long long i = 0; i < n; i++) h[i] = (float)(k + 1);
      CUDA_CHECK(cudaMemcpy(d_x[k], h, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemset(d_y[k], 0, (size_t)n * sizeof(float)));
    }
    delete[] h;
    ArmResult* r = &arms[n_arms++];
    r->n = n;
    r->matmul = 0;
    run_programmatic_arm("B-programmatic-elementwise", bkernels, reps, n, d_x, d_y, r);
    verify_ew(r, d_x, d_y, bkernels, n);
    for (int k = 0; k < bkernels; k++) {
      CUDA_CHECK(cudaFree(d_x[k]));
      CUDA_CHECK(cudaFree(d_y[k]));
    }
    delete[] d_x;
    delete[] d_y;
  }

  if (do_c) {
    const int M = 2048;
    size_t mbytes = (size_t)M * M * sizeof(float);
    float** d_a = new float*[c_streams];
    float** d_b = new float*[c_streams];
    float** d_c = new float*[c_streams];
    float* h = new float[(size_t)M * M];
    for (int s = 0; s < c_streams; s++) {
      CUDA_CHECK(cudaMalloc(&d_a[s], mbytes));
      CUDA_CHECK(cudaMalloc(&d_b[s], mbytes));
      CUDA_CHECK(cudaMalloc(&d_c[s], mbytes));
      for (long long i = 0; i < (long long)M * M; i++) h[i] = 0.0f;
      for (int i = 0; i < M; i++) {
        h[i * M + i] = sqrtf((float)M);  // A = B = sqrt(M)*I pins C = M*I
      }
      CUDA_CHECK(cudaMemcpy(d_a[s], h, mbytes, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_b[s], h, mbytes, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemset(d_c[s], 0, mbytes));
    }
    delete[] h;
    cudaStream_t* warm_st = new cudaStream_t[c_streams];
    for (int s = 0; s < c_streams; s++)
      CUDA_CHECK(cudaStreamCreateWithFlags(&warm_st[s], cudaStreamNonBlocking));
    dim3 block(MM_TILE, MM_TILE);
    unsigned nt = (unsigned)((M + MM_TILE - 1) / MM_TILE);
    MmCtx ctx = { d_a, d_b, d_c, M, dim3(nt, nt), block };
    ArmResult* r = &arms[n_arms++];
    r->n = 0;  // not elementwise; matmul MxMxM is reported via matmul
    r->matmul = M;
    run_capture_arm("C-2stream-matmul", c_streams, kernels, reps, mm_launch, &ctx, warm_st, r);
    verify_mm(r, d_c, c_streams, M);
    for (int s = 0; s < c_streams; s++) {
      CUDA_CHECK(cudaStreamDestroy(warm_st[s]));
      CUDA_CHECK(cudaFree(d_a[s]));
      CUDA_CHECK(cudaFree(d_b[s]));
      CUDA_CHECK(cudaFree(d_c[s]));
    }
    delete[] warm_st;
    delete[] d_a;
    delete[] d_b;
    delete[] d_c;
  }

  for (int i = 0; i < n_arms; i++) {
    const ArmResult* r = &arms[i];
    double mn = r->per_replay_spans_us[0], mx = r->per_replay_spans_us[0];
    for (int q = 1; q < r->reps; q++) {
      if (r->per_replay_spans_us[q] < mn) mn = r->per_replay_spans_us[q];
      if (r->per_replay_spans_us[q] > mx) mx = r->per_replay_spans_us[q];
    }
    fprintf(stderr, "%s method=%s streams=%d kernels=%d nodes=%lld capture=%s node_sum_source=%s "
                    "span[min/med/max]=%.1f/%.1f/%.1fus node_sum=%.1fus overlap=%.1f%% "
                    "numeric_ok=%d max_err=%.3e samples=%lld hash=0x%016llx\n",
            r->name, r->method, r->streams, r->kernels_per_stream, r->graph_node_count,
            r->capture_mode, r->node_sum_source, mn, r->span_us, mx, r->node_sum_us,
            r->overlap * 100.0, r->numeric_ok, r->max_err, r->samples_checked, r->hash);
  }

  write_json(stdout, &prop, driver, arm_sel, streams_opt, reps, n, kernels, arms, n_arms);
  if (json_path) {
    FILE* jf = fopen(json_path, "w");
    if (!jf) { fprintf(stderr, "cannot write %s\n", json_path); return 2; }
    write_json(jf, &prop, driver, arm_sel, streams_opt, reps, n, kernels, arms, n_arms);
    fclose(jf);
  }

  int all_ok = 1;
  for (int i = 0; i < n_arms; i++)
    if (!arms[i].numeric_ok) all_ok = 0;
  // Exit encodes only numerics + completion; the overlap gate lives in the record.
  return all_ok ? 0 : 1;
}
