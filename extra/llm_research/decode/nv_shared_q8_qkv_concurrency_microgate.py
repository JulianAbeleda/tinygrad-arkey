#!/usr/bin/env python3
"""Cache-cold producer-to-join Q || K/V concurrency microgate.

The control and candidate execute the same production-rendered shared-Q8
attention bodies and consume identical buffers.  Control serializes

  RMSNorm/Q8 provider -> Q -> paired K/V

on one CUDA stream.  Candidate keeps the provider on the parent stream, then
forks Q and paired K/V onto independent streams and joins both before the stop
event.  A 256 MiB device walk precedes (but is outside) every timed span so the
14.16 MiB of projection weights start cache-cold.  This is a physical overlap
gate only: it deliberately does not change tinygrad's HCQ scheduler.
"""
from __future__ import annotations

import argparse, json, os, re, statistics, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.llm.shared_q8_attention import (_emit_q4_cooperative, _emit_q4_cooperative_pair,
                                               _emit_rmsnorm_q8_provider)
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops, ReduceOutputSpec, UOp

K, Q_ROWS, KV_ROWS = 4096, 4096, 1024
Q_WORDS = Q_ROWS * (K // 256) * 36
KV_WORDS = KV_ROWS * (K // 256) * 36
PACKED = K // 4 + K // 32
FLUSH_WORDS = 64 * 1024 * 1024  # 256 MiB, larger than the device L2.
ENDPOINT_US = 4515.39571875
CUDA_BIN = "/usr/local/cuda-13.2/bin"


def _render() -> tuple[str, str, str]:
  p, extent = UOp.placeholder, UOp.const(dtypes.weakint, 4)
  provider = _emit_rmsnorm_q8_provider(
    ReduceOutputSpec(1, K, 1e-6, dtypes.float32), dtypes.float32, dtypes.float16)(
      p((PACKED,), dtypes.uint32, 0), p((K,), dtypes.float32, 1), p((K,), dtypes.float16, 2))
  q = _emit_q4_cooperative(Q_ROWS, extent, direct_output=True)(
    p((Q_ROWS,), dtypes.float32, 0), p((Q_WORDS,), dtypes.uint32, 1), p((PACKED,), dtypes.uint32, 2))
  kv = _emit_q4_cooperative_pair(KV_ROWS, extent)(
    p((KV_ROWS,), dtypes.float32, 0), p((KV_ROWS,), dtypes.float32, 1),
    p((KV_WORDS,), dtypes.uint32, 2), p((KV_WORDS,), dtypes.uint32, 3), p((PACKED,), dtypes.uint32, 4))
  ren = CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=False)

  def source(u: UOp) -> str:
    text = next(x.arg for x in to_program(u, ren).src if x.op is Ops.SOURCE)
    return text[text.index('extern "C" __global__'):]

  return source(provider), source(q), source(kv)


HARNESS = r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
#define K 4096
#define Q_ROWS 4096
#define KV_ROWS 1024
#define Q_WORDS 2359296
#define KV_WORDS 589824
#define PACKED 1152
#define FLUSH_WORDS 67108864
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }

__PROVIDER__
__Q__
__KV__

__global__ void evict_l2(unsigned int *buf, unsigned int salt) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < FLUSH_WORDS; i += stride) buf[i] = buf[i] + salt + i;
}

static void ck(cudaError_t e, const char *what) {
  if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); exit(2); }
}

struct State {
  cudaStream_t parent, qs, kvs;
  cudaEvent_t start, stop, ready, qdone, kvdone;
};

static void flush(State& st, unsigned int *trash, unsigned int salt) {
  evict_l2<<<8192, 256, 0, st.parent>>>(trash, salt);
}

static void provider(cudaStream_t s, unsigned int *xp, float *x, half *norm) {
  dim3 block(32, 16, 1);
  rmsnorm_q8_1_llama_provider_4096<<<8, block, 0, s>>>(xp, x, norm);
}

static void qproj(cudaStream_t s, float *out, unsigned int *w, unsigned int *xp) {
  q4k_warp_coop_q8_dp4a_direct_4096_4096<<<Q_ROWS, 128, 0, s>>>(out, w, xp);
}

static void kvproj(cudaStream_t s, float *ko, float *vo, unsigned int *kw, unsigned int *vw, unsigned int *xp) {
  q4k_warp_coop_q8_dp4a_pair_direct_1024_4096<<<KV_ROWS, 128, 0, s>>>(ko, vo, kw, vw, xp);
}

static double one_control(State& st, unsigned int *trash, unsigned int salt,
                          unsigned int *xp, float *x, half *norm, float *qo, float *ko, float *vo,
                          unsigned int *qw, unsigned int *kw, unsigned int *vw) {
  flush(st, trash, salt);
  ck(cudaEventRecord(st.start, st.parent), "control start");
  provider(st.parent, xp, x, norm);
  qproj(st.parent, qo, qw, xp);
  kvproj(st.parent, ko, vo, kw, vw, xp);
  ck(cudaEventRecord(st.stop, st.parent), "control stop");
  ck(cudaEventSynchronize(st.stop), "control sync");
  float ms = 0.0f; ck(cudaEventElapsedTime(&ms, st.start, st.stop), "control elapsed");
  return ms * 1000.0;
}

static double one_candidate(State& st, unsigned int *trash, unsigned int salt,
                            unsigned int *xp, float *x, half *norm, float *qo, float *ko, float *vo,
                            unsigned int *qw, unsigned int *kw, unsigned int *vw) {
  flush(st, trash, salt);
  ck(cudaEventRecord(st.start, st.parent), "candidate start");
  provider(st.parent, xp, x, norm);
  ck(cudaEventRecord(st.ready, st.parent), "provider ready");
  ck(cudaStreamWaitEvent(st.qs, st.ready), "q wait provider");
  ck(cudaStreamWaitEvent(st.kvs, st.ready), "kv wait provider");
  qproj(st.qs, qo, qw, xp);
  kvproj(st.kvs, ko, vo, kw, vw, xp);
  ck(cudaEventRecord(st.qdone, st.qs), "q done");
  ck(cudaEventRecord(st.kvdone, st.kvs), "kv done");
  ck(cudaStreamWaitEvent(st.parent, st.qdone), "join q");
  ck(cudaStreamWaitEvent(st.parent, st.kvdone), "join kv");
  ck(cudaEventRecord(st.stop, st.parent), "candidate stop");
  ck(cudaEventSynchronize(st.stop), "candidate sync");
  float ms = 0.0f; ck(cudaEventElapsedTime(&ms, st.start, st.stop), "candidate elapsed");
  return ms * 1000.0;
}

int main(int argc, char **argv) {
  int passes = argc > 1 ? atoi(argv[1]) : 80;
  int reps = argc > 2 ? atoi(argv[2]) : 9;
  State st{};
  ck(cudaStreamCreateWithFlags(&st.parent, cudaStreamNonBlocking), "parent stream");
  ck(cudaStreamCreateWithFlags(&st.qs, cudaStreamNonBlocking), "q stream");
  ck(cudaStreamCreateWithFlags(&st.kvs, cudaStreamNonBlocking), "kv stream");
  ck(cudaEventCreate(&st.start), "start event"); ck(cudaEventCreate(&st.stop), "stop event");
  ck(cudaEventCreateWithFlags(&st.ready, cudaEventDisableTiming), "ready event");
  ck(cudaEventCreateWithFlags(&st.qdone, cudaEventDisableTiming), "qdone event");
  ck(cudaEventCreateWithFlags(&st.kvdone, cudaEventDisableTiming), "kvdone event");

  unsigned int *trash=nullptr, *qw=nullptr, *kw=nullptr, *vw=nullptr, *xpc=nullptr, *xpp=nullptr;
  float *x=nullptr, *cq=nullptr, *ckout=nullptr, *cvout=nullptr, *pq=nullptr, *pkout=nullptr, *pvout=nullptr;
  half *norm=nullptr;
  ck(cudaMalloc(&trash, size_t(FLUSH_WORDS)*sizeof(unsigned int)), "trash");
  ck(cudaMalloc(&qw, size_t(Q_WORDS)*sizeof(unsigned int)), "qw");
  ck(cudaMalloc(&kw, size_t(KV_WORDS)*sizeof(unsigned int)), "kw");
  ck(cudaMalloc(&vw, size_t(KV_WORDS)*sizeof(unsigned int)), "vw");
  ck(cudaMalloc(&xpc, PACKED*sizeof(unsigned int)), "xpc"); ck(cudaMalloc(&xpp, PACKED*sizeof(unsigned int)), "xpp");
  ck(cudaMalloc(&x, K*sizeof(float)), "x"); ck(cudaMalloc(&norm, K*sizeof(half)), "norm");
  ck(cudaMalloc(&cq, Q_ROWS*sizeof(float)), "cq"); ck(cudaMalloc(&pq, Q_ROWS*sizeof(float)), "pq");
  ck(cudaMalloc(&ckout, KV_ROWS*sizeof(float)), "ck"); ck(cudaMalloc(&cvout, KV_ROWS*sizeof(float)), "cv");
  ck(cudaMalloc(&pkout, KV_ROWS*sizeof(float)), "pk"); ck(cudaMalloc(&pvout, KV_ROWS*sizeof(float)), "pv");

  unsigned int *hqw=(unsigned int*)malloc(size_t(Q_WORDS)*sizeof(unsigned int));
  unsigned int *hkw=(unsigned int*)malloc(size_t(KV_WORDS)*sizeof(unsigned int));
  unsigned int *hvw=(unsigned int*)malloc(size_t(KV_WORDS)*sizeof(unsigned int));
  float *hx=(float*)malloc(K*sizeof(float)); half *hn=(half*)malloc(K*sizeof(half));
  for (int i=0; i<Q_WORDS; i++) hqw[i]=(i*2654435761u)^0x9e3779b9u;
  for (int i=0; i<KV_WORDS; i++) { hkw[i]=(i*2246822519u)^0x85ebca6bu; hvw[i]=(i*3266489917u)^0xc2b2ae35u; }
  for (int row=0; row<Q_ROWS; row++) for (int b=0; b<K/256; b++) hqw[(row*(K/256)+b)*36]=0x24002400u;
  for (int row=0; row<KV_ROWS; row++) for (int b=0; b<K/256; b++) {
    hkw[(row*(K/256)+b)*36]=0x24002400u; hvw[(row*(K/256)+b)*36]=0x24002400u;
  }
  for (int i=0; i<K; i++) { hx[i]=float((i%257)-128)/1024.0f; hn[i]=__float2half(1.0f+float(i%13)/1024.0f); }
  ck(cudaMemcpy(qw,hqw,size_t(Q_WORDS)*sizeof(unsigned int),cudaMemcpyHostToDevice),"qw copy");
  ck(cudaMemcpy(kw,hkw,size_t(KV_WORDS)*sizeof(unsigned int),cudaMemcpyHostToDevice),"kw copy");
  ck(cudaMemcpy(vw,hvw,size_t(KV_WORDS)*sizeof(unsigned int),cudaMemcpyHostToDevice),"vw copy");
  ck(cudaMemcpy(x,hx,K*sizeof(float),cudaMemcpyHostToDevice),"x copy");
  ck(cudaMemcpy(norm,hn,K*sizeof(half),cudaMemcpyHostToDevice),"norm copy");
  ck(cudaMemset(trash, 0, size_t(FLUSH_WORDS)*sizeof(unsigned int)), "trash memset");
  free(hqw); free(hkw); free(hvw); free(hx); free(hn);

  one_control(st,trash,1,xpc,x,norm,cq,ckout,cvout,qw,kw,vw);
  one_candidate(st,trash,2,xpp,x,norm,pq,pkout,pvout,qw,kw,vw);
  unsigned int *hc=(unsigned int*)malloc((Q_ROWS+2*KV_ROWS)*sizeof(float));
  unsigned int *hp=(unsigned int*)malloc((Q_ROWS+2*KV_ROWS)*sizeof(float));
  ck(cudaMemcpy(hc,cq,Q_ROWS*sizeof(float),cudaMemcpyDeviceToHost),"cq copy");
  ck(cudaMemcpy(hc+Q_ROWS,ckout,KV_ROWS*sizeof(float),cudaMemcpyDeviceToHost),"ck copy");
  ck(cudaMemcpy(hc+Q_ROWS+KV_ROWS,cvout,KV_ROWS*sizeof(float),cudaMemcpyDeviceToHost),"cv copy");
  ck(cudaMemcpy(hp,pq,Q_ROWS*sizeof(float),cudaMemcpyDeviceToHost),"pq copy");
  ck(cudaMemcpy(hp+Q_ROWS,pkout,KV_ROWS*sizeof(float),cudaMemcpyDeviceToHost),"pk copy");
  ck(cudaMemcpy(hp+Q_ROWS+KV_ROWS,pvout,KV_ROWS*sizeof(float),cudaMemcpyDeviceToHost),"pv copy");
  int mismatch=0; for (int i=0; i<Q_ROWS+2*KV_ROWS; i++) mismatch += hc[i]!=hp[i];
  printf("mismatched_words=%d bitwise_identical=%d\n",mismatch,mismatch==0); free(hc); free(hp);

  unsigned int salt=3;
  for (int r=0; r<reps; r++) {
    double cs=0.0, ps=0.0;
    if ((r&1)==0) {
      for (int i=0;i<passes;i++) cs+=one_control(st,trash,salt++,xpc,x,norm,cq,ckout,cvout,qw,kw,vw);
      for (int i=0;i<passes;i++) ps+=one_candidate(st,trash,salt++,xpp,x,norm,pq,pkout,pvout,qw,kw,vw);
    } else {
      for (int i=0;i<passes;i++) ps+=one_candidate(st,trash,salt++,xpp,x,norm,pq,pkout,pvout,qw,kw,vw);
      for (int i=0;i<passes;i++) cs+=one_control(st,trash,salt++,xpc,x,norm,cq,ckout,cvout,qw,kw,vw);
    }
    printf("rep=%d control_us=%.6f candidate_us=%.6f recovery_us=%.6f\n",r,cs/passes,ps/passes,(cs-ps)/passes);
  }
  ck(cudaGetLastError(),"final CUDA status");
  return mismatch ? 5 : 0;
}
'''


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--passes", type=int, default=80)
  ap.add_argument("--reps", type=int, default=9)
  ap.add_argument("--out", type=Path, required=True)
  args = ap.parse_args()
  provider, q, kv = _render()
  source = HARNESS.replace("__PROVIDER__", provider).replace("__Q__", q).replace("__KV__", kv)
  with tempfile.TemporaryDirectory(prefix="nv_shared_q8_qkv_concurrency_") as td:
    src, binary = Path(td)/"gate.cu", Path(td)/"gate"
    src.write_text(source)
    env = {**os.environ, "PATH": f"{CUDA_BIN}:" + os.environ.get("PATH", "")}
    build = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-std=c++17", "--ptxas-options=-v",
                            str(src), "-o", str(binary)], capture_output=True, text=True, env=env)
    if build.returncode:
      print(build.stderr[-12000:], file=sys.stderr)
      return 3
    run = subprocess.run([str(binary), str(args.passes), str(args.reps)], capture_output=True, text=True)
    print(run.stdout.strip())
    if run.returncode not in (0, 5):
      print(run.stderr[-8000:], file=sys.stderr)
      return 4
    exact = re.search(r"mismatched_words=(\d+) bitwise_identical=(\d+)", run.stdout)
    controls, candidates, recoveries = [], [], []
    for line in run.stdout.splitlines():
      if m := re.search(r"control_us=([0-9.]+) candidate_us=([0-9.]+) recovery_us=([-0-9.]+)", line):
        controls.append(float(m.group(1))); candidates.append(float(m.group(2))); recoveries.append(float(m.group(3)))
    cm, pm = statistics.median(controls), statistics.median(candidates)
    recovery = cm-pm
    route_recovery, global_ceiling = recovery*9, recovery*36
    result = {
      "schema": "tinygrad.nv_shared_q8_qkv_concurrency_microgate.v1",
      "commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
      "method": "production-rendered kernels, cudaEvent fork/join, 256 MiB cache eviction before every span",
      "shape": {"q_rows": Q_ROWS, "kv_rows": KV_ROWS, "k": K, "weight_bytes": (Q_WORDS+2*KV_WORDS)*4},
      "passes": args.passes, "reps": args.reps,
      "bitwise_identical": bool(exact and int(exact.group(2))),
      "mismatched_words": int(exact.group(1)) if exact else None,
      "timing": {"unit": "us_per_complete_provider_to_join_span", "control_samples": controls,
                 "candidate_samples": candidates, "paired_recovery_samples": recoveries,
                 "control_median": cm, "candidate_median": pm, "recovery_us_per_layer": recovery,
                 "candidate_over_control": pm/cm},
      "projection": {
        "installed_shared_q4q4_layers": 9, "installed_route_recovery_us_per_token": route_recovery,
        "installed_route_projected_tok_s": 1e6/(ENDPOINT_US-route_recovery),
        "all_36_layers_unvalidated_ceiling_us": global_ceiling,
        "all_36_layers_unvalidated_ceiling_tok_s": 1e6/(ENDPOINT_US-global_ceiling),
        "endpoint_us": ENDPOINT_US, "endpoint_tok_s": 1e6/ENDPOINT_US,
      },
      "ptxas": build.stderr.strip().splitlines(),
      "verdict": "ADVANCE_TO_LIVE_HCQ" if exact and int(exact.group(2)) and recovery > 0.5 else "NO_GO",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["bitwise_identical"] else 5


if __name__ == "__main__": raise SystemExit(main())
