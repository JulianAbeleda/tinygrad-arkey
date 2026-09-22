#!/usr/bin/env python3
"""Cache-hot/cold Q6_K fp16 packed-lanemap microgate for FFN-down.

Control is the installed four-warp Q6_K FFN-down emitter.  Candidate keeps
the same one-row/four-warp geometry, fp16 activation contract, residual-add
epilogue, weight bytes, and output type, but changes ownership inside each
Q6_K block to llama's packed lane map.  Each lane forms two packed qwords
from ql/qh and consumes two half4 activation vectors (eight values total),
instead of evaluating the generic 16-group scalar-byte spelling.

The 210-byte Q6_K block is not widened wholesale.  That old surface is known
to misalign every other block.  This probe asks the narrower question: can a
packed lane map raise the same-byte DRAM rate of the already-landed four-warp
route?  A 256 MiB device walk precedes every cold timing span and is excluded
from the CUDA-event interval.
"""
from __future__ import annotations

import argparse, json, os, re, statistics, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.codegen.late.warp_reduce import _staged_shfl
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Target
from tinygrad.llm.decode_kernels import Q6K_HALFWORDS_PER_BLOCK, _f16_half, _half4_lane, _q6k_byte
from tinygrad.llm.q6k_ffn_down_mmvq import emit_q6k_four_warp_fp16_direct
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp

ROWS, K, QK = 4096, 12288, 256
K_BLOCKS = K // QK
WARP, WARPS = 32, 4
BLOCKS_PER_WARP = K_BLOCKS // WARPS
HALFS = ROWS * K_BLOCKS * Q6K_HALFWORDS_PER_BLOCK
FLUSH_WORDS = 64 * 1024 * 1024  # 256 MiB, larger than device L2.
WEIGHT_BYTES = HALFS * 2
ENDPOINT_US = 4515.39571875
LLAMA_RATE_TB_S = 1.449
CUDA_BIN = "/usr/local/cuda-13.2/bin"


def _i8f(v: UOp) -> UOp:
  return v.cast(dtypes.uint8).bitcast(dtypes.int8).cast(dtypes.float32)


def emit_q6k_fp16_packed_lanemap() -> callable:
  """Four-warp direct Q6_K with llama's packed per-block lane ownership."""
  def kernel(out: UOp, halfs: UOp, x: UOp, residual: UOp) -> UOp:
    row = UOp.special(ROWS, "gidx0")
    lid = UOp.special(WARP * WARPS, "lidx0")
    warp, lane = lid // WARP, lid % WARP
    block_rel = UOp.range(BLOCKS_PER_WARP, 0, axis_type=AxisType.REDUCE)
    block = warp * BLOCKS_PER_WARP + block_rel
    base = (row * K_BLOCKS + block) * Q6K_HALFWORDS_PER_BLOCK

    # Same packed ql/qh mapping as vec_dot_q6_K_q8_1_impl_mmvq.  Scalar
    # halfword spelling is deliberate: a Q6 block is 210 B, so every other
    # block is only 2-byte aligned.  The win under test comes from ownership
    # and deduplication, not an unsafe whole-block vector cast.
    vl = halfs[base + lane * 2].cast(dtypes.uint32).bitwise_or(
      halfs[base + lane * 2 + 1].cast(dtypes.uint32).lshift(16))
    qh_half = 16 * (lane // 16) + 2 * (lane % 8)
    vh = halfs[base + 64 + qh_half].cast(dtypes.uint32).bitwise_or(
      halfs[base + 65 + qh_half].cast(dtypes.uint32).lshift(16))
    vh = vh.rshift(2 * ((lane % 16) // 8))

    scale_idx = 8 * (lane // 16) + (lane % 16) // 4
    x_group0 = 4 * (lane // 16) + (lane % 16) // 8
    d = _f16_half(halfs[base + 104])
    contribution = UOp.const(dtypes.float32, 0.0)
    for term in range(2):
      low = vl.rshift(4 * term).bitwise_and(0x0F0F0F0F)
      high = vh.rshift(4 * term).lshift(4).bitwise_and(0x30303030)
      qword = low.bitwise_or(high)
      scale = _i8f(_q6k_byte(halfs, base, 192 + scale_idx + 4 * term))
      x_group = x_group0 + 2 * term
      xv = x.index(block * QK + x_group * 32 + (lane % 8) * 4).load(dtype=dtypes.float16.vec(4))
      for nib in range(4):
        q = qword.rshift(nib * 8).bitwise_and(0xff).cast(dtypes.float32) - 32.0
        weight = d * q * scale
        contribution = contribution + weight * _half4_lane(xv, nib)

    acc = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(block_rel)[0] + contribution).end(block_rel))
    total = acc[0]
    for slot, offset in enumerate((16, 8, 4, 2, 1), 90):
      total = total + _staged_shfl(total, offset, lane, slot)
    smem = UOp.placeholder((WARPS,), dtypes.float32, 40, addrspace=AddrSpace.LOCAL)
    published = smem[warp].store(total, lane.eq(0))
    ready = UOp.barrier(UOp.group(published))
    merged = UOp.const(dtypes.float32, 0.0)
    for wi in range(WARPS): merged = merged + smem.after(ready)[wi]
    return out[row].store(merged + residual[row], lid.eq(0)).sink(
      arg=KernelInfo(name="q6k_fp16_packed_lanemap_4096_12288_epi_ffnresadd", opts_to_apply=()))
  return kernel


def _render() -> tuple[str, str]:
  p = UOp.placeholder
  args = (p((ROWS,), dtypes.float32, 0), p((HALFS,), dtypes.uint16, 1),
          p((K,), dtypes.float16, 2), p((ROWS,), dtypes.float32, 3))
  control = emit_q6k_four_warp_fp16_direct()(*args)
  candidate = emit_q6k_four_warp_fp16_direct(packed_lanemap=True)(*args)
  ren = CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=False)

  def source(u: UOp) -> str:
    text = next(x.arg for x in to_program(u, ren).src if x.op is Ops.SOURCE)
    return text[text.index('extern "C" __global__'):]
  return source(control), source(candidate)


HARNESS = r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
#define ROWS 4096
#define K 12288
#define K_BLOCKS 48
#define Q6_HALFS_PER_BLOCK 105
#define HALFS_COUNT 20643840
#define FLUSH_WORDS 67108864
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
struct __align__(8) half4 { half x, y, z, w; };
__device__ half4 make_half4(half x, half y, half z, half w) { half4 r={x,y,z,w}; return r; }

__CONTROL__
__CANDIDATE__

__global__ void evict_l2(unsigned int *buf, unsigned int salt) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < FLUSH_WORDS; i += stride) buf[i] = buf[i] + salt + i;
}

static void ck(cudaError_t e, const char *what) {
  if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); exit(2); }
}

static void control(float *out, unsigned short *w, half *x, float *h, cudaStream_t s) {
  q6k_fp16_mmvq_direct_4096_12288_epi_ffnresadd<<<ROWS,128,0,s>>>(out,w,x,h);
}
static void candidate(float *out, unsigned short *w, half *x, float *h, cudaStream_t s) {
  q6k_fp16_packed_lanemap_4096_12288_epi_ffnresadd<<<ROWS,128,0,s>>>(out,w,x,h);
}

static double hot(bool cand, int passes, float *out, unsigned short *w, half *x, float *h,
                  cudaStream_t s, cudaEvent_t start, cudaEvent_t stop) {
  ck(cudaEventRecord(start,s),"hot start");
  for (int i=0;i<passes;i++) (cand ? candidate : control)(out,w,x,h,s);
  ck(cudaEventRecord(stop,s),"hot stop"); ck(cudaEventSynchronize(stop),"hot sync");
  float ms=0; ck(cudaEventElapsedTime(&ms,start,stop),"hot elapsed");
  return ms*1000.0/passes;
}

static double cold(bool cand, int passes, unsigned int& salt, unsigned int *trash,
                   float *out, unsigned short *w, half *x, float *h,
                   cudaStream_t s, cudaEvent_t start, cudaEvent_t stop) {
  double total=0.0;
  for (int i=0;i<passes;i++) {
    evict_l2<<<8192,256,0,s>>>(trash,salt++);
    ck(cudaEventRecord(start,s),"cold start");
    (cand ? candidate : control)(out,w,x,h,s);
    ck(cudaEventRecord(stop,s),"cold stop"); ck(cudaEventSynchronize(stop),"cold sync");
    float ms=0; ck(cudaEventElapsedTime(&ms,start,stop),"cold elapsed"); total += ms*1000.0;
  }
  return total/passes;
}

int main(int argc, char **argv) {
  int hot_passes=argc>1?atoi(argv[1]):100;
  int cold_passes=argc>2?atoi(argv[2]):40;
  int reps=argc>3?atoi(argv[3]):9;
  cudaStream_t s; cudaEvent_t start,stop;
  ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking),"stream");
  ck(cudaEventCreate(&start),"start event"); ck(cudaEventCreate(&stop),"stop event");
  unsigned short *w=nullptr; half *x=nullptr; float *h=nullptr,*co=nullptr,*ca=nullptr; unsigned int *trash=nullptr;
  ck(cudaMalloc(&w,size_t(HALFS_COUNT)*sizeof(unsigned short)),"weights");
  ck(cudaMalloc(&x,K*sizeof(half)),"x"); ck(cudaMalloc(&h,ROWS*sizeof(float)),"h");
  ck(cudaMalloc(&co,ROWS*sizeof(float)),"control out"); ck(cudaMalloc(&ca,ROWS*sizeof(float)),"candidate out");
  ck(cudaMalloc(&trash,size_t(FLUSH_WORDS)*sizeof(unsigned int)),"trash");
  unsigned short *hw=(unsigned short*)malloc(size_t(HALFS_COUNT)*sizeof(unsigned short));
  half *hx=(half*)malloc(K*sizeof(half)); float *hh=(float*)malloc(ROWS*sizeof(float));
  unsigned int state=0x1234567u;
  for (size_t i=0;i<size_t(HALFS_COUNT);i++) { state=1664525u*state+1013904223u; hw[i]=(unsigned short)(state>>16); }
  for (int row=0;row<ROWS;row++) for (int block=0;block<K_BLOCKS;block++) {
    size_t base=(size_t(row)*K_BLOCKS+block)*Q6_HALFS_PER_BLOCK;
    signed char *bytes=(signed char*)(hw+base);
    for (int i=0;i<16;i++) bytes[192+i]=(signed char)(((row*13+block*7+i*3)%15)-7);
    hw[base+104]=0x3800u;
  }
  for (int i=0;i<K;i++) hx[i]=__float2half(float((i%257)-128)/512.0f);
  for (int i=0;i<ROWS;i++) hh[i]=float((i%31)-15)/1024.0f;
  ck(cudaMemcpy(w,hw,size_t(HALFS_COUNT)*sizeof(unsigned short),cudaMemcpyHostToDevice),"weights copy");
  ck(cudaMemcpy(x,hx,K*sizeof(half),cudaMemcpyHostToDevice),"x copy");
  ck(cudaMemcpy(h,hh,ROWS*sizeof(float),cudaMemcpyHostToDevice),"h copy");
  ck(cudaMemset(trash,0,size_t(FLUSH_WORDS)*sizeof(unsigned int)),"trash memset");
  free(hw); free(hx); free(hh);

  control(co,w,x,h,s); candidate(ca,w,x,h,s); ck(cudaStreamSynchronize(s),"correctness sync");
  float *hc=(float*)malloc(ROWS*sizeof(float)), *hv=(float*)malloc(ROWS*sizeof(float));
  ck(cudaMemcpy(hc,co,ROWS*sizeof(float),cudaMemcpyDeviceToHost),"control copy");
  ck(cudaMemcpy(hv,ca,ROWS*sizeof(float),cudaMemcpyDeviceToHost),"candidate copy");
  double diff2=0.0,base2=0.0,max_abs=0.0; int finite=1,mismatch=0;
  for (int i=0;i<ROWS;i++) {
    double d=double(hv[i])-double(hc[i]); diff2+=d*d; base2+=double(hc[i])*double(hc[i]);
    max_abs=fmax(max_abs,fabs(d)); finite &= isfinite(hv[i]);
    unsigned int a,b; memcpy(&a,hc+i,4); memcpy(&b,hv+i,4); mismatch += a!=b;
  }
  printf("finite=%d mismatched_words=%d max_abs=%.9g rel_l2=%.9g\n",finite,mismatch,max_abs,sqrt(diff2/base2));
  free(hc); free(hv);

  // Warm both bodies before collecting hot and cold paired samples.
  for (int i=0;i<20;i++) { control(co,w,x,h,s); candidate(ca,w,x,h,s); }
  ck(cudaStreamSynchronize(s),"warm sync");
  unsigned int salt=1;
  for (int r=0;r<reps;r++) {
    double ch,vh,cc,vc;
    if ((r&1)==0) {
      ch=hot(false,hot_passes,co,w,x,h,s,start,stop); vh=hot(true,hot_passes,ca,w,x,h,s,start,stop);
      cc=cold(false,cold_passes,salt,trash,co,w,x,h,s,start,stop); vc=cold(true,cold_passes,salt,trash,ca,w,x,h,s,start,stop);
    } else {
      vh=hot(true,hot_passes,ca,w,x,h,s,start,stop); ch=hot(false,hot_passes,co,w,x,h,s,start,stop);
      vc=cold(true,cold_passes,salt,trash,ca,w,x,h,s,start,stop); cc=cold(false,cold_passes,salt,trash,co,w,x,h,s,start,stop);
    }
    printf("rep=%d hot_control_us=%.6f hot_candidate_us=%.6f cold_control_us=%.6f cold_candidate_us=%.6f\n",r,ch,vh,cc,vc);
  }
  ck(cudaGetLastError(),"final CUDA status");
  return finite ? 0 : 5;
}
'''


def _sass_census(binary: Path, symbol: str) -> dict:
  cuobjdump = Path(CUDA_BIN) / "cuobjdump"
  nvdisasm = ROOT / ".venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/nvdisasm"
  env = {**os.environ}
  if nvdisasm.is_file(): env["NVDISASM_PATH"] = str(nvdisasm)
  try:
    text = subprocess.check_output([str(cuobjdump), "--dump-sass", str(binary)], text=True,
                                   stderr=subprocess.STDOUT, env=env)
  except (OSError, subprocess.CalledProcessError) as exc:
    detail = getattr(exc, "output", None)
    return {"available": False, "reason": str(exc), "detail": detail[-2000:] if isinstance(detail, str) else None}
  marker = f"Function : {symbol}"
  if marker not in text: return {"available": False, "reason": f"missing {symbol}"}
  body = text.split(marker, 1)[1].split("Function :", 1)[0]
  opcodes = re.findall(r"/\*[0-9a-f]+\*/\s+(?:@[!P0-9]+\s+)?([A-Z][A-Z0-9_.]+)", body)
  ldg = [op for op in opcodes if op.startswith("LDG")]
  return {"available": True, "instructions": len(opcodes), "ldg": len(ldg),
          "ldg_u16": sum("U16" in op for op in ldg), "ldg_32": sum(op == "LDG.E" or op.endswith(".32") for op in ldg),
          "ldg_64": sum(".64" in op for op in ldg), "ldg_128": sum(".128" in op for op in ldg),
          "shfl": sum(op.startswith("SHFL") for op in opcodes), "barrier": sum(op.startswith("BAR") for op in opcodes)}


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--hot-passes", type=int, default=100)
  ap.add_argument("--cold-passes", type=int, default=40)
  ap.add_argument("--reps", type=int, default=9)
  ap.add_argument("--out", type=Path, required=True)
  args = ap.parse_args()
  control, candidate = _render()
  source = HARNESS.replace("__CONTROL__", control).replace("__CANDIDATE__", candidate)
  with tempfile.TemporaryDirectory(prefix="nv_q6k_fp16_packed_lanemap_") as td:
    src, binary = Path(td)/"gate.cu", Path(td)/"gate"
    src.write_text(source)
    env = {**os.environ, "PATH": f"{CUDA_BIN}:" + os.environ.get("PATH", "")}
    build = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-std=c++17", "--ptxas-options=-v",
                            str(src), "-o", str(binary)], capture_output=True, text=True, env=env)
    if build.returncode:
      print(build.stderr[-12000:], file=sys.stderr)
      return 3
    run = subprocess.run([str(binary), str(args.hot_passes), str(args.cold_passes), str(args.reps)],
                         capture_output=True, text=True)
    print(run.stdout.strip())
    if run.returncode not in (0, 5):
      print(run.stderr[-8000:], file=sys.stderr)
      return 4
    correctness = re.search(r"finite=(\d+) mismatched_words=(\d+) max_abs=([0-9.eE+-]+) rel_l2=([0-9.eE+-]+)", run.stdout)
    samples = {"hot_control": [], "hot_candidate": [], "cold_control": [], "cold_candidate": []}
    pat = re.compile(r"hot_control_us=([0-9.]+) hot_candidate_us=([0-9.]+) cold_control_us=([0-9.]+) cold_candidate_us=([0-9.]+)")
    for line in run.stdout.splitlines():
      if m := pat.search(line):
        for key, val in zip(samples, m.groups()): samples[key].append(float(val))
    med = {key: statistics.median(vals) for key, vals in samples.items()}
    hot_recovery, cold_recovery = med["hot_control"]-med["hot_candidate"], med["cold_control"]-med["cold_candidate"]
    cold_rate = WEIGHT_BYTES/med["cold_candidate"]/1e6
    llama_time = WEIGHT_BYTES/LLAMA_RATE_TB_S/1e6
    all_down_recovery = cold_recovery * 18
    result = {
      "schema": "tinygrad.nv_q6k_fp16_packed_lanemap_microgate.v1",
      "commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
      "method": "production-rendered bodies, cudaEvent, hot replay plus 256 MiB eviction before every cold span",
      "shape": {"rows": ROWS, "k": K, "weight_bytes": WEIGHT_BYTES, "q6_block_bytes": 210},
      "correctness": {"finite": bool(correctness and int(correctness.group(1))),
                      "mismatched_words": int(correctness.group(2)) if correctness else None,
                      "max_abs": float(correctness.group(3)) if correctness else None,
                      "relative_l2": float(correctness.group(4)) if correctness else None,
                      "pass": bool(correctness and int(correctness.group(1)) and float(correctness.group(4)) <= 1e-3)},
      "timing": {"unit": "us_per_kernel", "hot_passes": args.hot_passes, "cold_passes": args.cold_passes,
                 "reps": args.reps, "samples": samples, "medians": med,
                 "hot_recovery_us_per_layer": hot_recovery, "cold_recovery_us_per_layer": cold_recovery,
                 "control_cold_rate_tb_s": WEIGHT_BYTES/med["cold_control"]/1e6,
                 "candidate_cold_rate_tb_s": cold_rate},
      "llama_reference": {"rate_tb_s": LLAMA_RATE_TB_S, "same_weight_bytes_time_us": llama_time},
      "projection": {"installed_q6_down_layers": 18, "cold_recovery_us_per_token": all_down_recovery,
                     "endpoint_us": ENDPOINT_US, "endpoint_tok_s": 1e6/ENDPOINT_US,
                     "projected_us": ENDPOINT_US-all_down_recovery,
                     "projected_tok_s": 1e6/(ENDPOINT_US-all_down_recovery)},
      "sass": {"control": _sass_census(binary, "q6k_fp16_mmvq_direct_4096_12288_epi_ffnresadd"),
               "candidate": _sass_census(binary, "q6k_fp16_packed_lanemap_4096_12288_epi_ffnresadd")},
      "ptxas": build.stderr.strip().splitlines(),
      "verdict": "ADVANCE_TO_PRODUCTION_WALL" if correctness and int(correctness.group(1)) and float(correctness.group(4)) <= 1e-3 and cold_recovery > 0.5 else "NO_GO",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["correctness"]["pass"] else 5


if __name__ == "__main__": raise SystemExit(main())
