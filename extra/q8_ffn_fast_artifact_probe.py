#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, statistics, subprocess, tempfile, time

import numpy as np

from tinygrad import dtypes
from tinygrad.device import Buffer, Device
from tinygrad.runtime.support.compiler_amd import HIPCCCompiler
from tinygrad.runtime.support.elf import elf_loader
from extra.q8_ffn_hcq_artifact import NORM_SOURCE, MMVQ_SOURCE, copyin_array, copyout_array, make_buffer, q8_dequant
from extra.q8_ffn_handwritten_oracle import q4_ref_rows, tensor_info
from extra.qk_layout import Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, read_metadata

ROCM_LLVM = pathlib.Path("/opt/rocm/llvm/bin")

HIP_NORM_SOURCE_TEMPLATE = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

typedef struct { half d; half s; signed char qs[32]; } block_q8_1;

extern "C" __global__ __launch_bounds__(NT, 1) void q8_rmsnorm_side(float *out, block_q8_1 *q8,
                                                                     const float *x, const float *w) {
  __shared__ float red[NT];
  const int tid = threadIdx.x;
  const int n = 4096;
  const float eps = 1.0e-6f;
  float ss = 0.0f;
  for (int i = tid; i < n; i += NT) ss += x[i] * x[i];
  red[tid] = ss;
  __syncthreads();
  for (int off = NT / 2; off > 0; off >>= 1) {
    if (tid < off) red[tid] += red[tid + off];
    __syncthreads();
  }
  const float rinv = rsqrtf(red[0] / (float)n + eps);
  for (int i = tid; i < n; i += NT) out[i] = x[i] * rinv * w[i];
  __syncthreads();
  for (int b = tid; b < n / 32; b += NT) {
    float vals[32];
    float mx = 0.0f;
    #pragma unroll
    for (int j = 0; j < 32; j++) {
      const int idx = b * 32 + j;
      vals[j] = x[idx] * rinv * w[idx];
      mx = fmaxf(mx, fabsf(vals[j]));
    }
    const float scale = (mx == 0.0f) ? 1.0f : mx / 127.0f;
    q8[b].d = __float2half(scale);
    q8[b].s = __float2half(0.0f);
    #pragma unroll
    for (int j = 0; j < 32; j++) {
      int qi = (int)nearbyintf(vals[j] / scale);
      qi = qi < -128 ? -128 : (qi > 127 ? 127 : qi);
      q8[b].qs[j] = (signed char)qi;
    }
  }
}
"""

def hip_norm_source(nt:int) -> str:
  return HIP_NORM_SOURCE_TEMPLATE.replace("NT", str(nt))

HIP_MMVQ_SOURCE = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define QK_K 256
typedef struct { half d; half dmin; unsigned char scales[12]; unsigned char qs[128]; } block_q4_K;
typedef struct { half d; half s; signed char qs[32]; } block_q8_1;

__device__ __host__ static inline void get_scale_min(int j, const unsigned char *q, unsigned char *d, unsigned char *m) {
  if (j < 4) { *d = q[j] & 63; *m = q[j+4] & 63; }
  else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4); *m = (q[j+4] >> 4) | ((q[j] >> 6) << 4); }
}

#define NW 4
#define WS 32
extern "C" __global__ __launch_bounds__(NW*WS, 1) void q8_mmvq(float *dst, const block_q4_K *x, const block_q8_1 *y) {
  const int row = blockIdx.x;
  const int tid = threadIdx.y * WS + threadIdx.x;
  const int nbpr = 4096 / QK_K;
  float tmp = 0.0f;
  for (int kb = tid / 8; kb < nbpr; kb += (NW * WS) / 8) {
    const block_q4_K *bx = x + row * nbpr + kb;
    const int sub = tid & 7;
    const block_q8_1 *by = y + kb * 8 + sub;
    unsigned char sc, mn;
    get_scale_min(sub, bx->scales, &sc, &mn);
    const unsigned char *q = bx->qs + (sub / 2) * 32;
    int sumi = 0, sumq = 0;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
      int qb = ((const int*)q)[k];
      int nib = (sub & 1) ? ((qb >> 4) & 0x0F0F0F0F) : (qb & 0x0F0F0F0F);
      int u = ((const int*)by->qs)[k];
      sumi = __builtin_amdgcn_sudot4(false, nib, true, u, sumi, false);
      sumq = __builtin_amdgcn_sudot4(false, 0x01010101, true, u, sumq, false);
    }
    const float d8 = __half2float(by->d);
    const float dd = __half2float(bx->d), dm = __half2float(bx->dmin);
    tmp += d8 * (dd * (float)sc * (float)sumi - dm * (float)mn * (float)sumq);
  }
  __shared__ float sm[NW];
  for (int o = WS / 2; o > 0; o >>= 1) tmp += __shfl_xor(tmp, o);
  if (threadIdx.x == 0) sm[threadIdx.y] = tmp;
  __syncthreads();
  if (tid == 0) {
    float s = 0.0f;
    #pragma unroll
    for (int i = 0; i < NW; i++) s += sm[i];
    dst[row] = s;
  }
}
"""

HIP_MMVQ_GATEUP_SOURCE = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define QK_K 256
typedef struct { half d; half dmin; unsigned char scales[12]; unsigned char qs[128]; } block_q4_K;
typedef struct { half d; half s; signed char qs[32]; } block_q8_1;

__device__ __host__ static inline void get_scale_min(int j, const unsigned char *q, unsigned char *d, unsigned char *m) {
  if (j < 4) { *d = q[j] & 63; *m = q[j+4] & 63; }
  else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4); *m = (q[j+4] >> 4) | ((q[j] >> 6) << 4); }
}

#define NW 4
#define WS 32
extern "C" __global__ __launch_bounds__(NW*WS, 1) void q8_mmvq_gateup(float *dst0, float *dst1,
                                                                      const block_q4_K *x0, const block_q4_K *x1,
                                                                      const block_q8_1 *y) {
  const int row = blockIdx.x;
  const block_q4_K *x = blockIdx.y == 0 ? x0 : x1;
  const int tid = threadIdx.y * WS + threadIdx.x;
  const int nbpr = 4096 / QK_K;
  float tmp = 0.0f;
  for (int kb = tid / 8; kb < nbpr; kb += (NW * WS) / 8) {
    const block_q4_K *bx = x + row * nbpr + kb;
    const int sub = tid & 7;
    const block_q8_1 *by = y + kb * 8 + sub;
    unsigned char sc, mn;
    get_scale_min(sub, bx->scales, &sc, &mn);
    const unsigned char *q = bx->qs + (sub / 2) * 32;
    int sumi = 0, sumq = 0;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
      int qb = ((const int*)q)[k];
      int nib = (sub & 1) ? ((qb >> 4) & 0x0F0F0F0F) : (qb & 0x0F0F0F0F);
      int u = ((const int*)by->qs)[k];
      sumi = __builtin_amdgcn_sudot4(false, nib, true, u, sumi, false);
      sumq = __builtin_amdgcn_sudot4(false, 0x01010101, true, u, sumq, false);
    }
    const float d8 = __half2float(by->d);
    const float dd = __half2float(bx->d), dm = __half2float(bx->dmin);
    tmp += d8 * (dd * (float)sc * (float)sumi - dm * (float)mn * (float)sumq);
  }
  __shared__ float sm[NW];
  for (int o = WS / 2; o > 0; o >>= 1) tmp += __shfl_xor(tmp, o);
  if (threadIdx.x == 0) sm[threadIdx.y] = tmp;
  __syncthreads();
  if (tid == 0) {
    float s = 0.0f;
    #pragma unroll
    for (int i = 0; i < NW; i++) s += sm[i];
    if (blockIdx.y == 0) dst0[row] = s; else dst1[row] = s;
  }
}
"""

def run(cmd:list[str]) -> str:
  return subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout

def compile_hipcc_reloc(src:str, arch:str) -> bytes:
  return HIPCCCompiler(arch).compile(src)

def compile_hipcc_linked(src:str, arch:str) -> bytes:
  with tempfile.TemporaryDirectory(prefix="q8_fast_artifact_") as td:
    p = pathlib.Path(td)
    srcp, bcp, relp, linkedp = p/"src.cpp", p/"obj.bc", p/"rel.o", p/"linked.so"
    srcp.write_text(src)
    run(["hipcc", "-c", "-emit-llvm", "--cuda-device-only", "-O3", "-mcumode", f"--offload-arch={arch}",
         "-I/opt/rocm/include/hip", "-o", str(bcp), str(srcp)])
    run(["hipcc", "-target", "amdgcn-amd-amdhsa", f"-mcpu={arch}", "-O3", "-mllvm", "-amdgpu-internalize-symbols",
         "-c", "-o", str(relp), str(bcp)])
    run([str(ROCM_LLVM/"ld.lld"), "-flavor", "gnu", "-shared", "-o", str(linkedp), str(relp)])
    return linkedp.read_bytes()

def inspect_blob(blob:bytes, name:str) -> dict:
  with tempfile.NamedTemporaryFile(suffix=f"_{name}.hsaco") as f:
    f.write(blob); f.flush()
    readelf = run([str(ROCM_LLVM/"llvm-readelf"), "-r", "-S", "-s", f.name])
  image, sections, relocs = elf_loader(blob)
  reloc_lines = [ln.strip() for ln in readelf.splitlines() if "R_AMDGPU_" in ln]
  return {
    "bytes": len(blob),
    "image_bytes": image.nbytes,
    "sections": [{"name": s.name, "addr": int(s.header.sh_addr), "size": int(s.header.sh_size),
                  "type": int(s.header.sh_type), "flags": int(s.header.sh_flags)} for s in sections],
    "relocs_seen_by_tinygrad": [{"apply_image_offset": int(a), "rel_sym_offset": int(b), "type": int(c), "addend": int(d)}
                                for a,b,c,d in relocs],
    "readelf_relocations": reloc_lines,
  }

def write_audit(out:pathlib.Path, arch:str) -> dict:
  dev = Device["AMD"]
  comgr = dev.compiler.compile(MMVQ_SOURCE)
  raw_hipcc_reloc = compile_hipcc_reloc(MMVQ_SOURCE, arch)
  hipcc_reloc = compile_hipcc_reloc(HIP_MMVQ_SOURCE, arch)
  hipcc_linked = compile_hipcc_linked(HIP_MMVQ_SOURCE, arch)
  unlinked_load = {"ok": False, "error": ""}
  linked_load = {"ok": False, "error": ""}
  try:
    dev.runtime("q8_mmvq_hipcc_unlinked_audit", hipcc_reloc)
    unlinked_load["ok"] = True
  except Exception as e:
    unlinked_load["error"] = str(e)
  try:
    prg = dev.runtime("q8_mmvq_hipcc_linked_audit", hipcc_linked)
    linked_load.update({"ok": True, "kernarg_size": prg.kernargs_segment_size,
                        "group_segment_size": prg.group_segment_size, "private_segment_size": prg.private_segment_size})
  except Exception as e:
    linked_load["error"] = str(e)

  res = {
    "date": "2026-06-19",
    "phase": "A-F0",
    "arch": arch,
    "finding": "the header-free raw-C hipcc object emits unsupported REL32_LO/HI relocations; the HIP-style oracle object only needs supported REL64 before LLD, and the LLD-linked object has no relocations and loads in AMDProgram",
    "relocation_types": {
      "5": "R_AMDGPU_REL64",
      "10": "R_AMDGPU_REL32_LO",
      "11": "R_AMDGPU_REL32_HI",
    },
    "load": {"hipcc_relocatable": unlinked_load, "hipcc_lld_linked": linked_load},
    "objects": {
      "comgr_raw_c_baseline": inspect_blob(comgr, "comgr"),
      "hipcc_raw_c_relocatable": inspect_blob(raw_hipcc_reloc, "hipcc_raw_reloc"),
      "hipcc_oracle_relocatable": inspect_blob(hipcc_reloc, "hipcc_reloc"),
      "hipcc_oracle_lld_linked": inspect_blob(hipcc_linked, "hipcc_linked"),
    },
    "verdict": "PASS" if linked_load["ok"] else "FAIL",
  }
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(res, indent=2) + "\n")
  return res

def pctile(xs:list[float], p:float) -> float:
  ys = sorted(xs)
  return ys[min(len(ys)-1, max(0, round((len(ys)-1)*p)))]

def ms_stats(samples:list[float]) -> dict:
  return {
    "samples_ms": [round(x, 6) for x in samples],
    "min_ms": round(min(samples), 6),
    "median_ms": round(statistics.median(samples), 6),
    "mean_ms": round(statistics.fmean(samples), 6),
    "p10_ms": round(pctile(samples, 0.10), 6),
    "p90_ms": round(pctile(samples, 0.90), 6),
    "max_ms": round(max(samples), 6),
  }

def read_q4(gguf:pathlib.Path, tensor:str, rows_arg:int|None) -> tuple[bytes, int, int, list[int]]:
  meta = read_metadata(gguf)
  info, shape = tensor_info(meta, tensor)
  rows, k = min(rows_arg or shape[0], shape[0]), shape[1]
  row_bytes = k // Q4_K_BLOCK_ELEMS * Q4_K_BLOCK_BYTES
  with gguf.open("rb") as f:
    f.seek(meta.data_start + info.off)
    q4 = f.read(rows * row_bytes)
  return q4, rows, k, list(shape)

def perf_one(args, norm_prg, mmvq_prg, tensor:str) -> dict:
  rng = np.random.default_rng(args.seed)
  x = (rng.standard_normal(4096).astype(np.float32) * 0.9).astype(np.float32)
  w = (0.7 + rng.random(4096).astype(np.float32) * 0.2).astype(np.float32)
  rinv = np.float32(1.0 / np.sqrt(np.sum(x * x, dtype=np.float32) / np.float32(4096.0) + np.float32(1.0e-6)))
  ref_norm = (x * rinv * w).astype(np.float32)

  xbuf, wbuf = make_buffer(4096, dtypes.float32), make_buffer(4096, dtypes.float32)
  norm_out, q8buf = make_buffer(4096, dtypes.float32), make_buffer(128 * 36, dtypes.uint8)
  copyin_array(xbuf, x); copyin_array(wbuf, w)

  prod_ms = []
  for i in range(args.warmups + args.iters):
    ms = float(norm_prg(norm_out._buf, q8buf._buf, xbuf._buf, wbuf._buf, global_size=(1,1,1),
                        local_size=(args.producer_threads,1,1), wait=True)) * 1000.0
    if i >= args.warmups: prod_ms.append(ms)
  got_norm = copyout_array(norm_out, np.empty(4096, dtype=np.float32))
  got_q8 = bytearray(128 * 36); q8buf.copyout(memoryview(got_q8))
  q8_x = q8_dequant(bytes(got_q8), 4096)

  q4, rows, k, shape = read_q4(args.gguf, tensor, args.rows)
  ref = q4_ref_rows(q4, rows, k, q8_x)
  q4buf, dstbuf = make_buffer(len(q4), dtypes.uint8), make_buffer(rows, dtypes.float32)
  q4buf.copyin(memoryview(q4))

  mmvq_ms = []
  for i in range(args.warmups + args.iters):
    ms = float(mmvq_prg(dstbuf._buf, q4buf._buf, q8buf._buf, global_size=(rows,1,1), local_size=(32,4,1), wait=True)) * 1000.0
    if i >= args.warmups: mmvq_ms.append(ms)
  got = copyout_array(dstbuf, np.empty(rows, dtype=np.float32))
  err = np.abs(got - ref)
  return {
    "tensor": tensor,
    "shape": shape,
    "rows": rows,
    "producer": ms_stats(prod_ms),
    "consumer": ms_stats(mmvq_ms),
    "correctness": {
      "producer_fp_max_abs": float(np.abs(got_norm - ref_norm).max()),
      "q8_dequant_max_abs": float(np.abs(q8_x - ref_norm).max()),
      "consumer_max_abs": float(err.max()),
      "consumer_mean_abs": float(err.mean()),
    },
    "gates": {
      "producer_correct": float(np.abs(got_norm - ref_norm).max()) <= 1e-5,
      "consumer_correct": float(err.max()) <= 2e-3,
      "consumer_median_lte_60us": statistics.median(mmvq_ms) <= 0.060,
    },
  }

def perf_gateup(args, norm_prg, gateup_prg) -> dict:
  rng = np.random.default_rng(args.seed)
  x = (rng.standard_normal(4096).astype(np.float32) * 0.9).astype(np.float32)
  w = (0.7 + rng.random(4096).astype(np.float32) * 0.2).astype(np.float32)
  rinv = np.float32(1.0 / np.sqrt(np.sum(x * x, dtype=np.float32) / np.float32(4096.0) + np.float32(1.0e-6)))
  ref_norm = (x * rinv * w).astype(np.float32)
  xbuf, wbuf = make_buffer(4096, dtypes.float32), make_buffer(4096, dtypes.float32)
  norm_out, q8buf = make_buffer(4096, dtypes.float32), make_buffer(128 * 36, dtypes.uint8)
  copyin_array(xbuf, x); copyin_array(wbuf, w)
  prod_ms = []
  for i in range(args.warmups + args.iters):
    ms = float(norm_prg(norm_out._buf, q8buf._buf, xbuf._buf, wbuf._buf, global_size=(1,1,1),
                        local_size=(args.producer_threads,1,1), wait=True)) * 1000.0
    if i >= args.warmups: prod_ms.append(ms)
  got_norm = copyout_array(norm_out, np.empty(4096, dtype=np.float32))
  got_q8 = bytearray(128 * 36); q8buf.copyout(memoryview(got_q8))
  q8_x = q8_dequant(bytes(got_q8), 4096)

  q40, rows, k, shape0 = read_q4(args.gguf, "blk.0.ffn_gate.weight", args.rows)
  q41, rows1, k1, shape1 = read_q4(args.gguf, "blk.0.ffn_up.weight", args.rows)
  if rows != rows1 or k != k1: raise ValueError("gate/up shape mismatch")
  ref0, ref1 = q4_ref_rows(q40, rows, k, q8_x), q4_ref_rows(q41, rows, k, q8_x)
  q4b0, q4b1 = make_buffer(len(q40), dtypes.uint8), make_buffer(len(q41), dtypes.uint8)
  dst0, dst1 = make_buffer(rows, dtypes.float32), make_buffer(rows, dtypes.float32)
  q4b0.copyin(memoryview(q40)); q4b1.copyin(memoryview(q41))

  gateup_ms = []
  for i in range(args.warmups + args.iters):
    ms = float(gateup_prg(dst0._buf, dst1._buf, q4b0._buf, q4b1._buf, q8buf._buf,
                          global_size=(rows,2,1), local_size=(32,4,1), wait=True)) * 1000.0
    if i >= args.warmups: gateup_ms.append(ms)
  got0, got1 = copyout_array(dst0, np.empty(rows, dtype=np.float32)), copyout_array(dst1, np.empty(rows, dtype=np.float32))
  err0, err1 = np.abs(got0 - ref0), np.abs(got1 - ref1)
  lifecycle_us = (statistics.median(prod_ms) + statistics.median(gateup_ms)) * 1000.0
  return {
    "producer_threads": args.producer_threads,
    "shapes": {"gate": shape0, "up": shape1},
    "rows": rows,
    "producer": ms_stats(prod_ms),
    "gateup_consumer": ms_stats(gateup_ms),
    "gate_up_lifecycle_us": lifecycle_us,
    "correctness": {
      "producer_fp_max_abs": float(np.abs(got_norm - ref_norm).max()),
      "q8_dequant_max_abs": float(np.abs(q8_x - ref_norm).max()),
      "gate_max_abs": float(err0.max()),
      "gate_mean_abs": float(err0.mean()),
      "up_max_abs": float(err1.max()),
      "up_mean_abs": float(err1.mean()),
    },
    "gates": {
      "producer_correct": float(np.abs(got_norm - ref_norm).max()) <= 1e-5,
      "gate_correct": float(err0.max()) <= 2e-3,
      "up_correct": float(err1.max()) <= 2e-3,
      "gateup_lifecycle_lte_129p2us": lifecycle_us <= 129.2,
    },
  }

def write_perf(args) -> dict:
  dev = Device["AMD"]
  t0 = time.perf_counter()
  norm_lib = compile_hipcc_linked(hip_norm_source(args.producer_threads), args.arch)
  mmvq_lib = compile_hipcc_linked(HIP_MMVQ_SOURCE, args.arch)
  gateup_lib = compile_hipcc_linked(HIP_MMVQ_GATEUP_SOURCE, args.arch)
  norm_prg = dev.runtime("q8_rmsnorm_side_hipcc_linked", norm_lib)
  mmvq_prg = dev.runtime("q8_mmvq_hipcc_linked", mmvq_lib)
  gateup_prg = dev.runtime("q8_mmvq_gateup_hipcc_linked", gateup_lib)
  compile_s = time.perf_counter() - t0
  rows = [perf_one(args, norm_prg, mmvq_prg, t) for t in ("blk.0.ffn_gate.weight", "blk.0.ffn_up.weight")]
  separate_lifecycle_us = rows[0]["producer"]["median_ms"]*1000.0 + rows[0]["consumer"]["median_ms"]*1000.0 + rows[1]["consumer"]["median_ms"]*1000.0
  fused = perf_gateup(args, norm_prg, gateup_prg)
  maps = pathlib.Path("/proc/self/maps").read_text(errors="ignore")
  res = {
    "date": "2026-06-19",
    "phase": "A-F2",
    "arch": args.arch,
    "compiler": "HIP-style oracle device kernels, hipcc relocatable + ld.lld -shared, launched by tinygrad AMDProgram/HCQ",
    "compile_s": compile_s,
    "separate_consumers": rows,
    "separate_gate_up_lifecycle_us": separate_lifecycle_us,
    "fused_gateup": fused,
    "gate_up_lifecycle_us": fused["gate_up_lifecycle_us"],
    "target_lifecycle_us": 129.2,
    "no_hip_runtime_in_process": "libamdhip64.so" not in maps,
    "verdict": "PASS" if all(all(r["gates"].values()) for r in rows) and all(fused["gates"].values()) and "libamdhip64.so" not in maps else "FAIL",
  }
  args.perf_out.parent.mkdir(parents=True, exist_ok=True)
  args.perf_out.write_text(json.dumps(res, indent=2) + "\n")
  return res

def main() -> None:
  ap = argparse.ArgumentParser(description="A-F0/A-F2 q8 fast hipcc-linked artifact probe")
  ap.add_argument("gguf", type=pathlib.Path)
  ap.add_argument("--arch", default="gfx1100")
  ap.add_argument("--rows", type=int, default=12288)
  ap.add_argument("--seed", type=int, default=7)
  ap.add_argument("--warmups", type=int, default=10)
  ap.add_argument("--iters", type=int, default=30)
  ap.add_argument("--producer-threads", type=int, default=1024)
  ap.add_argument("--audit-out", type=pathlib.Path, default=pathlib.Path("bench/q8-ffn-handwritten-oracle/hipcc_object_audit.json"))
  ap.add_argument("--perf-out", type=pathlib.Path, default=pathlib.Path("bench/q8-ffn-handwritten-oracle/fast_artifact_perf.json"))
  args = ap.parse_args()
  audit = write_audit(args.audit_out, args.arch)
  perf = write_perf(args)
  print(json.dumps({"audit": audit["verdict"], "perf": perf["verdict"], "gate_up_lifecycle_us": perf["gate_up_lifecycle_us"]}, indent=2))
  if audit["verdict"] != "PASS" or perf["verdict"] != "PASS": raise SystemExit(1)

if __name__ == "__main__":
  main()
