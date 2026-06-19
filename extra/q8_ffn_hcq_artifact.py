#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, time

import numpy as np

from tinygrad import dtypes
from tinygrad.device import Buffer, Device
from extra.q8_ffn_handwritten_oracle import q4_ref_rows, tensor_info
from extra.qk_layout import Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, read_metadata

NORM_SOURCE = r"""
typedef unsigned short u16;
typedef unsigned int u32;
typedef struct { u16 d; u16 s; signed char qs[32]; } block_q8_1;

extern "C" __attribute__((device, const)) unsigned int __ockl_get_local_id(unsigned int);

static inline __attribute__((device)) u32 fbits(float x) { union { float f; u32 u; } v; v.f = x; return v.u; }
static inline __attribute__((device)) float ffrombits(u32 x) { union { float f; u32 u; } v; v.u = x; return v.f; }
static inline __attribute__((device)) float af(float x) { return x < 0.0f ? -x : x; }

static inline __attribute__((device)) u16 f32_to_f16(float f) {
  u32 x = fbits(f);
  u32 sign = (x >> 16) & 0x8000u;
  int exp = (int)((x >> 23) & 0xffu) - 127 + 15;
  u32 mant = x & 0x7fffffu;
  if (exp <= 0) {
    if (exp < -10) return (u16)sign;
    mant = (mant | 0x800000u) >> (1 - exp);
    return (u16)(sign | ((mant + 0x1000u) >> 13));
  }
  if (exp >= 31) return (u16)(sign | 0x7c00u);
  return (u16)(sign | ((u32)exp << 10) | ((mant + 0x1000u) >> 13));
}

extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(1, 256)))
void q8_rmsnorm_side(float *out, block_q8_1 *q8, const float *x, const float *w) {
  __attribute__((shared, aligned(16))) float red[256];
  const int tid = (int)__ockl_get_local_id(0);
  float ss = 0.0f;
  for (int i = tid; i < 4096; i += 256) ss += x[i] * x[i];
  red[tid] = ss;
  __builtin_amdgcn_s_barrier();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off) red[tid] += red[tid + off];
    __builtin_amdgcn_s_barrier();
  }
  const float rinv = 1.0f / __builtin_sqrtf(red[0] / 4096.0f + 1.0e-6f);
  for (int i = tid; i < 4096; i += 256) out[i] = x[i] * rinv * w[i];
  __builtin_amdgcn_s_barrier();
  for (int b = tid; b < 128; b += 256) {
    float vals[32];
    float mx = 0.0f;
    #pragma unroll
    for (int j = 0; j < 32; j++) {
      const int idx = b * 32 + j;
      vals[j] = x[idx] * rinv * w[idx];
      const float av = af(vals[j]);
      mx = mx > av ? mx : av;
    }
    const float scale = (mx == 0.0f) ? 1.0f : mx / 127.0f;
    q8[b].d = f32_to_f16(scale);
    q8[b].s = 0;
    #pragma unroll
    for (int j = 0; j < 32; j++) {
      float v = vals[j] / scale;
      int qi = (int)(v >= 0.0f ? v + 0.5f : v - 0.5f);
      qi = qi < -128 ? -128 : (qi > 127 ? 127 : qi);
      q8[b].qs[j] = (signed char)qi;
    }
  }
}
"""

MMVQ_SOURCE = r"""
#define QK_K 256
typedef unsigned short u16;
typedef unsigned int u32;
typedef struct { u16 d; u16 dmin; unsigned char scales[12]; unsigned char qs[128]; } block_q4_K;
typedef struct { u16 d; u16 s; signed char qs[32]; } block_q8_1;

extern "C" __attribute__((device, const)) unsigned int __ockl_get_local_id(unsigned int);
extern "C" __attribute__((device, const)) unsigned long __ockl_get_group_id(unsigned int);
extern "C" __attribute__((device, const)) float __ockl_wfred_add_f32(float);

static inline __attribute__((device)) void get_scale_min(int j, const unsigned char *q, unsigned char *d, unsigned char *m) {
  if (j < 4) { *d = q[j] & 63; *m = q[j+4] & 63; }
  else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4); *m = (q[j+4] >> 4) | ((q[j] >> 6) << 4); }
}

static inline __attribute__((device)) float ffrombits(u32 x) { union { float f; u32 u; } v; v.u = x; return v.f; }
static inline __attribute__((device)) float f16_to_f32(u16 h) {
  u32 sign = ((u32)h & 0x8000u) << 16;
  u32 exp = ((u32)h >> 10) & 0x1fu;
  u32 mant = (u32)h & 0x03ffu;
  if (exp == 0) {
    if (mant == 0) return ffrombits(sign);
    while ((mant & 0x0400u) == 0) { mant <<= 1; exp--; }
    exp++;
    mant &= 0x03ffu;
  } else if (exp == 31) {
    return ffrombits(sign | 0x7f800000u | (mant << 13));
  }
  exp = exp + (127 - 15);
  return ffrombits(sign | (exp << 23) | (mant << 13));
}

extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(1, 128)))
void q8_mmvq(float *dst, const block_q4_K *x, const block_q8_1 *y) {
  __attribute__((shared, aligned(16))) float sm[4];
  const int row = (int)__ockl_get_group_id(0);
  const int lx = (int)__ockl_get_local_id(0);
  const int ly = (int)__ockl_get_local_id(1);
  const int tid = ly * 32 + lx;
  float tmp = 0.0f;
  for (int kb = tid / 8; kb < 16; kb += 16) {
    const block_q4_K *bx = x + row * 16 + kb;
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
    const float d8 = f16_to_f32(by->d);
    const float dd = f16_to_f32(bx->d), dm = f16_to_f32(bx->dmin);
    tmp += d8 * (dd * (float)sc * (float)sumi - dm * (float)mn * (float)sumq);
  }
  tmp = __ockl_wfred_add_f32(tmp);
  if (lx == 0) sm[ly] = tmp;
  __builtin_amdgcn_s_barrier();
  if (tid == 0) dst[row] = sm[0] + sm[1] + sm[2] + sm[3];
}
"""

def make_buffer(n:int, dtype=dtypes.uint8) -> Buffer:
  return Buffer("AMD", n, dtype).ensure_allocated()

def copyin_array(buf:Buffer, arr:np.ndarray) -> None:
  buf.copyin(memoryview(np.ascontiguousarray(arr)))

def copyout_array(buf:Buffer, arr:np.ndarray) -> np.ndarray:
  buf.copyout(memoryview(arr))
  return arr

def q8_dequant(q8:bytes, n:int) -> np.ndarray:
  out = np.empty(n, dtype=np.float32)
  for bi in range(n // 32):
    off = bi * 36
    d = np.frombuffer(q8[off:off+2], dtype=np.float16).astype(np.float32)[0]
    q = np.frombuffer(q8[off+4:off+36], dtype=np.int8).astype(np.float32)
    out[bi*32:(bi+1)*32] = d * q
  return out

def main() -> None:
  parser = argparse.ArgumentParser(description="A1 HCQ launch proof for handwritten q8 FFN producer + consumer")
  parser.add_argument("gguf", type=pathlib.Path)
  parser.add_argument("--tensor", default="blk.0.ffn_gate.weight")
  parser.add_argument("--rows", type=int, default=12288)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bench/q8-ffn-handwritten-oracle/hcq_artifact.json"))
  args = parser.parse_args()

  dev = Device["AMD"]
  t0 = time.perf_counter()
  norm = dev.runtime("q8_rmsnorm_side", dev.compiler.compile(NORM_SOURCE))
  mmvq = dev.runtime("q8_mmvq", dev.compiler.compile(MMVQ_SOURCE))
  compile_s = time.perf_counter() - t0

  rng = np.random.default_rng(args.seed)
  x = (rng.standard_normal(4096).astype(np.float32) * 0.9).astype(np.float32)
  w = (0.7 + rng.random(4096).astype(np.float32) * 0.2).astype(np.float32)
  rinv = np.float32(1.0 / np.sqrt(np.sum(x * x, dtype=np.float32) / np.float32(4096.0) + np.float32(1.0e-6)))
  ref_norm = (x * rinv * w).astype(np.float32)

  xbuf, wbuf = make_buffer(4096, dtypes.float32), make_buffer(4096, dtypes.float32)
  outbuf, q8buf = make_buffer(4096, dtypes.float32), make_buffer(128 * 36, dtypes.uint8)
  copyin_array(xbuf, x)
  copyin_array(wbuf, w)
  norm(outbuf._buf, q8buf._buf, xbuf._buf, wbuf._buf, global_size=(1,1,1), local_size=(256,1,1), wait=True)
  got_norm = copyout_array(outbuf, np.empty(4096, dtype=np.float32))
  got_q8 = bytearray(128 * 36)
  q8buf.copyout(memoryview(got_q8))
  q8_x = q8_dequant(bytes(got_q8), 4096)

  meta = read_metadata(args.gguf)
  info, shape = tensor_info(meta, args.tensor)
  rows = min(args.rows, shape[0])
  if shape[1] != 4096: raise ValueError(f"A1 HCQ kernel is fixed at K=4096, got shape={shape}")
  row_bytes = shape[1] // Q4_K_BLOCK_ELEMS * Q4_K_BLOCK_BYTES
  with args.gguf.open("rb") as f:
    f.seek(meta.data_start + info.off)
    q4 = f.read(rows * row_bytes)
  ref_mmvq = q4_ref_rows(q4, rows, 4096, q8_x)

  q4buf, dstbuf = make_buffer(len(q4), dtypes.uint8), make_buffer(rows, dtypes.float32)
  q4buf.copyin(memoryview(q4))
  mmvq(dstbuf._buf, q4buf._buf, q8buf._buf, global_size=(rows,1,1), local_size=(32,4,1), wait=True)
  got_mmvq = copyout_array(dstbuf, np.empty(rows, dtype=np.float32))

  norm_abs = np.abs(got_norm - ref_norm)
  q8_abs = np.abs(q8_x - ref_norm)
  mmvq_abs = np.abs(got_mmvq - ref_mmvq)
  worst = int(np.argmax(mmvq_abs))
  result = {
    "date": "2026-06-19",
    "phase": "A1",
    "verdict": "PASS" if float(norm_abs.max()) <= 1e-5 and float(mmvq_abs.max()) <= 2e-3 else "FAIL",
    "compile_s": compile_s,
    "producer": {
      "n": 4096,
      "q8_bytes": len(got_q8),
      "fp_max_abs": float(norm_abs.max()),
      "q8_dequant_max_abs": float(q8_abs.max()),
    },
    "consumer": {
      "tensor": args.tensor,
      "shape": list(shape),
      "rows": rows,
      "q4_bytes": len(q4),
      "max_abs": float(mmvq_abs.max()),
      "mean_abs": float(mmvq_abs.mean()),
      "worst_row": worst,
      "worst_got": float(got_mmvq[worst]),
      "worst_ref": float(ref_mmvq[worst]),
    },
    "launch": {
      "runtime": "tinygrad AMD HCQ",
      "producer_global": [1,1,1],
      "producer_local": [256,1,1],
      "consumer_global": [rows,1,1],
      "consumer_local": [32,4,1],
      "hip_runtime_in_process": False,
    },
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))
  if result["verdict"] != "PASS": raise SystemExit(1)

if __name__ == "__main__":
  main()
