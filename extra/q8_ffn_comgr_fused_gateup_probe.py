#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, time

from tinygrad.device import Device
from extra.q8_ffn_fast_artifact_probe import perf_gateup
from extra.q8_ffn_hcq_artifact import NORM_SOURCE

COMGR_MMVQ_GATEUP_SOURCE = r"""
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
void q8_mmvq_gateup(float *dst0, float *dst1, const block_q4_K *x0, const block_q4_K *x1, const block_q8_1 *y) {
  __attribute__((shared, aligned(16))) float sm[4];
  const int row = (int)__ockl_get_group_id(0);
  const int which = (int)__ockl_get_group_id(1);
  const block_q4_K *x = which == 0 ? x0 : x1;
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
  if (tid == 0) {
    const float s = sm[0] + sm[1] + sm[2] + sm[3];
    if (which == 0) dst0[row] = s; else dst1[row] = s;
  }
}
"""

def main() -> None:
  ap = argparse.ArgumentParser(description="B2a probe: tinygrad COMGR-owned fused q8 gate/up consumer")
  ap.add_argument("--gguf", type=pathlib.Path, default=pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  ap.add_argument("--rows", type=int, default=12288)
  ap.add_argument("--seed", type=int, default=7)
  ap.add_argument("--warmups", type=int, default=8)
  ap.add_argument("--iters", type=int, default=30)
  ap.add_argument("--producer-threads", type=int, default=256)
  ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bench/q8-ffn-codegen-transfer/comgr_fused_gateup.json"))
  args = ap.parse_args()

  dev = Device["AMD"]
  t0 = time.perf_counter()
  norm_prg = dev.runtime("q8_rmsnorm_side_comgr_b2a", dev.compiler.compile(NORM_SOURCE))
  gateup_prg = dev.runtime("q8_mmvq_gateup_comgr_b2a", dev.compiler.compile(COMGR_MMVQ_GATEUP_SOURCE))
  compile_s = time.perf_counter() - t0

  result = perf_gateup(args, norm_prg, gateup_prg)
  result.update({
    "date": "2026-06-19",
    "phase": "B2a_comgr_fused_gateup",
    "route": "tinygrad_COMGR_raw_C_fused_gateup_consumer",
    "compile_s": compile_s,
    "external_hipcc_lld_artifact": False,
    "gates": {
      **result["gates"],
      "consumer_lte_60us": result["gateup_consumer"]["median_ms"] <= 0.060,
      "no_external_artifact": True,
    },
  })
  result["verdict"] = "PASS" if all(result["gates"].values()) else "FAIL"
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps({"out": str(args.out), "verdict": result["verdict"], "gateup_us": result["gateup_consumer"]["median_ms"]*1000.0,
                    "lifecycle_us": result["gate_up_lifecycle_us"]}, indent=2))

if __name__ == "__main__":
  main()
