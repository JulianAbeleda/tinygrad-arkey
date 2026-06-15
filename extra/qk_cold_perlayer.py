#!/usr/bin/env python3
"""Isolate the decode e2e penalty: does the per-layer-size GEMV saturate COLD + launch-amortized?

The vdot kernel saturates at LARGE size (131072 rows = 76% peak), but e2e the per-layer GEMVs run at ~13%.
Per-layer working sets (8-28MB) fit in the 96MB Infinity Cache, so naive reps cache them (or launch
overhead deflates). To get the REAL cold per-layer bandwidth: a ~2GB backing buffer, each rep reads a
DIFFERENT region (rotating offset) -> every rep cold, 30 reps amortize launch overhead.

If per-layer-cold ~= 76% -> the kernel saturates at per-layer size -> e2e 13% is JIT-graph/launch overhead.
If per-layer-cold ~= 13-40% -> small kernels don't sustain -> the per-layer kernel itself is the problem.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_cold_perlayer.py
"""
from __future__ import annotations
import statistics, sys
import numpy as np
from tinygrad.device import Device, Buffer
from tinygrad.dtype import dtypes

LOCAL, K = 64, 4096
WPB = 36
RW = (K // 256) * WPB  # 576 words/row
PEAK = 859

SRC = '''
extern "C" __attribute__((device, const)) unsigned long __ockl_get_group_id(unsigned int);
extern "C" __attribute__((device, const)) unsigned int __ockl_get_local_id(unsigned int);
typedef unsigned int u4v __attribute__((ext_vector_type(4)));
extern "C" __attribute__((global)) __attribute__((target("dot-insts"))) void
__attribute__((amdgpu_flat_work_group_size(1,%d))) gemv(
    float* out, unsigned int* words, unsigned int* q8, unsigned int off) {
  unsigned int row = __ockl_get_group_id(0)*%d + __ockl_get_local_id(0);
  u4v* W4 = (u4v*)(words + off + row*%d);
  unsigned int a0=0,a1=0,a2=0,a3=0;
  for (int blk=0; blk<%d; blk++) { int b=blk*9;
    for (int i=0; i<9; i++) { u4v v=W4[b+i]; int ii=i;
      for (int j=0; j<4; j++) { unsigned int qw=v[j]; int p=(ii*4+j)*2;
        unsigned int lo=qw&0x0f0f0f0fu, hi=(qw>>4)&0x0f0f0f0fu;
        a0 = __builtin_amdgcn_udot4(lo, q8[p],   a0, false);
        a1 = __builtin_amdgcn_udot4(hi, q8[p+1], a1, false); } } }
  out[row] = (float)(a0+a1+a2+a3); }
''' % (LOCAL, LOCAL, RW, K // 256)


def main():
  dev = Device["AMD"]
  rng = np.random.default_rng(7)
  prg = dev.runtime("gemv", dev.compiler.compile(SRC))
  q8 = Buffer("AMD", 72, dtypes.uint32).ensure_allocated(); q8.copyin(memoryview(rng.integers(0, 2**32, 72, dtype=np.uint32)))
  TARGET_BYTES = 2 * 1024**3  # ~2GB backing buffer -> cold across reps (> 96MB cache)
  print(f"ROWS    layer       cold Q4-GB/s   %peak    us/call   (per-layer GEMV, cold+launch-amortized)")
  for ROWS, label in [(4096, "attn (8MB)"), (12288, "ffn (28MB)"), (131072, "large (300MB)")]:
    region = ROWS * RW                      # uint32 per region
    nreg = max(2, TARGET_BYTES // (region * 4))
    words = Buffer("AMD", region * nreg, dtypes.uint32).ensure_allocated()
    words.copyin(memoryview(rng.integers(0, 2**32, region * nreg, dtype=np.uint32)))
    out = Buffer("AMD", ROWS, dtypes.float32).ensure_allocated()
    q4_bytes = ROWS * RW * 4
    gs = (ROWS // LOCAL, 1, 1)
    prg(out._buf, words._buf, q8._buf, global_size=gs, local_size=(LOCAL, 1, 1), vals=(0,), wait=True)  # warm
    tms = []
    for r in range(40):
      off = (r % nreg) * region            # rotate -> each rep reads a DIFFERENT (cold) region
      tms.append(prg(out._buf, words._buf, q8._buf, global_size=gs, local_size=(LOCAL, 1, 1), vals=(off,), wait=True))
    t = statistics.median(tms); gbs = q4_bytes/t/1e9
    print(f"{ROWS:<7} {label:<11} {gbs:8.1f}      {gbs/PEAK*100:4.1f}%   {t*1e6:7.1f}   (nreg={nreg})", file=sys.__stdout__)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
