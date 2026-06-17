#!/usr/bin/env python3
"""Phase 6 input — characterize WHICH workload pairings overlap on two AMD compute rings (gfx1100, KFD).

Phase 3 showed compute||compute overlaps 2x. But decode is HBM-BANDWIDTH-bound (the GEMV regime), and two
bandwidth-bound kernels share the one HBM bus -- so they may NOT overlap even on two rings. This probe measures
all three pairings to ground the decode-overlap design:
  A = bandwidth-bound (grid-stride read-sum over a buffer >> 96MB Infinity Cache, few workgroups)
  B = compute-bound (dependent-FMA register loop, tiny memory)
  A||A : two bandwidth   -> expect ~1.0x (HBM is the shared bottleneck) == two decode GEMVs don't overlap
  A||B : bandwidth||compute -> expect >1.0x (compute hides under the bandwidth shadow) == the real opportunity
  B||B : two compute     -> expect ~2.0x (control, matches Phase 3)
GPU-clock spans (timestamps), warm, median. Requires DEV=AMD AMD_COMPUTE_RINGS=2. Run:
  DEV=AMD AMD_COMPUTE_RINGS=2 PYTHONPATH=. .venv/bin/python extra/amd_ring_overlap_characterize.py
"""
from __future__ import annotations

import json, os, pathlib, statistics, sys
import numpy as np

_EXT = '''
extern "C" __attribute__((device, const)) unsigned long __ockl_get_group_id(unsigned int);
extern "C" __attribute__((device, const)) unsigned int __ockl_get_local_id(unsigned int);
'''
def bw_src(grid:int) -> str:
  return _EXT + f'''extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(1,256)))
void bw(float* out, const float* in, unsigned int n) {{
  unsigned int gid=(unsigned int)__ockl_get_group_id(0)*256u+__ockl_get_local_id(0); unsigned int stride={grid}u*256u;
  float acc=0.0f; for(unsigned int i=gid;i<n;i+=stride) acc+=in[i]; out[gid]=acc; }}'''
def cu_src() -> str:
  return _EXT + '''extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(1,256)))
void cu(float* out, unsigned int iters) {
  unsigned int gid=(unsigned int)__ockl_get_group_id(0)*256u+__ockl_get_local_id(0);
  float a=(float)gid*1e-6f,b=1.0000001f; for(unsigned int i=0;i<iters;i++) a=a*b+1.0f; out[gid]=a; }'''

def main():
  from tinygrad.runtime.ops_amd import AMD_COMPUTE_RINGS
  from tinygrad.device import Device, Buffer
  from tinygrad.dtype import dtypes
  MB, GA, GB, ITERS, REPS, WARM = 768, 16, 48, 200000, 40, 12
  dev = Device["AMD"]
  if AMD_COMPUTE_RINGS < 2:
    print("REQUIRES AMD_COMPUTE_RINGS=2 (got %d)" % AMD_COMPUTE_RINGS); sys.exit(2)
  dev.compute_queue_desc(1)
  n = (MB << 20) // 4
  inbuf = Buffer("AMD", n, dtypes.float32).ensure_allocated()
  inbuf.copyin(memoryview(np.random.default_rng(0).standard_normal(n).astype(np.float32)))
  oA = Buffer("AMD", GA*256, dtypes.float32).ensure_allocated(); oA2 = Buffer("AMD", GA*256, dtypes.float32).ensure_allocated()
  oB = Buffer("AMD", GB*256, dtypes.float32).ensure_allocated(); oB2 = Buffer("AMD", GB*256, dtypes.float32).ensure_allocated()
  pA = dev.runtime("bw", dev.compiler.compile(bw_src(GA))); pB = dev.runtime("cu", dev.compiler.compile(cu_src()))
  A = (pA, (oA, inbuf), (GA,1,1), (n,)); A2 = (pA, (oA2, inbuf), (GA,1,1), (n,))
  B = (pB, (oB,), (GB,1,1), (ITERS,)); B2 = (pB, (oB2,), (GB,1,1), (ITERS,))

  def solo(task):
    prg, bufs, gs, vals = task; spans=[]
    for i in range(WARM+REPS):
      s0,s1,d = dev.new_signal(),dev.new_signal(),dev.new_signal()
      q = dev.hw_compute_queue_t(queue_idx=0).memory_barrier()
      q.timestamp(s0); q.exec(prg, prg.fill_kernargs(tuple(b._buf for b in bufs), vals), gs, (256,1,1)); q.timestamp(s1); q.signal(d,1).submit(dev)
      d.wait(1)
      if i>=WARM: spans.append(float(s1.timestamp)-float(s0.timestamp))
    return statistics.median(spans)

  def conc(t1, t2, rings):
    spans=[]
    for i in range(WARM+REPS):
      sA0,sA1,sB0,sB1 = dev.new_signal(),dev.new_signal(),dev.new_signal(),dev.new_signal(); dA,dB=dev.new_signal(),dev.new_signal()
      (p1,b1,g1,v1),(p2,b2,g2,v2)=t1,t2
      q1=dev.hw_compute_queue_t(queue_idx=rings[0]).memory_barrier()
      q1.timestamp(sA0); q1.exec(p1,p1.fill_kernargs(tuple(b._buf for b in b1),v1),g1,(256,1,1)); q1.timestamp(sA1); q1.signal(dA,1)
      q2=dev.hw_compute_queue_t(queue_idx=rings[1]).memory_barrier()
      q2.timestamp(sB0); q2.exec(p2,p2.fill_kernargs(tuple(b._buf for b in b2),v2),g2,(256,1,1)); q2.timestamp(sB1); q2.signal(dB,1)
      q1.submit(dev); q2.submit(dev); dA.wait(1); dB.wait(1)
      if i>=WARM: spans.append(max(float(sA1.timestamp),float(sB1.timestamp))-min(float(sA0.timestamp),float(sB0.timestamp)))
    return statistics.median(spans)

  # CONTROL-RELATIVE metric (robust to clock-drift + duration-mismatch): for each pairing, measure the same two
  # kernels on ONE ring (serial) vs TWO rings (0,1), back-to-back. overlap = one_ring_span / two_ring_span.
  def pairing(t1, t2):
    two = conc(t1, t2, (0, 1)); one = conc(t1, t2, (0, 0))
    return round(one / two, 3) if two else None
  sA, sB = solo(A), solo(B)
  rows = {"A||A_bw_bw": pairing(A, A2), "A||B_bw_compute": pairing(A, B), "B||B_compute_compute": pairing(B, B2)}
  out = {"arch": dev.arch, "solo_A_bw": round(sA,1), "solo_B_compute": round(sB,1),
         "metric": "one_ring_span / two_ring_span (>1 = two rings reclaim time; ~1 = no overlap)", "overlap": rows,
         "interpretation": {
           "A||A": "two occupancy-limited bandwidth GEMVs (decode-like): do they co-run on 2 rings?",
           "A||B": "bandwidth||compute: compute hiding under the memory shadow",
           "B||B": "two compute kernels: ~2x control (matches Phase 3)"}}
  for k,v in rows.items(): print(f"{k:24s}: {v}x  (one_ring/two_ring)")
  print(f"solo A(bw)={sA:.0f}  solo B(compute)={sB:.0f} device-clk")
  art = pathlib.Path("bench/amd-ring-overlap-characterize/result.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2)); print(f"artifact: {art}")

if __name__ == "__main__":
  main()
