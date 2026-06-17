#!/usr/bin/env python3
"""Phase 3 — prove AMD same-process two-ring compute overlap (gfx1100, KFD).

The prior single-ring probe (extra/qk_two_queue_probe.py) showed every same-process pairing serializes ~1.0x
because all hw_compute_queue_t() builders funnel into ONE hardware ring. With the opt-in 2nd ring
(AMD_COMPUTE_RINGS=2 + compute_queue_desc/queue_idx), this probe routes queue A to ring 0 and queue B to ring 1
and asks: do they actually run CONCURRENTLY?

Measurement is GPU-clock (kernel start/end timestamps via q.timestamp()), NOT wall-clock: span =
max(ends) - min(starts) over the two kernels; overlap factor = (solo_A + solo_B) / concurrent_span (device-clock
units, unit-free ratio). Confound-controlled: warm clock, many reps, median, workloads >> dispatch overhead.

Primary case B||B (two compute-bound kernels each using ~half the CUs -> two fit -> SHOULD overlap on two rings).
Control: the SAME pairing on ONE ring must stay ~1.0x (proves the harness measures real ordering). Acceptance:
two-ring overlap >1.2x AND one-ring control ~1.0x AND outputs correct AND no hang.

Requires: DEV=AMD AMD_COMPUTE_RINGS=2. Run:
  DEV=AMD AMD_COMPUTE_RINGS=2 PYTHONPATH=. .venv/bin/python extra/amd_two_compute_ring_probe.py
"""
from __future__ import annotations

import json, os, pathlib, statistics, sys
import numpy as np

_EXT = '''
extern "C" __attribute__((device, const)) unsigned long __ockl_get_group_id(unsigned int);
extern "C" __attribute__((device, const)) unsigned int __ockl_get_local_id(unsigned int);
'''
def cu_src() -> str:
  # compute-bound: dependent-FMA chain on registers; tiny memory traffic. Two of these (few workgroups each)
  # should run concurrently on distinct CUs if two rings really overlap.
  return _EXT + '''
extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(1,256)))
void cu(float* out, unsigned int iters) {
  unsigned int gid = (unsigned int)__ockl_get_group_id(0)*256u + __ockl_get_local_id(0);
  float a = (float)gid * 1e-6f, b = 1.0000001f;
  for (unsigned int i = 0; i < iters; i++) a = a*b + 1.0f;
  out[gid] = a;
}'''

def main():
  from tinygrad.runtime.ops_amd import AMD_COMPUTE_RINGS
  from tinygrad.device import Device, Buffer
  from tinygrad.dtype import dtypes
  GRID, ITERS, REPS, WARMUP = int(os.environ.get("GRID", 48)), int(os.environ.get("ITERS", 200000)), 50, 15
  dev = Device["AMD"]
  if AMD_COMPUTE_RINGS < 2:
    print("REQUIRES AMD_COMPUTE_RINGS=2 (got %d)" % AMD_COMPUTE_RINGS); sys.exit(2)
  dev.compute_queue_desc(1)  # force-create ring 1 (fails clearly if unsupported)

  outB  = Buffer("AMD", GRID * 256, dtypes.float32).ensure_allocated()
  outB2 = Buffer("AMD", GRID * 256, dtypes.float32).ensure_allocated()
  pB = dev.runtime("cu", dev.compiler.compile(cu_src()))
  gs, ls = (GRID, 1, 1), (256, 1, 1)

  def solo_span(out, iters):
    spans = []
    for i in range(WARMUP + REPS):
      s0, s1, d = dev.new_signal(), dev.new_signal(), dev.new_signal()
      ka = pB.fill_kernargs((out._buf,), (iters,))
      q = dev.hw_compute_queue_t(queue_idx=0).memory_barrier()
      q.timestamp(s0); q.exec(pB, ka, gs, ls); q.timestamp(s1); q.signal(d, 1).submit(dev)
      d.wait(1)
      if i >= WARMUP: spans.append(float(s1.timestamp) - float(s0.timestamp))
    return statistics.median(spans)

  def concurrent_span(ring_a, ring_b):
    spans = []
    for i in range(WARMUP + REPS):
      sA0, sA1, sB0, sB1 = dev.new_signal(), dev.new_signal(), dev.new_signal(), dev.new_signal()
      dA, dB = dev.new_signal(), dev.new_signal()
      kaA = pB.fill_kernargs((outB._buf,), (ITERS,)); kaB = pB.fill_kernargs((outB2._buf,), (ITERS,))
      qa = dev.hw_compute_queue_t(queue_idx=ring_a).memory_barrier()
      qa.timestamp(sA0); qa.exec(pB, kaA, gs, ls); qa.timestamp(sA1); qa.signal(dA, 1)
      qb = dev.hw_compute_queue_t(queue_idx=ring_b).memory_barrier()
      qb.timestamp(sB0); qb.exec(pB, kaB, gs, ls); qb.timestamp(sB1); qb.signal(dB, 1)
      qa.submit(dev); qb.submit(dev)
      dA.wait(1); dB.wait(1)
      if i >= WARMUP: spans.append(max(float(sA1.timestamp), float(sB1.timestamp)) - min(float(sA0.timestamp), float(sB0.timestamp)))
    return statistics.median(spans)

  solo = solo_span(outB, ITERS)
  two_ring = concurrent_span(0, 1)   # B on ring 0 || B on ring 1
  one_ring = concurrent_span(0, 0)   # control: both on ring 0 -> must serialize ~1.0x

  # correctness: cu output is deterministic; recompute on CPU for a few lanes
  got = np.empty(GRID * 256, np.float32); outB.copyout(memoryview(got)); exp_ok = True
  for gid in (0, 100, GRID * 256 - 1):
    a, b = gid * 1e-6, 1.0000001
    for _ in range(ITERS): a = a * b + 1.0
    exp_ok = exp_ok and bool(abs(got[gid] - a) <= 1e-2 * max(1.0, abs(a)))

  f_two = round(2 * solo / two_ring, 3) if two_ring else None
  f_one = round(2 * solo / one_ring, 3) if one_ring else None
  passes = bool(f_two and f_two > 1.2 and f_one and f_one < 1.15 and exp_ok)
  out = {"arch": dev.arch, "rings": AMD_COMPUTE_RINGS, "grid": GRID, "iters": ITERS,
         "solo_span": round(solo, 1), "two_ring_span": round(two_ring, 1), "one_ring_span": round(one_ring, 1),
         "two_ring_overlap_x": f_two, "one_ring_control_x": f_one, "outputs_correct": exp_ok, "passes": passes,
         "verdict": (f"PASS: B||B overlaps {f_two}x on two rings (control {f_one}x on one ring) -> same-process "
                     f"compute overlap is REAL; scope dependency semantics (Phase 4)" if passes else
                     f"REFUTED: two-ring {f_two}x / one-ring {f_one}x / correct={exp_ok} -> rings serialize or "
                     f"harness off; stop and report")}
  print(f"solo B span      : {solo:10.1f} (device clk)")
  print(f"B||B  two rings  : {two_ring:10.1f}  -> overlap {f_two}x")
  print(f"B||B  one ring   : {one_ring:10.1f}  -> control {f_one}x (want ~1.0)")
  print(f"outputs correct  : {exp_ok}")
  print(out["verdict"])
  art = pathlib.Path("bench/amd-two-compute-ring-probe/result.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2)); print(f"artifact: {art}")

if __name__ == "__main__":
  main()
