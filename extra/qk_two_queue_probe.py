#!/usr/bin/env python3
"""Milestone 0 — two-queue overlap micro-prototype (gfx1100).

Decision gate for the decode-overlap build (docs/amd-decode-overlap-derisk-20260616.md):
can tinygrad issue two independent compute kernels to two AMD compute queues and get REAL
concurrent execution (wall_concurrent < t_A + t_B), intra-process? If yes -> the cross-layer
overlap scheduler is worth building; if they serialize -> KILL.

Two raw-HIP kernels (the extra/qk_flash_decode.py raw-C pattern):
  A (bandwidth, occupancy-limited): grid-stride read-sum over a >Infinity-Cache buffer with FEW
    workgroups -> runs at ~40-50% of HBM peak and leaves CUs idle (the decode-GEMV regime).
  B (compute-bound): dependent-FMA loop on registers -> needs CUs, ~no HBM -> the non-GEMV regime.

Measured on-device (GPU timestamps), confound-controlled (warm clock, many reps, median, big buf).
Sanity baselines anchor trust: A||A should NOT overlap (~1.0, both want HBM); B||B SHOULD (~2.0).

Standalone: consumes the runtime API, imports nothing from the decode path, changes no default.
"""
from __future__ import annotations

import argparse, statistics
import numpy as np

_EXT = '''
extern "C" __attribute__((device, const)) unsigned long __ockl_get_group_id(unsigned int);
extern "C" __attribute__((device, const)) unsigned int __ockl_get_local_id(unsigned int);
'''

def bw_src(grid:int) -> str:
  # bandwidth-bound: each thread strides over `in`, summing; few workgroups => occupancy-limited.
  return _EXT + f'''
extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(1,256)))
void bw(float* out, const float* in, unsigned int n) {{
  unsigned int gid = (unsigned int)__ockl_get_group_id(0)*256u + __ockl_get_local_id(0);
  unsigned int stride = {grid}u*256u;
  float acc = 0.0f;
  for (unsigned int i = gid; i < n; i += stride) acc += in[i];
  out[gid] = acc;
}}
'''

def cu_src() -> str:
  # compute-bound: dependent FMA chain on registers; `iters` tunes the cost; tiny memory traffic.
  return _EXT + '''
extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(1,256)))
void cu(float* out, unsigned int iters) {
  unsigned int gid = (unsigned int)__ockl_get_group_id(0)*256u + __ockl_get_local_id(0);
  float a = (float)gid * 1e-6f, b = 1.0000001f;
  for (unsigned int i = 0; i < iters; i++) a = a*b + 1.0f;
  out[gid] = a;
}
'''

def _ts(sig) -> float:
  return float(sig.timestamp)  # device clock units; ratios are unit-free, absolute used only for GB/s

def main():
  ap = argparse.ArgumentParser(description="two-queue overlap probe (gfx1100)")
  ap.add_argument("--mb", type=int, default=768, help="bandwidth buffer size in MiB (must exceed 96MB IC)")
  ap.add_argument("--grid-a", type=int, default=16, help="workgroups for kernel A (few => occupancy-limited)")
  ap.add_argument("--grid-b", type=int, default=512, help="workgroups for kernel B")
  ap.add_argument("--iters", type=int, default=24000, help="FMA iters for kernel B (tune so t_B ~ 0.3 t_A)")
  ap.add_argument("--reps", type=int, default=60)
  ap.add_argument("--warmup", type=int, default=20)
  ap.add_argument("--peak-gbs", type=float, default=859.0)
  args = ap.parse_args()

  from tinygrad.device import Device, Buffer
  from tinygrad.dtype import dtypes
  dev = Device["AMD"]

  n = (args.mb * 1024 * 1024) // 4
  rng = np.random.default_rng(0)
  inbuf = Buffer("AMD", n, dtypes.float32).ensure_allocated()
  inbuf.copyin(memoryview(np.ascontiguousarray(rng.standard_normal(n).astype(np.float32))))
  outA = Buffer("AMD", args.grid_a * 256, dtypes.float32).ensure_allocated()
  outA2 = Buffer("AMD", args.grid_a * 256, dtypes.float32).ensure_allocated()
  outB = Buffer("AMD", args.grid_b * 256, dtypes.float32).ensure_allocated()
  outB2 = Buffer("AMD", args.grid_b * 256, dtypes.float32).ensure_allocated()

  pA = dev.runtime("bw", dev.compiler.compile(bw_src(args.grid_a)))
  pB = dev.runtime("cu", dev.compiler.compile(cu_src()))

  gsA, lsA = (args.grid_a, 1, 1), (256, 1, 1)
  gsB, lsB = (args.grid_b, 1, 1), (256, 1, 1)

  def solo(prg, bufs, gs, ls, vals=()):
    ts = []
    raw = tuple(b._buf for b in bufs)
    for i in range(args.warmup + args.reps):
      t = prg(*raw, global_size=gs, local_size=ls, vals=vals, wait=True)
      if i >= args.warmup: ts.append(t)
    return statistics.median(ts)  # seconds

  def concurrent(prg1, bufs1, gs1, ls1, vals1, prg2, bufs2, gs2, ls2, vals2):
    """Launch two kernels on two compute queues with NO inter-queue wait; measure the GPU-clock
    span min(start)->max(end). Returns the span in device-clock units (ratios are unit-free)."""
    spans = []
    for i in range(args.warmup + args.reps):
      sA0, sA1, sB0, sB1 = dev.new_signal(), dev.new_signal(), dev.new_signal(), dev.new_signal()
      dA, dB = dev.new_signal(), dev.new_signal()
      kaA = prg1.fill_kernargs(tuple(b._buf for b in bufs1), vals1)
      kaB = prg2.fill_kernargs(tuple(b._buf for b in bufs2), vals2)
      q1 = dev.hw_compute_queue_t().memory_barrier()
      q1.timestamp(sA0); q1.exec(prg1, kaA, gs1, ls1); q1.timestamp(sA1); q1.signal(dA, 1)
      q2 = dev.hw_compute_queue_t().memory_barrier()
      q2.timestamp(sB0); q2.exec(prg2, kaB, gs2, ls2); q2.timestamp(sB1); q2.signal(dB, 1)
      q1.submit(dev); q2.submit(dev)
      dA.wait(1); dB.wait(1)
      if i >= args.warmup:
        span = max(_ts(sA1), _ts(sB1)) - min(_ts(sA0), _ts(sB0))
        spans.append(span)
    return statistics.median(spans)

  # solo timings
  tA = solo(pA, (outA, inbuf), gsA, lsA, (n,))
  tB = solo(pB, (outB,), gsB, lsB, (args.iters,))
  bw_gbs = (n * 4) / tA / 1e9
  print(f"kernel A (bandwidth): {tA*1e6:8.1f} us   {bw_gbs:6.1f} GB/s = {100*bw_gbs/args.peak_gbs:4.1f}% peak  (grid={args.grid_a})")
  print(f"kernel B (compute):   {tB*1e6:8.1f} us   t_B/t_A = {tB/tA:.2f}  (grid={args.grid_b}, iters={args.iters})")

  # device-clock unit calibration: time A both ways (solo seconds vs concurrent-span units) via A-with-tiny-B
  # We compute factors purely as ratios of concurrent spans, anchored by the solo ratio.
  span_AB = concurrent(pA, (outA, inbuf), gsA, lsA, (n,), pB, (outB,), gsB, lsB, (args.iters,))
  span_AA = concurrent(pA, (outA, inbuf), gsA, lsA, (n,), pA, (outA2, inbuf), gsA, lsA, (n,))
  span_BB = concurrent(pB, (outB,), gsB, lsB, (args.iters,), pB, (outB2,), gsB, lsB, (args.iters,))

  # convert spans (device units) to seconds using A: a solo-A span ~ tA. Calibrate k = tA / span_A_solo.
  span_A_solo = concurrent(pA, (outA, inbuf), gsA, lsA, (n,), pB, (outB,), (1,1,1), lsB, (0,))  # B≈no-op
  k = tA / span_A_solo if span_A_solo > 0 else 0.0
  ab_s, aa_s, bb_s = span_AB*k, span_AA*k, span_BB*k

  def factor(serial, wall): return serial / wall if wall > 0 else float('nan')
  print()
  # NOTE: tinygrad's AMD backend funnels ALL hw_compute_queue_t() builders into ONE hardware
  # compute ring (ops_amd.py:1001 self.compute_queue; _submit hardcodes it). So every pairing
  # below serializes (~1.0x) regardless of kernel shape — that uniformity IS the finding, and it
  # confirms the harness measures real ordering (a broken harness would show noise). Real overlap
  # needs a SECOND hardware compute ring (a [runtime] change), proven possible by the cross-process
  # +32% test. See docs/amd-decode-two-queue-probe-20260616.md.
  print(f"A||B  concurrent wall {ab_s*1e6:8.1f} us   serial(t_A+t_B) {(tA+tB)*1e6:8.1f} us   overlap factor {factor(tA+tB, ab_s):.2f}x")
  print(f"A||A  concurrent wall {aa_s*1e6:8.1f} us   serial(2*t_A)   {2*tA*1e6:8.1f} us   overlap factor {factor(2*tA, aa_s):.2f}x")
  print(f"B||B  concurrent wall {bb_s*1e6:8.1f} us   serial(2*t_B)   {2*tB*1e6:8.1f} us   overlap factor {factor(2*tB, bb_s):.2f}x  (small B that COULD overlap -> still ~1.0 = one-ring serialize)")
  print()
  f = factor(tA+tB, ab_s)
  reclaim = 100*(1 - ab_s/(tA+tB))
  print(f"VERDICT input: A||B overlap factor {f:.2f}x, reclaims {reclaim:.0f}% of the serial sum")

if __name__ == "__main__":
  main()
