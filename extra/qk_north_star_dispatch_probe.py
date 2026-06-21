#!/usr/bin/env python3
"""Audit probe: is the north-star FAIL_LOCAL_AB a dispatch-model confound, not kernel quality?

The local A/B timed coop as a batched TinyJit GRAPH (1 HCQ submit for 6 kernels) but the candidate as 2 UN-batched
raw dev.runtime dispatches with a per-iteration sync -> the candidate's ~flat ~180us total (does NOT scale with ctx,
combine delta DECREASES with ctx, streamk==serial) looks dispatch-latency-bound, not kernel-bound. This probe
re-times BOTH as THROUGHPUT (N back-to-back calls, ONE final sync) so kernels stream/overlap as they do in-model --
the fair comparison. If the candidate's throughput is competitive, the wall-clock latency A/B was confounded.

Run: DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_north_star_dispatch_probe.py
"""
from __future__ import annotations
import json, pathlib, statistics, sys, time
import numpy as np
from tinygrad import Tensor, Device, TinyJit
from tinygrad.device import Buffer
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp
from extra.qk_flash_decode import flash_reduce_src, flash_decode_attention
from extra.qk_decode_warp_flash_tile_ab import warp_tile_src, ref_attn
from extra.qk_clock_pin import pinned_peak

ROOT = pathlib.Path(__file__).resolve().parents[1]
Hd, Hq, Hkv, MAXC = 128, 32, 8, 4096
G = Hq // Hkv

def latency_us(fn, n=200):  # per-iteration sync (the original A/B method)
  Device["AMD"].synchronize(); ts = []
  for _ in range(n):
    t0 = time.perf_counter(); fn(); Device["AMD"].synchronize(); ts.append(time.perf_counter() - t0)
  return statistics.median(ts) * 1e6

def throughput_us(fn, n=300):  # back-to-back, ONE final sync (kernels stream/overlap)
  for _ in range(20): fn()
  Device["AMD"].synchronize()
  t0 = time.perf_counter()
  for _ in range(n): fn()
  Device["AMD"].synchronize()
  return (time.perf_counter() - t0) / n * 1e6

def main():
  dev = Device["AMD"]; rng = np.random.default_rng(0)
  q = rng.standard_normal((Hq, Hd)).astype(np.float16)
  k = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16)
  v = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16)
  qn, kn, vn = Tensor(q), Tensor(k), Tensor(v)
  rows = []
  with pinned_peak():
    time.sleep(0.4)
    for Tc in [512, 1024, 4096]:
      def buf(arr, dt):
        b = Buffer("AMD", arr.size, dt).ensure_allocated(); b.copyin(memoryview(np.ascontiguousarray(arr))); return b
      qb, kb, vb = buf(q, dtypes.half), buf(k, dtypes.half), buf(v, dtypes.half)
      vsp = UOp.variable("start_pos", 0, MAXC - 1)
      juop = TinyJit(lambda spb: flash_decode_attention(qn, kn, vn, spb + 1, vsp + 1, Hd, Hq, Hkv, MAXC, 128, variant="gqa_coop_vec").realize())
      for _ in range(8): juop(vsp.bind(Tc - 1))
      coop = lambda: juop(vsp.bind(Tc - 1))
      S = 64; per = (Tc + S - 1) // S
      pp = dev.runtime("warp_flash_tile", dev.compiler.compile(warp_tile_src(Hd, Hq, Hkv, S, MAXC, per)))
      prk = dev.runtime("flash_reduce", dev.compiler.compile(flash_reduce_src(Hd, Hq, Hkv, S, MAXC)))
      pout = Buffer("AMD", Hq * S * Hd, dtypes.float32).ensure_allocated()
      pm = Buffer("AMD", Hq * S, dtypes.float32).ensure_allocated(); pl = Buffer("AMD", Hq * S, dtypes.float32).ensure_allocated()
      ob = Buffer("AMD", Hq * Hd, dtypes.float32).ensure_allocated()
      def cand():
        pp(pout._buf, pm._buf, pl._buf, qb._buf, kb._buf, vb._buf, global_size=(Hkv * S, 1, 1), local_size=(128, 1, 1), vals=(Tc,), wait=False)
        prk(ob._buf, pout._buf, pm._buf, pl._buf, global_size=(Hq, 1, 1), local_size=(Hd, 1, 1), wait=False)
      cand(); dev.synchronize()
      _o = np.empty(Hq * Hd, np.float32); ob.copyout(memoryview(_o)); err = float(np.abs(_o.reshape(Hq, Hd) - ref_attn(q, k, v, Tc)).max())
      r = {"ctx": Tc, "S": S,
           "coop_latency_us": round(latency_us(coop), 1), "cand_latency_us": round(latency_us(cand), 1),
           "coop_throughput_us": round(throughput_us(coop), 1), "cand_throughput_us": round(throughput_us(cand), 1),
           "err": round(err, 5)}
      r["latency_speedup"] = round(r["coop_latency_us"] / r["cand_latency_us"], 3)
      r["throughput_speedup"] = round(r["coop_throughput_us"] / r["cand_throughput_us"], 3)
      rows.append(r)
      print(f"ctx{Tc}: LATENCY coop {r['coop_latency_us']:.1f} cand {r['cand_latency_us']:.1f} -> {r['latency_speedup']:.2f}x | "
            f"THROUGHPUT coop {r['coop_throughput_us']:.1f} cand {r['cand_throughput_us']:.1f} -> {r['throughput_speedup']:.2f}x err={err:.4f}", file=sys.__stderr__)
  out = ROOT / "bench/qk-north-star-flash-attn-tile/dispatch_probe.json"
  out.write_text(json.dumps({"date": "2026-06-21", "phase": "DISPATCH_MODEL_CONFOUND_PROBE", "rows": rows,
                             "note": "latency = per-iteration-sync (original A/B); throughput = back-to-back, kernels stream as in-model"}, indent=2))
  print(json.dumps({"out": str(out.relative_to(ROOT)),
                    "ctx1024_latency_speedup": next(r["latency_speedup"] for r in rows if r["ctx"] == 1024),
                    "ctx1024_throughput_speedup": next(r["throughput_speedup"] for r in rows if r["ctx"] == 1024)}, indent=2))

if __name__ == "__main__":
  main()
