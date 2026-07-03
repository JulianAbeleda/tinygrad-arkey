#!/usr/bin/env python3
"""Phase 2 Candidate A (decode-latency-hiding scope): fully-fused flash-decode tile prototype A/B.

The fully-fused flash tile already exists as RAW C in extra/qk/flash_decode.py (flash_partial_src + flash_reduce_src):
ONE kernel does Q·K score + online softmax + V accumulation per (head,split) tile (2 kernels total incl. the split
reduce), vs the UOp-integrated `gqa_coop_vec` path which the linearizer forces into 6 kernels (score matmul +
flash_max/prob/partial/gmax/den/combine). The raw path is the latency-hiding ideal (qk/softmax/V interleaved, no
materialized score/stat tensors); the blocker is UOp integration (custom_kernel takes a UOp builder, not raw C).

This harness times BOTH at real decode shapes (Hq=32,Hkv=8,Hd=128) and contexts, with correctness vs numpy, to
decide PROCEED / ROADMAP / STOP: does the fused tile actually hide enough work to beat the optimized-but-split
coop path? Clock-pinned local diagnostic (NOT a product headline). Default decode behavior NOT changed.

Run: DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk/decode_fused_flash_tile_ab.py
"""
from __future__ import annotations
import json, pathlib, statistics, sys, time
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.device import Buffer
from extra.qk.harness_contract import time_fn as _hc_time_fn   # the one per-call timing loop
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp
from extra.qk.flash_decode import flash_partial_src, flash_reduce_src, flash_decode_attention
from extra.qk.clock_pin import pinned_peak

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-decode-latency-hiding-lifecycle/fused_flash_tile_ab.json"
Hd, Hq, Hkv, MAXC = 128, 32, 8, 4096
G = Hq // Hkv

def ref_attn(q, k, v, Tc):
  qf, kf, vf = q.astype(np.float32), k[:, :Tc].astype(np.float32), v[:, :Tc].astype(np.float32)
  out = np.zeros((Hq, Hd), np.float32)
  for h in range(Hq):
    kv = h // G; sc = (qf[h] @ kf[kv].T) / np.sqrt(Hd)
    pw = np.exp(sc - sc.max()); pw /= pw.sum(); out[h] = pw @ vf[kv]
  return out

def time_fn(fn, n=200):
  return statistics.median(_hc_time_fn(fn, n))  # median us over the shared per-call loop (harness_contract)

def main():
  dev = Device["AMD"]
  rng = np.random.default_rng(0)
  q = rng.standard_normal((Hq, Hd)).astype(np.float16)
  k = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16)
  v = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16)
  qn, kn, vn = Tensor(q), Tensor(k), Tensor(v)
  rows = []
  with pinned_peak() as pin:
    time.sleep(0.4)
    for Tc, L in [(1024, 128), (4096, 128)]:
      S = (Tc + L - 1) // L
      ref = ref_attn(q, k, v, Tc)
      # ---- RAW fully-fused tile (2 kernels: flash_partial does qk+softmax+V, flash_reduce combines splits) ----
      p_partial = dev.runtime("flash_partial", dev.compiler.compile(flash_partial_src(Hd, Hq, Hkv, S, MAXC)))
      p_reduce = dev.runtime("flash_reduce", dev.compiler.compile(flash_reduce_src(Hd, Hq, Hkv, S, MAXC)))
      def buf(arr, dt):
        b = Buffer("AMD", arr.size, dt).ensure_allocated(); b.copyin(memoryview(np.ascontiguousarray(arr))); return b
      qb, kb, vb = buf(q, dtypes.half), buf(k, dtypes.half), buf(v, dtypes.half)
      pout = Buffer("AMD", Hq * S * Hd, dtypes.float32).ensure_allocated()
      pm = Buffer("AMD", Hq * S, dtypes.float32).ensure_allocated(); pl = Buffer("AMD", Hq * S, dtypes.float32).ensure_allocated()
      ob = Buffer("AMD", Hq * Hd, dtypes.float32).ensure_allocated()
      def run_raw():
        p_partial(pout._buf, pm._buf, pl._buf, qb._buf, kb._buf, vb._buf, global_size=(Hq * S, 1, 1), local_size=(Hd, 1, 1), vals=(Tc,), wait=False)
        p_reduce(ob._buf, pout._buf, pm._buf, pl._buf, global_size=(Hq, 1, 1), local_size=(Hd, 1, 1), wait=False)
      run_raw(); dev.synchronize()
      _o = np.empty(Hq * Hd, np.float32); ob.copyout(memoryview(_o)); raw_err = float(np.abs(_o.reshape(Hq, Hd) - ref).max())
      raw_us = time_fn(run_raw)

      # ---- UOp gqa_coop_vec path (6 kernels, the in-model lifecycle) -- JIT it so timing is the WARM kernel
      # replay (comparable to the raw precompiled dispatch), NOT a fresh graph rebuild every call. ----
      from tinygrad import TinyJit
      vsp = UOp.variable("start_pos", 0, MAXC - 1)
      juop = TinyJit(lambda spb: flash_decode_attention(qn, kn, vn, spb + 1, vsp + 1, Hd, Hq, Hkv, MAXC, L, variant="gqa_coop_vec").realize())
      for _ in range(8): juop(vsp.bind(Tc - 1))
      uop_out = juop(vsp.bind(Tc - 1)).numpy(); uop_err = float(np.abs(uop_out - ref).max())
      uop_us = time_fn(lambda: juop(vsp.bind(Tc - 1)))

      speedup = uop_us / raw_us if raw_us else 0
      rows.append({"ctx": Tc, "L": L, "splits": S,
                   "raw_fused_us": round(raw_us, 1), "uop_coop_6k_us": round(uop_us, 1),
                   "fused_speedup": round(speedup, 3), "raw_kernels": 2, "uop_kernels": 6,
                   "raw_max_err": round(raw_err, 4), "uop_max_err": round(uop_err, 4)})
      print(f"  ctx{Tc} S={S}: raw-fused(2k) {raw_us:.1f}us err={raw_err:.3f} | uop-coop(6k) {uop_us:.1f}us err={uop_err:.3f} "
            f"| fused {speedup:.2f}x", file=sys.__stderr__)
    pin_prov = pin
  verdict = ("PROCEED_fused_hides_work" if all(r["fused_speedup"] >= 1.15 for r in rows)
             else "ROADMAP_fused_correct_but_not_faster_needs_optimized_fused_tile" if all(r["raw_max_err"] < 2e-2 for r in rows)
             else "STOP")
  out = {"date": "2026-06-21", "phase": "FUSED_FLASH_TILE_AB", "candidate": "A", "shapes": {"Hq": Hq, "Hkv": Hkv, "Hd": Hd},
         "method": "clock-pinned local diagnostic; raw C fully-fused flash (2 kernels) vs UOp gqa_coop_vec (6 kernels)",
         "rows": rows, "verdict": verdict, "clock_pin": pin_prov, "default_behavior_changed": False}
  OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(out, indent=2))
  print(json.dumps({"verdict": verdict, "rows": [(r["ctx"], r["fused_speedup"]) for r in rows], "out": str(OUT.relative_to(ROOT))}))

if __name__ == "__main__":
  main()
