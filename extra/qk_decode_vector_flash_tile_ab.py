#!/usr/bin/env python3
"""VECTOR_FLASH_DECODE_TILE first gate (decode-vector-flash-tile-implementation-scope): does an LDS vector tile
with MANY KV-splits (llama's T=1 occupancy strategy) beat the current winner gqa_coop_vec by >=1.05x @ctx1024?

Lever #1 (cheapest): split-count sweep. The prior fused-LDS tile failed at a FIXED 8 splits (~64 workgroups,
occupancy-starved). llama uses ~48-144 parallel_blocks. Crank S (Hkv x S workgroups) and measure vs gqa_coop_vec.
Raw-C tile (flash_partial_lds + flash_reduce) for fast iteration; clock-pinned warm; byte-exact vs reference.
If no split count clears >=1.05x, classify and REST_DECODE. No model change.
"""
from __future__ import annotations
import json, pathlib, statistics, sys, time
import numpy as np
from tinygrad import Tensor, Device, TinyJit
from tinygrad.device import Buffer
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp
from extra.qk_flash_decode import flash_reduce_src, flash_decode_attention
from extra.qk_decode_fused_lds_tile_ab import flash_partial_lds_src
from extra.qk_clock_pin import pinned_peak

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-vector-flash-tile/split_sweep_ab.json"
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
  Device["AMD"].synchronize(); ts = []
  for _ in range(n):
    t0 = time.perf_counter(); fn(); Device["AMD"].synchronize(); ts.append(time.perf_counter() - t0)
  return statistics.median(ts) * 1e6

def main():
  dev = Device["AMD"]; rng = np.random.default_rng(0)
  q = rng.standard_normal((Hq, Hd)).astype(np.float16)
  k = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16)
  v = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16)
  qn, kn, vn = Tensor(q), Tensor(k), Tensor(v)
  results = []
  with pinned_peak():
    time.sleep(0.4)
    for Tc in [1024, 4096]:
      ref = ref_attn(q, k, v, Tc)
      def buf(arr, dt):
        b = Buffer("AMD", arr.size, dt).ensure_allocated(); b.copyin(memoryview(np.ascontiguousarray(arr))); return b
      qb, kb, vb = buf(q, dtypes.half), buf(k, dtypes.half), buf(v, dtypes.half)
      # gqa_coop_vec baseline (warm-JIT)
      vsp = UOp.variable("start_pos", 0, MAXC - 1)
      juop = TinyJit(lambda spb: flash_decode_attention(qn, kn, vn, spb + 1, vsp + 1, Hd, Hq, Hkv, MAXC, 128, variant="gqa_coop_vec").realize())
      for _ in range(8): juop(vsp.bind(Tc - 1))
      coop_us = time_fn(lambda: juop(vsp.bind(Tc - 1)))
      row = {"ctx": Tc, "gqa_coop_vec_us": round(coop_us, 1), "splits": []}
      for S in [8, 16, 32, 64, 96, 128]:
        per = (Tc + S - 1) // S  # keys per split (LDS tile size)
        try:
          pp = dev.runtime("flash_partial_lds", dev.compiler.compile(flash_partial_lds_src(Hd, Hq, Hkv, S, MAXC, per)))
          pr = dev.runtime("flash_reduce", dev.compiler.compile(flash_reduce_src(Hd, Hq, Hkv, S, MAXC)))
          pout = Buffer("AMD", Hq * S * Hd, dtypes.float32).ensure_allocated()
          pm = Buffer("AMD", Hq * S, dtypes.float32).ensure_allocated(); pl = Buffer("AMD", Hq * S, dtypes.float32).ensure_allocated()
          ob = Buffer("AMD", Hq * Hd, dtypes.float32).ensure_allocated()
          def run():
            pp(pout._buf, pm._buf, pl._buf, qb._buf, kb._buf, vb._buf, global_size=(Hkv * S, 1, 1), local_size=(Hd, 1, 1), vals=(Tc,), wait=False)
            pr(ob._buf, pout._buf, pm._buf, pl._buf, global_size=(Hq, 1, 1), local_size=(Hd, 1, 1), wait=False)
          run(); dev.synchronize()
          _o = np.empty(Hq * Hd, np.float32); ob.copyout(memoryview(_o)); err = float(np.abs(_o.reshape(Hq, Hd) - ref).max())
          us = time_fn(run); sp = coop_us / us if us else 0
          row["splits"].append({"S": S, "workgroups": Hkv * S, "keys_per_split": per, "tile_us": round(us, 1),
                                 "speedup_vs_coop": round(sp, 3), "err": round(err, 4)})
          print(f"  ctx{Tc} S={S:3} (wg={Hkv*S:4}, {per:3}k/split): tile {us:6.1f}us vs coop {coop_us:.1f} -> {sp:.2f}x err={err:.3f}", file=sys.__stderr__)
        except Exception as e:
          row["splits"].append({"S": S, "error": str(e)[:80]}); print(f"  ctx{Tc} S={S}: ERR {str(e)[:80]}", file=sys.__stderr__)
      best = max((s.get("speedup_vs_coop", 0) for s in row["splits"]), default=0)
      row["best_speedup_vs_coop"] = best
      results.append(row)
  gate = any(r["best_speedup_vs_coop"] >= 1.05 for r in results if r["ctx"] == 1024)
  out = {"date": "2026-06-21", "phase": "VECTOR_FLASH_TILE_SPLIT_SWEEP_GATE", "gate": ">=1.05x vs gqa_coop_vec @ctx1024",
         "results": results, "first_gate_pass": gate,
         "classification": None if gate else "split-count sweep alone does not clear 1.05x; per-thread q.k redundancy (128 d-threads recompute the full dot) bounds the tile -- next lever is warp/cooperative q.k + query-head column packing, OR REST_DECODE if that also misses.",
         "default_behavior_changed": False}
  OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(out, indent=2))
  print(json.dumps({"first_gate_pass": gate, "ctx1024_best": next(r["best_speedup_vs_coop"] for r in results if r["ctx"] == 1024), "out": str(OUT.relative_to(ROOT))}))

if __name__ == "__main__":
  main()
