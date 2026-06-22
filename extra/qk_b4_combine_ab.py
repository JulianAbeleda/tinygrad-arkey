#!/usr/bin/env python3
"""B5-lite cheaper-combine local A/B: base owned_flash_combine vs the owned_flash_combine_hd variants (thread-per-dim,
LDS-staged meta, 2D grid), per ctx x S. Standalone per-kernel GPU-busy (AMDProgram wait=True), no model.

Gate (local): new combine <= 8us @ S48/S64, correctness rel_rmse <= existing tol, no tile regression, total attention
improves. Emits bench/qk-decode-attention-route-b-b5-combine/latest.json.

  run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_b4_combine_ab.py
"""
from __future__ import annotations
import json, statistics, pathlib
import numpy as np
from tinygrad import Device
from tinygrad.device import Buffer
from tinygrad.dtype import dtypes
from tinygrad.runtime.ops_amd import AMDProgram
from extra.qk_owned_flash_decode_graph_node import (_compile, _specialize_tile, _specialize_combine, _combine_spec,
                                                    SRC, Hd, Hq, Hkv, G, SCALE)

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-attention-route-b-b5-combine"
MAXC = 4096
CTXS = [1024, 2048, 4096]
SPLITS = [32, 48, 56, 64]
VARIANTS = ["base", "hd64", "hw32", "hw64", "hw128"]
HBM_PEAK_GBs = 960.0

def _buf(nbytes): return Buffer("AMD", nbytes, dtypes.uint8).ensure_allocated()

def _gpu_us(prog, bufs, grid, block, vals, n=40):
  for _ in range(8): prog(*bufs, global_size=grid, local_size=block, vals=vals, wait=True)
  return statistics.median([prog(*bufs, global_size=grid, local_size=block, vals=vals, wait=True)*1e6 for _ in range(n)])

def _launch_floor(dev):
  # a trivial write-zero kernel measures the per-call wait=True launch/sync floor; in the JIT graph this is amortized,
  # so combine COMPUTE = standalone us - floor is the in-graph-relevant cost.
  triv = ('#include <hip/hip_runtime.h>\nextern "C" __global__ void cz(const float* part,const float* meta,'
          'float* out){ out[blockIdx.x*128+threadIdx.x]=0.f; }')
  pz = AMDProgram(dev, "cz", _compile(triv, "cz_probe"))
  b = [_buf(Hq*128*4)]*3
  return _gpu_us(pz, (b[0]._buf, b[1]._buf, b[2]._buf), (Hq,1,1), (128,1,1), ())

def main():
  assert Device.DEFAULT == "AMD"
  dev = Device["AMD"]
  rng = np.random.default_rng(0)
  Q  = rng.standard_normal((Hq, Hd)).astype(np.float16)
  Kf = (rng.standard_normal((Hkv, MAXC, Hd))*0.5).astype(np.float16)
  Vf = (rng.standard_normal((Hkv, MAXC, Hd))*0.5).astype(np.float16)
  bQ, bK, bV = _buf(Q.nbytes), _buf(Kf.nbytes), _buf(Vf.nbytes)
  bQ.copyin(memoryview(np.ascontiguousarray(Q))); bK.copyin(memoryview(np.ascontiguousarray(Kf)))
  bV.copyin(memoryview(np.ascontiguousarray(Vf)))
  src = SRC.read_text()
  floor = _launch_floor(dev); print(f"launch/sync floor (trivial kernel wait=True): {floor:.2f}us")

  rows = []
  for ck in CTXS:
    nvalid = ck; start_pos = nvalid - 1
    ref = np.zeros((Hq, Hd), np.float32)
    for h in range(Hq):
      kvh = h // G
      sc = (Q[h:h+1].astype(np.float32) @ Kf[kvh,:nvalid].astype(np.float32).T)[0]*SCALE
      p = np.exp(sc-sc.max()); p/=p.sum(); ref[h] = p @ Vf[kvh,:nvalid].astype(np.float32)
    for S in SPLITS:
      tile_elf = _compile(_specialize_tile(src, S, MAXC), f"tile_s{S}_m{MAXC}")
      tile = AMDProgram(dev, "owned_flash_tile_gqa", tile_elf)
      bPart, bMeta = _buf(Hq*S*Hd*4), _buf(Hq*S*2*4)
      tg, tb = (Hkv, S, 1), (128, 1, 1)
      tile_bufs = (bQ._buf, bK._buf, bV._buf, bPart._buf, bMeta._buf)
      tile(*tile_bufs, global_size=tg, local_size=tb, vals=(start_pos,), wait=True)  # fill part/meta once
      tile_us = _gpu_us(tile, tile_bufs, tg, tb, (start_pos,))
      var_rows = {}
      for v in VARIANTS:
        sym, defs, cg, cb = _combine_spec(v)
        comb_elf = _compile(_specialize_combine(src, S, MAXC, sym, defs), f"comb_{v}_s{S}_m{MAXC}")
        comb = AMDProgram(dev, sym, comb_elf)
        bOut = _buf(Hq*Hd*4)
        comb(bPart._buf, bMeta._buf, bOut._buf, global_size=cg, local_size=cb, vals=(), wait=True)
        ob = bytearray(Hq*Hd*4); bOut.copyout(memoryview(ob))
        out = np.frombuffer(bytes(ob), np.float32).reshape(Hq, Hd)
        rmse = float(np.sqrt(((out-ref)**2).mean())/(np.sqrt((ref**2).mean())+1e-9))
        comb_us = _gpu_us(comb, (bPart._buf, bMeta._buf, bOut._buf), cg, cb, ())
        compute = max(0.0, comb_us-floor)
        comb_bytes = Hq*S*(Hd+2)*4 + Hq*Hd*4
        var_rows[v] = {"combine_us": round(comb_us,2), "combine_compute_us": round(compute,2),
                       "rmse": round(rmse,8), "correct": rmse <= 1e-3, "grid": list(cg), "block": list(cb),
                       "workgroups": cg[0]*cg[1]*cg[2], "combine_bytes": comb_bytes,
                       "eff_bw_GBs": round(comb_bytes/(compute*1e-6)/1e9,1) if compute > 0 else None,
                       "eff_bw_pct_peak": round(100*comb_bytes/(compute*1e-6)/1e9/HBM_PEAK_GBs,1) if compute > 0 else None,
                       "combine_frac_of_total_compute": round(compute/(tile_us+compute),3),
                       "total_us": round(tile_us+comb_us,2)}
      base_us = var_rows["base"]["combine_us"]; base_cu = var_rows["base"]["combine_compute_us"]
      best_v = min((v for v in VARIANTS if v != "base" and var_rows[v]["correct"]),
                   key=lambda v: var_rows[v]["combine_us"], default="base")
      best_cu = var_rows[best_v]["combine_compute_us"]
      rows.append({"ctx": ck, "S": S, "tile_us": round(tile_us,2), "base_combine_us": base_us,
                   "base_combine_compute_us": base_cu, "best_variant": best_v,
                   "best_combine_us": var_rows[best_v]["combine_us"], "best_combine_compute_us": best_cu,
                   "combine_compute_speedup": round(base_cu/best_cu,2) if best_v!="base" and best_cu>0 else 1.0,
                   "variants": var_rows})
      b = rows[-1]
      print(f"ctx {ck:5} S{S:3}: tile {b['tile_us']:5.1f} | base comb {base_us:5.1f}us (compute {base_cu:4.1f}) -> "
            f"{best_v} {b['best_combine_us']:5.1f}us (compute {best_cu:4.1f}, {b['combine_compute_speedup']}x)")

  # local gate per ctx at the W==D-relevant split (S48 / S64@4096), launch-corrected combine COMPUTE.
  # Tiered targets (per principles): <=8us diagnostic/borderline, <=6-7us preferred for W==D, ~5us stretch w/ margin.
  gate_S = {1024: 48, 2048: 48, 4096: 64}
  gate_rows = [r for r in rows if r["S"] == gate_S[r["ctx"]]]
  worst_cu = max((r["best_combine_compute_us"] for r in gate_rows), default=99.0)
  tier = ("STRETCH(<=5us)" if worst_cu <= 5.0 else "PREFERRED(<=7us)" if worst_cu <= 7.0 else
          "DIAGNOSTIC(<=8us)" if worst_cu <= 8.0 else "FAIL(>8us)")
  gate_pass = worst_cu <= 7.0 and all(r["best_variant"] != "base" for r in gate_rows)  # preferred bar for W==D go
  out = {"date": "2026-06-22", "phase": "B5_CHEAPER_COMBINE_LOCAL_AB", "maxc": MAXC, "comparator": "base owned_flash_combine",
         "contexts": CTXS, "splits": SPLITS, "variants": VARIANTS, "launch_floor_us": round(floor,2),
         "method": "standalone per-kernel GPU-busy (AMDProgram wait=True), median-of-40, no model; compute = us - launch_floor",
         "rows": rows, "audit_classification": "COMBINE_TAX_DOMINATES",
         "targets": {"diagnostic_us": 8, "preferred_us": 7, "stretch_us": 5},
         "local_gate_S_by_ctx": gate_S, "worst_gate_combine_compute_us": round(worst_cu,2), "tier": tier,
         "local_gate_pass_preferred": gate_pass,
         "best_variant_by_gate_ctx": {r["ctx"]: r["best_variant"] for r in gate_rows}}
  OUT.mkdir(parents=True, exist_ok=True); (OUT/"latest.json").write_text(json.dumps(out, indent=2))
  print(f"\nlocal gate: worst combine compute {worst_cu:.2f}us -> {tier} | preferred-pass {gate_pass} | "
        f"best by ctx: {out['best_variant_by_gate_ctx']}\nartifact: {OUT/'latest.json'}")

if __name__ == "__main__":
  main()
