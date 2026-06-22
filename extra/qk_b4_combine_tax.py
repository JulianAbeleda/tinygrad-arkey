#!/usr/bin/env python3
"""B4 split-KV combine-tax attribution: per-kernel GPU-busy time (tile vs combine) by ctx x S, standalone (NO model
load). Answers whether the B4 W==D miss is the combine tax, over-splitting, or the Amdahl share.

Launches the B4 single-kernel ELFs (owned_flash_tile_gqa, owned_flash_combine) directly via tinygrad AMDProgram with
wait=True (signal-timestamp GPU time). Reports tile_us, combine_us, total_us, combine fraction, tile workgroups
(Hkv*S), combine bytes, per-ctx optimal S, and correctness vs numpy.

  run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_b4_combine_tax.py
"""
from __future__ import annotations
import json, statistics, pathlib, time
import numpy as np
from tinygrad import Device
from tinygrad.device import Buffer, BufferSpec
from tinygrad.dtype import dtypes
from tinygrad.runtime.ops_amd import AMDProgram
from extra.qk_owned_flash_decode_graph_node import _kernels, Hd, Hq, Hkv, G, SCALE

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-attention-route-b-b4-combine-tax"
MAXC = 4096
CTXS = [512, 1024, 2048, 4096]
SPLITS = [8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96]

def _buf(nbytes): return Buffer("AMD", nbytes, dtypes.uint8).ensure_allocated()

def _gpu_us(prog, bufs, grid, block, vals, n=40):
  ts = []
  for _ in range(8): prog(*bufs, global_size=grid, local_size=block, vals=vals, wait=True)  # warm
  for _ in range(n): ts.append(prog(*bufs, global_size=grid, local_size=block, vals=vals, wait=True) * 1e6)
  return statistics.median(ts)

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

  rows = []
  for ck in CTXS:
    nvalid = ck  # ctx == number of valid KV positions; kernel uses start_pos=nvalid-1 -> n_valid=nvalid
    start_pos = nvalid - 1
    # numpy GQA reference for correctness
    ref = np.zeros((Hq, Hd), np.float32)
    for h in range(Hq):
      kvh = h // G
      sc = (Q[h:h+1].astype(np.float32) @ Kf[kvh,:nvalid].astype(np.float32).T)[0]*SCALE
      p = np.exp(sc-sc.max()); p/=p.sum(); ref[h] = p @ Vf[kvh,:nvalid].astype(np.float32)
    for S in SPLITS:
      tile_elf, comb_elf, _, _ = _kernels(S, MAXC)
      tile = AMDProgram(dev, "owned_flash_tile_gqa", tile_elf)
      comb = AMDProgram(dev, "owned_flash_combine", comb_elf)
      bPart, bMeta, bOut = _buf(Hq*S*Hd*4), _buf(Hq*S*2*4), _buf(Hq*Hd*4)
      tg, tb = (Hkv, S, 1), (128, 1, 1)
      cg, cb = (Hq, 1, 1), (32, 1, 1)
      tile_bufs = (bQ._buf, bK._buf, bV._buf, bPart._buf, bMeta._buf)
      comb_bufs = (bPart._buf, bMeta._buf, bOut._buf)
      # correctness (run once)
      tile(*tile_bufs, global_size=tg, local_size=tb, vals=(start_pos,), wait=True)
      comb(*comb_bufs, global_size=cg, local_size=cb, vals=(), wait=True)
      ob = bytearray(Hq*Hd*4); bOut.copyout(memoryview(ob))
      out = np.frombuffer(bytes(ob), np.float32).reshape(Hq, Hd)
      rmse = float(np.sqrt(((out-ref)**2).mean())/(np.sqrt((ref**2).mean())+1e-9))
      tile_us = _gpu_us(tile, tile_bufs, tg, tb, (start_pos,))
      comb_us = _gpu_us(comb, comb_bufs, cg, cb, ())
      total = tile_us + comb_us
      comb_bytes = Hq*S*(Hd+2)*4 + Hq*Hd*4
      rows.append({"ctx": ck, "S": S, "tile_us": round(tile_us,2), "combine_us": round(comb_us,2),
                   "total_us": round(total,2), "combine_frac": round(comb_us/total,3), "tile_workgroups": Hkv*S,
                   "combine_bytes": comb_bytes, "rmse": round(rmse,2 if rmse>1 else 8),
                   "correct": rmse <= 1e-3})
    # per-ctx optimum
    ck_rows = [r for r in rows if r["ctx"] == ck]
    opt = min(ck_rows, key=lambda r: r["total_us"])
    print(f"ctx {ck:5}: opt S={opt['S']:3} total {opt['total_us']:6.2f}us "
          f"(tile {opt['tile_us']:.1f} + comb {opt['combine_us']:.1f}, comb {100*opt['combine_frac']:.0f}%, "
          f"wg {opt['tile_workgroups']})  | S48 total "
          f"{next(r['total_us'] for r in ck_rows if r['S']==48):.1f}us")

  opt_by_ctx = {ck: min((r for r in rows if r["ctx"]==ck), key=lambda r: r["total_us"])["S"] for ck in CTXS}
  out = {"date": "2026-06-21", "phase": "B4_SPLIT_KV_COMBINE_TAX_ATTRIBUTION", "maxc": MAXC,
         "contexts": CTXS, "splits": SPLITS, "comparator": "gqa_coop_vec (note: coop uses L=128 -> ~ctx/128 splits)",
         "method": "standalone per-kernel GPU-busy (AMDProgram wait=True signal timestamps), median-of-40, no model",
         "rows": rows, "optimal_S_by_ctx_min_total": opt_by_ctx,
         "note": "tile=owned_flash_tile_gqa, combine=owned_flash_combine; combine_bytes=Hq*S*(Hd+2)*4 read + Hq*Hd*4 write"}
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT/"latest.json").write_text(json.dumps(out, indent=2))
  print("optimal_S_by_ctx (min total attention):", opt_by_ctx)
  print(f"artifact: {OUT/'latest.json'}")

if __name__ == "__main__":
  main()
