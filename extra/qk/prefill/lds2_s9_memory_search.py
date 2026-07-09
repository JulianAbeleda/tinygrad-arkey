#!/usr/bin/env python3
"""Bounded S9 search over LDS2 PAD/memory-layout variants.

This search intentionally keeps register layout, lifecycle template, cadence, and wait policy at their defaults.
Every legal candidate must fit in 64 KiB LDS and pass correctness before timing.
"""
from __future__ import annotations

import argparse, json, os, pathlib, sys, time, traceback
from dataclasses import asdict
from typing import Iterable

import numpy as np

sys.path.insert(0, os.getcwd())

from tinygrad import Tensor, Device, TinyJit, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.engine.realize import Estimates
from tinygrad.helpers import colored
from tinygrad.uop.ops import KernelInfo, Ops, UOp

from extra.qk.prefill.hand_vs_generated_shape_matrix import _structure_metrics
from extra.qk.prefill.wmma import LDS2MemoryLayout, build_gemm_lds2, default_lds2_memory_layout
from extra.qk.timing_harness import add_clock_pin_arg, pinned_peak_from_env, set_clock_pin_env

ARTIFACT = pathlib.Path("bench/prefill-lds2-s9/memory-search.json")
PAD_VALUES = (0, 8, 16, 24, 32)
MAX_LDS_BYTES = 65536


def _tile_shape(wm: int, wn: int, waves_m: int, waves_n: int) -> tuple[int, int]:
  return waves_m * wm * 16, waves_n * wn * 16


def _layout_for_pad(bm: int, bn: int, bk: int, pad: int, dbuf: int) -> LDS2MemoryLayout:
  return default_lds2_memory_layout(bm, bn, bk, pad, dbuf)


def _memory_metrics(layout: LDS2MemoryLayout) -> dict[str, int]:
  return asdict(layout) | {"lds_bytes": layout.BUFSZ * layout.NBUF}


def candidate_proposals(wm: int, wn: int, waves_m: int, waves_n: int, bk: int, dbuf: int,
                        pads: Iterable[int] = PAD_VALUES) -> list[dict]:
  bm, bn = _tile_shape(wm, wn, waves_m, waves_n)
  out = []
  for pad in pads:
    rec = {"name": f"pad_{pad}", "pad": pad, "valid": False, "reason": "PAD-derived default LDS2MemoryLayout"}
    try:
      layout = _layout_for_pad(bm, bn, bk, pad, dbuf)
      rec["memory_layout"] = asdict(layout)
      rec.update(_memory_metrics(layout))
      layout.validate()
      rec["valid"] = True
    except Exception as e:
      rec["invalid_reason"] = str(e)
    out.append(rec)
  return out


def _kernel(m: int, n: int, k: int, wm: int, wn: int, waves_m: int, waves_n: int, bk: int, pad: int, dbuf: int,
            plrab: int, memory_layout: LDS2MemoryLayout):
  insts = build_gemm_lds2(m, n, k, waves_m, waves_n, wm, wn, bk, pad, dbuf, PLRAB=plrab,
                          memory_layout=memory_layout)
  bm, bn = _tile_shape(wm, wn, waves_m, waves_n)
  threads = waves_m * waves_n * 32
  lds_bytes = max(memory_layout.BUFSZ * memory_layout.NBUF, MAX_LDS_BYTES // 8)
  grid = (n // bn, m // bm, 1)
  name = f"hand_lds2_memory_s9_{m}_{n}_{k}_{wm}x{wn}_pad{pad}"

  def asm_kernel(A, Bt, C):
    lds = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(size=lds_bytes, addrspace=AddrSpace.LOCAL), (), "lds")
    g = [UOp.special(grid[0], "gidx0"), UOp.special(grid[1], "gidx1")]
    sink = UOp.sink(A.base, Bt.base, C.base, lds, *g, UOp.special(threads, "lidx0"),
                    arg=KernelInfo(name=colored(name, "cyan"),
                                   estimates=Estimates(ops=m*n*k*2, mem=(m*k+n*k+m*n)*2)))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT),
                                 UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=i) for i in insts))))
  cpr = bk // 8
  rstride = threads // cpr
  resource = {
    "tile_m": bm, "tile_n": bn, "threads": threads, "lds_bytes": memory_layout.BUFSZ * memory_layout.NBUF,
    "lds_alloc_bytes": lds_bytes, "loads_a": bm // rstride, "loads_b": bn // rstride,
  }
  return asm_kernel, resource | _memory_metrics(memory_layout) | _structure_metrics(insts)


def run_candidate(m: int, n: int, k: int, wm: int, wn: int, waves_m: int, waves_n: int, bk: int, pad: int, dbuf: int,
                  plrab: int, reps: int, iters: int, memory_layout: LDS2MemoryLayout) -> dict:
  row = {"shape": f"{wm}x{wn}", "wm": wm, "wn": wn, "status": "?", "tflops": 0.0}
  try:
    asm_kernel, resource = _kernel(m, n, k, wm, wn, waves_m, waves_n, bk, pad, dbuf, plrab, memory_layout)
    row.update(resource)
    rng = np.random.default_rng(0)
    a_np = (rng.standard_normal((m, k)) * 0.1).astype(np.float16)
    b_np = (rng.standard_normal((n, k)) * 0.1).astype(np.float16)
    ref = a_np.astype(np.float32) @ b_np.astype(np.float32).T
    refn = np.sqrt(np.mean(ref**2)) + 1e-9
    a, b = Tensor(a_np), Tensor(b_np)
    c = Tensor.empty(m, n, dtype=dtypes.half, device=a.device).contiguous()
    got = Tensor.custom_kernel(a, b, c, fxn=asm_kernel)[2].float().numpy()
    rr = float(np.sqrt(np.mean((got - ref)**2)) / refn)
    row["rel_rmse"] = rr
    if not np.isfinite(rr) or rr > 2e-2:
      row["status"] = f"WRONG rr={rr:.1e}"
      return row

    j = TinyJit(lambda: Tensor.custom_kernel(a, b, c, fxn=asm_kernel)[2].realize())
    dev = Device[Device.DEFAULT]
    with pinned_peak_from_env() as pin_prov:
      if pin_prov is not None: row["clock_pin"] = pin_prov
      for _ in range(5): j()
      dev.synchronize()
      ts = []
      for _ in range(reps):
        dev.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters): j()
        dev.synchronize()
        ts.append((time.perf_counter() - t0) / iters * 1e3)
    row["ms_min"] = round(min(ts), 4)
    row["tflops"] = round(2*m*n*k/min(ts)*1e-12*1e3, 2)
    row["status"] = "ok"
  except Exception as e:
    row["status"] = type(e).__name__
    row["message"] = str(e)
    row["traceback_tail"] = traceback.format_exc().splitlines()[-8:]
  return row


def _material_change(candidate: float, baseline: float, threshold: float) -> bool:
  return baseline > 0 and abs(candidate - baseline) / baseline >= threshold


def main(argv: Iterable[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--m", type=int, default=512)
  ap.add_argument("--n", type=int, default=12288)
  ap.add_argument("--k", type=int, default=4096)
  ap.add_argument("--wm", type=int, default=2)
  ap.add_argument("--wn", type=int, default=4)
  ap.add_argument("--waves-m", type=int, default=4)
  ap.add_argument("--waves-n", type=int, default=2)
  ap.add_argument("--bk", type=int, default=32)
  ap.add_argument("--dbuf", type=int, default=1)
  ap.add_argument("--plrab", type=int, default=1)
  ap.add_argument("--reps", type=int, default=2)
  ap.add_argument("--iters", type=int, default=5)
  ap.add_argument("--material-threshold", type=float, default=0.03)
  ap.add_argument("--artifact", default=str(ARTIFACT))
  ap.add_argument("--json", action="store_true")
  add_clock_pin_arg(ap)
  args = ap.parse_args(argv)

  set_clock_pin_env(os.environ, args.pin_clock)
  proposals = candidate_proposals(args.wm, args.wn, args.waves_m, args.waves_n, args.bk, args.dbuf)
  rows = []
  for idx, cand in enumerate(proposals):
    row = {"candidate_id": idx, **cand}
    if cand["valid"]:
      row.update(run_candidate(args.m, args.n, args.k, args.wm, args.wn, args.waves_m, args.waves_n, args.bk,
                               cand["pad"], args.dbuf, args.plrab, args.reps, args.iters,
                               LDS2MemoryLayout(**cand["memory_layout"])))
    else:
      row["status"] = "invalid-memory-layout"
      row["tflops"] = 0.0
    rows.append(row)

  baseline = next((r for r in rows if r["pad"] == 16), None)
  baseline_tflops = float(baseline.get("tflops", 0.0)) if baseline else 0.0
  ok_rows = [r for r in rows if r.get("status") == "ok"]
  best = max(ok_rows, key=lambda r: float(r.get("tflops", 0.0)), default=None)
  material = bool(best and _material_change(float(best.get("tflops", 0.0)), baseline_tflops, args.material_threshold))
  payload = {
    "schema": "prefill-lds2-s9-memory-search.v1",
    "shape": {"m": args.m, "n": args.n, "k": args.k, "wm": args.wm, "wn": args.wn,
              "waves_m": args.waves_m, "waves_n": args.waves_n, "bk": args.bk, "dbuf": args.dbuf,
              "plrab": args.plrab},
    "search_space": "memory_layout_pad_only_current_reg_layout_current_lifecycle_default_wait",
    "candidate_policy": "PAD in {0,8,16,24,32}; reject BUFSZ*NBUF > 65536 before correctness/timing",
    "pad_values": list(PAD_VALUES),
    "max_lds_bytes": MAX_LDS_BYTES,
    "material_threshold": args.material_threshold,
    "baseline_candidate_id": baseline.get("candidate_id") if baseline else None,
    "baseline_tflops": baseline_tflops,
    "best_candidate_id": best.get("candidate_id") if best else None,
    "best_tflops": float(best.get("tflops", 0.0)) if best else 0.0,
    "material_performance_change": material,
    "verdict": "S9_MEMORY_SEARCH_MATERIAL_CHANGE" if material else "S9_MEMORY_SEARCH_NO_MATERIAL_CHANGE",
    "rows": rows,
  }
  path = pathlib.Path(args.artifact)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2) + "\n")
  if args.json: print(json.dumps(payload, indent=2))
  else:
    print(f"{payload['verdict']} baseline={baseline_tflops:.2f} best={payload['best_tflops']:.2f} artifact={path}")
    for r in rows:
      print(f"  c{r['candidate_id']} pad={r['pad']} status={r.get('status')} tflops={r.get('tflops', 0.0)} rr={r.get('rel_rmse')} lds={r.get('lds_bytes')} layout={r.get('memory_layout')}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
